#import "decode_ane.h"
#import "decode_cpu.h"
#import "../../compiler/kernel_adapter.h"
#include "../../compiler/frontends/gpt2_decode.h"
#import "../../core/iosurface_tensor.h"
#import "../../core/ane_program_cache.h"
#import "../../core/kernel.h"
#import "../../model/configs/gpt2_124m.h"
#import <Accelerate/Accelerate.h>
#import <math.h>

// T101: ANE-accelerated decode step.
//
// Per layer: ANE decode_proj (LN1 + Q,K,V) → CPU attention → ANE decode_ffn (LN2 + FFN + residual)
// Final: CPU layernorm + logits on CPU (wte is too large for ANE).
//
// ANE tensors use [1, d_model, 1, ORION_DECODE_SEQ] layout. Token data at position 0.
// CPU attention reuses the same attention_decode_head logic from decode_cpu.m.

#pragma mark - IOSurface Helpers

/// Write x[d] into ANE surface [1, d, 1, seq] at position 0.
static void write_decode_input(IOSurfaceRef s, const float *x, int d, int seq, OrionIODtype dtype) {
    int count = d * seq;
    float *data = (float *)calloc(count, sizeof(float));
    for (int c = 0; c < d; c++) {
        data[c * seq + 0] = x[c];
    }
    orion_io_write_f32(s, data, count, dtype);
    free(data);
}

/// Read position 0 from ANE surface [1, d, 1, seq] into x[d].
static void read_decode_output(IOSurfaceRef s, float *x, int d, int seq, OrionIODtype dtype) {
    int count = d * seq;
    float *data = (float *)malloc(count * sizeof(float));
    orion_io_read_f32(s, data, count, dtype);
    for (int c = 0; c < d; c++) {
        x[c] = data[c * seq + 0];
    }
    free(data);
}

#pragma mark - Weight Dict Builders

static void add_blob(NSMutableDictionary *dict, NSString *mil_path, NSString *file_path) {
    NSData *data = [NSData dataWithContentsOfFile:file_path];
    if (data) {
        dict[mil_path] = @{@"offset": @0, @"data": data};
    }
}

/// Build weight dict for decode_proj: LN1 + Q,K,V projections (8 blobs).
static NSDictionary* build_decode_proj_wdict(int layer_idx, NSString *dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    const char *names[] = {"ln1_g", "ln1_b", "wq", "bq", "wk", "bk", "wv", "bv"};
    for (int i = 0; i < 8; i++) {
        NSString *mil_path = [NSString stringWithFormat:@"@model_path/layer%d/%s.bin",
                              layer_idx, names[i]];
        NSString *file_path = [NSString stringWithFormat:@"%@/layer%d/%s.bin",
                               dir, layer_idx, names[i]];
        add_blob(dict, mil_path, file_path);
    }
    return dict;
}

/// Build weight dict for decode_ffn: LN2 + FFN (6 blobs).
static NSDictionary* build_decode_ffn_wdict(int layer_idx, NSString *dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    const char *names[] = {"ln2_g", "ln2_b", "wfc", "bfc", "wproj", "bproj"};
    for (int i = 0; i < 6; i++) {
        NSString *mil_path = [NSString stringWithFormat:@"@model_path/layer%d/%s.bin",
                              layer_idx, names[i]];
        NSString *file_path = [NSString stringWithFormat:@"%@/layer%d/%s.bin",
                               dir, layer_idx, names[i]];
        add_blob(dict, mil_path, file_path);
    }
    return dict;
}

#pragma mark - OrionKernel Adapters

// Compiler MIL gen adapters — decode kernels ignore bucket
static NSString* compiler_decode_proj_adapter(int layer_idx, int bucket, const OrionModelConfig* cfg) {
    (void)bucket;
    return orion_kernel_adapter_generate_mil_2arg_io(orion_frontend_gpt2_decode_proj_io,
        layer_idx, cfg, orion_ane_io_policy()->dtype);
}

static NSString* compiler_decode_ffn_adapter(int layer_idx, int bucket, const OrionModelConfig* cfg) {
    (void)bucket;
    return orion_kernel_adapter_generate_mil_2arg_io(orion_frontend_gpt2_decode_ffn_io,
        layer_idx, cfg, orion_ane_io_policy()->dtype);
}

// Weight dict adapters
static NSDictionary* wdict_decode_proj_adapter(int layer_idx, int bucket, const char* blob_dir) {
    (void)bucket;
    return build_decode_proj_wdict(layer_idx, @(blob_dir));
}

static NSDictionary* wdict_decode_ffn_adapter(int layer_idx, int bucket, const char* blob_dir) {
    (void)bucket;
    return build_decode_ffn_wdict(layer_idx, @(blob_dir));
}

// Kernel instances — using compiler-generated MIL
static const OrionKernel kDecodeProj = {
    .name = "decode_proj",
    .generate_mil = compiler_decode_proj_adapter,
    .build_wdict = wdict_decode_proj_adapter,
    .n_inputs = 1, .n_outputs = 3,
};

static const OrionKernel kDecodeFFN = {
    .name = "decode_ffn",
    .generate_mil = compiler_decode_ffn_adapter,
    .build_wdict = wdict_decode_ffn_adapter,
    .n_inputs = 1, .n_outputs = 1,
};

#pragma mark - T101: ANE Decode Step

bool orion_ane_decode_step(const OrionGPT2Weights* w,
                            OrionKVCache* kv,
                            int token,
                            const char* blob_dir,
                            float* logits) {
    int d = w->d_model;
    int n_layer = w->n_layer;
    int n_head = kv->n_head;
    int head_dim = kv->head_dim;
    int vocab = w->vocab;
    int pos = kv->current_len;
    const OrionANEIO *policy = orion_ane_io_policy();
    if (!policy) return false;
    OrionIODtype dtype = policy->dtype;
    int seq = orion_io_decode_seq(dtype);
    OrionKernel proj = kDecodeProj, ffn = kDecodeFFN;
    proj.io_dtype = ffn.io_dtype = dtype;
    proj.pack_weights = ffn.pack_weights = policy->pack_weights;

    OrionModelConfig cfg = kGPT2_124M;
    OrionWeightsBinding wb = { .weights_id = "base", .bucket = seq };

    // x[d] — hidden state for single token
    float *x = (float *)malloc(d * sizeof(float));

    // Embed: wte[token] + wpe[pos]
    const float *tok_emb = w->wte + token * d;
    const float *pos_emb = w->wpe + pos * d;
    vDSP_vadd(tok_emb, 1, pos_emb, 1, x, 1, d);

    size_t layer_stride = (size_t)n_head * kv->max_seq * head_dim;

    // Reusable input surface
    IOSurfaceRef ioInput = orion_io_tensor_create(d, seq, dtype);

    for (int layer = 0; layer < n_layer; layer++) {
        const OrionGPT2LayerWeights *l = &w->layers[layer];

        // === ANE decode_proj: x → LN1 → Q, K, V ===
        write_decode_input(ioInput, x, d, seq, dtype);

        IOSurfaceRef ioQ = orion_io_tensor_create(d, seq, dtype);
        IOSurfaceRef ioK = orion_io_tensor_create(d, seq, dtype);
        IOSurfaceRef ioV = orion_io_tensor_create(d, seq, dtype);

        // ANE orders multi-output surfaces alphabetically by MIL variable name.
        // Both boundary formats sort K before Q before V.
        IOSurfaceRef proj_ins[]  = {ioInput};
        IOSurfaceRef proj_outs[] = {ioK, ioQ, ioV};
        bool ok = orion_kernel_eval(&proj, layer, seq, &cfg,
                                    blob_dir, &wb, proj_ins, 1, proj_outs, 3);
        if (!ok) {
            fprintf(stderr, "ANE decode: proj L%d failed\n", layer);
            CFRelease(ioInput); CFRelease(ioQ); CFRelease(ioK); CFRelease(ioV);
            free(x);
            return false;
        }

        // Extract Q[0], K_new[0], V_new[0] from position 0
        float *q     = (float *)malloc(d * sizeof(float));
        float *k_new = (float *)malloc(d * sizeof(float));
        float *v_new = (float *)malloc(d * sizeof(float));
        read_decode_output(ioQ, q, d, seq, dtype);
        read_decode_output(ioK, k_new, d, seq, dtype);
        read_decode_output(ioV, v_new, d, seq, dtype);
        CFRelease(ioQ); CFRelease(ioK); CFRelease(ioV);

        // Append K, V to cache
        orion_kv_cache_append(kv, layer, k_new, v_new);

        // === CPU multi-head attention ===
        int cache_len = pos + 1;
        float *attn_out = (float *)calloc(d, sizeof(float));
        float scale = 1.0f / sqrtf((float)head_dim);

        for (int h = 0; h < n_head; h++) {
            float *q_h = q + h * head_dim;
            float *k_cache_h = kv->k_cache + layer * layer_stride + h * kv->max_seq * head_dim;
            float *v_cache_h = kv->v_cache + layer * layer_stride + h * kv->max_seq * head_dim;

            // scores[cache_len] = scale * K_cache_h @ q_h
            float *scores = (float *)malloc(cache_len * sizeof(float));
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        cache_len, head_dim,
                        scale, k_cache_h, head_dim, q_h, 1,
                        0.0f, scores, 1);

            // Softmax
            float max_val = scores[0];
            for (int j = 1; j < cache_len; j++) {
                if (scores[j] > max_val) max_val = scores[j];
            }
            float sum = 0.0f;
            for (int j = 0; j < cache_len; j++) {
                scores[j] = expf(scores[j] - max_val);
                sum += scores[j];
            }
            for (int j = 0; j < cache_len; j++) {
                scores[j] /= sum;
            }

            // attn_h[head_dim] = scores @ V_cache_h
            cblas_sgemv(CblasRowMajor, CblasTrans,
                        cache_len, head_dim,
                        1.0f, v_cache_h, head_dim, scores, 1,
                        0.0f, attn_out + h * head_dim, 1);

            free(scores);
        }

        free(q); free(k_new); free(v_new);

        // Output projection: proj = attn_out @ Wo^T + bo
        // Wo stored [d, d] row-major. sgemv(NoTrans) gives Wo @ attn_out = attn_out @ Wo^T.
        float *proj_out = (float *)malloc(d * sizeof(float));
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    d, d, 1.0f, l->wo, d, attn_out, 1, 0.0f, proj_out, 1);
        // Add bias
        vDSP_vadd(proj_out, 1, l->bo, 1, proj_out, 1, d);

        // Residual: x += proj_out
        vDSP_vadd(x, 1, proj_out, 1, x, 1, d);
        free(proj_out);
        free(attn_out);

        // === ANE decode_ffn: x → LN2 → FFN → residual → hidden ===
        write_decode_input(ioInput, x, d, seq, dtype);

        IOSurfaceRef ioFFNOut = orion_io_tensor_create(d, seq, dtype);
        IOSurfaceRef ffn_ins[]  = {ioInput};
        IOSurfaceRef ffn_outs[] = {ioFFNOut};
        ok = orion_kernel_eval(&ffn, layer, seq, &cfg,
                               blob_dir, &wb, ffn_ins, 1, ffn_outs, 1);
        if (!ok) {
            fprintf(stderr, "ANE decode: FFN L%d failed\n", layer);
            CFRelease(ioInput); CFRelease(ioFFNOut);
            free(x);
            return false;
        }

        // Extract hidden[0] → x for next layer
        read_decode_output(ioFFNOut, x, d, seq, dtype);
        CFRelease(ioFFNOut);
    }

    CFRelease(ioInput);

    // Final LayerNorm (CPU — single vector, not worth ANE round-trip)
    orion_cpu_layernorm(x, w->ln_f_g, w->ln_f_b, d, x);

    // Logits: x @ wte^T → logits[vocab]
    // wte is [vocab, d]. sgemv(NoTrans) gives wte @ x = logits[v] = sum_c wte[v,c] * x[c].
    cblas_sgemv(CblasRowMajor, CblasNoTrans, vocab, d, 1.0f,
                w->wte, d, x, 1, 0.0f, logits, 1);

    kv->current_len = pos + 1;

    free(x);
    return true;
}

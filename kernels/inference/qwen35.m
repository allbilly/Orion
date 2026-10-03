#import "qwen35.h"
#import "../../compiler/frontends/qwen35.h"
#import "../../compiler/kernel_adapter.h"
#import "../../core/ane_io.h"
#import <Accelerate/Accelerate.h>
#include <math.h>
#include <time.h>

static const int D = 1024, F = 3584, V = 248320, L = 24, QKV = 6144, LINEAR = 2048, KV = 512;
static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000. + t.tv_nsec / 1e6;
}
static float sigmoid(float x) {
    return x >= 0 ? 1.f / (1.f + expf(-x)) : expf(x) / (1.f + expf(x));
}
static float silu(float x) { return x * sigmoid(x); }
static float softplus(float x) { return fmaxf(x, 0.f) + log1pf(expf(-fabsf(x))); }

void orion_qwen35_rmsnorm(const float *x, const float *w, float *y, int dim, float eps) {
    float sum = 0;
    for (int i = 0; i < dim; i++)
        sum += x[i] * x[i];
    float scale = 1.f / sqrtf(sum / dim + eps);
    for (int i = 0; i < dim; i++)
        y[i] = x[i] * scale * (w ? w[i] : 1.f);
}
void orion_qwen35_rope(float *x, int headDim, int rotaryDim, int position, float theta) {
    (void)headDim;
    for (int i = 0; i < rotaryDim / 2; i++) {
        float angle = position * powf(theta, -2.f * i / rotaryDim), c = cosf(angle),
              s = sinf(angle);
        float a = x[i], b = x[i + rotaryDim / 2];
        x[i] = a * c - b * s;
        x[i + rotaryDim / 2] = a * s + b * c;
    }
}
void orion_qwen35_delta_step(const float *q, const float *k, const float *v, float decay,
                             float beta, float *state, float *out, int dk, int dv) {
    for (int j = 0; j < dv; j++) {
        float *row = state + j * dk, estimate = 0;
        for (int i = 0; i < dk; i++) {
            row[i] *= decay;
            estimate += row[i] * k[i];
        }
        float delta = (v[j] - estimate) * beta, value = 0;
        for (int i = 0; i < dk; i++) {
            row[i] += k[i] * delta;
            value += row[i] * q[i];
        }
        out[j] = value;
    }
}
// Token-major matrices; weights are [out, in]. Column slicing preserves the
// original row stride, which matters for the CPU remainder of an ANE split.
static void linear(const float *x, const float *w, float *y, int n, int in, int out, int stride,
                   float accumulate) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, out, in, 1.f, x, in, w, stride,
                accumulate, y, out);
}
static NSString *layer_key(int layer, NSString *suffix) {
    return [NSString stringWithFormat:@"model.layers.%d.%@", layer, suffix];
}

@interface OrionQwenANE : NSObject {
  @public
    OrionProgram *program;
    IOSurfaceRef input, output, output2;
    OrionANEIO policy;
    int inDim, outDim, tile, channels;
}
- (BOOL)evaluate:(const float *)x count:(int)n output:(float *)y;
@end
@implementation OrionQwenANE
- (void)dealloc {
    if (program)
        orion_release_program(program);
    if (input)
        CFRelease(input);
    if (output)
        CFRelease(output);
    if (output2)
        CFRelease(output2);
}
- (BOOL)evaluate:(const float *)x count:(int)n output:(float *)y {
    if (n < 1 || n > tile)
        return NO;
    NSMutableData *a = [NSMutableData dataWithLength:(size_t)inDim * tile * sizeof(float)];
    NSMutableData *b = [NSMutableData dataWithLength:(size_t)outDim * tile * sizeof(float)];
    float *ap = a.mutableBytes, *bp = b.mutableBytes;
    // ANE [1,C,1,T] uses channel-major data. Pad only token-local projections;
    // padded positions never enter attention or recurrent state.
    for (int t = 0; t < n; t++)
        for (int c = 0; c < inDim; c++)
            ap[c * tile + t] = x[t * inDim + c];
    orion_io_write_f32(input, ap, inDim * tile, policy.dtype);
    IOSurfaceRef outputs[2] = {output, output2};
    if (!orion_eval(program, &input, 1, outputs, output2 ? 2 : 1))
        return NO;
    int surfaceDim = output2 ? outDim / 2 : outDim;
    orion_io_read_f32(output, bp, surfaceDim * tile, policy.dtype);
    if (output2)
        orion_io_read_f32(output2, bp + surfaceDim * tile, surfaceDim * tile, policy.dtype);
    for (int t = 0; t < n; t++)
        for (int c = 0; c < outDim; c++) {
            float value = bp[c * tile + t];
            if (!isfinite(value))
                return NO;
            y[t * outDim + c] = value;
        }
    return YES;
}
@end

static NSData *blob_slice(const float *w, int rows, int cols, int stride) {
    size_t bytes = (size_t)rows * cols * 2;
    NSMutableData *blob = [NSMutableData dataWithLength:128 + bytes];
    uint8_t *h = blob.mutableBytes;
    h[0] = 1;
    h[4] = 2;
    uint32_t magic = 0xdeadbeef, size = (uint32_t)bytes, offset = 128;
    memcpy(h + 64, &magic, 4);
    h[68] = 1;
    memcpy(h + 72, &size, 4);
    memcpy(h + 80, &offset, 4);
    __fp16 *p = (__fp16 *)(h + 128);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            p[r * cols + c] = (__fp16)w[r * stride + c];
    return blob;
}
static OrionQwenANE *compile_projection(const float *w, int in, int out, int tile,
                                        const OrionANEIO *p, const char *tag) {
    NSString *path = @"@model_path/weights/w.bin";
    OrionModelConfig cfg = {.d_model = in, .hidden_dim = out};
    NSString *mil = orion_kernel_adapter_generate_mil_io(orion_frontend_qwen35_projection_io, 0,
                                                         tile, &cfg, p->dtype);
    if (!mil)
        return nil;
    OrionQwenANE *a = [OrionQwenANE new];
    a->inDim = in;
    a->outDim = out;
    a->tile = tile;
    a->channels = out;
    a->policy = *p;
    a->program = orion_ane_io_compile(
        mil,
        @{path : @{@"offset" : @0, @"data" : blob_slice(w, out, in, in)}}, p, tag);
    if (!a->program)
        return nil;
    a->input = orion_io_tensor_create(in, tile, p->dtype);
    a->output = orion_io_tensor_create(out, tile, p->dtype);
    return a->input && a->output ? a : nil;
}
static OrionQwenANE *compile_mlp(const float *gate, const float *up, int channels, int tile,
                                 const OrionANEIO *p, const char *tag) {
    OrionModelConfig cfg = {.d_model = D, .hidden_dim = channels};
    NSString *mil = orion_kernel_adapter_generate_mil_io(orion_frontend_qwen35_gate_up_io, 0, tile,
                                                         &cfg, p->dtype);
    if (!mil)
        return nil;
    NSDictionary *weights = @{
        @"@model_path/weights/g.bin" :
            @{@"offset" : @0, @"data" : blob_slice(gate, channels, D, D)},
        @"@model_path/weights/u.bin" : @{@"offset" : @0, @"data" : blob_slice(up, channels, D, D)}
    };
    OrionQwenANE *a = [OrionQwenANE new];
    a->inDim = D;
    a->outDim = 2 * channels;
    a->tile = tile;
    a->channels = channels;
    a->policy = *p;
    a->program = orion_ane_io_compile(mil, weights, p, tag);
    if (!a->program)
        return nil;
    a->input = orion_io_tensor_create(D, tile, p->dtype);
    a->output = orion_io_tensor_create(channels, tile, p->dtype);
    a->output2 = orion_io_tensor_create(channels, tile, p->dtype);
    return a->input && a->output && a->output2 ? a : nil;
}

// Verify actual execution, weight binding, token/channel layout and both MLP
// outputs against independent CPU matrix products before accepting a program.
static BOOL verify_projection(OrionQwenANE *a, const float *w, const float *up) {
    int n = 2, width = a->channels;
    NSMutableData *xd = [NSMutableData dataWithLength:(size_t)n * a->inDim * 4];
    NSMutableData *yd = [NSMutableData dataWithLength:(size_t)n * a->outDim * 4];
    NSMutableData *refd = [NSMutableData dataWithLength:(size_t)n * a->outDim * 4];
    float *x = xd.mutableBytes, *y = yd.mutableBytes, *ref = refd.mutableBytes;
    for (int i = 0; i < n * a->inDim; i++)
        x[i] = .25f * sinf(i * .13f);
    if (![a evaluate:x count:n output:y])
        return NO;
    if (up) {
        NSMutableData *gd = [NSMutableData dataWithLength:(size_t)n * width * 4],
                      *ud = [NSMutableData dataWithLength:(size_t)n * width * 4];
        linear(x, w, gd.mutableBytes, n, a->inDim, width, a->inDim, 0);
        linear(x, up, ud.mutableBytes, n, a->inDim, width, a->inDim, 0);
        for (int t = 0; t < n; t++) {
            memcpy(ref + t * a->outDim, (float *)gd.mutableBytes + t * width, width * 4);
            memcpy(ref + t * a->outDim + width, (float *)ud.mutableBytes + t * width, width * 4);
        }
    } else
        linear(x, w, ref, n, a->inDim, a->outDim, a->inDim, 0);
    double error = 0, energy = 0;
    for (int i = 0; i < n * a->outDim; i++) {
        double delta = y[i] - ref[i];
        error += delta * delta;
        energy += (double)ref[i] * ref[i];
    }
    double relative = sqrt(error / fmax(energy, 1e-12));
    if (!isfinite(relative) || relative > .003) {
        fprintf(stderr, "qwen35: ANE projection verification failed (relative RMSE %.6f)\n",
                relative);
        return NO;
    }
    return YES;
}

@implementation OrionQwen35 {
    NSDictionary<NSString *, NSData *> *_weights;
    NSMutableArray<NSMutableData *> *_states, *_convs, *_keys, *_values;
    NSMutableDictionary<NSNumber *, OrionQwenANE *> *_mlpANE, *_gdnANE;
    float _eps, _theta;
    int _tile;
}
- (const float *)weight:(NSString *)name {
    return _weights[name].bytes;
}
- (const float *)layer:(int)layer weight:(NSString *)name {
    return [self weight:layer_key(layer, name)];
}
+ (instancetype)loadDirectory:(NSString *)directory maxContext:(int)context {
    if (context < 1 || context > 32768) {
        fprintf(stderr, "qwen35: context must be 1..32768\n");
        return nil;
    }
    NSData *data =
        [NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:@"manifest.json"]];
    NSDictionary *manifest =
        data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:nil] : nil;
    if (![manifest isKindOfClass:NSDictionary.class] ||
        ![manifest[@"config"] isKindOfClass:NSDictionary.class]) {
        fprintf(stderr, "qwen35: missing/invalid manifest in %s\n", directory.UTF8String);
        return nil;
    }
    NSDictionary *c = manifest[@"config"];
    NSDictionary *expected = @{
        @"hidden_size" : @1024,
        @"num_hidden_layers" : @24,
        @"num_attention_heads" : @8,
        @"num_key_value_heads" : @2,
        @"head_dim" : @256,
        @"intermediate_size" : @3584,
        @"vocab_size" : @248320,
        @"linear_num_key_heads" : @16,
        @"linear_num_value_heads" : @16,
        @"linear_key_head_dim" : @128,
        @"linear_value_head_dim" : @128,
        @"linear_conv_kernel_dim" : @4,
        @"full_attention_interval" : @4
    };
    if (![manifest[@"format"] isEqual:@"orion-qwen35-f32-v1"]) {
        fprintf(stderr, "qwen35: missing/unsupported manifest in %s\n", directory.UTF8String);
        return nil;
    }
    for (NSString *k in expected)
        if (![c[k] isEqual:expected[k]]) {
            fprintf(stderr, "qwen35: unsupported %s\n", k.UTF8String);
            return nil;
        }
    NSDictionary *rope = c[@"rope_parameters"] ?: c[@"rope_scaling"];
    float theta = [rope[@"rope_theta"] floatValue];
    if (theta == 0)
        theta = [c[@"rope_theta"] floatValue];
    float fraction = [rope[@"partial_rotary_factor"] floatValue];
    if (fraction == 0)
        fraction = [c[@"partial_rotary_factor"] floatValue];
    if (![c[@"tie_word_embeddings"] boolValue] || [c[@"num_experts"] intValue] ||
        fraction != 0.25f || theta != 10000000.f || [c[@"attention_bias"] boolValue] ||
        (c[@"attn_output_gate"] && ![c[@"attn_output_gate"] boolValue]) ||
        ![c[@"hidden_act"] isEqual:@"silu"] ||
        (rope[@"rope_type"] && ![rope[@"rope_type"] isEqual:@"default"])) {
        fprintf(stderr, "qwen35: unsupported embedding/rope configuration\n");
        return nil;
    }
    if (c[@"layer_types"]) {
        NSArray *types = c[@"layer_types"];
        if (types.count != L)
            return nil;
        for (int l = 0; l < L; l++)
            if (![types[l] isEqual:(l + 1) % 4 ? @"linear_attention" : @"full_attention"])
                return nil;
    }
    NSMutableDictionary *shapes = [NSMutableDictionary dictionary];
    shapes[@"model.embed_tokens.weight"] = @[ @(V), @(D) ];
    shapes[@"model.norm.weight"] = @[ @(D) ];
    for (int l = 0; l < L; l++) {
        NSDictionary *common = @{
            @"input_layernorm.weight" : @[ @(D) ],
            @"post_attention_layernorm.weight" : @[ @(D) ],
            @"mlp.gate_proj.weight" : @[ @(F), @(D) ],
            @"mlp.up_proj.weight" : @[ @(F), @(D) ],
            @"mlp.down_proj.weight" : @[ @(D), @(F) ]
        };
        for (NSString *k in common)
            shapes[layer_key(l, k)] = common[k];
        NSDictionary *specific = (l + 1) % 4
                                     ? @{
                                           @"linear_attn.in_proj_qkv.weight" : @[ @(QKV), @(D) ],
                                           @"linear_attn.in_proj_z.weight" : @[ @(LINEAR), @(D) ],
                                           @"linear_attn.in_proj_a.weight" : @[ @16, @(D) ],
                                           @"linear_attn.in_proj_b.weight" : @[ @16, @(D) ],
                                           @"linear_attn.out_proj.weight" : @[ @(D), @(LINEAR) ],
                                           @"linear_attn.conv1d.weight" : @[ @(QKV), @4, @1 ],
                                           @"linear_attn.norm.weight" : @[ @128 ],
                                           @"linear_attn.A_log" : @[ @16 ],
                                           @"linear_attn.dt_bias" : @[ @16 ]
                                       }
                                     : @{
                                           @"self_attn.q_proj.weight" : @[ @4096, @(D) ],
                                           @"self_attn.k_proj.weight" : @[ @(KV), @(D) ],
                                           @"self_attn.v_proj.weight" : @[ @(KV), @(D) ],
                                           @"self_attn.o_proj.weight" : @[ @(D), @(LINEAR) ],
                                           @"self_attn.q_norm.weight" : @[ @256 ],
                                           @"self_attn.k_norm.weight" : @[ @256 ]
                                       };
        for (NSString *k in specific)
            shapes[layer_key(l, k)] = specific[k];
    }
    NSMutableDictionary *weights = [NSMutableDictionary dictionary];
    for (NSString *name in shapes) {
        NSDictionary *entry = manifest[@"tensors"][name];
        NSString *file = entry[@"file"];
        size_t size = sizeof(float);
        for (NSNumber *v in shapes[name])
            size *= v.unsignedIntegerValue;
        if (![entry[@"shape"] isEqual:shapes[name]] || ![file isKindOfClass:NSString.class] ||
            ![file.lastPathComponent isEqual:file] || [file isEqual:@"."] || [file isEqual:@".."]) {
            fprintf(stderr, "qwen35: invalid tensor metadata %s\n", name.UTF8String);
            return nil;
        }
        NSData *mapped =
            [NSData dataWithContentsOfFile:[directory stringByAppendingPathComponent:file]
                                   options:NSDataReadingMappedAlways
                                     error:nil];
        if (mapped.length != size) {
            fprintf(stderr, "qwen35: wrong byte count for %s\n", name.UTF8String);
            return nil;
        }
        weights[name] = mapped;
    }
    OrionQwen35 *m = [self new];
    m->_weights = weights;
    m->_maxContext = context;
    m->_eps = [c[@"rms_norm_eps"] floatValue];
    m->_theta = theta;
    m->_tile = 256;
    if (!isfinite(m->_eps) || m->_eps <= 0)
        return nil;
    m->_states = [NSMutableArray array];
    m->_convs = [NSMutableArray array];
    m->_keys = [NSMutableArray array];
    m->_values = [NSMutableArray array];
    m->_mlpANE = [NSMutableDictionary dictionary];
    m->_gdnANE = [NSMutableDictionary dictionary];
    for (int l = 0; l < L; l++) {
        BOOL gdn = (l + 1) % 4;
        [m->_states
            addObject:[NSMutableData dataWithLength:gdn ? 16 * 128 * 128 * sizeof(float) : 0]];
        [m->_convs addObject:[NSMutableData dataWithLength:gdn ? 3 * QKV * sizeof(float) : 0]];
        [m->_keys addObject:[NSMutableData
                                dataWithLength:gdn ? 0 : (size_t)context * KV * sizeof(float)]];
        [m->_values addObject:[NSMutableData
                                  dataWithLength:gdn ? 0 : (size_t)context * KV * sizeof(float)]];
    }
    return m;
}
- (void)reset {
    _position = 0;
    _aneEvaluations = 0;
    _aneMLPEvaluations = 0;
    _aneGDNEvaluations = 0;
    _aneMilliseconds = 0;
    for (NSArray *list in @[ _states, _convs, _keys, _values ])
        for (NSMutableData *d in list)
            memset(d.mutableBytes, 0, d.length);
}
- (BOOL)enableANEWithMLPFraction:(float)mlp gdnFraction:(float)gdn tile:(int)tile {
    if (_anePrograms || _position || !isfinite(mlp) || !isfinite(gdn) || mlp < 0 || mlp > 1 ||
        gdn < 0 || gdn > 1 || (mlp == 0 && gdn == 0) || tile < 32 || tile > 256 || tile % 32)
        return NO;
    const OrionANEIO *policy = orion_ane_io_policy();
    if (!policy)
        return NO;
    // Quantize splits to multiples of 32, with a nonzero fraction always selecting a tile.
    int mf = mlp > 0 ? MIN(F, MAX(32, (int)lroundf(F * mlp / 32) * 32)) : 0;
    int gf = gdn > 0 ? MIN(QKV, MAX(32, (int)lroundf(QKV * gdn / 32) * 32)) : 0;
    NSMutableDictionary *mlps = [NSMutableDictionary dictionary],
                        *gdns = [NSMutableDictionary dictionary];
    for (int l = 0; l < L; l++) {
        @autoreleasepool {
            if (mf) {
                OrionQwenANE *a =
                    compile_mlp([self layer:l weight:@"mlp.gate_proj.weight"],
                                [self layer:l weight:@"mlp.up_proj.weight"], mf, tile, policy,
                                [NSString stringWithFormat:@"qwen35_mlp_%d", l].UTF8String);
                if (!a || !verify_projection(a, [self layer:l weight:@"mlp.gate_proj.weight"],
                                             [self layer:l weight:@"mlp.up_proj.weight"]))
                    return NO;
                mlps[@(l)] = a;
            }
            if (gf && (l + 1) % 4) {
                OrionQwenANE *a = compile_projection(
                    [self layer:l weight:@"linear_attn.in_proj_qkv.weight"], D, gf, tile, policy,
                    [NSString stringWithFormat:@"qwen35_gdn_qkv_%d", l].UTF8String);
                if (!a || !verify_projection(
                              a, [self layer:l weight:@"linear_attn.in_proj_qkv.weight"], NULL))
                    return NO;
                gdns[@(l)] = a;
            }
        }
    }
    _mlpANE = mlps;
    _gdnANE = gdns;
    _anePrograms = (int)(mlps.count + gdns.count);
    _tile = tile;
    _aneVerifications = _anePrograms;
    fprintf(stderr,
            "[qwen35-ane] compiled_programs=%d mlp_channels=%d/%d gdn_qkv_channels=%d/%d tile=%d "
            "io=%s\n",
            _anePrograms, mf, F, gf, QKV, tile, policy->dtype == ORION_IO_FP16 ? "fp16" : "fp32");
    return YES;
}
- (BOOL)runANE:(OrionQwenANE *)a input:(const float *)x count:(int)n output:(float *)y {
    double start = now_ms();
    BOOL ok = [a evaluate:x count:n output:y];
    _aneMilliseconds += now_ms() - start;
    if (ok) {
        _aneEvaluations++;
        if (a->output2)
            _aneMLPEvaluations++;
        else
            _aneGDNEvaluations++;
    } else
        fprintf(stderr, "qwen35: ANE evaluation failed\n");
    return ok;
}
- (BOOL)forwardTokens:(const int *)tokens count:(int)n logits:(float *)logits useANE:(BOOL)ane {
    if (!tokens || n < 1 || n > (ane ? _tile : 256) || _position + n > _maxContext || (ane && !_anePrograms))
        return NO;
    for (int t = 0; t < n; t++)
        if (tokens[t] < 0 || tokens[t] >= V)
            return NO;
    NSMutableData *hd = [NSMutableData dataWithLength:(size_t)n * D * 4],
                  *nd = [NSMutableData dataWithLength:(size_t)n * D * 4];
    NSMutableData *rd = [NSMutableData dataWithLength:(size_t)n * D * 4],
                  *pd = [NSMutableData dataWithLength:(size_t)n * QKV * 4];
    NSMutableData *zd = [NSMutableData dataWithLength:(size_t)n * LINEAR * 4],
                  *od = [NSMutableData dataWithLength:(size_t)n * LINEAR * 4];
    NSMutableData *ad = [NSMutableData dataWithLength:(size_t)n * 16 * 4],
                  *bd = [NSMutableData dataWithLength:(size_t)n * 16 * 4];
    float *h = hd.mutableBytes, *norm = nd.mutableBytes, *r = rd.mutableBytes,
          *proj = pd.mutableBytes;
    float *z = zd.mutableBytes, *out = od.mutableBytes, *av = ad.mutableBytes,
          *bv = bd.mutableBytes;
    const float *emb = [self weight:@"model.embed_tokens.weight"];
    for (int t = 0; t < n; t++)
        memcpy(h + t * D, emb + (size_t)tokens[t] * D, D * 4);
    for (int l = 0; l < L; l++) {
        @autoreleasepool {
            for (int t = 0; t < n; t++)
                orion_qwen35_rmsnorm(h + t * D, [self layer:l weight:@"input_layernorm.weight"],
                                     norm + t * D, D, _eps);
            if ((l + 1) % 4) {
                const float *w = [self layer:l weight:@"linear_attn.in_proj_qkv.weight"];
                OrionQwenANE *a = ane ? _gdnANE[@(l)] : nil;
                if (a) {
                    NSMutableData *part = [NSMutableData dataWithLength:(size_t)n * a->outDim * 4];
                    if (![self runANE:a input:norm count:n output:part.mutableBytes])
                        return NO;
                    int rest = QKV - a->outDim;
                    NSMutableData *cpu = [NSMutableData dataWithLength:(size_t)n * rest * 4];
                    if (rest)
                        linear(norm, w + (size_t)a->outDim * D, cpu.mutableBytes, n, D, rest, D, 0);
                    for (int t = 0; t < n; t++) {
                        memcpy(proj + t * QKV, (float *)part.mutableBytes + t * a->outDim,
                               a->outDim * 4);
                        if (rest)
                            memcpy(proj + t * QKV + a->outDim, (float *)cpu.mutableBytes + t * rest,
                                   rest * 4);
                    }
                } else
                    linear(norm, w, proj, n, D, QKV, D, 0);
                linear(norm, [self layer:l weight:@"linear_attn.in_proj_z.weight"], z, n, D, LINEAR,
                       D, 0);
                linear(norm, [self layer:l weight:@"linear_attn.in_proj_a.weight"], av, n, D, 16, D,
                       0);
                linear(norm, [self layer:l weight:@"linear_attn.in_proj_b.weight"], bv, n, D, 16, D,
                       0);
                const float *convW = [self layer:l weight:@"linear_attn.conv1d.weight"];
                const float *A = [self layer:l weight:@"linear_attn.A_log"],
                            *dt = [self layer:l weight:@"linear_attn.dt_bias"];
                float *history = _convs[l].mutableBytes, *state = _states[l].mutableBytes;
                float conv[6144];
                for (int t = 0; t < n; t++) {
                    for (int c = 0; c < QKV; c++) {
                        float value = proj[t * QKV + c] * convW[c * 4 + 3];
                        for (int j = 0; j < 3; j++)
                            value += history[j * QKV + c] * convW[c * 4 + j];
                        conv[c] = silu(value);
                    }
                    memmove(history, history + QKV, 2 * QKV * 4);
                    memcpy(history + 2 * QKV, proj + t * QKV, QKV * 4);
                    for (int head = 0; head < 16; head++) {
                        float *q = conv + head * 128, *k = conv + LINEAR + head * 128,
                              *v = conv + 2 * LINEAR + head * 128;
                        float qs = 1e-6f, ks = 1e-6f;
                        for (int i = 0; i < 128; i++) {
                            qs += q[i] * q[i];
                            ks += k[i] * k[i];
                        }
                        qs = 1.f / sqrtf(qs * 128.f);
                        ks = 1.f / sqrtf(ks);
                        for (int i = 0; i < 128; i++) {
                            q[i] *= qs;
                            k[i] *= ks;
                        }
                        float decay = expf(-expf(A[head]) * softplus(av[t * 16 + head] + dt[head]));
                        float *y = out + t * LINEAR + head * 128;
                        orion_qwen35_delta_step(q, k, v, decay, sigmoid(bv[t * 16 + head]),
                                                state + head * 128 * 128, y, 128, 128);
                        orion_qwen35_rmsnorm(y, [self layer:l weight:@"linear_attn.norm.weight"], y,
                                             128, _eps);
                        for (int i = 0; i < 128; i++)
                            y[i] *= silu(z[t * LINEAR + head * 128 + i]);
                    }
                }
                linear(out, [self layer:l weight:@"linear_attn.out_proj.weight"], r, n, LINEAR, D,
                       LINEAR, 0);
            } else {
                // Interleaved per-head Q/gate, two KV heads shared by four Q heads.
                linear(norm, [self layer:l weight:@"self_attn.q_proj.weight"], proj, n, D, 4096, D,
                       0);
                linear(norm, [self layer:l weight:@"self_attn.k_proj.weight"], z, n, D, KV, D, 0);
                linear(norm, [self layer:l weight:@"self_attn.v_proj.weight"], out, n, D, KV, D, 0);
                float *keys = _keys[l].mutableBytes, *values = _values[l].mutableBytes;
                NSMutableData *attnData = [NSMutableData dataWithLength:(size_t)n * LINEAR * 4];
                float *attn = attnData.mutableBytes;
                NSMutableData *scoreData =
                    [NSMutableData dataWithLength:(size_t)(_position + n) * 4];
                float *scores = scoreData.mutableBytes;
                for (int t = 0; t < n; t++) {
                    int position = _position + t;
                    for (int head = 0; head < 2; head++) {
                        float *k = keys + (size_t)position * KV + head * 256;
                        orion_qwen35_rmsnorm(z + t * KV + head * 256,
                                             [self layer:l weight:@"self_attn.k_norm.weight"], k,
                                             256, _eps);
                        orion_qwen35_rope(k, 256, 64, position, _theta);
                    }
                    memcpy(values + (size_t)position * KV, out + t * KV, KV * 4);
                    for (int head = 0; head < 8; head++) {
                        float q[256];
                        float *gate = proj + t * 4096 + head * 512 + 256;
                        orion_qwen35_rmsnorm(proj + t * 4096 + head * 512,
                                             [self layer:l weight:@"self_attn.q_norm.weight"], q,
                                             256, _eps);
                        orion_qwen35_rope(q, 256, 64, position, _theta);
                        float max = -INFINITY;
                        for (int p = 0; p <= position; p++) {
                            scores[p] =
                                cblas_sdot(256, q, 1, keys + (size_t)p * KV + (head / 4) * 256, 1) /
                                16.f;
                            max = fmaxf(max, scores[p]);
                        }
                        float sum = 0;
                        for (int p = 0; p <= position; p++) {
                            scores[p] = expf(scores[p] - max);
                            sum += scores[p];
                        }
                        float *y = attn + t * LINEAR + head * 256;
                        memset(y, 0, 256 * 4);
                        for (int p = 0; p <= position; p++)
                            cblas_saxpy(256, scores[p] / sum,
                                        values + (size_t)p * KV + (head / 4) * 256, 1, y, 1);
                        for (int i = 0; i < 256; i++)
                            y[i] *= sigmoid(gate[i]);
                    }
                }
                linear(attn, [self layer:l weight:@"self_attn.o_proj.weight"], r, n, LINEAR, D,
                       LINEAR, 0);
            }
            for (int i = 0; i < n * D; i++)
                h[i] += r[i];
            for (int t = 0; t < n; t++)
                orion_qwen35_rmsnorm(h + t * D,
                                     [self layer:l weight:@"post_attention_layernorm.weight"],
                                     norm + t * D, D, _eps);
            OrionQwenANE *a = ane ? _mlpANE[@(l)] : nil;
            int prefix = a ? a->channels : 0, rest = F - prefix;
            NSMutableData *g = [NSMutableData dataWithLength:(size_t)n * F * 4],
                          *u = [NSMutableData dataWithLength:(size_t)n * F * 4];
            float *gp = g.mutableBytes, *up = u.mutableBytes;
            if (a) {
                NSMutableData *part = [NSMutableData dataWithLength:(size_t)n * 2 * prefix * 4];
                if (![self runANE:a input:norm count:n output:part.mutableBytes])
                    return NO;
                float *pp = part.mutableBytes;
                for (int t = 0; t < n; t++) {
                    memcpy(gp + t * F, pp + t * 2 * prefix, prefix * 4);
                    memcpy(up + t * F, pp + t * 2 * prefix + prefix, prefix * 4);
                }
            }
            if (rest) {
                NSMutableData *cg = [NSMutableData dataWithLength:(size_t)n * rest * 4],
                              *cu = [NSMutableData dataWithLength:(size_t)n * rest * 4];
                linear(norm, [self layer:l weight:@"mlp.gate_proj.weight"] + (size_t)prefix * D,
                       cg.mutableBytes, n, D, rest, D, 0);
                linear(norm, [self layer:l weight:@"mlp.up_proj.weight"] + (size_t)prefix * D,
                       cu.mutableBytes, n, D, rest, D, 0);
                for (int t = 0; t < n; t++) {
                    memcpy(gp + t * F + prefix, (float *)cg.mutableBytes + t * rest, rest * 4);
                    memcpy(up + t * F + prefix, (float *)cu.mutableBytes + t * rest, rest * 4);
                }
            }
            for (int i = 0; i < n * F; i++)
                gp[i] = silu(gp[i]) * up[i];
            linear(gp, [self layer:l weight:@"mlp.down_proj.weight"], r, n, F, D, F, 0);
            for (int i = 0; i < n * D; i++) {
                h[i] += r[i];
                if (!isfinite(h[i])) {
                    fprintf(stderr, "qwen35: non-finite activation at layer %d\n", l);
                    return NO;
                }
            }
        }
    }
    if (logits) {
        orion_qwen35_rmsnorm(h + (n - 1) * D, [self weight:@"model.norm.weight"], norm, D, _eps);
        linear(norm, emb, logits, 1, D, V, D, 0);
        for (int i = 0; i < V; i++)
            if (!isfinite(logits[i]))
                return NO;
    }
    _position += n;
    return YES;
}
@end

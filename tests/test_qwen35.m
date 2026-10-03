#import "core/model_registry.h"
#import "kernels/inference/qwen35.h"
#import "tokenizer/gpt2_bpe.h"
#include <math.h>
#define CHECK(c)                                                                                   \
    do {                                                                                           \
        if (!(c)) {                                                                                \
            fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c);                                        \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)
int main(void) {
    @autoreleasepool {
        // Independent small delta-rule reference, including decay, correction and
        // readout. Nonzero initial state catches transposed-state implementations.
        float state[] = {.1, .2, .3, .4}, q[] = {.2, -.1}, k[] = {.6, .8}, v[] = {1, -.5}, out[2];
        orion_qwen35_delta_step(q, k, v, .9, .7, state, out, 2, 2);
        float expected[] = {.42684, .62912, -.129, -.172};
        for (int i = 0; i < 4; i++)
            CHECK(fabsf(state[i] - expected[i]) < 1e-6);
        CHECK(fabsf(out[0] - .022456f) < 1e-6 && fabsf(out[1] + .0086f) < 1e-6);
        orion_qwen35_delta_step(q, k, v, 0, 1, state, out, 2, 2);
        CHECK(fabsf(out[0] - .04f) < 1e-6 && fabsf(out[1] + .02f) < 1e-6);
        float x[] = {1, 2, -3, 4}, w[] = {1, .5, 2, 1}, y[4];
        orion_qwen35_rmsnorm(x, w, y, 4, 1e-6);
        for (int i = 0; i < 4; i++)
            CHECK(fabs(y[i] - x[i] * w[i] / sqrt(7.5 + 1e-6)) < 1e-6);
        orion_qwen35_rmsnorm(x, w, x, 4, 1e-6);
        for (int i = 0; i < 4; i++)
            CHECK(x[i] == y[i]);
        float rotary[256], original[256];
        for (int i = 0; i < 256; i++)
            rotary[i] = original[i] = sinf(i * .3f);
        orion_qwen35_rope(rotary, 256, 64, 37, 1e7);
        for (int i = 64; i < 256; i++)
            CHECK(rotary[i] == original[i]);
        for (int i = 0; i < 32; i++)
            CHECK(fabsf(rotary[i] * rotary[i] + rotary[i + 32] * rotary[i + 32] -
                        original[i] * original[i] - original[i + 32] * original[i + 32]) < 3e-7);
        const OrionModelSpec *spec = orion_model_lookup("qwen35_0_8b");
        CHECK(spec && spec->config.n_layer == 24 && spec->config.vocab == 248320);
        CHECK(![OrionQwen35 loadDirectory:@"/nonexistent/orion-qwen35" maxContext:0]);
        CHECK(![OrionQwen35 loadDirectory:@"/nonexistent/orion-qwen35" maxContext:32]);
        // A tiny byte-level vocabulary with sparse special-token IDs. Include all
        // UTF-8 bytes so Unicode round trips are tested without downloading weights.
        NSMutableDictionary *vocab = [NSMutableDictionary dictionary];
        int extra = 0;
        for (int b = 0; b < 256; b++) {
            unichar c =
                ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) ? b : 256 + extra++;
            vocab[[NSString stringWithFormat:@"%C", c]] = @(b);
        }
        vocab[@"he"] = @256;
        vocab[@"hel"] = @257;
        vocab[@"hell"] = @258;
        vocab[@"hello"] = @259;
        NSDictionary *json = @{
            @"model" : @{
                @"type" : @"BPE",
                @"vocab" : vocab,
                @"merges" :
                    @[ @[ @"h", @"e" ], @[ @"he", @"l" ], @[ @"hel", @"l" ], @[ @"hell", @"o" ] ]
            },
            @"added_tokens" : @[ @{
                @"content" : @"<|im_start|>",
                @"id" : @512,
                @"special" : @YES,
                @"lstrip" : @NO,
                @"rstrip" : @NO
            } ],
            @"pre_tokenizer" : @{
                @"pretokenizers" : @[
                    @{@"pattern" : @{@"Regex" : @"\\p{L}+|\\p{N}|[^\\p{L}\\p{N}]+"}}
                ]
            }
        };
        NSString *path = [NSTemporaryDirectory()
            stringByAppendingPathComponent:[NSUUID.UUID.UUIDString
                                               stringByAppendingString:@".json"]];
        CHECK([[NSJSONSerialization dataWithJSONObject:json options:0
                                                 error:nil] writeToFile:path atomically:YES]);
        OrionGPT2Tokenizer *tok = orion_bpe_tokenizer_load_json(path.UTF8String);
        [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
        CHECK(tok);
        int ids[100];
        int count = orion_gpt2_encode(tok, "hello<|im_start|>你好 😀123", ids, 100);
        CHECK(count > 2 && ids[0] == 259 && ids[1] == 512);
        char *text = orion_gpt2_decode(tok, ids, count);
        CHECK(!strcmp(text, "hello<|im_start|>你好 😀123"));
        free(text);
        CHECK(orion_gpt2_encode(tok, "hello hello", ids, 1) == -1);
        orion_gpt2_tokenizer_free(tok);
        puts("PASS Qwen3.5: delta state, RMSNorm, partial RoPE, Unicode/special BPE, bounds");
        return 0;
    }
}

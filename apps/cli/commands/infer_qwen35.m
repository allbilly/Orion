#import "../../../kernels/inference/decode_cpu.h"
#import "../../../kernels/inference/qwen35.h"
#import "../../../tokenizer/gpt2_bpe.h"
#import <Foundation/Foundation.h>
#include <math.h>
#include <time.h>

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}
static void help(void) {
    fprintf(stderr,
            "Usage: orion infer --model qwen35_0_8b --prompt TEXT [options]\n"
            "  --weights PATH         Export directory (default: model/blobs/qwen35_0_8b)\n"
            "  --tokenizer PATH       tokenizer.json (default: in export directory)\n"
            "  --dump-tokens PATH     Write actual prompt/generated IDs and stop token as JSON\n"
            "  --max_tokens N         Generated tokens (default: 128)\n"
            "  --context N            Cache capacity, 1..32768 (default: 4096)\n"
            "  --temperature FLOAT    Sampling temperature (default: 0, greedy)\n"
            "  --top_p FLOAT          Nucleus threshold (default: 0.9)\n"
            "  --seed N               Sampling seed (default: 42)\n"
            "  --raw                  Use prompt verbatim; default: single-turn chat\n"
            "  --thinking             Enable Qwen thinking in the chat template\n"
            "  --ane-prefill          ANE dense projections during prefill, CPU decode\n"
            "  --ane                  ANE dense projections during prefill and decode\n"
            "  --ane-mlp-fraction F   MLP channel fraction on ANE, 0..1 (default: 0.6)\n"
            "  --ane-gdn-fraction F   GDN QKV channel fraction on ANE (default: 0.25)\n"
            "  --ane-tile N           Token tile, multiples of 32, 32..256 (default: 32)\n"
            "  --help                 Show help\n");
}
static BOOL integer(const char *s, int *out) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!*s || *end || v < 0 || v > 32768)
        return NO;
    *out = (int)v;
    return YES;
}
static BOOL real(const char *s, float *out) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (!*s || *end || !isfinite(v))
        return NO;
    *out = v;
    return YES;
}
static int invalid(void) {
    fprintf(stderr, "qwen35: invalid or missing argument\n");
    help();
    return 1;
}

int orion_cmd_infer_qwen35(int argc, const char *argv[]) {
    NSString *directory = @"model/blobs/qwen35_0_8b", *tokenizerPath = nil, *prompt = nil;
    NSString *dumpPath = nil;
    int maximum = 128, context = 4096, tile = 32;
    float temperature = 0, topP = .9f, mf = .6f, gf = .25f;
    uint64_t seed = 42;
    BOOL ane = NO, decodeANE = NO, raw = NO, thinking = NO;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--help")) {
            help();
            return 0;
        }
        if (!strcmp(arg, "--ane")) {
            ane = YES;
            decodeANE = YES;
            continue;
        }
        if (!strcmp(arg, "--ane-prefill")) {
            ane = YES;
            decodeANE = NO;
            continue;
        }
        if (!strcmp(arg, "--raw")) {
            raw = YES;
            continue;
        }
        if (!strcmp(arg, "--thinking")) {
            thinking = YES;
            continue;
        }
        if (i + 1 >= argc)
            return invalid();
        const char *value = argv[++i];
        if (!strcmp(arg, "--model")) {
            if (strcmp(value, "qwen35_0_8b"))
                return invalid();
        } else if (!strcmp(arg, "--prompt"))
            prompt = @(value);
        else if (!strcmp(arg, "--weights"))
            directory = @(value);
        else if (!strcmp(arg, "--tokenizer"))
            tokenizerPath = @(value);
        else if (!strcmp(arg, "--dump-tokens"))
            dumpPath = @(value);
        else if (!strcmp(arg, "--max_tokens")) {
            if (!integer(value, &maximum))
                return invalid();
        } else if (!strcmp(arg, "--context")) {
            if (!integer(value, &context))
                return invalid();
        } else if (!strcmp(arg, "--ane-tile")) {
            if (!integer(value, &tile))
                return invalid();
        } else if (!strcmp(arg, "--temperature")) {
            if (!real(value, &temperature))
                return invalid();
        } else if (!strcmp(arg, "--top_p")) {
            if (!real(value, &topP))
                return invalid();
        } else if (!strcmp(arg, "--ane-mlp-fraction")) {
            if (!real(value, &mf))
                return invalid();
        } else if (!strcmp(arg, "--ane-gdn-fraction")) {
            if (!real(value, &gf))
                return invalid();
        } else if (!strcmp(arg, "--seed")) {
            char *end;
            seed = strtoull(value, &end, 10);
            if (!*value || *end || *value == '-')
                return invalid();
        } else
            return invalid();
    }
    if (!prompt || !prompt.length || maximum < 1 || context < 1 || temperature < 0 || topP <= 0 ||
        topP > 1 || mf < 0 || mf > 1 || gf < 0 || gf > 1 || tile < 32 || tile > 256 || tile % 32 ||
        (ane && mf == 0 && gf == 0))
        return invalid();
    if (!raw)
        prompt = [NSString
            stringWithFormat:@"<|im_start|>user\n%@<|im_end|>\n<|im_start|>assistant\n%@", prompt,
                             thinking ? @"<think>\n" : @"<think>\n\n</think>\n\n"];
    OrionGPT2Tokenizer *tok = orion_bpe_tokenizer_load_json(
        (tokenizerPath ?: [directory stringByAppendingPathComponent:@"tokenizer.json"]).UTF8String);
    if (!tok) {
        fprintf(stderr, "qwen35: cannot load byte-level tokenizer.json\n");
        return 1;
    }
    NSMutableData *tokenData = [NSMutableData dataWithLength:(size_t)context * sizeof(int)];
    int *tokens = tokenData.mutableBytes;
    int count = orion_gpt2_encode(tok, prompt.UTF8String, tokens, context);
    if (count < 1 || count + maximum > context) {
        fprintf(stderr,
                "qwen35: prompt plus generation exceeds context (%d); raise --context or lower "
                "--max_tokens\n",
                context);
        orion_gpt2_tokenizer_free(tok);
        return 1;
    }
    OrionQwen35 *model = [OrionQwen35 loadDirectory:directory maxContext:context];
    if (!model) {
        orion_gpt2_tokenizer_free(tok);
        return 1;
    }
    double compileStart = seconds();
    if (ane && ![model enableANEWithMLPFraction:mf gdnFraction:gf tile:tile]) {
        fprintf(stderr, "qwen35: ANE setup failed; try CPU inference for diagnosis\n");
        orion_gpt2_tokenizer_free(tok);
        return 1;
    }
    double compileTime = seconds() - compileStart;
    NSMutableData *logitData = [NSMutableData dataWithLength:248320 * sizeof(float)];
    float *logits = logitData.mutableBytes;
    double start = seconds();
    int batch = ane ? tile : 128;
    for (int p = 0; p < count; p += batch) {
        int n = MIN(batch, count - p);
        if (![model forwardTokens:tokens + p
                            count:n
                           logits:p + n == count ? logits : NULL
                           useANE:ane]) {
            fprintf(stderr, "qwen35: inference failed at position %d\n", model.position);
            orion_gpt2_tokenizer_free(tok);
            return 1;
        }
    }
    double prefill = seconds() - start, decodeStart = seconds();
    int generated = 0;
    int stopToken = -1;
    NSMutableData *generatedData = [NSMutableData dataWithLength:(size_t)maximum * sizeof(int)];
    int *ids = generatedData.mutableBytes;
    for (int i = 0; i < maximum; i++) {
        int next = orion_sample_token(logits, 248320, temperature, topP, &seed);
        if (next == 248044 || next == 248046) {
            stopToken = next;
            break;
        }
        ids[generated++] = next;
        if (i + 1 < maximum && ![model forwardTokens:&next
                                               count:1
                                              logits:logits
                                              useANE:decodeANE]) {
            fprintf(stderr, "qwen35: inference failed at position %d\n", model.position);
            orion_gpt2_tokenizer_free(tok);
            return 1;
        }
    }
    if (dumpPath) {
        NSMutableArray *promptIDs = [NSMutableArray arrayWithCapacity:count];
        NSMutableArray *outputIDs = [NSMutableArray arrayWithCapacity:generated];
        for (int i = 0; i < count; i++)
            [promptIDs addObject:@(tokens[i])];
        for (int i = 0; i < generated; i++)
            [outputIDs addObject:@(ids[i])];
        NSDictionary *report = @{
            @"prompt_tokens" : promptIDs,
            @"generated_tokens" : outputIDs,
            @"stop_token" : stopToken >= 0 ? @(stopToken) : NSNull.null
        };
        NSError *error = nil;
        NSData *data = [NSJSONSerialization dataWithJSONObject:report
                                                       options:NSJSONWritingPrettyPrinted
                                                         error:&error];
        if (!data || ![data writeToFile:dumpPath options:NSDataWritingAtomic error:&error]) {
            fprintf(stderr, "qwen35: cannot write token report: %s\n",
                    error.localizedDescription.UTF8String);
            orion_gpt2_tokenizer_free(tok);
            return 1;
        }
    }
    // Decode the whole sequence once, preserving UTF-8 code points split across tokens.
    char *text = orion_gpt2_decode(tok, ids, generated);
    printf("%s\n", text);
    free(text);
    fprintf(stderr,
            "[qwen35] prompt_tokens=%d generated_tokens=%d compile_s=%.3f prefill_s=%.3f "
            "prefill_tps=%.2f decode_s=%.3f\n",
            count, generated, compileTime, prefill, count / prefill, seconds() - decodeStart);
    fprintf(stderr,
            "[qwen35-ane] compiled_programs=%d verified_programs=%d evaluations=%lu "
            "mlp_evaluations=%lu gdn_evaluations=%lu eval_ms=%.3f "
            "decode=%s\n",
            model.anePrograms, model.aneVerifications, model.aneEvaluations,
            model.aneMLPEvaluations, model.aneGDNEvaluations, model.aneMilliseconds,
            decodeANE ? "hybrid" : "cpu");
    orion_gpt2_tokenizer_free(tok);
    return 0;
}

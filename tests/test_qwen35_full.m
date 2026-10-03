#import "kernels/inference/decode_cpu.h"
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
static BOOL prefill(OrionQwen35 *m, NSArray *ids, int batch, float *logits, BOOL ane) {
    int count = (int)ids.count, tokens[256];
    for (int p = 0; p < count; p += batch) {
        int n = MIN(batch, count - p);
        for (int t = 0; t < n; t++)
            tokens[t] = [ids[p + t] intValue];
        if (![m forwardTokens:tokens count:n logits:p + n == count ? logits : NULL useANE:ane])
            return NO;
    }
    return YES;
}
static float difference(const float *a, const float *b) {
    float worst = 0;
    for (int i = 0; i < 248320; i++) {
        if (!isfinite(a[i]) || !isfinite(b[i]))
            return INFINITY;
        worst = fmaxf(worst, fabsf(a[i] - b[i]));
    }
    return worst;
}
int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSString *dir = argc > 1 ? @(argv[1]) : @"model/blobs/qwen35_0_8b";
        BOOL ane = argc > 2 && !strcmp(argv[2], "--ane");
        NSData *data =
            [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"reference.json"]];
        CHECK(data);
        NSDictionary *ref = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
        CHECK(ref);
        OrionGPT2Tokenizer *tok = orion_bpe_tokenizer_load_json(
            [dir stringByAppendingPathComponent:@"tokenizer.json"].UTF8String);
        CHECK(tok);
        int tokens[1024];
        for (NSDictionary *c in ref[@"tokenizer_cases"]) {
            int n = orion_gpt2_encode(tok, [c[@"text"] UTF8String], tokens, 1024);
            if (n != (int)[c[@"tokens"] count]) {
                fprintf(stderr, "tokenizer mismatch text=%s n=%d expected=%s\n",
                        [c[@"text"] UTF8String], n, [c[@"tokens"] description].UTF8String);
                for (int i = 0; i < n; i++)
                    fprintf(stderr, "%d ", tokens[i]);
                fprintf(stderr, "\n");
            }
            CHECK(n == (int)[c[@"tokens"] count]);
            for (int i = 0; i < n; i++)
                CHECK(tokens[i] == [c[@"tokens"][i] intValue]);
            char *text = orion_gpt2_decode(tok, tokens, n);
            CHECK(!strcmp(text, [(c[@"decoded"] ?: c[@"text"]) UTF8String]));
            free(text);
        }
        puts("PASS native tokenizer matches Hugging Face: English, Chinese, accents, emoji, "
             "digits, special tokens");
        OrionQwen35 *model = [OrionQwen35 loadDirectory:dir maxContext:256];
        CHECK(model);
        float mf = argc > 3 ? atof(argv[3]) : .6f, gf = argc > 4 ? atof(argv[4]) : .25f;
        if (ane)
            CHECK([model enableANEWithMLPFraction:mf gdnFraction:gf tile:32]);
        float *logits = malloc(248320 * 4), *batchLogits = malloc(248320 * 4);
        CHECK(prefill(model, ref[@"prompt_tokens"], ane ? 32 : 128, logits, ane));
        memcpy(batchLogits, logits, 248320 * 4);
        NSArray *expected = ref[@"generated_tokens"];
        for (int step = 0; step < (int)expected.count; step++) {
            NSData *gold = [NSData
                dataWithContentsOfFile:[dir stringByAppendingPathComponent:
                                                [NSString
                                                    stringWithFormat:@"reference_logits_%d.f32",
                                                                     step]]];
            CHECK(gold.length == 248320 * 4);
            float err = difference(logits, gold.bytes);
            fprintf(stderr, "%s step=%d max_logit_error=%.7f\n", ane ? "ANE" : "CPU", step, err);
            double sq = 0, base = 0;
            const float *reference = gold.bytes;
            for (int i = 0; i < 248320; i++) {
                sq += (double)(logits[i] - reference[i]) * (logits[i] - reference[i]);
                base += (double)reference[i] * reference[i];
            }
            double relative = sqrt(sq / fmax(base, 1e-12));
            fprintf(stderr, "relative_logit_rmse=%.7f\n", relative);
            int next = orion_sample_token(logits, 248320, 0, 1, NULL);
            fprintf(stderr, "greedy_token=%d expected=%d\n", next, [expected[step] intValue]);
            CHECK(next == [expected[step] intValue]);
            CHECK(err < (ane ? .05f : .001f) && relative < (ane ? .0035 : .0001));
            if (step + 1 < (int)expected.count)
                CHECK([model forwardTokens:&next count:1 logits:logits useANE:ane]);
        }
        if (ane) {
            CHECK(model.aneVerifications == model.anePrograms);
            CHECK(model.aneEvaluations == model.aneMLPEvaluations + model.aneGDNEvaluations);
            CHECK(model.anePrograms == (mf > 0 ? 24 : 0) + (gf > 0 ? 18 : 0) &&
                  model.aneEvaluations > 0);
            fprintf(stderr, "ANE programs=%d evaluations=%lu eval_ms=%.3f\n", model.anePrograms,
                    model.aneEvaluations, model.aneMilliseconds);
        }
        // Incremental CPU vs batched CPU checks causal masks, convolution carry and
        // recurrent state independently of ANE's token padding.
        if (!ane) {
            [model reset];
            CHECK(model.position == 0);
            CHECK(prefill(model, ref[@"prompt_tokens"], 1, logits, NO));
            float err = difference(logits, batchLogits);
            fprintf(stderr, "CPU batch vs incremental max_logit_error=%.7f\n", err);
            CHECK(err < .01f);
            int bad = -1;
            int before = model.position;
            CHECK(![model forwardTokens:&bad count:1 logits:logits useANE:NO] &&
                  model.position == before);
            CHECK(![model forwardTokens:tokens count:257 logits:logits useANE:NO] &&
                  model.position == before);
        }
        [model reset];
        CHECK(model.position == 0 && model.aneEvaluations == 0);
        CHECK(prefill(model, ref[@"prompt_tokens"], ane ? 32 : 128, logits, ane));
        CHECK(difference(logits, batchLogits) < 1e-5f);
        if (ane) {
            // Cross two tile boundaries and use a partial final tile. Padded tokens
            // must not advance the recurrent/KV caches between chunks.
            NSMutableArray *longPrompt = [NSMutableArray array];
            for (int i = 0; i < 3; i++)
                [longPrompt addObjectsFromArray:ref[@"prompt_tokens"]];
            CHECK(longPrompt.count > 64 && longPrompt.count < 128);
            [model reset];
            CHECK(prefill(model, longPrompt, 128, batchLogits, NO));
            [model reset];
            CHECK(prefill(model, longPrompt, 32, logits, YES));
            CHECK(model.position == (int)longPrompt.count);
            float err = difference(logits, batchLogits);
            fprintf(stderr, "ANE multi-tile vs CPU max_logit_error=%.7f position=%d\n", err,
                    model.position);
            CHECK(err < .1f);
            CHECK(orion_sample_token(logits, 248320, 0, 1, NULL) ==
                  orion_sample_token(batchLogits, 248320, 0, 1, NULL));
        }
        free(logits);
        free(batchLogits);
        orion_gpt2_tokenizer_free(tok);
        puts("PASS full Qwen3.5-0.8B: reference logits, greedy decode, cached prefill and reset");
        return 0;
    }
}

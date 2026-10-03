#import "ane_io.h"
#import "iosurface_tensor.h"
#import "mil_builder.h"
#import <dispatch/dispatch.h>
#include <math.h>

bool orion_ane_io_choose(OrionANEIOProbe probe, void *context, OrionANEIO *policy) {
    if (!probe || !policy) return false;
    const OrionIODtype candidates[] = {ORION_IO_FP32, ORION_IO_FP16};
    for (int i = 0; i < 2; i++) {
        OrionIODtype dtype = candidates[i];
        if (!probe(dtype, false, false, context)) continue;
        bool pack = false;
        if (!probe(dtype, true, false, context)) {
            if (!probe(dtype, true, true, context)) continue;
            pack = true;
        }
        *policy = (OrionANEIO){.dtype = dtype, .pack_weights = pack};
        return true;
    }
    return false;
}

IOSurfaceRef orion_io_tensor_create(int channels, int seq, OrionIODtype dtype) {
    return dtype == ORION_IO_FP16 ? orion_tensor_create(channels, seq)
                                 : orion_tensor_create_f32(channels, seq);
}

void orion_io_write_f32(IOSurfaceRef surface, const float *data, int count, OrionIODtype dtype) {
    if (dtype == ORION_IO_FP16) orion_tensor_write_f32(surface, data, count);
    else orion_tensor_write(surface, data, (size_t)count * sizeof(float));
}

void orion_io_read_f32(IOSurfaceRef surface, float *data, int count, OrionIODtype dtype) {
    if (dtype == ORION_IO_FP16) orion_tensor_read_f32(surface, data, count);
    else orion_tensor_read_f32_direct(surface, data, count);
}

void orion_io_prefill_outputs(OrionIODtype dtype, IOSurfaceRef hidden,
                              IOSurfaceRef k, IOSurfaceRef v, IOSurfaceRef outputs[3]) {
    outputs[0] = dtype == ORION_IO_FP16 ? k : hidden;
    outputs[1] = dtype == ORION_IO_FP16 ? hidden : k;
    outputs[2] = v;
}

OrionProgram *orion_ane_io_compile(NSString *mil, NSDictionary *weights,
                                  const OrionANEIO *policy, const char *tag) {
    if (!policy || !mil) return NULL;
    if (policy->pack_weights && weights.count &&
        !orion_mil_pack_weights(mil, weights, &mil, &weights)) return NULL;
    return orion_compile_mil(mil.UTF8String, weights, tag);
}

static NSData *probe_blob(int count, float value, int in_dim) {
    NSMutableData *blob = [NSMutableData dataWithLength:128 + count * 2];
    uint8_t *bytes = blob.mutableBytes;
    bytes[0] = 1; bytes[4] = 2; bytes[68] = 1;
    uint32_t magic = 0xdeadbeef, size = count * 2;
    uint64_t payload = 128;
    memcpy(bytes + 64, &magic, 4);
    memcpy(bytes + 72, &size, 4);
    memcpy(bytes + 80, &payload, 8);
    for (int i = 0; i < count; i++)
        ((_Float16 *)(bytes + 128))[i] = !in_dim || i / in_dim == i % in_dim ? value : 0;
    return blob;
}

static void probe_failure(void *context, NSString *type, bool weighted, bool packed,
                          NSString *reason) {
    NSMutableArray *failures = (__bridge NSMutableArray *)context;
    [failures addObject:[NSString stringWithFormat:@"%@: %@, %@: %@", type,
        weighted ? @"weighted" : @"boundary", packed ? @"packed" : @"separate", reason]];
}

static bool probe_io(OrionIODtype dtype, bool weighted, bool packed, void *context) {
    @autoreleasepool {
        NSString *type = dtype == ORION_IO_FP16 ? @"fp16" : @"fp32";
        int seq = orion_io_decode_seq(dtype);
        int channels = weighted ? 768 : 32;
        NSMutableString *body = [NSMutableString stringWithFormat:
            @"        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
             "        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to16, x=x)[name=string(\"x16\")];\n", channels, seq];
        NSDictionary *weights = @{};
        if (weighted) {
            // Match GPT-2's norm + two projections and real weight sizes. Tiny
            // elementwise/conv probes can compile despite bundle verification
            // failing for the full separate-file GPT-2 programs.
            int hidden = 3072;
            [body appendString:orion_mil_layernorm("ln", "x16", channels, seq,
                "@model_path/layer0/g.bin", "@model_path/layer0/b.bin", 1e-5f)];
            [body appendString:orion_mil_linear("fc", "ln_out", channels, hidden, seq,
                "@model_path/layer0/w1.bin", "@model_path/layer0/b1.bin")];
            [body appendString:orion_mil_linear("proj", "fc_out", hidden, channels, seq,
                "@model_path/layer0/w2.bin", "@model_path/layer0/b2.bin")];
            [body appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y16 = add(x=x16, y=proj_out)[name=string(\"y16\")];\n", channels, seq];
            weights = @{
                @"@model_path/layer0/g.bin": @{@"offset": @0, @"data": probe_blob(channels, 1, 0)},
                @"@model_path/layer0/b.bin": @{@"offset": @0, @"data": probe_blob(channels, 0, 0)},
                @"@model_path/layer0/w1.bin": @{@"offset": @0, @"data": probe_blob(channels*hidden, 2, channels)},
                @"@model_path/layer0/b1.bin": @{@"offset": @0, @"data": probe_blob(hidden, 0, 0)},
                @"@model_path/layer0/w2.bin": @{@"offset": @0, @"data": probe_blob(channels*hidden, 2, hidden)},
                @"@model_path/layer0/b2.bin": @{@"offset": @0, @"data": probe_blob(channels, 3, 0)},
            };
        } else {
            [body appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y16 = add(x=x16, y=x16)[name=string(\"y16\")];\n", channels, seq];
        }
        [body appendFormat:
            @"        string to_io = const()[name=string(\"to_io\"), val=string(\"%@\")];\n"
             "        tensor<%@, [1,%d,1,%d]> out = cast(dtype=to_io, x=y16)[name=string(\"out\")];\n", type, type, channels, seq];
        NSString *mil = orion_mil_program_multi(body,
            @[[NSString stringWithFormat:@"tensor<%@, [1,%d,1,%d]> x", type, channels, seq]], @[@"out"]);
        if (packed && !orion_mil_pack_weights(mil, weights, &mil, &weights)) {
            probe_failure(context, type, weighted, packed, @"weight packing failed");
            return false;
        }
        OrionProgram *program = orion_compile_mil_probe(mil.UTF8String, weights, "io_capability_probe");
        if (!program) {
            probe_failure(context, type, weighted, packed, @"compile/load failed");
            return false;
        }
        int count = channels * seq;
        float *x = malloc(count * sizeof(float)), *y = malloc(count * sizeof(float));
        IOSurfaceRef input = orion_io_tensor_create(channels, seq, dtype);
        IOSurfaceRef output = orion_io_tensor_create(channels, seq, dtype);
        bool ok = input && output && x && y;
        if (!ok) probe_failure(context, type, weighted, packed, @"buffer allocation failed");
        if (ok) {
            // Balanced +/- inputs give mean 0 and nonzero variance. LN rounds
            // to +/-1 in fp16; both diagonal projections must contribute 2*2.
            // Vary positions too, so the check catches layout errors.
            for (int i = 0; i < count; i++) {
                int c = i / seq, s = i % seq;
                float sign = (c + s) % 2 ? -1 : 1;
                x[i] = weighted ? sign * (0.5f + (s % 4) * 0.125f)
                                : (float)(i % 8) / 8;
            }
            orion_io_write_f32(input, x, count, dtype);
            ok = orion_eval(program, &input, 1, &output, 1);
            if (!ok) probe_failure(context, type, weighted, packed, @"evaluation failed");
        }
        if (ok) {
            orion_io_read_f32(output, y, count, dtype);
            for (int i = 0; i < count; i++) {
                float expected = weighted ? x[i] + (x[i] > 0 ? 4 : -4) + 3 : 2*x[i];
                if (!isfinite(y[i]) || fabsf(y[i] - expected) > 0.02f) {
                    probe_failure(context, type, weighted, packed,
                        [NSString stringWithFormat:@"numerical check failed at %d (expected %g, got %g)",
                            i, expected, y[i]]);
                    ok = false;
                    break;
                }
            }
        }
        free(x); free(y);
        orion_tensor_release(input); orion_tensor_release(output);
        orion_release_program(program);
        return ok;
    }
}

const OrionANEIO *orion_ane_io_policy(void) {
    static dispatch_once_t once;
    static OrionANEIO policy;
    static bool available;
    dispatch_once(&once, ^{
        NSMutableArray *failures = [NSMutableArray array];
        bool initialized = orion_ane_init();
        available = initialized && orion_ane_io_choose(probe_io, (__bridge void *)failures, &policy);
        if (available) fprintf(stderr, "ANE I/O: %s, decode seq %d, weights %s\n",
            policy.dtype == ORION_IO_FP16 ? "fp16" : "fp32",
            orion_io_decode_seq(policy.dtype), policy.pack_weights ? "packed" : "separate");
        else {
            fprintf(stderr, "ANE I/O: %s\n", initialized ? "capability probes failed" : "runtime initialization failed");
            for (NSString *failure in failures) fprintf(stderr, "  %s\n", failure.UTF8String);
        }
    });
    return available ? &policy : NULL;
}

#import "core/ane_io.h"
#import "core/iosurface_tensor.h"
#include <math.h>
#include <stdio.h>

typedef struct {
    bool io[2];
    bool separate[2];
    bool packed[2];
    int calls;
} Capabilities;

static bool fake_probe(OrionIODtype dtype, bool weights, bool packed, void *context) {
    Capabilities *caps = context;
    caps->calls++;
    return !weights ? caps->io[dtype] : (packed ? caps->packed[dtype] : caps->separate[dtype]);
}

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); return 1; \
} } while (0)

int main(void) { @autoreleasepool {
    // FP32-capable devices keep the original format and padding.
    Capabilities modern = {.io = {true, true}, .separate = {true, true}};
    OrionANEIO policy;
    CHECK(orion_ane_io_choose(fake_probe, &modern, &policy));
    CHECK(policy.dtype == ORION_IO_FP32 && !policy.pack_weights && modern.calls == 2);
    CHECK(orion_io_decode_seq(policy.dtype) == 16);

    // Older devices can require both fp16 boundaries and packed weights.
    Capabilities older = {.io = {false, true}, .packed = {false, true}};
    CHECK(orion_ane_io_choose(fake_probe, &older, &policy));
    CHECK(policy.dtype == ORION_IO_FP16 && policy.pack_weights && older.calls == 4);
    CHECK(orion_io_decode_seq(policy.dtype) == 32);

    // These capabilities must not be inferred from one another or a chip name.
    Capabilities fp16_separate = {.io = {false, true}, .separate = {false, true}};
    CHECK(orion_ane_io_choose(fake_probe, &fp16_separate, &policy));
    CHECK(policy.dtype == ORION_IO_FP16 && !policy.pack_weights);
    Capabilities fp32_packed = {.io = {true, true}, .packed = {true, true}};
    CHECK(orion_ane_io_choose(fake_probe, &fp32_packed, &policy));
    CHECK(policy.dtype == ORION_IO_FP32 && policy.pack_weights);

    // A boundary-only success must not select a format that cannot use weights.
    Capabilities weighted_fallback = {.io = {true, true}, .separate = {false, true}};
    CHECK(orion_ane_io_choose(fake_probe, &weighted_fallback, &policy));
    CHECK(policy.dtype == ORION_IO_FP16 && !policy.pack_weights);
    Capabilities unavailable = {0};
    policy = (OrionANEIO){.dtype = ORION_IO_FP32, .pack_weights = true};
    CHECK(!orion_ane_io_choose(fake_probe, &unavailable, &policy));
    CHECK(policy.dtype == ORION_IO_FP32 && policy.pack_weights);
    CHECK(!orion_ane_io_choose(NULL, NULL, &policy));

    // Verify host encoding, decoding and output mapping for both formats.
    for (int variant = 0; variant < 2; variant++) {
        OrionIODtype dtype = (OrionIODtype)variant;
        int seq = orion_io_decode_seq(dtype), count = 32 * seq;
        float x[1024], y[1024];
        for (int i = 0; i < count; i++) x[i] = (float)(i % 31 - 15) / 8;
        IOSurfaceRef hidden = orion_io_tensor_create(32, seq, dtype);
        IOSurfaceRef k = orion_io_tensor_create(32, seq, dtype);
        IOSurfaceRef v = orion_io_tensor_create(32, seq, dtype);
        CHECK(hidden && k && v);
        CHECK(IOSurfaceGetWidth(hidden) == (size_t)count * orion_io_element_size(dtype));
        orion_io_write_f32(hidden, x, count, dtype);
        orion_io_read_f32(hidden, y, count, dtype);
        for (int i = 0; i < count; i++) CHECK(isfinite(y[i]) && y[i] == x[i]);
        IOSurfaceRef outputs[3];
        orion_io_prefill_outputs(dtype, hidden, k, v, outputs);
        CHECK(outputs[0] == (dtype == ORION_IO_FP16 ? k : hidden));
        CHECK(outputs[1] == (dtype == ORION_IO_FP16 ? hidden : k));
        CHECK(outputs[2] == v);
        orion_tensor_release(hidden); orion_tensor_release(k); orion_tensor_release(v);
    }

    // The real probe verifies compile + evaluate + numerical output, once.
    const OrionANEIO *actual = orion_ane_io_policy();
    CHECK(actual);
    int compiles = orion_compile_count();
    CHECK(orion_ane_io_policy() == actual && orion_compile_count() == compiles);
    fprintf(stderr, "ANE I/O: selection cases, host round trips, runtime probe PASS\n");
    return 0;
} }

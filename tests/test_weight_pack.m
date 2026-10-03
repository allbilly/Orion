#import "core/ane_runtime.h"
#import "core/mil_builder.h"
#import "core/iosurface_tensor.h"
#import "core/kernel.h"
#include <math.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); return false; \
} } while (0)

static NSData *blob(float value) {
    // A trailing byte makes the next blob require alignment padding.
    NSMutableData *data = [NSMutableData dataWithLength:128 + 32 * 2 + 1];
    uint8_t *b = data.mutableBytes;
    b[0] = 1; b[4] = 2;
    uint32_t magic = 0xdeadbeef, size = 64;
    uint64_t payload = 128;
    memcpy(b + 64, &magic, 4); b[68] = 1;
    memcpy(b + 72, &size, 4); memcpy(b + 80, &payload, 8);
    for (int i = 0; i < 32; i++) ((_Float16 *)(b + 128))[i] = value;
    return data;
}

static bool rejects(NSString *mil, NSDictionary *files) {
    NSString *out = @"unchanged";
    NSDictionary *weights = @{@"unchanged": @1};
    CHECK(!orion_mil_pack_weights(mil, files, &out, &weights));
    CHECK([out isEqualToString:@"unchanged"] && [weights isEqual:@{@"unchanged": @1}]);
    return true;
}

static bool test_packing_validation(void) {
    NSString *ref = @"BLOBFILE(path=string(\"@model_path/a.bin\"), offset=uint64(64))";
    NSData *valid = blob(2);
    NSDictionary *files = @{@"@model_path/a.bin": @{@"offset": @0, @"data": valid}};
    NSString *out = nil; NSDictionary *packed = nil;
    // Reusing a chunk at a nonzero base must relocate its pointer only once.
    NSString *zref = [ref stringByReplacingOccurrencesOfString:@"a.bin" withString:@"z.bin"];
    NSDictionary *both = @{@"@model_path/a.bin": files[@"@model_path/a.bin"],
                            @"@model_path/z.bin": @{@"offset": @0, @"data": blob(3)}};
    CHECK(orion_mil_pack_weights([zref stringByAppendingFormat:@" %@", zref], both, &out, &packed));
    uint64_t payload;
    NSUInteger base = (valid.length + 63) / 64 * 64;
    memcpy(&payload, (const uint8_t *)[packed[@"@model_path/weights/packed.bin"][@"data"] bytes] + base + 80, 8);
    CHECK(payload == base + 128);
    CHECK(rejects([ref stringByAppendingString:
        @" BLOBFILE(path=string(\"@model_path/a.bin\"), offset=int64(64))"], files));
    CHECK(rejects([ref stringByReplacingOccurrencesOfString:@"a.bin" withString:@"missing.bin"], files));
    CHECK(rejects([ref stringByReplacingOccurrencesOfString:@"uint64(64)"
        withString:@"uint64(18446744073709551616)"], files));
    CHECK(rejects(ref, @{@"@model_path/a.bin": @{@"offset": @1, @"data": valid}}));
    CHECK(rejects(ref, @{@"@model_path/a.bin": @{@"offset": @0, @"data": [valid subdataWithRange:NSMakeRange(0, 100)]}}));

    // Corrupt each part of the chunk independently.
    for (int variant = 0; variant < 4; variant++) {
        NSMutableData *bad = [valid mutableCopy];
        uint8_t *bytes = bad.mutableBytes;
        if (variant == 0) bytes[64] = 0; // Invalid magic.
        if (variant == 1) { uint32_t size = 4096; memcpy(bytes + 72, &size, 4); }
        if (variant == 2) { uint64_t offset = 64; memcpy(bytes + 80, &offset, 8); }
        if (variant == 3) { uint64_t offset = 4096; memcpy(bytes + 80, &offset, 8); }
        CHECK(rejects(ref, @{@"@model_path/a.bin": @{@"offset": @0, @"data": bad}}));
    }
    return true;
}

static NSString *s_cache_mil;
static NSDictionary *s_cache_weights;
static int s_generations;

static NSString *cache_mil(int layer, int bucket, const OrionModelConfig *cfg) {
    (void)layer; (void)bucket; (void)cfg;
    // Distinct descriptors avoid sharing an ANE temporary directory.
    return [s_cache_mil stringByReplacingOccurrencesOfString:@"name=string(\"scaled\")"
        withString:[NSString stringWithFormat:@"name=string(\"scaled_%d\")", ++s_generations]];
}

static NSDictionary *cache_weights(int layer, int bucket, const char *dir) {
    (void)layer; (void)bucket; (void)dir;
    return s_cache_weights;
}

static bool test_kernel_cache(NSString *mil, NSDictionary *files) {
    s_cache_mil = mil; s_cache_weights = files;
    orion_cache_clear();
    char name[401]; memset(name, 'k', 400); name[400] = 0;
    OrionKernel kernel = {.name = name, .generate_mil = cache_mil, .build_wdict = cache_weights,
                          .n_inputs = 1, .n_outputs = 1};
    OrionWeightsBinding binding = {.weights_id = "packing_test", .bucket = 32};
    OrionProgram *programs[4];
    int before = orion_compile_count();
    // All programs use the supported fp16 fixture. Vary descriptor metadata
    // independently to verify cache identity without requiring fp32 hardware.
    for (int variant = 0; variant < 4; variant++) {
        kernel.io_dtype = variant & 1 ? ORION_IO_FP16 : ORION_IO_FP32;
        kernel.pack_weights = (variant & 2) != 0;
        programs[variant] = orion_kernel_compile(&kernel, 0, 32, NULL, NULL, &binding);
        CHECK(programs[variant]);
        for (int j = 0; j < variant; j++) CHECK(programs[variant] != programs[j]);
        CHECK(orion_kernel_compile(&kernel, 0, 32, NULL, NULL, &binding) == programs[variant]);
        CHECK(s_generations == variant + 1);
    }
    CHECK(orion_cache_size() == 4 && orion_compile_count() == before + 4);
    name[399] = 'z'; // Names sharing the first 399 bytes must remain distinct.
    CHECK(orion_kernel_compile(&kernel, 0, 32, NULL, NULL, &binding));
    CHECK(orion_cache_size() == 5 && s_generations == 5);
    orion_cache_evict("packing_test");
    CHECK(orion_cache_size() == 0);
    s_cache_mil = nil; s_cache_weights = nil;
    return true;
}

int main(void) { @autoreleasepool {
    if (!test_packing_validation()) return 1;
    NSString *mil = orion_mil_program_multi(
        @"        tensor<fp16, [1,32,1,1]> a = const()[name=string(\"a\"), val=tensor<fp16, [1,32,1,1]>(BLOBFILE(path=string(\"@model_path/a.bin\"), offset=uint64(64)))];\n"
         "        tensor<fp16, [1,32,1,1]> z = const()[name=string(\"z\"), val=tensor<fp16, [1,32,1,1]>(BLOBFILE ( path = string ( \"@model_path/z.bin\" ) ,\n offset = uint64 ( 64 ) ))];\n"
         "        tensor<fp16, [1,32,1,32]> scaled = mul(x=x, y=a)[name=string(\"scaled\")];\n"
         "        tensor<fp16, [1,32,1,32]> out = add(x=scaled, y=z)[name=string(\"out\")];\n",
        @[@"tensor<fp16, [1,32,1,32]> x"], @[@"out"]);
    NSDictionary *files = @{@"@model_path/a.bin": @{@"offset":@0, @"data":blob(2)},
                             @"@model_path/z.bin": @{@"offset":@0, @"data":blob(3)}};
    NSString *packed = nil; NSDictionary *weights = nil;
    if (!orion_mil_pack_weights(mil, files, &packed, &weights) || weights.count != 1 ||
        [packed containsString:@"@model_path/a.bin"] || [packed containsString:@"@model_path/z.bin"]) return 1;
    if (!orion_ane_init()) return 2;
    OrionProgram *program = orion_compile_mil(packed.UTF8String, weights, "packed_weights_test");
    if (!program) return 1;
    IOSurfaceRef input = orion_tensor_create(32,32), output = orion_tensor_create(32,32);
    float x[1024], y[1024];
    for (int i = 0; i < 1024; i++) x[i] = (float)(i % 8) / 8;
    orion_tensor_write_f32(input,x,1024);
    bool ok = orion_eval(program,&input,1,&output,1);
    if (ok) {
        orion_tensor_read_f32(output,y,1024);
        for (int i = 0; i < 1024; i++) ok &= isfinite(y[i]) && fabsf(y[i] - (2*x[i]+3)) < 0.003f;
    }
    orion_tensor_release(input); orion_tensor_release(output); orion_release_program(program);
    if (!test_kernel_cache(mil, files)) return 1;
    fprintf(stderr,"packed weights: relocation + ANE numerical check %s\n",ok?"PASS":"FAIL");
    return ok?0:1;
}}

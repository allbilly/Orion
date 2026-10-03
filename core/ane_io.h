#ifndef ORION_ANE_IO_H
#define ORION_ANE_IO_H

#include "ane_io_types.h"

#ifdef __OBJC__
#import "ane_runtime.h"

// Probe GPT-2 I/O once per process without model files. NULL means no
// boundary/weight-layout combination passed compile + numerical evaluation.
const OrionANEIO *orion_ane_io_policy(void);
IOSurfaceRef orion_io_tensor_create(int channels, int seq, OrionIODtype dtype);
void orion_io_write_f32(IOSurfaceRef surface, const float *data, int count, OrionIODtype dtype);
void orion_io_read_f32(IOSurfaceRef surface, float *data, int count, OrionIODtype dtype);

// Compile direct frontend users with the same independently probed weight policy.
OrionProgram *orion_ane_io_compile(NSString *mil, NSDictionary *weights,
                                  const OrionANEIO *policy, const char *tag);

// Output casts disappear in fp16 mode; ANE binds surfaces by sorted variable name.
void orion_io_prefill_outputs(OrionIODtype dtype, IOSurfaceRef hidden,
                              IOSurfaceRef k, IOSurfaceRef v, IOSurfaceRef outputs[3]);
#endif
#endif

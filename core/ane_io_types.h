#ifndef ORION_ANE_IO_TYPES_H
#define ORION_ANE_IO_TYPES_H

#include <stdbool.h>
#include <stddef.h>

// Program boundaries are independent of the fp16 computation inside GPT-2.
typedef enum { ORION_IO_FP32 = 0, ORION_IO_FP16 = 1 } OrionIODtype;

typedef struct {
    OrionIODtype dtype;
    bool pack_weights;
} OrionANEIO;

static inline size_t orion_io_element_size(OrionIODtype dtype) {
    return dtype == ORION_IO_FP16 ? 2 : 4;
}

// A channel occupies at least 64 bytes in a flat ANE IOSurface.
static inline int orion_io_decode_seq(OrionIODtype dtype) {
    return (int)(64 / orion_io_element_size(dtype));
}

// Injectable capability check: compile, evaluate, and verify numerical output.
typedef bool (*OrionANEIOProbe)(OrionIODtype dtype, bool weights, bool packed, void *context);
bool orion_ane_io_choose(OrionANEIOProbe probe, void *context, OrionANEIO *policy);

#endif

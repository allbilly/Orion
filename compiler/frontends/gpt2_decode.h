// compiler/frontends/gpt2_decode.h — T135: GPT-2 decode frontend
#ifndef ORION_FRONTEND_GPT2_DECODE_H
#define ORION_FRONTEND_GPT2_DECODE_H

#include "../graph.h"
#include "../model_config.h"
#include "../../core/ane_io_types.h"

// Legacy FP32 frontend padding; runtime callers use orion_io_decode_seq().
#define ORION_GRAPH_DECODE_SEQ 16

// Build decode projection graph (LN1 -> QKV).
// Outputs: q32, k32, v32 (fp32, multi-output)
OrionGraph* orion_frontend_gpt2_decode_proj(int layer, const OrionModelConfig* cfg);
// Explicit boundary format; padding follows its element size.
OrionGraph* orion_frontend_gpt2_decode_proj_io(int layer, const OrionModelConfig* cfg, OrionIODtype dtype);

// Build decode FFN graph (LN2 -> FFN -> residual).
// Output: hidden (fp32)
OrionGraph* orion_frontend_gpt2_decode_ffn(int layer, const OrionModelConfig* cfg);
OrionGraph* orion_frontend_gpt2_decode_ffn_io(int layer, const OrionModelConfig* cfg, OrionIODtype dtype);

#endif // ORION_FRONTEND_GPT2_DECODE_H

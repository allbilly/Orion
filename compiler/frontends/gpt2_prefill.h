// compiler/frontends/gpt2_prefill.h — T134: GPT-2 prefill frontend
#ifndef ORION_FRONTEND_GPT2_PREFILL_H
#define ORION_FRONTEND_GPT2_PREFILL_H

#include "../graph.h"
#include "../model_config.h"
#include "../../core/ane_io_types.h"

// Build prefill attention graph for one layer.
// Equivalent to orion_milgen_gpt2_prefill_attn.
// Outputs: hidden (fp32), k_cache (fp32), v_cache (fp32)
OrionGraph* orion_frontend_gpt2_prefill_attn(int layer, int bucket, const OrionModelConfig* cfg);
// Explicit boundary format; the legacy entry point above keeps FP32.
OrionGraph* orion_frontend_gpt2_prefill_attn_io(int layer, int bucket, const OrionModelConfig* cfg, OrionIODtype dtype);

// Build prefill FFN graph for one layer.
// Equivalent to orion_milgen_gpt2_prefill_ffn.
// Output: hidden (fp32)
OrionGraph* orion_frontend_gpt2_prefill_ffn(int layer, int bucket, const OrionModelConfig* cfg);
OrionGraph* orion_frontend_gpt2_prefill_ffn_io(int layer, int bucket, const OrionModelConfig* cfg, OrionIODtype dtype);

#endif // ORION_FRONTEND_GPT2_PREFILL_H

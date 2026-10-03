#ifndef ORION_FRONTEND_QWEN35_H
#define ORION_FRONTEND_QWEN35_H
#include "../graph.h"
#ifdef __OBJC__
#import "../../core/ane_runtime.h"
#else
#include "../model_config.h"
#endif
#include "core/ane_io_types.h"
// cfg.d_model is the input width; cfg.hidden_dim is the selected output prefix.
OrionGraph *orion_frontend_qwen35_gate_up_io(int layer, int tile, const OrionModelConfig *cfg,
                                             OrionIODtype dtype);
OrionGraph *orion_frontend_qwen35_projection_io(int layer, int tile, const OrionModelConfig *cfg,
                                                OrionIODtype dtype);
#endif

#include "qwen35.h"
#include "../builder.h"
#include "../patterns.h"

static OrionGraph *projection_graph(int tile, const OrionModelConfig *cfg, OrionIODtype dtype,
                                    bool pair) {
    if (!cfg || cfg->d_model < 1 || cfg->hidden_dim < 1 || tile < 1)
        return NULL;
    OrionGraph *g = orion_graph_create();
    int shape[4] = {1, cfg->d_model, 1, tile};
    OrionDtype io = dtype == ORION_IO_FP16 ? ORION_DTYPE_FP16 : ORION_DTYPE_FP32;
    int input = orion_gb_input(g, "x", io, shape);
    int x = orion_pattern_cast_to_fp16(g, input, "x16", cfg->d_model, tile);
    int (*cast)(OrionGraph *, int, const char *, int, int) =
        dtype == ORION_IO_FP16 ? orion_pattern_cast_to_fp16 : orion_pattern_cast_to_fp32;
    int a = orion_gb_linear(g, x, pair ? "gate" : "proj", cfg->d_model, cfg->hidden_dim, tile,
                            pair ? "@model_path/weights/g.bin" : "@model_path/weights/w.bin", NULL);
    a = cast(g, a, pair ? "gate_io" : "out", cfg->hidden_dim, tile);
    orion_gb_output(g, a, pair ? "gate_io" : "out");
    if (pair) {
        int b = orion_gb_linear(g, x, "up", cfg->d_model, cfg->hidden_dim, tile,
                                "@model_path/weights/u.bin", NULL);
        b = cast(g, b, "up_io", cfg->hidden_dim, tile);
        orion_gb_output(g, b, "up_io");
    }
    return g;
}
OrionGraph *orion_frontend_qwen35_gate_up_io(int layer, int tile, const OrionModelConfig *cfg,
                                             OrionIODtype dtype) {
    (void)layer;
    return projection_graph(tile, cfg, dtype, true);
}
OrionGraph *orion_frontend_qwen35_projection_io(int layer, int tile, const OrionModelConfig *cfg,
                                                OrionIODtype dtype) {
    (void)layer;
    return projection_graph(tile, cfg, dtype, false);
}

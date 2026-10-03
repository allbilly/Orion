#ifndef ORION_QWEN35_H
#define ORION_QWEN35_H
#import <Foundation/Foundation.h>

// Native, single-sequence Qwen3.5-0.8B text inference. Stateful caches belong to
// the model instance; use reset before a new prompt. Vision is not supported.
@interface OrionQwen35 : NSObject
@property(nonatomic, readonly) int position;
@property(nonatomic, readonly) int maxContext;
@property(nonatomic, readonly) int anePrograms;
@property(nonatomic, readonly) int aneVerifications;
@property(nonatomic, readonly) unsigned long aneEvaluations;
@property(nonatomic, readonly) unsigned long aneMLPEvaluations;
@property(nonatomic, readonly) unsigned long aneGDNEvaluations;
@property(nonatomic, readonly) double aneMilliseconds;
+ (instancetype)loadDirectory:(NSString *)directory maxContext:(int)context;
- (void)reset;
// Fractions refer to MLP gate/up channels and GDN QKV output channels.
// Compiles once. Returns NO on unavailable ANE or any compile/verification error.
- (BOOL)enableANEWithMLPFraction:(float)mlp gdnFraction:(float)gdn tile:(int)tile;
// Count must fit the remaining context and tile (CPU maximum batch: 256).
// logits may be NULL during prefill; otherwise write 248320 last-token logits.
// On an evaluation error the cache may be partially updated; reset before retry.
- (BOOL)forwardTokens:(const int *)tokens count:(int)count logits:(float *)logits useANE:(BOOL)ane;
@end

// Numerical primitives shared by inference and independent reference tests.
void orion_qwen35_rmsnorm(const float *x, const float *weight, float *out, int dim, float eps);
void orion_qwen35_rope(float *x, int headDim, int rotaryDim, int position, float theta);
// q and k already L2-normalized, with 1/sqrt(keyDim) folded into q.
// State is [valueDim, keyDim] in fp32.
void orion_qwen35_delta_step(const float *q, const float *k, const float *v, float decay,
                             float beta, float *state, float *out, int keyDim, int valueDim);
#endif

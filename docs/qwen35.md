# Qwen3.5-0.8B text inference

Branch: `feat/qwen35-ane`. The CLI runs Qwen3.5-0.8B natively in Objective-C/C with
Accelerate and Orion's private ANE runtime. Python/MLX are used only to import
weights and generate independent test references.

## Import and run

Use a local Hugging Face safetensors checkpoint or an affine-quantized MLX
checkpoint. The checkpoint already installed for oMLX can be reused:

```sh
# Use an interpreter with numpy and mlx installed.
# For this Mac, /opt/homebrew/opt/omlx/libexec/bin/python has the dependencies.
python3 model/convert/hf_to_qwen35.py \
  --source ~/.omlx/models/mlx-community/Qwen3.5-0.8B-4bit \
  --output model/blobs/qwen35_0_8b

make -j4
./orion infer --model qwen35_0_8b \
  --prompt 'What is 2 + 2? Answer briefly.' --max_tokens 32

./orion infer --model qwen35_0_8b \
  --prompt 'What is 2 + 2? Answer briefly.' --max_tokens 32 --ane-prefill

./orion infer --model qwen35_0_8b \
  --prompt 'What is 2 + 2? Answer briefly.' --max_tokens 32 --ane \
  --ane-mlp-fraction 0.6 --ane-gdn-fraction 0.25 --ane-tile 32
```

The default prompt is single-turn ChatML with thinking disabled. `--thinking`
opens the reasoning section; `--raw` uses the input verbatim. The tokenizer
loads the checkpoint's BPE merges, NFC normalization, Unicode regex and added
tokens. Decode stops at `<|im_end|>` or `<|endoftext|>`.

The importer writes 320 text tensors as little-endian float32 files and a
validated manifest. It removes vision/MTP tensors, dequantizes MLX weights,
transposes HF depthwise-convolution weights and converts HF zero-centered
RMSNorm parameters. This initial export occupies about **2.8 GiB**, even when
its source is a 4-bit checkpoint. The native runtime memory-maps these files;
it does not preserve packed 4-bit inference. Exported weights are ignored by
Git. The importer refuses to overwrite an existing manifest.

## CPU and ANE work

All 24 decoder layers are implemented: 18 Gated DeltaNet layers and 6 gated
attention layers, with tied embeddings and SwiGLU MLPs. CPU inference includes
causal depthwise convolution, normalized Q/K, fp32 delta-rule state, per-head
gated RMSNorm, grouped-query attention, Q/K RMSNorm, partial RoPE, KV caching,
and greedy or temperature/top-p sampling.

ANE programs are built through Orion's compiler frontends. They run token-local
MLP gate/up projections and GDN QKV projections. Fractions select a contiguous
output-channel prefix, rounded to multiples of 32. CPU computes the remaining
channels. SwiGLU, down projection, attention and recurrent state stay on CPU.
The split therefore changes placement rather than dropping model channels.

`--ane-prefill` uses this split only for the prompt; `--ane` also uses it during
decode. The default split creates 24 paired MLP programs and 18 GDN programs.
Each program is evaluated and numerically checked against CPU matrix products
before it is accepted. Compile/load/evaluation errors fail explicitly.

Fixed token tiles are padded to ANE surface requirements, and only real tokens
update convolution history, recurrent state, KV caches or position. Tile sizes
are 32, 64, ..., 256. A capability probe selects fp16/fp32 boundaries and weight
packing independently; there is no chip-name whitelist.

Example diagnostics:

```text
[qwen35-ane] compiled_programs=42 mlp_channels=2144/3584 gdn_qkv_channels=1536/6144 tile=32 io=fp16
[qwen35-ane] compiled_programs=42 verified_programs=42 evaluations=210 mlp_evaluations=120 gdn_evaluations=90 eval_ms=... decode=hybrid
```

`verified_programs` counts startup checks. `evaluations` counts successful
inference calls, separately from startup checks. `eval_ms` includes host tensor
packing, evaluation and readback. These counters establish actual execution;
compilation alone does not.

The 0.6/0.25 defaults are informed by the earlier oMLX run on this M1, but Orion
uses CPU for the remainder while oMLX uses MLX/GPU. They are starting values,
not an Orion autotuning result. Startup compilation is charged separately from
prefill timing. No performance advantage is assumed for short prompts or decode.

## Verification

For a direct token-by-token check against the original quantized MLX checkpoint:

```sh
python3 scripts/compare_qwen35_tokens.py \
  --prompt 'What is 2 + 2? Answer briefly.' --max-tokens 16
```

This runs greedy generation with thinking disabled and compares the official
chat template's prompt IDs, every generated ID and the stop token against both
Orion CPU and hybrid ANE inference. It writes `build/qwen35-token-comparison.json`
and exits nonzero on any mismatch. `--source`, `--weights`, `--orion` and `--output`
can select other paths. The reference uses MLX-LM directly, without the oMLX
server's memory guard. The CLI's `--dump-tokens PATH` also exports actual IDs for
other reference tools. One matching prompt does not establish parity for all
prompts or different quantizations.

For the arithmetic prompt above, the original 4-bit MLX checkpoint, Orion CPU
and Orion hybrid ANE all produced exactly
`[17, 478, 220, 17, 283, 220, 19, 13]` (`2 + 2 = 4.`), followed by stop token
`248046` (`<|im_end|>`). All 23 prompt IDs also matched.

```sh
# Requires mlx-lm and tokenizers in the conversion interpreter.
python3 model/convert/hf_to_qwen35.py \
  --output model/blobs/qwen35_0_8b --reference

make test-qwen35
# A different export can be selected:
make test-qwen35 QWEN35_WEIGHTS=/path/to/export
```

Reference logits come from MLX-LM's fp32 text model loaded with exactly the
exported tensors. This isolates native math/layout correctness from checkpoint
quantization. The tests check five greedy decode steps, all 248320 logits per
step, batched versus incremental prefill, cache reset, bounds, and tokenizer
agreement for English, Chinese, accents, emoji, numbers and added tokens.
An additional 69-token prefill crosses three ANE tiles, including a partial
final tile, and checks the cache position and next token against CPU inference.
Small delta-rule/RMSNorm/RoPE/tokenizer tests run under `make test` without
requiring Qwen weights; the full-model target fails if reference data is absent.

Observed on Apple M1 / macOS 27.0.1 with the local 4-bit source on 2026-10-03:

| Check | Result |
|---|---|
| Native CPU vs reference maximum logit error | 0.0000458 across five steps |
| CPU batched vs incremental maximum logit error | 0.0000608 |
| Hybrid ANE vs reference maximum logit error | 0.0295458 across five steps |
| Hybrid ANE maximum relative logit RMSE | 0.0017323 |
| Greedy tokens | All five match for CPU and hybrid |
| Actual hybrid execution | 42 programs; 210 inference evaluations |
| 69-token tiled ANE vs CPU maximum logit error | 0.0108299; cache position 69 |

CPU tests require max error below 0.001; hybrid tests require max error below
0.05 and relative RMSE below 0.0035. These tolerances cover fp16 boundaries while
still checking every logit and requiring exact greedy-token agreement.

The existing regression suite passed 20/26 suites including the new unit suite.
The other six fail identically at the unchanged baseline commit `ae3a518`:
`test_ane_runtime`, `test_mil_builder`, `test_data_loader`, `test_train_kernels`,
`test_train_smoke`, `test_program_cache`. The data-loader fixture is absent;
the other five encounter ANE compile/load failures on this Mac in their existing
programs. Existing GPT-2 CPU/prefill/decode/golden, ANE I/O, weight
packing and compiler tests pass.

## Scope and references

This is text inference, not vision input, training, continuous batching, a server
or an automatic split tuner. Native context defaults to 4096 and is capped at
32768 to bound KV-cache allocation; the model's full advertised context is not
implemented here. The model/cache instance is for one sequence and must be reset
between prompts. An evaluation error can partially update caches; reset before
retrying.

Architecture: [official Qwen3.5-0.8B model card](https://huggingface.co/Qwen/Qwen3.5-0.8B).
Offload reference: [oMLX v0.7.0 Qwen3.5 ANE prefill](https://github.com/jundot/omlx/blob/v0.7.0/omlx/patches/qwen35_ane_prefill.py).
Numerical reference: [MLX-LM Qwen3.5](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py)
and [gated delta rule](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py).

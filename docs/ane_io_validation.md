# GPT-2 ANE I/O validation

The selected policy depends on compilation and numerical evaluation on the
current device and macOS runtime. Chip names alone do not establish support.
Synthetic selection tests and graph checks cover both boundary formats; hardware
tests validate the policy actually selected on that machine.

## Hardware results

| Chip | macOS | Selected I/O | Decode padding | Weights | Validation |
| --- | --- | --- | --- | --- | --- |
| Apple M1 | 27.0.1 (26A434) | FP16 | 32 | Packed | Compiler 4/4; policy and packing/cache tests; prefill 34/34; decode 7/7; decode-step 3/3; golden 8/8. All five kernel benchmarks and the 100-iteration swap benchmark pass. |
| M4 | Pending | Pending | Pending | Pending | Run the commands below on this checkout before claiming compatibility. |
| Other chips | Pending | Pending | Pending | Pending | No hardware validation yet. |

The golden test requires successful ANE prefill and decode without fallback, and
compares greedy tokens against the CPU reference. Packing tests verify alignment,
payload relocation, repeated chunk references, rejection of malformed input, and
cache separation across dtype/packing metadata and long kernel names.
Golden tests cover prefill buckets 32, 64 and 128. Negative controls run with
temporary copies of the probe rejected both a zeroed first projection and a
zeroed second projection, reporting the numerical mismatch in each case.

## Reproduce on another Mac

Record the checkout revision, chip, macOS version/build and the `ANE I/O:` line:

```sh
git rev-parse HEAD
sysctl -n machdep.cpu.brand_string
sw_vers
make -j4 BUILDDIR=build/ane-validation orion test-compiler \
  build/ane-validation/tests/test_ane_io \
  build/ane-validation/tests/test_weight_pack \
  build/ane-validation/tests/test_ane_prefill \
  build/ane-validation/tests/test_decode_ane \
  build/ane-validation/tests/test_decode_ane_step \
  build/ane-validation/tests/test_infer_golden_ane
build/ane-validation/tests/test_ane_io
build/ane-validation/tests/test_weight_pack
build/ane-validation/tests/test_ane_prefill
build/ane-validation/tests/test_decode_ane
build/ane-validation/tests/test_decode_ane_step
build/ane-validation/tests/test_infer_golden_ane
./orion bench kernels --iters 10 > kernels.jsonl
./orion bench swap --weights_a A --weights_b B --iters 100
```

Model-dependent checks require the converted GPT-2 files in
`model/blobs/gpt2_124m`. Run hardware tests sequentially: identical ANE descriptors
can share a temporary directory. Kernel benchmark JSON records `io_dtype`,
`decode_seq` and `pack_weights` so measurements identify the selected policy.

The weighted startup probe uses balanced nonzero channel inputs. LayerNorm
produces approximately +/-1, and both diagonal projections contribute to the
expected output. Replacing either projection with zero weights must make its
numerical check fail. Expected candidate failures are quiet when another policy
succeeds; if all candidates fail, the log lists the failed stage for each one.

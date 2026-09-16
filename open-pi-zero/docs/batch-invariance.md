# Batch-invariance experiment

## Question

For a fixed input `x₀`, does evaluating `x₀` alone produce exactly the same
result as evaluating `[x₀, x₁, …]` and taking the first result? In idealized
math the answer is yes. On accelerators, kernels may choose different tiling,
parallel reduction, or accumulation schedules for different shapes. Floating
point addition is not associative, so those schedules can produce different
rounding and change a downstream result.

This project uses the following protocol:

1. Seed Python, NumPy, and PyTorch.
2. Construct one fixed reference sample and one fixed initial action noise.
3. Evaluate the reference sample as a batch of one.
4. Create fresh distractors for each trial and put the reference at index zero.
5. Compare output index zero and every traced intermediate tensor.

Run it with:

```console
cd open-pi-zero
uv run python scripts/run_sample.py --batch_sizes 1 2 4 8 --num_trials 10
```

The JSON output contains mean and worst-case `max_abs_diff`, mean absolute
error, RMSE, relative L2 error, and per-stage values. A large intermediate
error before the final action is more useful diagnostically than the final
action alone: it identifies the first operation that needs attention.

## Changes under test

The companion implementation makes two targeted changes:

- `torch.mm` dispatches to the batch-invariant Triton matmul when the CUDA
  operator mode is enabled. The joint model uses an explicit `loop_bmm` so each
  matrix product has a stable reduction scope rather than relying on a fused
  batched matmul schedule.
- SigLIP's non-overlapping patch projection uses `_UnfoldConv2d`, which performs
  the matrix reduction independently for each sample. This is intentionally
  simple and is a reference implementation for the blog, not a drop-in
  replacement for every convolution configuration.

Tracing is opt-in through `src.utils.trace.trace_context`; it clones recorded
tensors and therefore has a real memory cost. It should not be enabled in a
training or production inference path.

## Limitations

- The Triton dispatch layer currently targets CUDA and depends on the installed
  PyTorch/Triton ABI. CPU and MPS runs exercise the model and helper paths but
  do not validate the CUDA kernels.
- The experiment does not prove bitwise determinism across devices, driver
  versions, compiler versions, or launches. It tests the narrower property of
  independence from unrelated batch members in one environment.
- RMSNorm, attention softmax, compiler fusion, KV-cache behavior, and all model
  branches require separate validation. The included RMSNorm script is a
  diagnostic, not a fix.
- Random weights and synthetic inputs make the sample reproducible and cheap;
  they do not represent policy quality.

## Provenance

`open-pi-zero/` is based on the external open-pi-zero implementation. Keep its
license and attribution notices when publishing a derived copy, and link to
the blog's exact commit when reporting numbers.

# Batch-invariant PiZero

This is the minimal companion repository for a blog post about a subtle inference
failure mode: the output for one example can change when unrelated examples are
added to the same batch. It contains a reference PiZero implementation, a
small set of batch-invariant CUDA/Triton operators, a per-sample SigLIP patch
projection, and tracing utilities used to locate the first divergence.

The repository is deliberately an experiment, not a claim that all of PiZero
is batch invariant. The current intervention covers the operations exercised by
the sample; RMSNorm and other reductions still need independent validation.

## Repository map

- `open-pi-zero/` — the PiZero implementation and reproducible experiment.
- `open-pi-zero/batch_invariant_ops/` — the operator shim and focused tests.
- `open-pi-zero/scripts/run_sample.py` — compares a fixed sample alone with
  that sample at index zero in larger batches and records intermediate errors.
- `custom_conv2d.py` — standalone reference/Triton Conv2d experiment.

Training, dataset, simulator, and deployment code are intentionally excluded.

## Run the sample

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run
the sample from the `open-pi-zero` project directory:

```console
cd open-pi-zero
uv sync --extra dev
uv run python scripts/run_sample.py --batch_sizes 1 2 4 --num_trials 10
```

Run the lightweight regression tests with `uv run pytest`.

The script automatically uses CUDA, Apple Silicon MPS, or CPU when available.
Use `--device cpu`, `--device mps`, or `--device cuda` to select one explicitly.
Results are written to `open-pi-zero/batch_invariance_results.json`.

The sample uses randomly initialized model weights and synthetic inputs. It is
intended to measure whether the output for a fixed sample changes when other
samples are added to its batch.

For the CUDA/Triton kernels, use a CUDA-capable environment. CPU and MPS are
useful for checking model wiring and the per-sample convolution, but the
operator shim becomes a no-op there. The experiment is not a performance or
numerical-equivalence benchmark: it measures sensitivity of sample zero to
batch composition under a fixed seed and fixed initial action noise.

See [`open-pi-zero/docs/batch-invariance.md`](open-pi-zero/docs/batch-invariance.md)
for the experiment protocol, interpretation, and known limitations.

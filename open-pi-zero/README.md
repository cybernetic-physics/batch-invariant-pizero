# Inference-only PiZero batch-invariance companion

This directory contains only the PiZero inference path and the tools needed to
study whether an output changes when unrelated samples are added to a batch.
It is based on the external open-pi-zero implementation, with targeted changes
for batch-invariant matmul and SigLIP patch projection.

## Quick start

```console
uv sync --extra dev
uv run python scripts/run_sample.py --batch_sizes 1 2 4 --num_trials 10
uv run pytest
```

The sample uses random weights and synthetic inputs. It is a diagnostic, not a
policy-quality benchmark. CUDA is required to exercise the Triton operator
replacement; CPU and MPS are useful for model and helper tests.

The experiment protocol and limitations are documented in
[`docs/batch-invariance.md`](docs/batch-invariance.md).

## Layout

```text
config/inference.yaml                 model-only inference configuration
scripts/run_sample.py                 end-to-end batch-composition experiment
examples/compare_matmul.py            minimal operator example
tests/                                regression tests for the custom helpers
batch_invariant_ops/                  optional CUDA/Triton operator package
src/model/                            PiZero, PaliGemma, KV-cache inference code
src/utils/                             timing and opt-in tracing helpers
```

The model still expects a PaliGemma checkpoint when running real inference.
The sample itself constructs random weights and does not download a checkpoint.

## Provenance

The model implementation is derived from the upstream open-pi-zero project;
preserve its license and attribution when publishing a derived copy. The
batch-invariant operator package is a focused subset used by this experiment.

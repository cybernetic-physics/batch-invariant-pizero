"""Check whether GemmaRMSNorm gives batch-size-independent outputs.

Run from the repository root with:

    python scripts/check_gemma_rmsnorm_batch_invariance.py

The first sample's output from a batch-size-one run is used as the reference,
then compared with that same sample evaluated in batches of size 1, 4, 8, 16,
and 32.
"""

import argparse
from contextlib import nullcontext

import torch
from omegaconf import OmegaConf

from src.model.paligemma.modules import GemmaRMSNorm


def check_batch_invariance(norm, inputs, batch_sizes, batch_invariant):
    """Print the error for each batch size against the batch-one reference."""
    if batch_invariant:
        from batch_invariant_ops import set_batch_invariant_mode

        mode_context = set_batch_invariant_mode(True)
    else:
        mode_context = nullcontext()

    with mode_context, torch.inference_mode():
        reference = norm(inputs[:1])

        for batch_size in batch_sizes:
            output = norm(inputs[:batch_size])
            difference = (output[:1] - reference).abs()
            max_abs_error = difference.max().item()
            print(
                f"  batch_size={batch_size:>2} "
                f"max_abs_error={max_abs_error:.8g} "
                f"exact={torch.equal(output[:1], reference)}",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/inference.yaml",
        help="Evaluation config used to select the Gemma hidden size and epsilon.",
    )
    parser.add_argument("--device", help="Override the device from the config.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device or cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The bridge config requests CUDA, but CUDA is unavailable.")

    hidden_size = cfg.mixture.vlm.hidden_size
    eps = cfg.joint.config.rms_norm_eps
    dtype = torch.bfloat16 if cfg.use_bf16 else torch.float32

    torch.manual_seed(cfg.seed)
    inputs = torch.randn(
        32,
        4,
        hidden_size,
        device=device,
        dtype=dtype,
    )
    print(f'inputs.shape: {inputs.shape}')

    norm = GemmaRMSNorm(hidden_size, eps=eps).to(device=device, dtype=dtype)
    with torch.no_grad():
        # Use non-default weights so the complete GemmaRMSNorm path is tested.
        norm.weight.copy_(torch.linspace(-0.25, 0.25, hidden_size, device=device))

    batch_sizes = (1, 4, 8, 16, 32)
    print(
        f"config={args.config} device={device} dtype={dtype} "
        f"hidden_size={hidden_size} eps={eps}"
    )

    for mode in (False, True):
        try:
            mode_name = "batch-invariant" if mode else "standard"
            print(f"{mode_name}:", flush=True)
            check_batch_invariance(norm, inputs, batch_sizes, mode)
        except ImportError as error:
            if mode:
                print(f"batch-invariant mode: unavailable ({error})")
                continue
            raise


if __name__ == "__main__":
    main()

"""Compare a sample matrix product with the same sample in a larger batch.

This is the smallest runnable demonstration of the property used by the
PiZero experiment. On CPU/MPS the optional Triton shim is intentionally a
no-op; use CUDA to exercise the replacement kernel.
"""

import argparse

import torch

from batch_invariant_ops import set_batch_invariant_mode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, choices=("cpu", "cuda", "mps"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    )
    torch.manual_seed(0)
    left = torch.randn(args.batch_size, 128, 128, device=device)
    right = torch.randn(args.batch_size, 128, 128, device=device)

    with set_batch_invariant_mode(True):
        reference = torch.mm(left[0], right[0])
        batched = torch.stack(
            [torch.mm(left[i], right[i]) for i in range(args.batch_size)]
        )[:1]

    print(f"device={device} batch_size={args.batch_size}")
    print(f"max_abs_diff={(reference - batched).abs().max().item():.8g}")


if __name__ == "__main__":
    main()

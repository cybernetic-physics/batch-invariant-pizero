"""Check whether ``torch.nn.functional.softmax`` is batch invariant."""

import argparse
import json
import random

import numpy as np
import torch
import torch.nn.functional as F


SEED = 42
BATCH_SIZES = (1, 2, 4, 8, 16, 32)


def main(args: argparse.Namespace) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(args.device)
    generator = torch.Generator(device="cpu").manual_seed(SEED)

    # One fixed sample is the reference in every batch. The remaining samples
    # are distractors and cannot affect softmax when dim=-1.
    reference = torch.randn(
        1, args.sequence_length, args.features, generator=generator
    )
    reference_output = F.softmax(reference.to(device), dim=args.dim).cpu()

    print(
        f"softmax batch-invariance test: device={device}, "
        f"shape=[batch, {args.sequence_length}, {args.features}], dim={args.dim}"
    )

    results = []
    for batch_size in BATCH_SIZES:
        distractors = torch.randn(
            batch_size - 1,
            args.sequence_length,
            args.features,
            generator=generator,
        )
        batch_input = torch.cat((reference, distractors), dim=0).to(device)
        batch_output = F.softmax(batch_input, dim=args.dim).cpu()
        difference = batch_output[:1] - reference_output

        result = {
            "batch_size": batch_size,
            "max_abs_diff": difference.abs().max().item(),
            "mean_abs_diff": difference.abs().mean().item(),
            "rmse": torch.sqrt(torch.mean(difference.square())).item(),
            "reference_sum_error": (
                (batch_output[0].sum(dim=-1) - 1).abs().max().item()
            ),
        }
        results.append(result)
        print(json.dumps(result))

    passed = all(result["max_abs_diff"] == 0 for result in results)
    print(f"PASS: {passed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--sequence_length", type=int, default=16)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument(
        "--dim",
        type=int,
        default=-1,
        help="Softmax dimension; -1 is the usual per-token feature dimension",
    )
    main(parser.parse_args())

"""Optional CUDA/Triton batch-invariant operators.

The PiZero sample also runs on CPU and MPS for inspection.  Triton kernels are
CUDA-only, so importing this package must not make those backends unusable.
"""

try:
    from .batch_invariant_ops import (
        set_batch_invariant_mode,
        is_batch_invariant_mode_enabled,
        disable_batch_invariant_mode,
        enable_batch_invariant_mode,
        matmul_persistent,
        log_softmax,
        mean_dim,
        get_batch_invariant_attention_block_size,
        AttentionBlockSize,
    )
except ImportError as error:  # pragma: no cover - depends on the environment
    _IMPORT_ERROR = error

    def set_batch_invariant_mode(enabled: bool = True):
        """No-op context when Triton is unavailable."""
        from contextlib import nullcontext

        return nullcontext()

    def is_batch_invariant_mode_enabled():
        return False

    def disable_batch_invariant_mode():
        return None

    def enable_batch_invariant_mode():
        raise RuntimeError("Batch-invariant CUDA ops require Triton") from _IMPORT_ERROR

    def _unavailable(*args, **kwargs):
        raise RuntimeError("Batch-invariant CUDA ops require Triton") from _IMPORT_ERROR

    matmul_persistent = _unavailable
    log_softmax = _unavailable
    mean_dim = _unavailable

    from collections import namedtuple

    AttentionBlockSize = namedtuple("AttentionBlockSize", ["block_m", "block_n"])

    def get_batch_invariant_attention_block_size():
        return AttentionBlockSize(block_m=16, block_n=16)

__version__ = "0.1.0"

__all__ = [
    "set_batch_invariant_mode",
    "is_batch_invariant_mode_enabled",
    "disable_batch_invariant_mode", 
    "enable_batch_invariant_mode",
    "matmul_persistent",
    "log_softmax",
    "mean_dim",
    "get_batch_invariant_attention_block_size",
    "AttentionBlockSize",
]

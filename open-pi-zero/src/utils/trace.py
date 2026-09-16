"""Opt-in tensor tracing for the batch-invariance experiment."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

import torch


_ACTIVE_TRACE: ContextVar[dict | None] = ContextVar("active_trace", default=None)


@contextmanager
def trace_context(trace_dict: dict | None = None) -> Iterator[dict]:
    trace = {} if trace_dict is None else trace_dict
    token = _ACTIVE_TRACE.set(trace)
    try:
        yield trace
    finally:
        _ACTIVE_TRACE.reset(token)


def get_trace() -> dict | None:
    return _ACTIVE_TRACE.get()


def _clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        # Trace snapshots are only used after the model call for comparison.
        # Keeping them on the model device means every recorded activation is
        # retained in VRAM until the call finishes, which becomes prohibitive
        # for larger batch sizes. Copy directly to CPU instead.
        return value.detach().to(device="cpu")
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return value


class _GlobalTrace:
    def record(self, name: str, value: Any) -> None:
        trace = get_trace()
        if trace is not None:
            trace[name] = _clone(value)


GLOBAL_TRACE = _GlobalTrace()

import torch

from src.model.paligemma.siglip import _UnfoldConv2d
from src.utils.trace import trace_context


def test_unfold_conv_matches_torch_conv2d():
    torch.manual_seed(0)
    reference = torch.nn.Conv2d(3, 5, kernel_size=3, stride=2, padding=1)
    candidate = _UnfoldConv2d(3, 5, kernel_size=3, stride=2, padding=1)
    candidate.load_state_dict(reference.state_dict())

    inputs = torch.randn(2, 3, 9, 11)
    torch.testing.assert_close(candidate(inputs), reference(inputs))


def test_unfold_conv_is_independent_of_batch_size():
    torch.manual_seed(0)
    conv = _UnfoldConv2d(3, 5, kernel_size=3, stride=2, padding=1)
    inputs = torch.randn(8, 3, 9, 11)

    reference = conv(inputs[:1])
    for batch_size in (1, 2, 4, 8):
        torch.testing.assert_close(reference, conv(inputs[:batch_size])[:1])


def test_trace_context_records_nested_tensors():
    value = torch.tensor([1.0])
    with trace_context() as trace:
        from src.utils.trace import GLOBAL_TRACE

        GLOBAL_TRACE.record("nested", {"value": value})

    assert torch.equal(trace["nested"]["value"], value)
    assert trace["nested"]["value"] is not value

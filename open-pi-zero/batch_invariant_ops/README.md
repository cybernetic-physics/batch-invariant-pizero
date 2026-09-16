# Batch-invariant CUDA operators

This small package provides the CUDA/Triton operators used by the PiZero
inference experiment. `set_batch_invariant_mode()` temporarily overrides the
selected ATen reductions through `torch.library`.

```python
from batch_invariant_ops import set_batch_invariant_mode

with set_batch_invariant_mode():
    output = model(inputs)
```

The current implementation targets CUDA and Triton. On CPU or MPS the context
is inert so the rest of the inference example remains inspectable. The package
does not claim universal determinism across devices, drivers, or compiler
versions.

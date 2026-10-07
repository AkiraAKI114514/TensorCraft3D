"""Fixed graph buffers shared by local execution and standalone Python export."""
import math
import torch
from torch import nn


class TensorLabConstantAdd(nn.Module):
    def __init__(self, shape, values, sequence_dim=None):
        super().__init__()
        if not isinstance(shape, (list, tuple)) or len(shape) > 5 or any(type(v) is not int or not 1 <= v <= 65536 for v in shape):
            raise ValueError("Invalid constant shape")
        if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 65536 or math.prod(shape) != len(values) or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 3.4028234663852886e38 for v in values):
            raise ValueError("Invalid constant values")
        if sequence_dim is not None and (type(sequence_dim) is not int or not 1 <= sequence_dim <= 4):
            raise ValueError("Constant sequence_dim must preserve the batch axis")
        self.sequence_dim = sequence_dim
        self.register_buffer("constant", torch.tensor(values, dtype=torch.float32).reshape(shape))

    def forward(self, x):
        if not x.is_floating_point() or self.constant.ndim > x.ndim:
            raise ValueError("ConstantAdd requires floating input with compatible rank")
        offset = x.ndim - self.constant.ndim
        shape = (1,) * offset + tuple(self.constant.shape)
        if shape[0] != 1:
            raise ValueError("ConstantAdd must broadcast across the batch axis")
        constant = self.constant
        for dim, (size, actual) in enumerate(zip(shape, x.shape)):
            if dim == self.sequence_dim:
                if dim < offset or actual > size:
                    raise ValueError("Input sequence exceeds constant buffer capacity")
                constant = constant.narrow(dim - offset, 0, actual)
            elif size not in (1, actual):
                raise ValueError("Constant shape does not broadcast to the input")
        if self.sequence_dim is not None and self.sequence_dim >= x.ndim:
            raise ValueError("Constant sequence_dim exceeds input rank")
        return x + constant

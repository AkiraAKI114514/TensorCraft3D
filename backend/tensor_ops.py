"""Fixed graph buffers shared by local execution and standalone Python export."""
import math
import torch
from torch import nn


def constant_buffer(shape, values, sequence_dim):
    """Validate and materialise a fixed graph buffer, shared by both constant ops."""
    if not isinstance(shape, (list, tuple)) or len(shape) > 5 or any(type(v) is not int or not 1 <= v <= 65536 for v in shape):
        raise ValueError("Invalid constant shape")
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 65536 or math.prod(shape) != len(values) or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 3.4028234663852886e38 for v in values):
        raise ValueError("Invalid constant values")
    if sequence_dim is not None and (type(sequence_dim) is not int or not 1 <= sequence_dim <= 4):
        raise ValueError("Constant sequence_dim must preserve the batch axis")
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


class TensorLabConstantAdd(nn.Module):
    def __init__(self, shape, values, sequence_dim=None):
        super().__init__()
        self.sequence_dim = sequence_dim
        self.register_buffer("constant", constant_buffer(shape, values, sequence_dim))

    def broadcast(self, x, operation):
        """Align the buffer to `x`'s rank and narrow a sequence-axis prefix to the input length."""
        if not x.is_floating_point() or self.constant.ndim > x.ndim:
            raise ValueError(f"{operation} requires floating input with compatible rank")
        offset = x.ndim - self.constant.ndim
        shape = (1,) * offset + tuple(self.constant.shape)
        if shape[0] != 1:
            raise ValueError(f"{operation} must broadcast across the batch axis")
        if self.sequence_dim is not None and self.sequence_dim >= x.ndim:
            raise ValueError("Constant sequence_dim exceeds input rank")
        constant = self.constant
        for dim, (size, actual) in enumerate(zip(shape, x.shape)):
            if dim == self.sequence_dim:
                if dim < offset or actual > size:
                    raise ValueError("Input sequence exceeds constant buffer capacity")
                constant = constant.narrow(dim - offset, 0, actual)
            elif size not in (1, actual):
                raise ValueError("Constant shape does not broadcast to the input")
        return constant

    def forward(self, x):
        return x + self.broadcast(x, "ConstantAdd")


class TensorLabConstantMul(nn.Module):
    """Multiplies by a fixed buffer -- the operation rotary position embedding needs.

    `rotary(x) = x * cos + rotate_half(x) * sin` is unreachable with an additive-only constant
    (a rotation is not a translation), so this is the primitive that makes RoPE expressible.
    """

    def __init__(self, shape, values, sequence_dim=None):
        super().__init__()
        self.sequence_dim = sequence_dim
        self.register_buffer("constant", constant_buffer(shape, values, sequence_dim))

    def forward(self, x):
        return x * TensorLabConstantAdd.broadcast(self, x, "ConstantMul")


class TensorLabCausalMask(nn.Module):
    """Sets every attention score whose key comes after its query to -inf, as a decoder does.

    `query_dim` and `key_dim` name the two position axes of the score tensor: the last two for
    `q @ k.transpose(-2, -1)`, others for layouts such as `bsht`. Both must be equally long. A
    softmax over the key axis then weighs each position and the ones before it only.
    """

    def __init__(self, query_dim=-2, key_dim=-1):
        super().__init__()
        self.query_dim, self.key_dim = query_dim, key_dim

    def forward(self, x):
        query, key = self.query_dim % x.ndim, self.key_dim % x.ndim
        if query == key or 0 in (query, key) or x.shape[query] != x.shape[key]:
            raise ValueError("CausalMask needs two distinct, equally long non-batch position axes")
        size = x.shape[query]
        blocked = torch.ones(size, size, dtype=torch.bool, device=x.device).triu(1)
        # `blocked` is laid out [query, key]; give it the score's rank with both axes in place.
        shape = [1] * x.ndim
        shape[query] = shape[key] = size
        return x.masked_fill((blocked if query < key else blocked.T).reshape(shape), float("-inf"))


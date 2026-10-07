"""Bounded, torch-free evaluator for static float32 tensor constants."""
from __future__ import annotations

import itertools
import math
import struct
from dataclasses import dataclass

MAX_ELEMENTS = 65_536
MAX_WORK = 1_000_000


class StaticTensorError(ValueError):
    pass


class _Budget:
    def __init__(self):
        self.used = 0

    def charge(self, count):
        if count < 0 or self.used + count > MAX_WORK:
            raise StaticTensorError("静态张量计算超过 1000000 次运算限制")
        self.used += count


def checked_shape(shape):
    if not isinstance(shape, (list, tuple)) or len(shape) > 5 or any(type(v) is not int or not 0 <= v <= MAX_ELEMENTS for v in shape):
        raise StaticTensorError("静态张量 shape 需要最多 5 维非负整数")
    count = math.prod(shape)
    if count > MAX_ELEMENTS: raise StaticTensorError("静态张量元素数超过 65536")
    return count


def f32(value):
    try:
        value = struct.unpack("!f", struct.pack("!f", float(value)))[0]
    except (OverflowError, struct.error) as exc:
        raise StaticTensorError("静态张量包含超出 float32 范围的数值") from exc
    if not math.isfinite(value):
        raise StaticTensorError("静态张量不能包含 NaN 或 Inf")
    return value


@dataclass
class StaticTensor:
    shape: list[int]
    values: list[float]
    budget: _Budget
    dtype: str = "float32"

    def __post_init__(self):
        count = checked_shape(self.shape)
        self.budget.charge(count)
        self.shape = list(self.shape)
        if len(self.values) != count:
            raise StaticTensorError("静态张量形状和值数量不一致")
        if self.dtype != "float32":
            raise StaticTensorError("静态张量仅支持 float32")
        self.values = [f32(v) for v in self.values]

    @classmethod
    def filled(cls, shape, value, budget):
        count = checked_shape(shape)
        budget.charge(count)
        return cls(list(shape), [f32(value)] * count, budget)

    @classmethod
    def arange(cls, start, end, step, budget):
        if not all(math.isfinite(float(v)) for v in (start, end, step)) or step == 0:
            raise StaticTensorError("arange 参数无效")
        count = max(0, int(math.ceil((end - start) / step))) if (end - start) * step > 0 else 0
        if count > MAX_ELEMENTS:
            raise StaticTensorError("静态张量元素数超过 65536")
        budget.charge(count)
        return cls([count], [f32(start + i * step) for i in range(count)], budget)

    def unary(self, fn):
        self.budget.charge(len(self.values))
        return StaticTensor(self.shape, [f32(fn(v)) for v in self.values], self.budget)

    def _flat_at(self, index):
        if not self.shape:
            return self.values[0]
        offset = 0
        for size, item in zip(self.shape, index):
            offset = offset * size + item
        return self.values[offset]

    def binary(self, other, fn):
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            other = StaticTensor([], [f32(other)], self.budget)
        if not isinstance(other, StaticTensor):
            raise StaticTensorError("静态张量运算需要数值或静态张量")
        rank = max(len(self.shape), len(other.shape))
        left = [1] * (rank - len(self.shape)) + self.shape
        right = [1] * (rank - len(other.shape)) + other.shape
        shape = [a if a == b or b == 1 else b if a == 1 else -1 for a, b in zip(left, right)]
        if -1 in shape:
            raise StaticTensorError("静态张量形状无法广播")
        count = checked_shape(shape)
        self.budget.charge(count)
        out = []
        for index in itertools.product(*(range(v) for v in shape)):
            li = tuple(i if size != 1 else 0 for i, size in zip(index, left))
            ri = tuple(i if size != 1 else 0 for i, size in zip(index, right))
            out.append(f32(fn(self._flat_at(li[-len(self.shape):]) if self.shape else self.values[0], other._flat_at(ri[-len(other.shape):]) if other.shape else other.values[0])))
        return StaticTensor(shape, out, self.budget)

    def unsqueeze(self, dim):
        rank = len(self.shape)
        if dim < 0:
            dim += rank + 1
        if not 0 <= dim <= rank:
            raise StaticTensorError("unsqueeze 维度越界")
        return StaticTensor(self.shape[:dim] + [1] + self.shape[dim:], list(self.values), self.budget)

    def squeeze(self, dim="all"):
        if dim == "all":
            dims = [i for i, size in enumerate(self.shape) if size == 1]
        else:
            dims = [dim] if type(dim) is int else list(dim)
            if any(type(d) is not int for d in dims): raise StaticTensorError("squeeze 维度需要整数")
            rank = max(1, len(self.shape))
            if any(not -rank <= d < rank for d in dims): raise StaticTensorError("squeeze 维度越界")
            dims = [d % rank for d in dims]
            if len(set(dims)) != len(dims): raise StaticTensorError("squeeze 维度不能重复")
        return StaticTensor([size for i, size in enumerate(self.shape) if i not in set(dims) or size != 1], list(self.values), self.budget)

    def index(self, indices):
        if len(indices) != len(self.shape):
            raise StaticTensorError("静态索引维度不匹配")
        ranges = []
        output_shape = []
        for size, item in zip(self.shape, indices):
            if isinstance(item, int):
                value = item + size if item < 0 else item
                if not 0 <= value < size:
                    raise StaticTensorError("静态索引越界")
                ranges.append([value])
            elif isinstance(item, slice):
                if item.step == 0:
                    raise StaticTensorError("切片步长不能为零")
                values = list(range(*item.indices(size)))
                ranges.append(values)
                output_shape.append(len(values))
            else:
                raise StaticTensorError("不支持此静态索引")
        count = math.prod(len(v) for v in ranges)
        self.budget.charge(count)
        strides = []
        stride = 1
        for size in reversed(self.shape):
            strides.append(stride)
            stride *= size
        strides.reverse()
        flat = [self.values[sum(i * s for i, s in zip(index, strides))] for index in itertools.product(*ranges)]
        return StaticTensor(output_shape, flat, self.budget)

    def assign(self, indices, value):
        selected = self.index(indices)
        expanded = value if value.shape == selected.shape else value.broadcast_to(selected.shape)
        ranges = []
        for size, item in zip(self.shape, indices):
            if isinstance(item, int):
                ranges.append([item + size if item < 0 else item])
            elif isinstance(item, slice):
                ranges.append(list(range(*item.indices(size))))
            else:
                raise StaticTensorError("静态赋值只支持整数和切片")
        strides = []
        stride = 1
        for size in reversed(self.shape):
            strides.append(stride)
            stride *= size
        strides.reverse()
        self.budget.charge(len(expanded.values))
        for offset, index in enumerate(itertools.product(*ranges)):
            self.values[sum(i * s for i, s in zip(index, strides))] = expanded.values[offset]

    def broadcast_to(self, shape):
        if len(shape) < len(self.shape):
            raise StaticTensorError("静态张量无法广播")
        padded = [1] * (len(shape) - len(self.shape)) + self.shape
        if any(a not in (1, b) for a, b in zip(padded, shape)):
            raise StaticTensorError("静态张量无法广播")
        self.budget.charge(checked_shape(shape))
        strides = []
        stride = 1
        for size in reversed(self.shape):
            strides.append(stride)
            stride *= size
        strides.reverse()
        out = []
        for index in itertools.product(*(range(v) for v in shape)):
            source = tuple(i if size != 1 else 0 for i, size in zip(index, padded))
            if not self.shape:
                out.append(self.values[0])
            else:
                source = source[-len(self.shape):]
                out.append(self.values[sum(i * s for i, s in zip(source, strides))])
        return StaticTensor(list(shape), out, self.budget)


@dataclass(frozen=True)
class DynamicStaticSlice:
    tensor: StaticTensor
    buffer_dim: int
    sequence_dim: int
    source: object


def new_budget():
    return _Budget()

"""Bounded, single-sample graph inference and tensor inspection."""
from __future__ import annotations

import copy
import math
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, field_validator, model_validator

from .graph import analyze_expanded_graph, analyze_graph, expand_subgraphs

MAX_CAPTURE_BYTES = 64 * 1024 * 1024
# Layers the Attention Inspector reads: Q/K/V, scores, probabilities and outputs per head.
ATTENTION_OPS = ("Transformer", "MultiHeadAttention", "LatentAttention")
MAX_INPUT_ELEMENTS = 4_000_000
# An MoE layer's routing keeps its tokens-by-experts gate scores until the call returns.
MAX_ROUTING_ELEMENTS = 1_048_576
MAX_PARAMETERS = 5_000_000
MAX_PARAMETER_BYTES = 20 * 1024 * 1024
MAX_SLICE_SIDE = 16
# A channel-first tensor is also shown as a page of feature maps. Each map is one bounded
# MAX_SLICE_SIDE window, so a page is at most 16 windows of 16 x 16 values.
MAX_CHANNEL_MAPS = 16
# `allNodes` observes every executed layer at once, for the 3D view. Each layer is described as
# boundedly as a selected one; this caps how many layers one response can carry.
MAX_ALL_NODES = 256
# A layer's weights and biases, each described like an activation; an attention module or a
# Transformer layer has the most, about twenty.
MAX_PARAMETER_TENSORS = 32
HISTOGRAM_BINS = 12
# Explaining one output value lists its largest terms and input-channel sums; the rest are summed.
MAX_LISTED_TERMS = 16
MAX_LISTED_CHANNELS = 16
# Every term is recomputed in float64, a few arrays of this length; it bounds that scratch.
MAX_EXPLAINED_TERMS = 1_048_576
EXPLAINED_RULES = {
    **{op: "weighted-sum" for op in ("Linear", "Conv1d", "Conv2d", "Conv3d")},
    **{op: "mean" for op in ("AvgPool1d", "AvgPool2d", "AvgPool3d", "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d")},
    **{op: "max" for op in ("MaxPool1d", "MaxPool2d", "MaxPool3d", "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d")},
}


class AttentionSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    branch: StrictInt = Field(0, ge=0, le=7)
    head: StrictInt = Field(0, ge=0, le=15)
    queryStart: StrictInt = Field(0, ge=0, le=65535)
    keyStart: StrictInt = Field(0, ge=0, le=65535)


class RoutingSelection(BaseModel):
    """The position x expert window of an MoE layer's routing to show; the load covers every expert.
    A position is one row the layer routes (the token of a language model; without tokenizer
    metadata the workbench calls it a position)."""
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    positionStart: StrictInt = Field(0, ge=0, le=4_000_000)
    expertStart: StrictInt = Field(0, ge=0, le=63)


class ValueSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    nodeId: StrictStr = Field(min_length=1, max_length=512)
    # The output element to explain, without the batch axis: single-sample inference has batch 1.
    index: list[StrictInt] = Field(min_length=1, max_length=4)


class GradientSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    # The class whose cross-entropy is backpropagated; logits have at most 256 classes.
    target: StrictInt = Field(ge=0, le=255)


class TensorInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    graph: dict[str, Any]
    nodeIds: list[StrictStr] = Field(default_factory=list, max_length=8)
    allNodes: StrictBool = False
    # One layer whose weights and biases to describe, from the model this request runs.
    parameters: StrictStr | None = Field(default=None, min_length=1, max_length=512)
    parameterSlices: dict[StrictStr, list[StrictInt]] = Field(default_factory=dict, max_length=8)
    seed: StrictInt = Field(42, ge=0, le=2_147_483_647)
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    modelId: StrictStr | None = Field(default=None, min_length=1, max_length=128)
    inputs: dict[str, list[StrictFloat | StrictInt]] | None = Field(default=None, max_length=8)
    slices: dict[str, list[StrictInt]] = Field(default_factory=dict, max_length=8)
    attention: dict[StrictStr, AttentionSelection] = Field(default_factory=dict, max_length=8)
    # Routing windows of selected MoE layers; every selected MoE layer reports its routing, at the
    # top-left window unless one is chosen here.
    routing: dict[StrictStr, RoutingSelection] = Field(default_factory=dict, max_length=8)
    # One output element of a selected Linear, convolution or pooling layer, split into its terms.
    explain: ValueSelection | None = None
    # One backward pass of the output's cross-entropy against this class.
    gradient: GradientSelection | None = None

    @field_validator("nodeIds")
    @classmethod
    def unique_node_ids(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("nodeIds must not contain duplicates")
        return value

    @field_validator("slices")
    @classmethod
    def bounded_slice_indices(cls, value: dict[str, list[int]]) -> dict[str, list[int]]:
        if any(len(indices) > 3 for indices in value.values()):
            raise ValueError("Each slice may contain at most 3 prefix indices")
        return value

    @model_validator(mode="after")
    def one_kind_of_selection(self) -> "TensorInferenceRequest":
        # Either named layers (with optional attention windows), or every layer and nothing else.
        if self.allNodes:
            if self.nodeIds or self.attention or self.routing or self.explain or self.gradient:
                raise ValueError("allNodes observes every layer; leave nodeIds, attention, routing, explain and gradient empty")
        elif not self.nodeIds:
            raise ValueError("nodeIds must name at least one node")
        return self


class InferenceError(ValueError):
    """A user-correctable inference request error."""


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _share(value: float) -> float | None:
    """A summary that lies in [0, 1] by definition, without the float32 rounding that can step past
    an end; a non-finite input has no share at all."""
    return min(1.0, max(0.0, float(value))) if _finite(value) else None


def _shape_elements(shape: list[int]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


def _activation_cone(info: dict[str, Any], selected: list[str]) -> set[str]:
    """被检节点及其全部祖先。

    单样本检查只需要这些层的中间张量；其余层即使存在也不参与捕获。用整模型激活总和
    做闸门会让「只想看第 1 层」在大模型上直接失败，这才是它的实际瓶颈。
    """
    by_id = {node["id"]: node for node in info["nodes"]}
    keep: set[str] = set()
    stack = [node_id for node_id in selected if node_id in by_id]
    while stack:
        key = stack.pop()
        if key in keep:
            continue
        keep.add(key)
        for edge in info["baseEdges"].get(key, []):
            stack.append(edge["source"])
        for edge in info.get("overrideEdges", {}).get(key, []):
            stack.append(edge["source"])
    return keep


def _validate_limits(info: dict[str, Any], label: str, selected: list[str] | None = None) -> None:
    # 有选中节点时按祖先锥计账，没有选中就退回到整模型（那正是完整前向的真实占用）。
    if selected:
        cone = _activation_cone(info, selected)
        peak = sum(_shape_elements(info["shapes"][key][1:]) * 4 for key in cone if key in info["shapes"])
    else:
        peak = info["activationBytesPerSample"]
    if peak > MAX_CAPTURE_BYTES:
        raise InferenceError(f"{label} activation inspection exceeds 64 MB")
    if info["totalParameters"] > MAX_PARAMETERS or info["totalParameters"] * 4 > MAX_PARAMETER_BYTES:
        raise InferenceError("Inference model exceeds the 5M parameter / 20 MB limit")
    if any(_shape_elements(shape) > MAX_INPUT_ELEMENTS for key, shape in info["shapes"].items() if key in info["inputs"]):
        raise InferenceError("An input tensor exceeds the 4M element limit")
    by_id = {node["id"]: node for node in info["nodes"]}
    for key, node in by_id.items():
        shape = info["shapes"][key]
        if node["op"].startswith("InstanceNorm") and math.prod(shape[2:]) == 1:
            raise InferenceError(f"{key}: InstanceNorm requires more than one spatial element for single-sample eval")
        if node["op"] not in ATTENTION_OPS:
            continue
        p = node["params"]
        heads = int(p.get("num_heads", 4 if node["op"] == "LatentAttention" else 1)); branches = int(p.get("branches", 1))
        query = shape[1]
        score_keys = []
        for branch in range(branches):
            port = f"b{branch}:k0"
            source_shape = info["portShapes"].get(key, {}).get(port)
            if source_shape is None:
                source_shape = shape
            score_keys.append(source_shape[1])
        if heads * query * sum(score_keys) > 1_000_000:
            raise InferenceError(f"{key}: attention score matrix exceeds the 1M inference limit")


def _embedding_inputs(info: dict[str, Any]) -> dict[str, int]:
    by_id = {node["id"]: node for node in info["nodes"]}
    result: dict[str, int] = {}
    for node in info["nodes"]:
        if node["op"] != "Embedding":
            continue
        edges = info["baseEdges"].get(node["id"], [])
        if not edges:
            continue
        source = edges[0]["source"]
        if by_id.get(source, {}).get("op") != "Input":
            continue
        vocabulary = int(node["params"].get("num_embeddings", 100))
        result[source] = min(result.get(source, vocabulary), vocabulary)
    # A single input cannot be both integer IDs and floating point data.
    for source in result:
        direct_ops = []
        for key, edges in info["baseEdges"].items():
            if any(edge["source"] == source for edge in edges):
                direct_ops.append(by_id[key]["op"])
        if any(op != "Embedding" for op in direct_ops):
            raise InferenceError(f"Input {source} feeds Embedding and non-Embedding consumers")
    return result


def _prepare_graph(graph: dict[str, Any], selected: list[str] | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    # Analyze the submitted graph first: rewriting B must not make an invalid source valid.
    analyze_graph(graph)
    # Folded blocks carry no operator semantics, so inference expands them into the real
    # operators before the single-sample batch rewrite below touches every node.
    copied = expand_subgraphs(copy.deepcopy(graph))
    for node in copied["nodes"]:
        if node["op"] == "Input":
            node["params"]["shape"] = [1, *node["params"]["shape"][1:]]
    inference_info = analyze_expanded_graph(copied)
    _validate_limits(inference_info, "Single-sample graph", selected)
    return copied, inference_info, _embedding_inputs(inference_info)


def _input_tensors(provided: dict[str, list[Any]] | None, info: dict[str, Any], embedding: dict[str, int], torch: Any, generator: Any) -> tuple[dict[str, Any], str]:
    """One sample per Input node: the flattened values given, or a seeded synthetic sample."""
    input_ids = list(info["inputs"])
    if provided is not None and set(provided) != set(input_ids):
        raise InferenceError("Provided inputs must contain exactly every Input node ID")
    if provided is None:
        source = "synthetic"
    else:
        source = "provided"
    tensors: dict[str, Any] = {}
    for key in input_ids:
        shape = info["shapes"][key]
        count = _shape_elements(shape[1:])
        values = None if provided is None else provided[key]
        if values is not None:
            if len(values) != count:
                raise InferenceError(f"Input {key} requires exactly {count} values")
            if any(not _finite(value) or abs(float(value)) > 3.4028234663852886e38 for value in values):
                raise InferenceError(f"Input {key} contains non-finite or out-of-range values")
            if key in embedding:
                if any(float(value) != int(float(value)) for value in values):
                    raise InferenceError(f"Embedding input {key} requires integer IDs")
                limit = embedding[key]
                if any(int(float(value)) < 0 or int(float(value)) >= limit for value in values):
                    raise InferenceError(f"Embedding input {key} IDs must be in [0,{limit - 1}]")
                tensors[key] = torch.tensor([int(float(value)) for value in values], dtype=torch.int64).reshape(shape)
            else:
                tensors[key] = torch.tensor(values, dtype=torch.float32).reshape(shape)
            continue
        if key in embedding:
            tensors[key] = torch.randint(embedding[key], shape, generator=generator, dtype=torch.int64)
        else:
            tensors[key] = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.25
    return tensors, source


def _histogram(values: Any, torch: Any) -> dict[str, list[Any]]:
    finite = values[torch.isfinite(values)].to(torch.float64)
    if finite.numel() == 0:
        return {"edges": [float(i) / HISTOGRAM_BINS for i in range(HISTOGRAM_BINS + 1)], "counts": [0] * HISTOGRAM_BINS}
    low = float(finite.min().item())
    high = float(finite.max().item())
    if low == high:
        pad = max(abs(low) * 0.01, 0.5)
        lo, hi = low - pad, low + pad
        if not math.isfinite(lo) or not math.isfinite(hi):
            lo, hi = math.nextafter(low, -math.inf), math.nextafter(low, math.inf)
    else:
        lo, hi = low, high
    scale = max(abs(lo), abs(hi), 1.0)
    normalized_lo, normalized_hi = lo / scale, hi / scale
    if not math.isfinite(normalized_lo) or not math.isfinite(normalized_hi) or normalized_lo == normalized_hi:
        raise InferenceError("Tensor histogram range cannot be represented safely")
    edges = [lo + (hi - lo) * i / HISTOGRAM_BINS for i in range(HISTOGRAM_BINS + 1)]
    if not all(math.isfinite(edge) for edge in edges):
        edges = [normalized_lo + (normalized_hi - normalized_lo) * i / HISTOGRAM_BINS for i in range(HISTOGRAM_BINS + 1)]
    normalized = finite / scale
    boundaries = torch.tensor([normalized_lo + (normalized_hi - normalized_lo) * i / HISTOGRAM_BINS for i in range(1, HISTOGRAM_BINS)], dtype=torch.float64, device=normalized.device)
    counts = torch.bucketize(normalized, boundaries, right=True)
    counts = torch.bincount(counts, minlength=HISTOGRAM_BINS).tolist()
    return {"edges": edges, "counts": [int(value) for value in counts[:HISTOGRAM_BINS]]}


def _slice(value: Any, indices: list[int], torch: Any) -> dict[str, Any]:
    shape = list(value.shape)
    rank = len(shape)
    trailing = shape[-2:] if rank >= 2 else ([shape[0]] if rank == 1 else [1])
    rows = int(trailing[0]) if len(trailing) == 2 else 1
    columns = int(trailing[-1])
    # Prefix indexes select all leading axes; remaining display is top-left bounded 16x16.
    selected = value
    if rank > 2:
        selected = value[tuple(indices)]
        display_shape = list(selected.shape)
        rows, columns = (int(display_shape[-2]), int(display_shape[-1])) if len(display_shape) >= 2 else (1, int(display_shape[-1]))
    elif rank == 1:
        selected = value
    elif rank == 0:
        selected = value.reshape(1)
    display_rows, display_cols = min(rows, MAX_SLICE_SIDE), min(columns, MAX_SLICE_SIDE)
    if rank == 0:
        matrix = [[selected.detach().cpu().item()]]
    elif rank == 1:
        matrix = [selected.detach().cpu()[:display_cols].tolist()]
    else:
        matrix = selected[..., :display_rows, :display_cols].detach().cpu().reshape(display_rows, display_cols).tolist()
    values: list[list[float | None]] = []
    for row in matrix:
        values.append([float(item) if _finite(item) else None for item in row])
    return {
        "indices": indices,
        "shape": [rows, columns] if rank != 1 else [columns],
        "rows": display_rows,
        "columns": display_cols,
        "values": values,
        "truncated": rows > display_rows or columns > display_cols,
    }


def _channel_page(value: Any, indices: list[int], torch: Any) -> dict[str, Any] | None:
    """Feature maps for the page of channels around the selected one, or None.

    Only a rank-4/5 tensor has a channel axis followed by spatial axes, so only `[1,C,H,W]`
    and `[1,C,D,H,W]` get this view; a sequence `[1,S,E]` has no channel axis to page through
    and its position x feature slice already is its semantic view. Every map is the same
    top-left window the main slice shows, taken at the same depth index for a rank-5 tensor.
    The statistics describe the whole channel, not the window, so a dead channel reads as dead
    even when the window happens to miss its few active elements.

    Channels are reduced one at a time. The scratch this needs never exceeds one channel, which
    is less than `_describe` already uses to summarise the whole tensor.
    """
    shape = [int(size) for size in value.shape]
    if len(shape) not in (4, 5) or len(indices) != len(shape) - 2:
        return None
    channels, selected = shape[1], int(indices[1])
    start = selected // MAX_CHANNEL_MAPS * MAX_CHANNEL_MAPS
    count = min(MAX_CHANNEL_MAPS, channels - start)
    rows, columns = shape[-2], shape[-1]
    display_rows, display_cols = min(rows, MAX_SLICE_SIDE), min(columns, MAX_SLICE_SIDE)
    maps: list[list[list[float | None]]] = []
    stats: list[dict[str, float | None]] = []
    for channel in range(start, start + count):
        whole = value[indices[0], channel]
        plane = whole[tuple(indices[2:])] if len(shape) == 5 else whole
        window = plane[:display_rows, :display_cols].detach().cpu().tolist()
        maps.append([[float(item) if _finite(item) else None for item in row] for row in window])
        finite = whole[torch.isfinite(whole)].to(torch.float64)
        if finite.numel():
            stats.append({"min": float(finite.min().item()), "max": float(finite.max().item()), "mean": float(finite.mean().item())})
        else:
            stats.append({"min": None, "max": None, "mean": None})
    return {
        "start": start,
        "count": count,
        "total": channels,
        "selected": selected,
        "shape": [rows, columns],
        "rows": display_rows,
        "columns": display_cols,
        "truncated": rows > display_rows or columns > display_cols,
        "maps": maps,
        "stats": stats,
    }


def _describe(node_id: str, value: Any, indices: list[int], torch: Any) -> dict[str, Any]:
    detached = value.detach()
    flat = detached.reshape(-1)
    finite_mask = torch.isfinite(flat)
    finite = flat[finite_mask]
    finite_count = int(finite.numel())
    stats: dict[str, float | None]
    if finite_count:
        robust = finite.to(torch.float64)
        mean = float(robust.mean().item())
        variance = float(((robust - mean) ** 2).mean().item())
        stats = {"min": float(robust.min().item()), "max": float(robust.max().item()), "mean": mean, "std": math.sqrt(max(0.0, variance))}
    else:
        stats = {"min": None, "max": None, "mean": None, "std": None}
    return {
        "nodeId": node_id,
        "shape": [int(size) for size in detached.shape],
        "dtype": str(detached.dtype).replace("torch.", ""),
        "elements": int(detached.numel()),
        "finiteCount": finite_count,
        "nonFiniteCount": int(flat.numel()) - finite_count,
        "stats": stats,
        "histogram": _histogram(flat, torch),
        "slice": _slice(detached, indices, torch),
        "channels": _channel_page(detached, indices, torch),
    }


def _attention_matrix(name: str, value: Any, row_start: int, column_start: int, torch: Any) -> dict[str, Any]:
    """Describe one full sample/head matrix and expose only a bounded offset window."""
    matrix = value.detach().to(torch.float32)
    if matrix.ndim != 2:
        matrix = matrix.reshape(matrix.shape[-2], matrix.shape[-1])
    rows, columns = (int(matrix.shape[0]), int(matrix.shape[1]))
    end_row = min(rows, row_start + MAX_SLICE_SIDE)
    end_column = min(columns, column_start + MAX_SLICE_SIDE)
    window = matrix[row_start:end_row, column_start:end_column].detach().cpu().tolist()
    values = [[float(item) if _finite(item) else None for item in row] for row in window]
    described = _describe(name, matrix, [], torch)
    described["slice"] = {
        "indices": [],
        "shape": [rows, columns],
        "rows": len(values),
        "columns": len(values[0]) if values else 0,
        "values": values,
        "truncated": row_start != 0 or column_start != 0 or len(values) != rows or (len(values[0]) if values else 0) != columns,
    }
    described["rowStart"] = row_start
    described["columnStart"] = column_start
    return described


def _explain_scratch(explain: ValueSelection | None, info: dict[str, Any], selected: list[str]) -> int:
    """Check the value to explain before anything is allocated; return the float64 scratch its
    terms need, which joins the attention scratch in the 64 MiB inspection budget."""
    if explain is None:
        return 0
    node = next((item for item in info["nodes"] if item["id"] == explain.nodeId), None)
    if node is None or explain.nodeId not in selected:
        raise InferenceError("explain must name one of the selected nodeIds")
    op = node["op"]
    if op not in EXPLAINED_RULES:
        raise InferenceError(f"{explain.nodeId}: {op} output values cannot be explained; choose a Linear, convolution or pooling layer")
    shape = info["shapes"][explain.nodeId]
    if len(explain.index) != len(shape) - 1 or any(index < 0 or index >= int(size) for index, size in zip(explain.index, shape[1:])):
        raise InferenceError(f"{explain.nodeId}: explain index needs {len(shape) - 1} entries inside the output shape {shape[1:]}")
    edge = info["baseEdges"][explain.nodeId][0]
    source = info["portShapes"][edge["source"]][edge["sourcePort"]] if edge.get("sourcePort") else info["shapes"][edge["source"]]
    params, spatial = node["params"], len(source) - 2
    if op == "Linear":
        terms = int(source[-1])
    elif op.startswith("Conv"):
        terms = int(source[1]) // int(params.get("groups", 1)) * int(params.get("kernel_size", 3)) ** spatial
    elif op.startswith("Adaptive"):
        terms = math.prod(-(-int(size) // int(out)) + 1 for size, out in zip(source[2:], shape[2:]))
    else:
        terms = int(params.get("kernel_size", 2)) ** spatial
    if terms > MAX_EXPLAINED_TERMS:
        raise InferenceError(f"{explain.nodeId}: the explained value has {terms} terms, more than the {MAX_EXPLAINED_TERMS} limit")
    # Window, mask, weights, products, the sort key and its order.
    return terms * 8 * 6


def _validate_gradient(gradient: GradientSelection | None, info: dict[str, Any]) -> None:
    """A backward pass needs classification logits to take the cross-entropy of, and memory for
    every activation autograd keeps plus its gradient; both are checked before allocation."""
    if gradient is None:
        return
    shape = info["shapes"][info["output"]]
    if len(shape) != 2 or not 2 <= int(shape[1]) <= 256:
        raise InferenceError("Gradient inspection needs classification logits [B, C] with 2-256 classes at the Output")
    if gradient.target >= int(shape[1]):
        raise InferenceError(f"Gradient target class must be in [0,{int(shape[1]) - 1}]")
    # Autograd keeps what the backward pass needs of every activation, and each gets a gradient.
    if int(info["activationBytesPerSample"]) * 3 > MAX_CAPTURE_BYTES:
        raise InferenceError("Gradient inspection exceeds 64 MB: the backward pass keeps every activation and its gradient")


def _resolve_routing(requested: dict[str, RoutingSelection], info: dict[str, Any], selected: list[str]) -> dict[str, RoutingSelection]:
    """Every selected MoE layer with the window it reports: the requested one, or the top-left."""
    by_id = {node["id"]: node for node in info["nodes"]}
    if set(requested) - set(selected):
        raise InferenceError("routing keys must refer to selected nodeIds")
    resolved = {}
    for node_id in selected:
        node = by_id[node_id]
        if node["op"] != "MoE":
            if node_id in requested:
                raise InferenceError(f"{node_id}: routing selection requires an MoE node")
            continue
        selection = requested.get(node_id, RoutingSelection())
        tokens = math.prod(info["shapes"][node_id][1:-1])
        experts = int(node["params"].get("experts", 8))
        if selection.positionStart >= tokens:
            raise InferenceError(f"{node_id}: positionStart is outside the {tokens} positions")
        if selection.expertStart >= experts:
            raise InferenceError(f"{node_id}: expertStart is outside the {experts} experts")
        if tokens * experts > MAX_ROUTING_ELEMENTS:
            raise InferenceError(f"{node_id}: routing capture exceeds {MAX_ROUTING_ELEMENTS} position-expert scores")
        resolved[node_id] = selection
    return resolved


def _capture_routing(node_id: str, selection: RoutingSelection, record: dict[str, Any], params: dict[str, Any], torch: Any) -> dict[str, Any]:
    """One MoE call's routing for a single sample: the gate's unbiased scores and the weight each
    token gives each expert (zero where it chose another), both at the selected window; the experts
    each windowed token chose; and every expert's load, the tokens it received."""
    scores = record["scores"].detach().float()
    indices = record["indices"].detach()
    weights = torch.zeros_like(scores).scatter_(1, indices, record["weights"].detach().float())
    tokens, experts = (int(size) for size in scores.shape)
    rows = range(selection.positionStart, min(tokens, selection.positionStart + MAX_SLICE_SIDE))
    return {
        "nodeId": node_id, "positions": tokens, "experts": experts, "topK": int(indices.shape[1]),
        "sharedExperts": int(params.get("shared_experts", 1)), "scoreFunc": str(params.get("score_func", "sigmoid")),
        "routeScale": float(params.get("route_scale", 1.0)), "positionStart": selection.positionStart, "expertStart": selection.expertStart,
        "scores": _attention_matrix("scores", scores, selection.positionStart, selection.expertStart, torch),
        "weights": _attention_matrix("weights", weights, selection.positionStart, selection.expertStart, torch),
        "chosen": [[int(expert) for expert in indices[row].tolist()] for row in rows],
        "load": [int(count) for count in torch.bincount(indices.flatten(), minlength=experts).tolist()],
    }


def _validate_attention(attention: dict[str, AttentionSelection], info: dict[str, Any], selected: list[str], other_scratch: int = 0) -> None:
    by_id = {node["id"]: node for node in info["nodes"]}
    unknown = set(attention) - set(selected)
    if unknown:
        raise InferenceError("attention keys must refer to selected nodeIds")
    extra_scratch = 0
    for node_id, selection in attention.items():
        node = by_id.get(node_id)
        if node is None or node["op"] not in ATTENTION_OPS:
            raise InferenceError(f"{node_id}: attention selection requires an attention node")
        params = node["params"]
        latent = node["op"] == "LatentAttention"
        branches = int(params.get("branches", 1))
        heads = int(params.get("num_heads", 4 if latent else 1))
        if selection.branch >= branches:
            raise InferenceError(f"{node_id}: attention branch is outside [0,{branches - 1}]")
        if selection.head >= heads:
            raise InferenceError(f"{node_id}: attention head is outside [0,{heads - 1}]")
        query_length = int(info["shapes"][node_id][1])
        # Latent attention has no projection ports: its keys are the query's own positions.
        key_shape = info["portShapes"].get(node_id, {}).get(f"b{selection.branch}:k0")
        key_length = int(key_shape[1]) if key_shape is not None else query_length
        if selection.queryStart >= query_length:
            raise InferenceError(f"{node_id}: queryStart is outside the query length")
        if selection.keyStart >= key_length:
            raise InferenceError(f"{node_id}: keyStart is outside the key length")
        if query_length * key_length > 262144:
            raise InferenceError(f"{node_id}: selected attention score matrix exceeds the 262144 element limit")
        embed = int(params.get("embed_dim", 64))
        if latent:
            # Q and K heads are the position-free part plus the rotary part; V heads have their own width.
            qk_dim = int(params.get("qk_nope_head_dim", 16)) + int(params.get("qk_rope_head_dim", 8))
            projection_elements = query_length * (embed * 2 + heads * qk_dim) + key_length * heads * (qk_dim + int(params.get("v_head_dim", 16)) * 2)
        else:
            head_dim = embed // heads
            kv_heads = int(params.get("kv_heads", 1 if params.get("attention_type") == "multi_query" else heads))
            # Captures retain all native projection/head stacks until the merged callback.
            projection_elements = query_length * embed * 4 + key_length * head_dim * kv_heads * 2
        extra_scratch += query_length * key_length * 64 + projection_elements * 32
    # 与张量检查同一口径：只按被检节点的祖先锥计账，否则深层模型里查看浅层注意力
    # 会被无关深层的激活拖到超限。
    cone = _activation_cone(info, selected)
    captured = sum(_shape_elements(info["shapes"][key][1:]) * 4 for key in cone if key in info["shapes"])
    if captured + extra_scratch + other_scratch > MAX_CAPTURE_BYTES:
        what = "Value explanation" if not extra_scratch else "Attention inspection and value explanation" if other_scratch else "Attention inspection"
        raise InferenceError(f"{what} scratch exceeds 64 MB")


def _head_overview(record: dict[str, Any], selection: AttentionSelection, causal: bool, torch: Any) -> list[dict[str, Any]]:
    """Every query head of the selected branch, side by side: each head's softmax probabilities at
    the selected window, at most 16 x 16, with two summaries over the whole matrix.

    `entropy` is the mean over query rows of the row's entropy divided by the largest it could be
    (the log of the keys that row may see, so causal rows count only their visible keys): near 0
    the head looks at one key, near 1 it spreads evenly. `peak` is the mean of each row's largest
    probability. Heads are computed one at a time, so the scratch is one score matrix, which the
    selected-head limit already bounds.
    """
    q_all, k_all, repeats = record["q"][0], record["k"][0], record["repeats"]
    overview = []
    for head in range(int(q_all.shape[0])):
        q, k = q_all[head].float(), k_all[head // repeats].float()
        with torch.no_grad():
            # The detail's exact expression, so the selected head's thumbnail equals its matrix.
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))
            rows, keys = int(scores.shape[-2]), int(scores.shape[-1])
            if causal:
                scores = scores.masked_fill(torch.ones((rows, keys), dtype=torch.bool, device=scores.device).triu(1), float("-inf"))
                visible = torch.arange(1, rows + 1, device=scores.device).clamp(max=keys).float()
            else:
                visible = torch.full((rows,), float(keys), device=scores.device)
            probabilities = torch.softmax(scores, dim=-1)
            entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-30))).sum(-1)
            largest = torch.log(visible)
            spread = torch.where(largest > 0, entropy / largest.clamp_min(1e-12), torch.zeros_like(entropy))
            peak = probabilities.max(-1).values
        window = probabilities[selection.queryStart:selection.queryStart + MAX_SLICE_SIDE, selection.keyStart:selection.keyStart + MAX_SLICE_SIDE].detach().cpu().tolist()
        overview.append({
            "head": head, "kvHead": head // repeats, "rows": len(window), "columns": len(window[0]) if window else 0,
            "values": [[float(item) if _finite(item) else None for item in row] for row in window],
            "entropy": _share(spread.mean().item()), "peak": _share(peak.mean().item()),
        })
    return overview


def _capture_attention(node_id: str, selection: AttentionSelection, record: dict[str, Any], torch: Any) -> dict[str, Any]:
    q = record["q"][0, selection.head]
    native_k = record["k"][0, selection.head // record["repeats"]]
    native_v = record["v"][0, selection.head // record["repeats"]]
    weighted = record["weighted"][0, selection.head]
    scale = 1.0 / math.sqrt(q.shape[-1])
    causal = bool(record["params"].get("causal"))
    with torch.no_grad():
        scores = (q.float() @ native_k.float().transpose(-2, -1)) * scale
        if causal:
            # The executed SDPA sets every key after the query's own position to -inf before the
            # softmax. `scores` stays the unmasked product, so its statistics are those of real
            # numbers; the mask is applied to what the softmax receives, which makes the masked
            # probabilities exactly zero.
            future = torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device).triu(1)
            probabilities = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
        else:
            probabilities = torch.softmax(scores, dim=-1)
    tensors = {
        "q": _attention_matrix("q", q, selection.queryStart, 0, torch),
        "k": _attention_matrix("k", native_k, selection.keyStart, 0, torch),
        "v": _attention_matrix("v", native_v, selection.keyStart, 0, torch),
        "scores": _attention_matrix("scores", scores, selection.queryStart, selection.keyStart, torch),
        "probabilities": _attention_matrix("probabilities", probabilities, selection.queryStart, selection.keyStart, torch),
        "headOutput": _attention_matrix("headOutput", weighted, selection.queryStart, 0, torch),
        "branchOutput": _attention_matrix("branchOutput", record["output"][0], selection.queryStart, 0, torch),
        "mergedOutput": _attention_matrix("mergedOutput", record["merged"][0], selection.queryStart, 0, torch),
    }
    params = record["params"]
    return {
        "nodeId": node_id, "branch": selection.branch, "head": selection.head,
        "queryStart": selection.queryStart, "keyStart": selection.keyStart,
        "kvHead": selection.head // record["repeats"], "numHeads": int(params["heads"]),
        "kvHeads": int(params["kv_heads"]), "branches": int(params["branches"]),
        "headDim": int(q.shape[-1]), "queryLength": int(q.shape[-2]), "keyLength": int(native_k.shape[-2]),
        "scale": scale, "mask": "causal" if causal else "none", "dropout": 0, "scoreSource": "projected-qk",
        "outputSource": "scaled_dot_product_attention", "tensors": tensors,
        "overview": _head_overview(record, selection, causal, torch),
    }


def _resolve_slices(requested: dict[str, list[int]], info: dict[str, Any], by_id: dict[str, Any]) -> dict[str, list[int]]:
    """Leading indices for every node: the requested ones, checked, and zeros elsewhere."""
    slices: dict[str, list[int]] = {}
    for node_id, indices in requested.items():
        if node_id not in by_id:
            raise InferenceError(f"Unknown slice node {node_id}")
        rank = len(info["shapes"][node_id])
        prefix_rank = max(0, rank - 2)
        if len(indices) > 3 or len(indices) != prefix_rank:
            raise InferenceError(f"Slice {node_id} requires {prefix_rank} leading indices (maximum 3)")
        for index, size in zip(indices, info["shapes"][node_id]):
            if index < 0 or index >= size:
                raise InferenceError(f"Slice {node_id} index is outside tensor shape")
        slices[node_id] = list(indices)
    for node_id, shape in info["shapes"].items():
        prefix_rank = max(0, len(shape) - 2)
        if prefix_rank > 3:
            raise InferenceError(f"Slice {node_id} has more than 3 leading axes")
        slices.setdefault(node_id, [0] * prefix_rank)
    return slices


def _device_name(requested: str, torch: Any) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise InferenceError("CUDA is unavailable in this PyTorch installation")
    return "cuda" if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()) else "cpu"


def describe_parameters(model: Any, node_id: str, slices: dict[str, list[int]], torch: Any, gradients: bool = False) -> list[dict[str, Any]]:
    """The weights and biases a layer runs with, each described like an activation.

    They come from the very module the forward pass used, so they are the weights behind the
    measured values. A layer that shares weights shows its owner's (`owner` names it), and a
    folded layer its first instance. A rank-4 convolution weight `[out, in, kh, kw]` is read like a
    feature map: the window is one filter's kernel for one input channel (leading indices, zeros
    by default) and the channel page shows that filter's kernels across input channels.

    With `gradients`, each tensor is the gradient a backward pass left on that parameter instead;
    a parameter the pass did not reach is left out. Shared weights hold the sum over every use.
    """
    from torch import nn
    owner = getattr(model, "shared", {}).get(node_id, node_id)
    module = model.layers[owner] if owner in model.layers else None
    if module is None:
        return []
    if isinstance(module, nn.ModuleList):
        module = module[0]
    described = []
    for name, value in list(module.named_parameters())[:MAX_PARAMETER_TENSORS]:
        if gradients:
            if value.grad is None:
                continue
            value = value.grad
        prefix_rank = max(0, value.dim() - 2)
        indices = list(slices.get(name, [0] * prefix_rank))
        if len(indices) != prefix_rank or prefix_rank > 3 or any(index < 0 or index >= size for index, size in zip(indices, value.shape)):
            raise InferenceError(f"Parameter {name} of {node_id} needs {prefix_rank} leading indices inside its shape")
        entry = _describe(node_id, value.detach(), indices, torch)
        entry["name"] = name
        if owner != node_id:
            entry["owner"] = owner
        described.append(entry)
    return described


def _number(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _per_axis(value: Any, count: int) -> list[int]:
    """A module's size argument, an int or one per spatial axis, as one int per axis."""
    return [int(value)] * count if isinstance(value, int) else [int(item) for item in value]


def _unravel(flat: int, shape: list[int]) -> list[int]:
    position = []
    for size in reversed(shape):
        flat, rest = divmod(flat, size)
        position.append(rest)
    return position[::-1]


def _window(x: Any, first: int, channels: int, starts: list[int], extents: list[int], steps: list[int], torch: Any) -> tuple[Any, Any, list[list[int]]]:
    """`x[0, first:first + channels]` at `start + k * step` (k < extent) on every spatial axis, in
    float64 on the CPU, with the mask of positions inside the tensor and each axis's positions.
    A position outside reads the padding; its value here means nothing until it is masked."""
    block = x[0, first:first + channels]
    inside = None
    coordinates = []
    for axis, (start, extent, step) in enumerate(zip(starts, extents, steps)):
        size = int(block.shape[axis + 1])
        positions = torch.arange(extent, device=block.device) * step + start
        block = block.index_select(axis + 1, positions.clamp(0, size - 1))
        shape = [1] * len(starts)
        shape[axis] = extent
        valid = ((positions >= 0) & (positions < size)).cpu().reshape(shape)
        inside = valid if inside is None else inside & valid
        coordinates.append(positions.cpu().tolist())
    return block.double().cpu(), inside, coordinates


def _explain_value(node_id: str, op: str, module: Any, x: Any, y: Any, index: list[int], torch: Any) -> dict[str, Any]:
    """One output element as the measured terms it is made of.

    A Linear or convolution output is Σ x·w + b over the input values and weights the forward
    pass used, where padding reads zeros. Average pooling is Σ x ÷ divisor over its window, as
    PyTorch counts it; max pooling is the largest value inside the window. Every term is
    recomputed in float64 from the module's actual input and weights, so `recomputed` checks the
    decomposition against the measured float32 `value`. The largest terms are listed (for max
    pooling, the largest values) and the rest are summed.
    """
    rule = EXPLAINED_RULES[op]
    spatial = x.dim() - 2
    bias = divisor = weights = channel_rest = None
    channels: list[dict[str, Any]] = []
    if op == "Linear":
        feature = index[-1]
        xs = x[(0, *index[:-1])].double().cpu()
        weights = module.weight[feature].detach().double().cpu()
        terms = xs * weights
        inside = considered = torch.ones(terms.shape, dtype=torch.bool)
        def place(position: list[int]) -> tuple[list[int], list[int] | None]:
            return [*index[:-1], position[0]], [feature, position[0]]
        if module.bias is not None:
            bias = float(module.bias[feature].item())
    elif rule == "weighted-sum":
        if isinstance(module.padding, str) or module.padding_mode != "zeros":
            raise InferenceError(f"{node_id}: only explicit zero padding can be explained")
        out_channel, position = index[0], index[1:]
        per_group = module.in_channels // module.groups
        first = out_channel // (module.out_channels // module.groups) * per_group
        starts = [p * s - pad for p, s, pad in zip(position, _per_axis(module.stride, spatial), _per_axis(module.padding, spatial))]
        window, inside, coordinates = _window(x, first, per_group, starts, _per_axis(module.kernel_size, spatial), _per_axis(module.dilation, spatial), torch)
        inside = inside.expand(window.shape)
        xs = window * inside
        weights = module.weight[out_channel].detach().double().cpu()
        terms = xs * weights
        considered = torch.ones(terms.shape, dtype=torch.bool)
        def place(position: list[int]) -> tuple[list[int], list[int] | None]:
            return [first + position[0], *(coordinates[axis][k] for axis, k in enumerate(position[1:]))], [out_channel, *position]
        sums = terms.reshape(per_group, -1).sum(1)
        chosen = torch.argsort(torch.where(torch.isnan(sums), math.inf, sums.abs()), descending=True, stable=True)[:MAX_LISTED_CHANNELS]
        channels = [{"channel": first + channel, "sum": _number(sums[channel])} for channel in chosen.tolist()]
        remaining = torch.ones(per_group, dtype=torch.bool)
        remaining[chosen] = False
        channel_rest = {"count": int(remaining.sum()), "sum": _number(sums[remaining].sum())}
        if module.bias is not None:
            bias = float(module.bias[out_channel].item())
    else:
        channel, position = index[0], index[1:]
        adaptive = op.startswith("Adaptive")
        paddings = [0] * spatial
        if adaptive:
            # PyTorch's adaptive windows: floor(i * size / out) up to ceil((i + 1) * size / out).
            sizes, outs = [int(v) for v in x.shape[2:]], [int(v) for v in y.shape[2:]]
            starts = [p * size // out for p, size, out in zip(position, sizes, outs)]
            extents = [-(-(p + 1) * size // out) - start for p, size, out, start in zip(position, sizes, outs, starts)]
            steps = [1] * spatial
        else:
            if module.ceil_mode:
                raise InferenceError(f"{node_id}: pooling with ceil_mode cannot be explained")
            extents, paddings = _per_axis(module.kernel_size, spatial), _per_axis(module.padding, spatial)
            starts = [p * s - pad for p, s, pad in zip(position, _per_axis(module.stride, spatial), paddings)]
            steps = _per_axis(getattr(module, "dilation", 1), spatial)
        window, inside, coordinates = _window(x, channel, 1, starts, extents, steps, torch)
        window, inside = window[0], inside.expand(window.shape[1:])
        def place(position: list[int]) -> tuple[list[int], list[int] | None]:
            return [channel, *(coordinates[axis][k] for axis, k in enumerate(position))], None
        if rule == "mean":
            include_pad = not adaptive and bool(module.count_include_pad)
            if not adaptive and module.divisor_override:
                divisor = int(module.divisor_override)
            elif include_pad:
                # The window clipped to the padded extent; without ceil_mode, the whole kernel.
                divisor = math.prod(min(start + extent, int(size) + pad) - start for start, extent, size, pad in zip(starts, extents, x.shape[2:], paddings))
            else:
                divisor = int(inside.sum())
            considered = torch.ones(window.shape, dtype=torch.bool) if include_pad else inside.clone()
            xs = window * inside
            terms = xs / divisor
        else:
            # Max pooling pads with -inf: positions outside never compete.
            considered, xs, terms = inside.clone(), window, window
    flat_terms, flat_x = terms.reshape(-1), xs.reshape(-1)
    flat_inside, flat_considered = inside.reshape(-1), considered.reshape(-1)
    flat_weights = None if weights is None else weights.reshape(-1)
    shape = [int(size) for size in terms.shape]
    count = int(flat_considered.sum())
    key = flat_terms if rule == "max" else flat_terms.abs()
    key = torch.where(flat_considered, torch.where(torch.isnan(flat_terms), math.inf, key), -math.inf)
    order = torch.argsort(key, descending=True, stable=True)[:min(MAX_LISTED_TERMS, count)]
    listed = []
    for flat in order.tolist():
        source, parameter = place(_unravel(flat, shape))
        listed.append({
            "input": source if bool(flat_inside[flat]) else None, "x": _number(flat_x[flat]),
            "parameter": parameter, "w": None if flat_weights is None else _number(flat_weights[flat]),
            "term": _number(flat_terms[flat]),
        })
    rest = flat_considered.clone()
    rest[order] = False
    chosen_terms = flat_terms[flat_considered]
    if rule == "max":
        recomputed = chosen_terms.max() if count else math.nan
    else:
        recomputed = chosen_terms.sum() + (bias or 0.0)
    return {
        "rule": rule, "value": _number(y[(0, *index)]), "recomputed": _number(recomputed),
        "bias": None if bias is None else _number(bias), "divisor": divisor,
        "count": count, "padded": int((~flat_inside).sum()), "terms": listed,
        "rest": {"count": int(rest.sum()), "sum": None if rule == "max" else _number(flat_terms[rest].sum())},
        "channels": channels, "channelRest": channel_rest,
    }


def _backward(output: Any, gradient: GradientSelection, selected: list[str], retained: dict[str, Any], slices: dict[str, list[int]], model: Any, torch: Any) -> dict[str, Any]:
    """Backpropagate the cross-entropy of the [1, C] logits against the chosen class, once, and
    describe each retained layer output's gradient like its value. The pass frees the graph."""
    if output.dim() != 2 or int(output.shape[0]) != 1 or not 2 <= int(output.shape[1]) <= 256:
        raise InferenceError("Gradient inspection needs classification logits [1, C] with 2-256 classes")
    logits = output.float()
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([gradient.target], device=logits.device))
    if not loss.requires_grad:
        raise InferenceError("The output depends on nothing trainable, so it has no gradient")
    model.zero_grad(set_to_none=True)
    loss.backward()
    with torch.no_grad():
        probabilities = torch.softmax(logits.detach(), dim=-1)[0]
        tensors = [_describe(node_id, retained[node_id].grad, slices[node_id], torch) for node_id in selected if node_id in retained and retained[node_id].grad is not None]
        return {
            "target": gradient.target, "classes": int(logits.shape[1]), "loss": _number(loss.detach()),
            "predicted": int(probabilities.argmax()), "probability": _number(probabilities[gradient.target]), "tensors": tensors,
        }


def _execute(model: Any, info: dict[str, Any], by_id: dict[str, Any], inputs: dict[str, Any], device_name: str, selected: list[str], slices: dict[str, list[int]], attention: dict[str, AttentionSelection], torch: Any, explain: ValueSelection | None = None, gradient: GradientSelection | None = None, routing: dict[str, RoutingSelection] | None = None) -> dict[str, Any]:
    """Run one eval forward pass and describe the selected nodes and attention windows.

    The inspector endpoint and the offline snapshot writer both measure through this, so a
    snapshot holds exactly the numbers the inspector would have shown for the same weights and
    sample. The caller builds the model, loads its weights and seeds the generators.

    `explain` splits one output value of a selected layer into its terms. Forward hooks on the
    layer's module record each call's input and output; the call the layer itself made is the
    latest one when the layer's value is observed, which holds for a layer that shares its
    owner's module and for the last instance of a folded one.

    `gradient` runs the same pass with autograd instead of inference mode, then backpropagates
    the output's cross-entropy against one class once. The model stays in eval mode, so dropout is
    off and BatchNorm uses its running statistics, and no weight changes. Floating-point inputs
    get a gradient too; integer IDs have none. The values are described as they are produced,
    from detached tensors, so they equal an inference-mode run; the graph is freed by the one
    backward pass and nothing here keeps it. Parameter gradients stay on the model for the caller.
    """
    from torch import nn
    model.to(device_name)
    model.eval()
    inputs = {key: value.to(device_name) for key, value in inputs.items()}
    wanted = set(selected)
    captured: dict[str, dict[str, Any]] = {}
    attention_records: dict[str, dict[str, Any]] = {}
    attention_results: dict[str, dict[str, Any]] = {}
    latest: dict[str, Any] = {}
    explained: dict[str, Any] = {}
    retained: dict[str, Any] = {}
    hooks = []
    instances: list[Any] = []
    if explain is not None:
        owner = getattr(model, "shared", {}).get(explain.nodeId, explain.nodeId)
        if owner not in model.layers:
            raise InferenceError(f"{explain.nodeId}: the explained layer has no module")
        layer = model.layers[owner]
        instances = list(layer) if isinstance(layer, nn.ModuleList) else [layer]
        for position, instance in enumerate(instances):
            hooks.append(instance.register_forward_hook(lambda module, args, output, position=position: latest.update(call=(module, args[0], output, position))))
    def observe(node_id: str, value: Any) -> None:
        if node_id in wanted:
            if gradient is not None:
                if value.requires_grad:
                    if not value.is_leaf:
                        value.retain_grad()
                    retained[node_id] = value
                value = value.detach()
            captured[node_id] = _describe(node_id, value, slices[node_id], torch)
        if explain is not None and node_id == explain.nodeId and "call" in latest:
            explained["call"] = latest["call"]
    def observe_attention(node_id: str, stage: str, branch: int, tensors: dict[str, Any]) -> None:
        selection = attention.get(node_id)
        if selection is None or (stage != "merged" and branch != selection.branch):
            return
        record = attention_records.setdefault(node_id, {})
        if stage == "merged":
            params = by_id[node_id]["params"]
            # Latent attention defaults to four causal heads, each with its own keys and values.
            latent = by_id[node_id]["op"] == "LatentAttention"
            heads = int(params.get("num_heads", 4 if latent else 1))
            kv_heads = heads if latent else int(params.get("kv_heads", 1 if params.get("attention_type") == "multi_query" else heads))
            record["params"] = {"heads": heads, "kv_heads": kv_heads, "branches": int(params.get("branches", 1)), "causal": bool(params.get("causal", 1 if latent else 0))}
            record["repeats"] = heads // kv_heads
            record["merged"] = tensors["output"].detach()
            attention_results[node_id] = _capture_attention(node_id, selection, record, torch)
            del attention_records[node_id]
        else:
            wanted_stage = {"q": "q", "k": "k", "v": "v", "weighted": "weighted", "output": "output"}[stage]
            record[wanted_stage] = tensors[wanted_stage].detach()
    routing = routing or {}
    routing_results: dict[str, dict[str, Any]] = {}
    def observe_routing(node_id: str, record: dict[str, Any]) -> None:
        if node_id in routing:
            routing_results[node_id] = _capture_routing(node_id, routing[node_id], record, by_id[node_id]["params"], torch)
    explanation = backward = None
    try:
        with torch.enable_grad() if gradient is not None else torch.inference_mode():
            if gradient is not None:
                inputs = {key: value.requires_grad_(True) if value.is_floating_point() else value for key, value in inputs.items()}
            output = model(inputs if len(info["inputs"]) > 1 else inputs[info["inputs"][0]], observer=observe, attention_observer=observe_attention if attention else None, routing_observer=observe_routing if routing else None)
            if gradient is not None:
                backward = _backward(output, gradient, selected, retained, slices, model, torch)
            del output
            if explain is not None:
                if "call" not in explained:
                    raise InferenceError(f"{explain.nodeId}: the explained layer did not run")
                module, value_in, value_out, position = explained.pop("call")
                op = by_id[explain.nodeId]["op"]
                with torch.no_grad():
                    explanation = {"nodeId": explain.nodeId, "op": op, "index": list(explain.index), **_explain_value(explain.nodeId, op, module, value_in, value_out, list(explain.index), torch)}
                edge = info["baseEdges"][explain.nodeId][0]
                explanation.update({"inputNode": edge["source"], "inputShape": [int(size) for size in value_in.shape[1:]], "instance": position, "instances": len(instances)})
                if owner != explain.nodeId:
                    explanation["owner"] = owner
    finally:
        for hook in hooks:
            hook.remove()
        latest.clear()
        explained.clear()
        retained.clear()
    if set(attention_results) != set(attention):
        raise InferenceError("Attention observer did not capture the selected nodes")
    if set(routing_results) != set(routing):
        raise InferenceError("Routing observer did not capture the selected MoE layers")
    return {"tensors": [captured[node_id] for node_id in selected], "attentions": [attention_results[node_id] for node_id in attention], "routings": [routing_results[node_id] for node_id in routing], "explanation": explanation, "gradient": backward}


def run_inference(request: TensorInferenceRequest) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed") from exc
    selected = None if request.allNodes else list(dict.fromkeys(request.nodeIds))
    # Without a selection the activation budget covers the whole model, which every layer needs.
    copied, info, embedding = _prepare_graph(request.graph, selected)
    if selected is None:
        if len(info["order"]) > MAX_ALL_NODES:
            raise InferenceError(f"allNodes observes at most {MAX_ALL_NODES} executed layers; select layers instead")
        selected = list(info["order"])
    from .trained_models import trained_models
    snapshot = trained_models.get(request.modelId, request.graph) if request.modelId is not None else None
    by_id = {node["id"]: node for node in copied["nodes"]}
    if any(node_id not in by_id for node_id in selected):
        raise InferenceError("nodeIds must refer to graph nodes")
    _validate_attention(request.attention, info, selected, _explain_scratch(request.explain, info, selected))
    _validate_gradient(request.gradient, info)
    routing = _resolve_routing(request.routing, info, selected)
    slices = _resolve_slices(request.slices, info, by_id)
    device_name = _device_name(request.device, torch)
    rng_devices = [torch.cuda.current_device()] if device_name == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices):
        torch.random.default_generator.manual_seed(request.seed)
        if device_name == "cuda":
            torch.cuda.manual_seed(request.seed)
        generator = torch.Generator(device="cpu").manual_seed(request.seed)
        # Validate and materialize the bounded sample before constructing parameters.
        inputs, input_source = _input_tensors(request.inputs, info, embedding, torch, generator)
        from .training import build_model
        model, _ = build_model(copied)
        if snapshot is not None:
            model.load_state_dict(snapshot.state, strict=True)
            inputs = {key: (value - snapshot.preprocessing[key]["mean"]) / snapshot.preprocessing[key]["scale"] if key in snapshot.preprocessing else value for key, value in inputs.items()}
        measured = _execute(model, info, by_id, inputs, device_name, selected, slices, request.attention, torch, request.explain, request.gradient, routing)
        if request.parameters is not None and request.parameters not in by_id:
            raise InferenceError("parameters must name a graph node")
        parameters = describe_parameters(model, request.parameters, request.parameterSlices, torch) if request.parameters is not None else []
        gradient = measured["gradient"]
        if gradient is not None:
            # The backward pass left these on the very parameters described above.
            gradient["parameters"] = describe_parameters(model, request.parameters, request.parameterSlices, torch, gradients=True) if request.parameters is not None else []
            model.zero_grad(set_to_none=True)
    return {
        "runId": str(uuid.uuid4()),
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "device": device_name.upper(),
        "seed": request.seed,
        "sampleIndex": 0,
        "inputSource": input_source,
        "weights": "trained" if snapshot is not None else "random-initialized",
        "model": copy.deepcopy(snapshot.metadata) if snapshot is not None else None,
        "inputTransform": snapshot.metadata["preprocessing"] if snapshot is not None else "none",
        "mode": "eval",
        "tensors": measured["tensors"],
        "attentions": measured["attentions"],
        "routings": measured["routings"],
        "parameters": parameters,
        "explanation": measured["explanation"],
        "gradient": gradient,
    }

"""Bounded, single-sample graph inference and tensor inspection."""
from __future__ import annotations

import copy
import math
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr, field_validator

from .graph import analyze_graph

MAX_CAPTURE_BYTES = 64 * 1024 * 1024
MAX_INPUT_ELEMENTS = 4_000_000
MAX_PARAMETERS = 5_000_000
MAX_PARAMETER_BYTES = 20 * 1024 * 1024
MAX_SLICE_SIDE = 16
HISTOGRAM_BINS = 12


class TensorInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    graph: dict[str, Any]
    nodeIds: list[StrictStr] = Field(min_length=1, max_length=8)
    seed: StrictInt = Field(42, ge=0, le=2_147_483_647)
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    inputs: dict[str, list[StrictFloat | StrictInt]] | None = Field(default=None, max_length=8)
    slices: dict[str, list[StrictInt]] = Field(default_factory=dict, max_length=8)

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


class InferenceError(ValueError):
    """A user-correctable inference request error."""


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _shape_elements(shape: list[int]) -> int:
    result = 1
    for value in shape:
        result *= int(value)
    return result


def _validate_limits(info: dict[str, Any], label: str) -> None:
    if info["activationBytesPerSample"] > MAX_CAPTURE_BYTES:
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
        if node["op"] not in ("Transformer", "MultiHeadAttention"):
            continue
        p = node["params"]
        heads = int(p.get("num_heads", 1)); branches = int(p.get("branches", 1))
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


def _prepare_graph(graph: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    # Analyze the submitted graph first: rewriting B must not make an invalid source valid.
    analyze_graph(graph)
    copied = copy.deepcopy(graph)
    for node in copied["nodes"]:
        if node["op"] == "Input":
            node["params"]["shape"] = [1, *node["params"]["shape"][1:]]
    inference_info = analyze_graph(copied)
    _validate_limits(inference_info, "Single-sample graph")
    return copied, inference_info, _embedding_inputs(inference_info)


def _input_tensors(request: TensorInferenceRequest, info: dict[str, Any], embedding: dict[str, int], torch: Any, generator: Any) -> tuple[dict[str, Any], str]:
    input_ids = list(info["inputs"])
    if request.inputs is not None and set(request.inputs) != set(input_ids):
        raise InferenceError("Provided inputs must contain exactly every Input node ID")
    if request.inputs is None:
        source = "synthetic"
    else:
        source = "provided"
    tensors: dict[str, Any] = {}
    for key in input_ids:
        shape = info["shapes"][key]
        count = _shape_elements(shape[1:])
        values = None if request.inputs is None else request.inputs[key]
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
    }


def run_inference(request: TensorInferenceRequest) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed") from exc
    copied, info, embedding = _prepare_graph(request.graph)
    selected = list(dict.fromkeys(request.nodeIds))
    by_id = {node["id"]: node for node in copied["nodes"]}
    if any(node_id not in by_id for node_id in selected):
        raise InferenceError("nodeIds must refer to graph nodes")
    slices: dict[str, list[int]] = {}
    for node_id, requested in request.slices.items():
        if node_id not in by_id:
            raise InferenceError(f"Unknown slice node {node_id}")
        rank = len(info["shapes"][node_id])
        prefix_rank = max(0, rank - 2)
        if len(requested) > 3 or len(requested) != prefix_rank:
            raise InferenceError(f"Slice {node_id} requires {prefix_rank} leading indices (maximum 3)")
        for index, size in zip(requested, info["shapes"][node_id]):
            if index < 0 or index >= size:
                raise InferenceError(f"Slice {node_id} index is outside tensor shape")
        slices[node_id] = list(requested)
    for node_id, shape in info["shapes"].items():
        prefix_rank = max(0, len(shape) - 2)
        if prefix_rank > 3:
            raise InferenceError(f"Slice {node_id} has more than 3 leading axes")
        slices.setdefault(node_id, [0] * prefix_rank)
    requested_device = request.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise InferenceError("CUDA is unavailable in this PyTorch installation")
    device_name = "cuda" if requested_device == "cuda" or (requested_device == "auto" and torch.cuda.is_available()) else "cpu"
    rng_devices = [torch.cuda.current_device()] if device_name == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices):
        torch.random.default_generator.manual_seed(request.seed)
        if device_name == "cuda":
            torch.cuda.manual_seed(request.seed)
        generator = torch.Generator(device="cpu").manual_seed(request.seed)
        # Validate and materialize the bounded sample before constructing parameters.
        inputs, input_source = _input_tensors(request, info, embedding, torch, generator)
        from .training import build_model
        model, _ = build_model(copied)
        model.to(device_name)
        model.eval()
        inputs = {key: value.to(device_name) for key, value in inputs.items()}
        captured: dict[str, dict[str, Any]] = {}
        def observe(node_id: str, value: Any) -> None:
            if node_id in selected:
                captured[node_id] = _describe(node_id, value, slices[node_id], torch)
        with torch.inference_mode():
            model(inputs if len(info["inputs"]) > 1 else inputs[info["inputs"][0]], observer=observe)
    return {
        "runId": str(uuid.uuid4()),
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "device": device_name.upper(),
        "seed": request.seed,
        "sampleIndex": 0,
        "inputSource": input_source,
        "weights": "random-initialized",
        "mode": "eval",
        "tensors": [captured[node_id] for node_id in selected],
    }

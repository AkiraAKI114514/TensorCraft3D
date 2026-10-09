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


class AttentionSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    branch: StrictInt = Field(0, ge=0, le=7)
    head: StrictInt = Field(0, ge=0, le=15)
    queryStart: StrictInt = Field(0, ge=0, le=65535)
    keyStart: StrictInt = Field(0, ge=0, le=65535)


class TensorInferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    graph: dict[str, Any]
    nodeIds: list[StrictStr] = Field(min_length=1, max_length=8)
    seed: StrictInt = Field(42, ge=0, le=2_147_483_647)
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    modelId: StrictStr | None = Field(default=None, min_length=1, max_length=128)
    inputs: dict[str, list[StrictFloat | StrictInt]] | None = Field(default=None, max_length=8)
    slices: dict[str, list[StrictInt]] = Field(default_factory=dict, max_length=8)
    attention: dict[StrictStr, AttentionSelection] = Field(default_factory=dict, max_length=8)

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


def _prepare_graph(graph: dict[str, Any], selected: list[str] | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    # Analyze the submitted graph first: rewriting B must not make an invalid source valid.
    analyze_graph(graph)
    copied = copy.deepcopy(graph)
    for node in copied["nodes"]:
        if node["op"] == "Input":
            node["params"]["shape"] = [1, *node["params"]["shape"][1:]]
    inference_info = analyze_graph(copied)
    _validate_limits(inference_info, "Single-sample graph", selected)
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


def _validate_attention(request: TensorInferenceRequest, info: dict[str, Any], selected: list[str]) -> None:
    by_id = {node["id"]: node for node in info["nodes"]}
    unknown = set(request.attention) - set(selected)
    if unknown:
        raise InferenceError("attention keys must refer to selected nodeIds")
    extra_scratch = 0
    for node_id, selection in request.attention.items():
        node = by_id.get(node_id)
        if node is None or node["op"] not in ("Transformer", "MultiHeadAttention"):
            raise InferenceError(f"{node_id}: attention selection requires an attention node")
        params = node["params"]
        branches = int(params.get("branches", 1))
        heads = int(params.get("num_heads", 1))
        if selection.branch >= branches:
            raise InferenceError(f"{node_id}: attention branch is outside [0,{branches - 1}]")
        if selection.head >= heads:
            raise InferenceError(f"{node_id}: attention head is outside [0,{heads - 1}]")
        query_length = int(info["shapes"][node_id][1])
        key_shape = info["portShapes"][node_id].get(f"b{selection.branch}:k0")
        key_length = int(key_shape[1]) if key_shape is not None else query_length
        if selection.queryStart >= query_length:
            raise InferenceError(f"{node_id}: queryStart is outside the query length")
        if selection.keyStart >= key_length:
            raise InferenceError(f"{node_id}: keyStart is outside the key length")
        if query_length * key_length > 262144:
            raise InferenceError(f"{node_id}: selected attention score matrix exceeds the 262144 element limit")
        embed = int(params.get("embed_dim", 64))
        head_dim = embed // heads
        kv_heads = int(params.get("kv_heads", 1 if params.get("attention_type") == "multi_query" else heads))
        # Captures retain all native projection/head stacks until the merged callback.
        projection_elements = query_length * embed * 4 + key_length * head_dim * kv_heads * 2
        extra_scratch += query_length * key_length * 64 + projection_elements * 32
    # 与张量检查同一口径：只按被检节点的祖先锥计账，否则深层模型里查看浅层注意力
    # 会被无关深层的激活拖到超限。
    cone = _activation_cone(info, selected)
    captured = sum(_shape_elements(info["shapes"][key][1:]) * 4 for key in cone if key in info["shapes"])
    if captured + extra_scratch > MAX_CAPTURE_BYTES:
        raise InferenceError("Attention inspection scratch exceeds 64 MB")


def _capture_attention(node_id: str, selection: AttentionSelection, record: dict[str, Any], torch: Any) -> dict[str, Any]:
    q = record["q"][0, selection.head]
    native_k = record["k"][0, selection.head // record["repeats"]]
    native_v = record["v"][0, selection.head // record["repeats"]]
    weighted = record["weighted"][0, selection.head]
    scale = 1.0 / math.sqrt(q.shape[-1])
    with torch.no_grad():
        scores = (q.float() @ native_k.float().transpose(-2, -1)) * scale
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
        "scale": scale, "mask": "none", "dropout": 0, "scoreSource": "projected-qk",
        "outputSource": "scaled_dot_product_attention", "tensors": tensors,
    }


def run_inference(request: TensorInferenceRequest) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed") from exc
    selected = list(dict.fromkeys(request.nodeIds))
    copied, info, embedding = _prepare_graph(request.graph, selected)
    from .trained_models import trained_models
    snapshot = trained_models.get(request.modelId, request.graph) if request.modelId is not None else None
    by_id = {node["id"]: node for node in copied["nodes"]}
    if any(node_id not in by_id for node_id in selected):
        raise InferenceError("nodeIds must refer to graph nodes")
    _validate_attention(request, info, selected)
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
        if snapshot is not None:
            model.load_state_dict(snapshot.state, strict=True)
            inputs = {key: (value - snapshot.preprocessing[key]["mean"]) / snapshot.preprocessing[key]["scale"] if key in snapshot.preprocessing else value for key, value in inputs.items()}
        model.to(device_name)
        model.eval()
        inputs = {key: value.to(device_name) for key, value in inputs.items()}
        captured: dict[str, dict[str, Any]] = {}
        attention_records: dict[str, dict[str, Any]] = {}
        attention_results: dict[str, dict[str, Any]] = {}
        def observe(node_id: str, value: Any) -> None:
            if node_id in selected:
                captured[node_id] = _describe(node_id, value, slices[node_id], torch)
        def observe_attention(node_id: str, stage: str, branch: int, tensors: dict[str, Any]) -> None:
            selection = request.attention.get(node_id)
            if selection is None or (stage != "merged" and branch != selection.branch):
                return
            record = attention_records.setdefault(node_id, {})
            if stage == "merged":
                params = by_id[node_id]["params"]
                heads = int(params.get("num_heads", 1))
                kv_heads = int(params.get("kv_heads", 1 if params.get("attention_type") == "multi_query" else heads))
                record["params"] = {"heads": heads, "kv_heads": kv_heads, "branches": int(params.get("branches", 1))}
                record["repeats"] = heads // kv_heads
                record["merged"] = tensors["output"].detach()
                attention_results[node_id] = _capture_attention(node_id, selection, record, torch)
                del attention_records[node_id]
            else:
                wanted = {"q": "q", "k": "k", "v": "v", "weighted": "weighted", "output": "output"}[stage]
                record[wanted] = tensors[wanted].detach()
        with torch.inference_mode():
            model(inputs if len(info["inputs"]) > 1 else inputs[info["inputs"][0]], observer=observe, attention_observer=observe_attention if request.attention else None)
        if set(attention_results) != set(request.attention):
            raise InferenceError("Attention observer did not capture the selected nodes")
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
        "tensors": [captured[node_id] for node_id in selected],
        "attentions": [attention_results[node_id] for node_id in request.attention],
    }

"""One bounded, process-local training snapshot; never loads uploaded weights."""
import copy
import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024


def _computation_node(node):
    # `shares` changes which weights a layer uses, and `repeat` and a block's definition change
    # which operators run, so they are part of the computation. Each is added only when present,
    # which keeps the fingerprint of every graph without them unchanged.
    return {
        "id": node["id"], "op": node["op"], "params": node["params"],
        **({"repeat": node["repeat"]} if node.get("repeat") not in (None, 1) else {}),
        **({"subgraph": node["subgraph"]} if node.get("subgraph") is not None else {}),
        **({"shares": node["shares"]} if node.get("shares") is not None else {}),
    }


def _computation_edges(edges):
    return [{key: edge[key] for key in ("source", "target", "sourcePort", "targetPort") if key in edge} for edge in edges]


def graph_fingerprint(graph):
    computation = {
        "nodes": [_computation_node(node) for node in graph["nodes"]],
        "edges": _computation_edges(graph["edges"]),
    }
    # Only the block definitions the graph actually runs count, nested ones included.
    definitions = graph.get("subgraphs") or {}
    used, pending = set(), [node.get("subgraph") for node in graph["nodes"] if node.get("op") == "Group"]
    while pending:
        name = pending.pop()
        if name in used or name not in definitions:
            continue
        used.add(name)
        pending.extend(inner.get("subgraph") for inner in definitions[name]["nodes"] if inner.get("op") == "Group")
    if used:
        computation["subgraphs"] = {name: {"nodes": [_computation_node(node) for node in definitions[name]["nodes"]], "edges": _computation_edges(definitions[name]["edges"])} for name in sorted(used)}
    return hashlib.sha256(json.dumps(computation, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class TrainedModel:
    metadata: dict
    state: dict
    preprocessing: dict


class TrainedModelStore:
    # All access is serialized by the app's training/inference lock.
    def __init__(self):
        self._latest = None

    def save(self, graph, model, preprocessing, provenance):
        state = model.state_dict()
        tensors = [*state.values(), *(value for pair in preprocessing.values() for value in pair.values())]
        size = sum(value.numel() * value.element_size() for value in tensors)
        if size > MAX_SNAPSHOT_BYTES:
            return None
        metadata = {
            **provenance,
            "modelId": str(uuid.uuid4()),
            "graphFingerprint": graph_fingerprint(graph),
            "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "preprocessing": "csv-standardized" if preprocessing else "none",
            "storage": "backend-memory",
        }
        self._latest = TrainedModel(
            metadata=metadata,
            state={key: value.detach().cpu().clone() for key, value in state.items()},
            preprocessing={key: {name: value.detach().cpu().clone() for name, value in pair.items()} for key, pair in preprocessing.items()},
        )
        return copy.deepcopy(metadata)

    def get(self, model_id, graph):
        snapshot = self._latest
        if snapshot is None or snapshot.metadata["modelId"] != model_id:
            raise ValueError("Trained model snapshot is unavailable (backend restarted or a newer training run replaced it). Retrain or explicitly select random weights.")
        if snapshot.metadata["graphFingerprint"] != graph_fingerprint(graph):
            raise ValueError("Trained model does not match the current computation graph. Restore the training graph or retrain.")
        return snapshot


trained_models = TrainedModelStore()

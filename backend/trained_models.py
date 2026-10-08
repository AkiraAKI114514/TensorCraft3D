"""One bounded, process-local training snapshot; never loads uploaded weights."""
import copy
import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024


def graph_fingerprint(graph):
    computation = {
        "nodes": [{"id": node["id"], "op": node["op"], "params": node["params"]} for node in graph["nodes"]],
        "edges": [{key: edge[key] for key in ("source", "target", "sourcePort", "targetPort") if key in edge} for edge in graph["edges"]],
    }
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

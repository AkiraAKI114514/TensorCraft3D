"""Offline observation snapshots, written by a Python process outside the workbench.

The workbench never loads weight files or user data. A script that trained the exported model
elsewhere -- on its own data, on another machine -- runs one sample through the same runtime the
inspector uses and writes the bounded observations together with the graph they were measured
on. The workbench opens that file read-only and binds it to the graph's computation, the way it
binds its own trained weights.

    from backend.observation_snapshot import capture_snapshot, write_snapshot
    snapshot = capture_snapshot(graph, model.state_dict(), {"input": image})
    write_snapshot(snapshot, "sample.observation.json")

or from a shell, in the project root:

    python -m backend.observation_snapshot project.json --weights weights.pt --inputs sample.json --out sample.observation.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .inference import ATTENTION_OPS, AttentionSelection, InferenceError, _device_name, _execute, _input_tensors, _prepare_graph, _resolve_slices, _validate_attention, describe_parameters

FORMAT = "tensorcraft-observation"
VERSION = 1
# The workbench refuses a larger file before parsing it.
MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024
MAX_LABEL_LENGTH = 120
def _flatten_inputs(inputs: Any, info: dict[str, Any], torch: Any) -> dict[str, list[Any]]:
    """One sample per Input node, flattened the way the inspector's custom sample is.

    A value may keep its batch axis of 1, drop it, or already be flat. Any other shape is
    refused even when the element count happens to match, because reshaping it would silently
    reorder the sample.
    """
    ids = list(info["inputs"])
    if not isinstance(inputs, Mapping):
        if len(ids) != 1:
            raise InferenceError("A graph with several Input nodes needs inputs keyed by Input node ID")
        inputs = {ids[0]: inputs}
    if set(inputs) != set(ids):
        raise InferenceError(f"inputs must contain exactly the Input node IDs {ids}")
    flat: dict[str, list[Any]] = {}
    for key in ids:
        try:
            value = torch.as_tensor(inputs[key]).detach().cpu()
        except (TypeError, ValueError, RuntimeError) as error:
            raise InferenceError(f"Input {key} is not a numeric tensor: {error}") from error
        shape = [int(size) for size in info["shapes"][key]]
        if list(value.shape) not in (shape, shape[1:], [math.prod(shape[1:])]):
            raise InferenceError(f"Input {key} must have shape {shape} or {shape[1:]}, or be flat; got {list(value.shape)}")
        flat[key] = value.reshape(-1).tolist()
    return flat


def _sample(keys: list[str]) -> str:
    return ", ".join(keys[:3]) + (f" (+{len(keys) - 3})" if len(keys) > 3 else "") if keys else "none"


def _aligned_state(weights: Any, model: Any) -> Mapping[str, Any]:
    """The given weights, keyed the way the runtime model names its parameters.

    The exported `VisualModel` keeps top-level layers in the same `layers` dictionary as the
    runtime, so its state dict loads as it is. A folded block is the exception: export writes it
    as a module list of block classes (`layers.g1.0.fc.weight`), while the runtime expands it
    (`layers.g1/0/fc.weight`). Only expanded IDs contain `/`, so each runtime layer inside a
    block has exactly one exported name, and the longest one is matched first for nested blocks.
    """
    state = weights.state_dict() if callable(getattr(weights, "state_dict", None)) else weights
    if not isinstance(state, Mapping) or not all(isinstance(key, str) for key in state):
        raise InferenceError("weights must be a state dict or a module with one")
    expected = set(model.state_dict())
    if set(state) == expected:
        return state
    aliases = sorted(((f"layers.{key.replace('/', '.')}.", f"layers.{key}.") for key in model.layers if "/" in key), key=lambda pair: -len(pair[0]))
    aligned: dict[str, Any] = {}
    for name, value in state.items():
        for exported, runtime in aliases:
            if name.startswith(exported):
                name = runtime + name[len(exported):]
                break
        aligned[name] = value
    missing, unexpected = sorted(expected - set(aligned)), sorted(set(aligned) - expected)
    if missing or unexpected:
        raise InferenceError(f"weights do not match this graph: missing {_sample(missing)}; unexpected {_sample(unexpected)}")
    return aligned


def capture_snapshot(graph: dict[str, Any], weights: Any = None, inputs: Any = None, *, seed: int = 42, device: str = "cpu", slices: dict[str, list[int]] | None = None, attention: dict[str, Any] | None = None, label: str | None = None) -> dict[str, Any]:
    """Observe every node of `graph` for one sample, in eval mode.

    `weights` is the state dict of the graph's model, or a module that has one -- typically the
    exported `VisualModel` after training. None keeps the seeded random initialization, and the
    snapshot says so. `inputs` maps each Input node ID to one sample; a single tensor is enough
    when there is one Input, and None draws the seeded synthetic sample the inspector uses.

    Each node gets what the inspector shows: statistics, a histogram, a window of at most
    16 x 16 at the leading indices in `slices` (zeros by default) and, for feature maps, one
    page of 16 channels. Every attention node also gets its Q/K/V, scores and probabilities at
    the branch, head and window in `attention` (branch 0, head 0, top-left by default); map a
    node to None to keep only its output, for instance when its score matrix is too large.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed") from exc
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2_147_483_647:
        raise InferenceError("seed must be an integer in [0, 2147483647]")
    if label is not None and (not isinstance(label, str) or len(label) > MAX_LABEL_LENGTH):
        raise InferenceError(f"label must be text of at most {MAX_LABEL_LENGTH} characters")
    if device not in ("cpu", "cuda", "auto"):
        raise InferenceError("device must be cpu, cuda or auto")
    copied, info, embedding = _prepare_graph(graph)
    by_id = {node["id"]: node for node in copied["nodes"]}
    order = list(info["order"])
    selections: dict[str, AttentionSelection] = {node_id: AttentionSelection() for node_id in order if by_id[node_id]["op"] in ATTENTION_OPS}
    for node_id, selection in (attention or {}).items():
        if node_id not in selections:
            raise InferenceError(f"{node_id}: attention selection requires an attention node")
        if selection is None:
            del selections[node_id]
        else:
            selections[node_id] = selection if isinstance(selection, AttentionSelection) else AttentionSelection.model_validate(selection)
    _validate_attention(selections, info, order)
    resolved = _resolve_slices(slices or {}, info, by_id)
    device_name = _device_name(device, torch)
    provided = None if inputs is None else _flatten_inputs(inputs, info, torch)
    rng_devices = [torch.cuda.current_device()] if device_name == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices):
        torch.random.default_generator.manual_seed(seed)
        if device_name == "cuda":
            torch.cuda.manual_seed(seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        sample, input_source = _input_tensors(provided, info, embedding, torch, generator)
        from .training import build_model
        model, _ = build_model(copied)
        if weights is not None:
            try:
                model.load_state_dict(_aligned_state(weights, model), strict=True)
            except RuntimeError as error:
                # Same names, different shapes: the weights belong to another configuration.
                raise InferenceError(f"weights do not match this graph: {str(error)[:400]}") from error
        measured = _execute(model, info, by_id, sample, device_name, order, resolved, selections, torch)
        tensors, attentions = measured["tensors"], measured["attentions"]
        # The weights a run outside the workbench trained are the reason to bring it in, so every
        # layer that has any carries their summaries, at the default leading indices.
        parameters = {node_id: described for node_id in order if (described := describe_parameters(model, node_id, {}, torch))}
    return {
        "format": FORMAT,
        "version": VERSION,
        "snapshotId": str(uuid.uuid4()),
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "label": label,
        "graph": copy.deepcopy(graph),
        "device": device_name.upper(),
        "seed": seed,
        "sampleIndex": 0,
        "inputSource": input_source,
        "weights": "external" if weights is not None else "random-initialized",
        "mode": "eval",
        "tensors": tensors,
        "attentions": attentions,
        "parameters": parameters,
    }


def snapshot_json(snapshot: dict[str, Any]) -> str:
    """Serialize a snapshot, refusing one the workbench would not open."""
    text = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    size = len(text.encode("utf-8"))
    if size > MAX_SNAPSHOT_BYTES:
        raise InferenceError(f"Observation snapshot is {size} bytes, more than the {MAX_SNAPSHOT_BYTES} bytes the workbench opens")
    return text


def write_snapshot(snapshot: dict[str, Any], path: str | Path) -> int:
    """Write a snapshot as UTF-8 JSON and return its size in bytes."""
    text = snapshot_json(snapshot)
    Path(path).write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def _load_tensor_file(path: Path) -> Any:
    import torch
    # weights_only refuses pickled code: a weight or sample file can only hold tensors and containers.
    return torch.load(path, map_location="cpu", weights_only=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m backend.observation_snapshot", description="Write an observation snapshot that the TensorCraft3D workbench opens read-only.")
    parser.add_argument("project", type=Path, help="project JSON saved from the workbench")
    parser.add_argument("--out", type=Path, required=True, help="snapshot file to write")
    parser.add_argument("--weights", type=Path, help="state dict saved with torch.save(model.state_dict(), path); omit for seeded random weights")
    parser.add_argument("--inputs", type=Path, help="one sample: JSON {input id: values}, or a tensor or dict of tensors saved with torch.save; omit for a seeded synthetic sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--label", help=f"short note shown with the snapshot, at most {MAX_LABEL_LENGTH} characters")
    args = parser.parse_args(argv)
    try:
        graph = json.loads(args.project.read_text(encoding="utf-8"))
        weights = None if args.weights is None else _load_tensor_file(args.weights)
        inputs = None
        if args.inputs is not None:
            inputs = json.loads(args.inputs.read_text(encoding="utf-8")) if args.inputs.suffix.lower() == ".json" else _load_tensor_file(args.inputs)
        snapshot = capture_snapshot(graph, weights, inputs, seed=args.seed, device=args.device, label=args.label)
        size = write_snapshot(snapshot, args.out)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote {args.out}: {len(snapshot['tensors'])} tensors, {len(snapshot['attentions'])} attention windows, {size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())

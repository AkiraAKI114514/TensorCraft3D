# Offline observation snapshots

The Tensor Inspector normally runs one sample in the workbench's own Python service, with random weights or the weights of its latest training run. An observation snapshot brings in a sample measured somewhere else. A Python process outside the workbench runs one sample through the graph with weights it trained itself, and writes what the inspector would show into a JSON file. That process can be your own training script, a notebook, or another machine with a GPU and the real dataset. The workbench opens the file read-only.

Only bounded numbers and the graph they were measured on reach the workbench. It never loads a weight file or your data, never executes anything from the file, and does not need its Python service to show one.

## Writing a snapshot

A typical run:

1. Build the model in the workbench, save the project JSON and export the Python code.
2. Train the exported `VisualModel` wherever you like.
3. Capture one sample with `backend/observation_snapshot.py`. Run it from the TensorCraft3D project root, or put that root on `sys.path`.

```python
import json
import torch
from backend.observation_snapshot import capture_snapshot, write_snapshot

graph = json.load(open("project.json", encoding="utf-8"))
model = VisualModel()  # the exported class
model.load_state_dict(torch.load("weights.pt", weights_only=True))
# Inputs are keyed by the Input node's ID in project.json.
snapshot = capture_snapshot(graph, model, {"layer_0": image}, label="validation digit 7")
write_snapshot(snapshot, "digit7.observation.json")
```

The same from a shell, with weights saved by `torch.save(model.state_dict(), "weights.pt")`:

```powershell
.\.venv\Scripts\python.exe -m backend.observation_snapshot project.json --weights weights.pt --inputs sample.json --out digit7.observation.json --label "validation digit 7"
```

| Argument | Meaning |
| --- | --- |
| `graph` | The project JSON saved from the workbench. The snapshot carries it, and the workbench checks it like any project file. |
| `weights` | A state dict, or a module that has one. Omit it (no `--weights`) to use the seeded random initialization; the snapshot then says `random-initialized`. |
| `inputs` | One sample per Input node ID: a tensor, array or nested list, with its batch axis of 1, without it, or flat. Any other shape is refused even when the element count matches, because reshaping would reorder the sample. A graph with one Input also takes the bare tensor. Embedding inputs must be integer IDs in range. Omit it to use the seeded synthetic sample the inspector uses. `--inputs` reads JSON `{input id: values}`, or a tensor or dict of tensors saved with `torch.save`. |
| `seed`, `device` | The sample and initialization seed, and `cpu`, `cuda` or `auto`. |
| `slices` | Leading indices per node ID, zeros by default; the same rule as the inspector's slice axes. Python API only. |
| `attention` | `{node id: {branch, head, queryStart, keyStart}}`. Every attention node is captured at branch 0, head 0 and the top-left window unless you choose otherwise. Map a node to `None` to keep only its output, for example when its full score matrix is over 262,144 elements. Python API only. |
| `label` | Up to 120 characters, shown with the snapshot. |

The exported `VisualModel` names its parameters the way the workbench runtime does, so its state dict loads as it is. Folded blocks are the one difference: export writes `layers.<block>.<instance>.<layer>…`, while the runtime expands blocks into `layers.<block>/<instance>/<layer>…`. The writer accepts both layouts. Weights whose keys or shapes do not match the graph are an error that names the keys that differ. `--weights` and tensor `--inputs` files are loaded with `weights_only=True`, so they can only hold tensors and containers.

The sample runs through the same runtime and the same inspection code as `/api/infer`, in `eval()` and `inference_mode()` with batch size 1. So the snapshot holds exactly what the inspector would show for the same weights and sample. The same limits apply as for one inspector run over the whole model: 5 million parameters / 20 MiB, 4 million elements per input, 64 MiB of single-sample activations, and attention inspection scratch within that budget.

## What a snapshot holds

```json
{
  "format": "tensorcraft-observation",
  "version": 1,
  "snapshotId": "…", "createdAt": "…Z", "label": "validation digit 7",
  "graph": { "version": 1, "nodes": ["…"], "edges": ["…"] },
  "device": "CPU", "seed": 42, "sampleIndex": 0,
  "inputSource": "provided", "weights": "external", "mode": "eval",
  "tensors": ["one entry per executed node"],
  "attentions": ["one entry per captured attention node"]
}
```

Every node of the executed graph has one `tensors` entry. Folded blocks are expanded, so their layers appear as `block/0/fc`. An entry has the same fields as an inspector report:

- shape, dtype, and finite and non-finite element counts
- min, max, mean and population std over the whole tensor
- a 12-bin histogram
- a window of at most 16 × 16 at the chosen leading indices
- for `[1,C,H,W]` and `[1,C,D,H,W]`, the page of up to 16 channels that contains the selected one

An `attentions` entry holds Q/K/V, scores, probabilities and outputs for one branch and head. Each matrix has statistics over the full matrix and a window of at most 16 × 16. Non-finite values are written as `null`. The writer refuses to write a file over 16 MiB, which is also the most the workbench will read.

## Opening it in the workbench

In **张量观测**, set **观测来源** to **外部 Python 观测快照 · 只读** and open the file. The panel shows where the numbers came from: device, sample source and seed, which weights were used, the label, the creation time and the snapshot ID.

The numbers are bound to the computation the snapshot measured, the same way trained weights are. Names, layout and the project title do not matter. Operators, parameters, input shapes, wiring, weight sharing, repeat counts and the contents of folded blocks do. If the current project computes something else, the panel says so and offers to open the snapshot's own graph. Opening it replaces the project the way importing one does, and undo brings the previous project back. A later edit hides the numbers again until the graph matches.

While the snapshot matches the current graph, **把快照画到三维视图** paints every layer's values on the 3D blocks, the same way the inspector's all-layer run does (see the workbench guide).

Read-only means:

- No inference runs, and nothing is sent to the Python service, which may be offline.
- The slice and attention windows stay where they were captured. The channel grid switches between the captured channels without a rerun.
- Folded blocks are looked up the way live inference names them: a selected block shows its last instance's exit layer, and inside an entered block the instance selector switches between the captured instances without a rerun.
- The snapshot stays in this page's memory. It is not written to the project, browser storage or disk, and reloading the page drops it.

The file is checked before anything is shown:

- format and version
- a valid project graph
- observations only for nodes of that graph, and attention only for attention nodes
- bounded windows and channel pages
- consistent element counts
- finite numbers or `null`

A refused file names the field that failed.

## Not covered

A snapshot is one sample, measured once. There is no live connection to a running process, no streaming during training and no gradient observation. A model the workbench cannot express cannot be snapshotted either; import its source first if [the importer supports it](pytorch-import-support.md).

# TensorCraft3D workbench guide

[← Back to the project overview](../README.md)

This guide contains the detailed usage, development and implementation notes. For a quick introduction and setup, start with the project README. Run the commands below from the project root.

## Running on Windows

Requires Node.js 20.19+ (22/24 recommended) and Python 3.10+ (3.12/3.13 recommended).

```powershell
# First-time install, including real training
powershell -ExecutionPolicy Bypass -File .\install.ps1 -Training

# Start, opening the browser on a free local port
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

Once dependencies are installed and a compiled `dist` is present, you can also just double-click `start.cmd`. The default address is http://127.0.0.1:8765, and it listens on localhost only. Closing the terminal stops the service. For non-interactive startup, use `powershell -File .\start.ps1 -NoBrowser -Port 8766` (rebuilds first), or `start.cmd --no-browser --port 8766` (uses the existing build).

If you don't need real training, `npm install`, `npm run build` and `python run.py` are enough. The standard-library server still provides the 3D view, model editing, warning demos and export.

## Development, testing and packaging

```powershell
# Terminal 1: real training backend
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8765

# Terminal 2: dev frontend, proxying /api to the service above
npm run dev

npm test
.\.venv\Scripts\python.exe -m unittest backend.test_training backend.test_pytorch_import backend.test_tensor_ops backend.test_cuda_environment backend.test_graph_contracts
# Focused shape/parameter/resource contracts: both sides read the same JSON, no PyTorch required
npm exec --no -- vitest run src/analysisContracts.test.ts
.\.venv\Scripts\python.exe -m unittest backend.test_graph_contracts
# With the service running on 8765, run the browser flow tests using the installed Microsoft Edge
npm run test:e2e
# If run.py selected another port, point browser tests at that server
$env:TENSORLAB_TEST_URL = 'http://127.0.0.1:8766'
npm run test:e2e -- e2e/runtime.spec.ts
# Windows startup smoke checks: built assets, API/static modes, wrappers and port fallback
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_startup.py -v
# Focused crash recovery: saved-project backup, reload and storage failures
npm run test:e2e -- e2e/errorBoundary.spec.ts
# Focused single-sample inference, tensor observation and lazy code export
npm exec --no -- vitest run src/CodeExport.test.tsx src/TensorInspector.test.tsx src/trainedModel.test.ts
.\.venv\Scripts\python.exe -m unittest backend.test_inference backend.test_trained_models
npm run test:e2e -- e2e/inference.spec.ts e2e/trainedInference.spec.ts
# Focused dimension contracts, Attention observation, examples and executable roundtrips
npm exec --no -- vitest run src/channelRanks.test.ts src/analysisContracts.test.ts src/AttentionInspector.test.tsx src/importExamples.test.ts src/pytorchImport.test.ts
.\.venv\Scripts\python.exe -m unittest backend.test_channel_ranks backend.test_attention_inference backend.test_import_examples backend.test_graph_contracts
npm run test:e2e -- e2e/attentionInference.spec.ts
npm run build
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

CI runs frontend checks on Ubuntu and automatically discovers every `backend/test_*.py` module, including import-parameter contracts. A Windows job checks the real PowerShell/CMD launchers, API and standard-library static serving, built assets and occupied-port fallback, then runs focused Edge regressions for editing/history, import/export, tensor/attention/trained inference, language switching, recovery and runtime behavior. In CI, Playwright starts and stops its own local backend and waits for `/api/health`; normal local browser runs still use the already-running server. Failed browser runs retain traces/screenshots and upload diagnostics. These hosted checks use CPU computation, not CUDA hardware validation.

`build.ps1` produces `release/TensorCraft3D.zip`, containing source, build output, the backend and the launch scripts, but not `node_modules` or the virtual environment. Recipients need Python to run it, and real training additionally needs the backend dependencies installed; this is not a runtime-free Windows EXE.

## The workbench

- Ships with Compact CNN, Feature MLP, Residual CNN, Transformer Encoder, MQA, GQA, Cross-Attention and a blank template. The layer library supports both clicking and dragging into the scene. New layers connect to the currently selected layer and are inserted before the final output.
- The 3D view generates solid layers from each tensor's spatial dimensions and channels; geometric sizes use a logarithmic mapping so extreme sizes don't occlude each other.
- Clicking any 3D layer centres rotation and zoom on that layer; attention heads and their QKV, weight matrices, LN, FFN, merge and add modules can each be clicked and centred individually. Add-focused framing keeps the full residual bypass intact while centring on the Add node.
- The generic Add residual and the Transformer Encoder share the pink outer bypass, directional arrows and GPU particle styling. The residual branches off at the source layer's output, bypasses the intermediate modules, enters the pink "+" node from above, and continues after being summed with the main path. Camera focus and PNG / SVG export include the full bypass; the topology graph shows the pink outer curve as well. Skip branches are identified from reachability in the graph, while independent parallel branches keep ordinary edges.
- The topology graph supports dragging nodes and pulling wires from an output port to an input port; Add / Multiply / Concat accept multiple inputs. Press Delete after clicking an edge to remove it, or remove the connection from the properties panel.
- The topology toolbar can add standalone input/output objects and delete the selected object; the properties panel's "connected objects" section lets you pick upstream/downstream objects and add or remove input/output connections individually. Q/K/V objects have their own ports, so input sources and output targets can be configured per vertex. Invalid connections stay in the graph with the reason shown; cycles, shape mismatches and illegal multi-inputs block training and Python export. Computation graphs support 1–8 model Inputs and one Output.
- Editing layer properties re-derives tensor shapes and checks for cycles, rank-4/rank-2 mismatches, residual shapes, concatenation dimensions, pooling parameters and resource limits. Projects are saved automatically in local browser storage, with undo/redo and JSON import/export. Use Ctrl/Cmd+Z to undo, Ctrl/Cmd+Shift+Z or Ctrl/Cmd+Y to redo when focus is outside an editor field.
- Exports a PyTorch `nn.Module`, including a runnable random-input test, with logits as output. Python code uploaded from the browser is never executed.
- Image export defaults to a compact layout: topological stages flow left-to-right and wrap into rows, parallel branches stay aligned, and cross-row/residual edges use outer routing lanes. Choose a 16:9, 4:3 or square frame and an 8–18 label font size; long captions wrap and labels avoid one another. PNG (1920 / 3840 / 7680 pixel widths, optional transparency) and editable SVG share the geometry and label placement. "Current viewpoint" preserves the camera and viewport aspect instead of rearranging the model. Export hides the floor grid and renders a separate scene, leaving the interactive camera, model connections and GPU animation intact.

### Attention and Transformer

`MultiHeadAttention` and `Transformer` use a shared PyTorch `scaled_dot_product_attention` runtime. `Transformer` is a pre-norm encoder layer (LN → Attention → residual → LN → GELU FFN → residual). Inputs and outputs use `[B,S,E]`. `attention_type` supports `self`, `multi_query` (MQA), `grouped_query` (GQA), `cross` and `multi_branch`; `num_heads` (1–16) is the number of queries and defaults to 1; `kv_heads` is the number of K/V groups; `branches` (1–8) is the number of independently computed branches that are then averaged. Embedding must equal the last input dimension and be divisible by the query count, and the query count must be divisible by the number of K/V groups. MQA uses a single K/V group, while standard self-attention uses one K/V group per query. Cross-Attention takes two inputs, Query and Context, and allows the two sequence lengths to differ. Every parameter changes the real computation, the parameter count and the exported code.

One polygon is drawn per shared K/V group, with all Q vertices that use that K/V placed in the same face; K and V are each drawn once, with no extra floating shared layer. The number of sides equals the number of queries in the group plus 2: a single query gives a triangle; MQA's 4 queries sharing one K/V give a hexagon; GQA's 8 queries over 2 K/V groups give two hexagons. `qkv_count` has been removed from the parameters panel, and the field is ignored in older projects. Groups are stacked along the data-flow X axis in the same direction, with attention heads computing in parallel; multiple branches are displayed separately per branch. Each query has its own softmax and V-weighting path, and the shared K/V connect directly to those compute nodes. The Transformer continues to show LN, FFN and the two pink residual paths. Rotation stops on selection or hover; once per-vertex connections are configured, the polygon orientation is fixed so that external particles stay aligned with the vertices. PNG / SVG share the same geometry and port layout as the scene. In image export, heads that split one shared input into Q/K/V subspaces are drawn as a stepped stack instead of side by side: every face keeps its full size, the front face shows the Q/K/V detail and the head range (e.g. `H1–H8`), and each face behind it shows only a stepped edge, with a `×N` count. A head whose own Q/K/V port is wired to another layer keeps its own column, and branches are never stacked together.

Clicking a 3D Q/K/V cube centres it and opens that object's connection configuration, and it can also be selected from the "Q/K/V objects" selector in the properties panel. Each vertex inherits the model layer's inputs by default (for Cross-Attention, Q inherits Query and K/V inherit Context); specifying an upstream object overrides the corresponding projection source, and deleting the override connection restores the default source. A vertex outputs a real projected tensor `[B,S,E/num_heads]` and can connect to other model layers or to the input ports of other Q/K/V objects. All vertex inputs require `[B,S,E]`; a query's sequence length must match the layer's Query, and within a branch all K/V must share a sequence length. The Transformer's Q and self-attention overrides also pass through LN. Each projection port takes at most one specified input, while outputs can fan out to multiple objects. A complete model still needs a base input, and the Transformer's residual comes from the base Query.

Project connections store vertex ports via `sourcePort` / `targetPort`, for example `{"source":"memory","target":"attention","targetPort":"b0:k0"}` specifies the input of K1 in the first branch; `{"source":"attention","sourcePort":"b0:q0","target":"flatten"}` passes the projected Q1 of the first branch to Flatten. The same pair of model layers can be connected multiple times through different ports. Training, shape validation and standalone Python export all use the same connection semantics.

The colours and particles of the 3D weight matrices are still a **simulated illustration**, and the properties panel labels this "weight schematic". The on-demand **Attention Inspector** separately observes actual Q/K/V projections and head/branch outputs during single-sample inference, and recomputes scaled QK scores and softmax probabilities from those projections. It does not collect token-level gradients. Loss and per-layer gradients in real training monitoring come from actual PyTorch training. The example input is a token feature vector; there is no text tokenizer or causal/padding mask yet. Imported static float32 positional buffers are supported as `ConstantAdd` nodes; they are not automatically added to attention templates.

## GPU animation and training

Three.js uses WebGL, GPU instancing and custom vertex shaders. Each particle's Bézier path is computed on the GPU, and the CPU only updates the time/direction uniforms per frame rather than rewriting positions per particle. Ordinary connections are cyan forward and coral backward; residual bypasses stay pink in both directions, with arrows and particle direction reversed when running backward. Pause, speed adjustment, orbit rotation, zoom, pan and expanded spacing are supported. The animation represents simulated propagation, not actual tensor values.

The browser needs hardware acceleration enabled. A software renderer is reported as "software rendering"; the app cannot force a browser or operating system to provide a physical GPU. WebGL GPU acceleration and PyTorch CUDA training are two separate device mechanisms.

Real training uses Adam + CrossEntropyLoss, with CPU / CUDA / automatic selection, over synthetic classification data or a local CSV. CSVs support an optional header row, with the last column as a zero-based integer class and the preceding columns as numeric features; the feature count must equal the number of elements in the input tensor excluding batch, with images flattened in C/H/W order. CSV values are standardised and then reshaped to the model's input shape. The validation split is a fixed random split. Epochs, learning rate, batch size, validation fraction and the early-stopping window are configurable.

Training pushes measured train/val loss, accuracy, global and per-layer gradient norms, and the ReLU zero-activation fraction over WebSocket. Gradient clipping is at a threshold of 100; early stopping triggers on a run of unimproved validation loss windows. Only one local training job runs at a time, and disconnecting the page or pressing stop halts it on the next batch. Inputs, data and activations have resource limits, and complex models need to run in a professional training environment.

### Local CUDA diagnostics and setup

Open "CUDA 环境与配置" in the training panel to inspect the backend's actual interpreter, PyTorch version, CUDA build, GPU/driver and selected environment variables. The folded panel does not request full diagnostics. Starting training is disabled until any diagnostic/probe request finishes, even if the panel is folded or the dialog is closed and reopened. `GET /api/environment` is read-only and distinguishes a missing or broken PyTorch import, a CPU wheel, a hidden GPU and CUDA initialization failures. Full inspection runs a bounded `nvidia-smi` query; the regular health check does not. This is an on-demand snapshot, not continuous monitoring.

CUDA training runs in the local Python backend, not in the browser or a separate CUDA service. Official CUDA PyTorch wheels supply their runtime libraries and still need a compatible NVIDIA driver. Installing a system CUDA toolkit or setting `CUDA_PATH` cannot turn a CPU wheel into a CUDA build. The app does not bundle PyTorch/CUDA, install drivers, alter environment variables or connect to a remote GPU service.

The setup script changes only this project's existing `.venv`. It pins `torch==2.9.1+cu128` for CUDA and `torch==2.14.1+cpu` for CPU, using the corresponding official PyTorch index. It skips installation when that exact version is already present, then checks import and device availability. A different Python/platform or driver may need a different wheel from the [official installation selector](https://pytorch.org/get-started/locally/).

Stop all project Python backends and training jobs before replacing PyTorch. On Windows, a running process can hold PyTorch DLLs open. The script refuses to start package replacement while it detects a project Python process; it does not stop processes itself.

```powershell
# Run from the project root, after stopping its Python processes
.\setup-training.ps1 -Variant cu128

# Explicit CPU alternative
.\setup-training.ps1 -Variant cpu
```

After successful setup, restart the Python backend, refresh diagnostics and explicitly click "测试 GPU 前向与反向". That button calls `POST /api/environment/smoke` to perform a small CUDA matrix operation and backward pass; opening or refreshing diagnostics never runs this probe automatically. Diagnostics and the probe share the training lock and return HTTP 409 during training. CUDA-unavailable or failed probes return HTTP 422. Finally, select CUDA in the training panel and run a classification job to verify the full training path.

If installation fails during uninstall/replacement, PyTorch may be left incomplete. Do not assume the old wheel still works: inspect `/api/environment` after restarting, or run the diagnostic below in a new process. Resolve permissions and stop DLL-holding processes before an explicitly authorized repair.

```powershell
.\.venv\Scripts\python.exe -c "import json; from backend.cuda_environment import inspect_environment; print(json.dumps(inspect_environment(), ensure_ascii=False))"
```

Local verification on 2026-10-07 used Python 3.13, `torch==2.9.1+cu128` (CUDA runtime 12.8), an NVIDIA GeForce RTX 4060 Ti and driver 591.74. The browser's explicit GPU forward/backward probe and a two-epoch CUDA classification run with imported position buffers and slices passed. An earlier replacement failed at a locked `c10.dll`; stopping the confirmed project backends and repairing `.venv` resolved it without changing the system driver or environment variables. This verifies that local configuration, not every GPU/driver combination.

### Single-sample inference and Tensor Inspector

Select a layer in the 3D scene or topology graph, open **张量观测**, and explicitly run one sample. The local PyTorch backend executes the same graph runtime in `eval()` and `inference_mode()` with batch size 1. It captures the selected layer and model Output; selecting a different layer requires another run unless that layer was already captured. This works with non-classification outputs as well as CNN/MLP/Attention graphs.

**Choose local trained weights or explicitly choose fresh, seeded random weights.** A successfully completed training run retains its final parameters and buffers; early stopping retains the restored best-validation state and identifies that state’s epoch separately from the completed epoch count. Stopped/failed runs, demonstrations and imported metrics do not produce a new trained model. The panel automatically selects a newly confirmed trained model, identifies the model ID, training run ID, graph fingerprint, weight epoch and training source, and separately identifies each inference run’s UTC timestamp, sample seed, device and input source. This does not load the original imported trainable weights.

The backend keeps **only the latest successful training snapshot in process memory**, bounded to 64 MiB including buffers and preprocessing. A later retained training model (including training in another page) replaces it, and restarting the backend removes it. Weights are not written to project JSON, browser storage or disk; refreshing the page does not reconnect to a previous model automatically. A graph edit that changes operations, parameters, input shapes or wiring disables the current trained selection; restoring the corresponding graph re-enables it. Graph/layer names and layout changes do not invalidate the weight binding. An unavailable or mismatched model ID is an explicit error, never an automatic fallback to random weights.

Choose a reproducible synthetic sample (integer IDs for Embedding) or provide JSON mapping every Input node ID to one flat sample array, without the batch axis. For CSV-trained weights, supply **raw CSV feature values**: inference replays the per-input mean/scale fitted only on the training split, including for synthetic inference inputs, before observing the Input node and subsequent layers. Integer Embedding inputs remain unnormalised. Random-weight and synthetic-trained models apply no CSV normalisation. With provided inputs, changing the inference seed does not reinitialise trained weights. This version infers integer input types when an Input connects directly to Embedding; an input shared between Embedding and non-Embedding consumers is rejected rather than silently cast.

The inspector shows measured shape/dtype, finite/non-finite element counts, full-tensor min/max/mean/population std, and a 12-bin histogram with keyboard/hover readouts and a data table. The numerical slice fixes the leading axes (sample/channel/depth as applicable) and shows up to 16 × 16 values from the final two dimensions; truncation is labelled. Change the leading-axis indices and rerun to inspect another channel or depth. Non-finite values are explicitly marked and excluded from finite statistics/histograms. No autograd graphs or full activations are sent to the browser. Editing the graph or sampling configuration invalidates the previous snapshot, including late responses from an earlier graph.

`POST /api/infer` accepts `graph`, `nodeIds`, optional `seed`, `device` (`cpu` by default, `cuda` or `auto`), optional `modelId` (a retained trained model; omitted/null means random weights), optional flat `inputs`, and optional leading-axis `slices` keyed by captured node ID. The report includes `weights`, `model` metadata (null for random weights) and `inputTransform`. Inference shares the training/diagnostics lock: concurrent work returns HTTP 409, invalid graphs/inputs or resource limits return HTTP 422. Opening the tab never starts inference, installs dependencies or probes CUDA. The existing particle animation and attention weight colours remain illustrative; attention internals are shown only in the separately labelled, explicit inference observation.

It accepts at most 8 captured nodes, 5 million parameters / 20 MiB parameter-storage cap, 4 million elements per input, 64 MiB of summed single-sample activation estimates, and 1 million attention-score elements per attention layer. These limits are checked before model allocation; displayed slices are capped at 16 × 16 and leading-axis indices at 3.

### Attention Inspector

Select a `MultiHeadAttention` or `Transformer` node in **张量观测**, choose a branch and Query head, and run the sample. The same inference report, seed/input, random or retained trained model, and lock are used for layer and attention observations. Native K/V groups are identified correctly for MQA/GQA; cross-attention and projection overrides use their actual Query/Key sequence lengths.

The panel exposes actual projected Q/K/V, actual SDPA head output before concatenation, branch output after Wₒ, and the branch-averaged attention output. For Transformer, this attention merge is **not** the final residual/FFN layer output, which remains separately available below. Scores `QKᵀ / √d` and `softmax(scores)` are recomputed in float32 from the actual projections for inspection; they can differ numerically from fused SDPA and do not replace its executed result. This eval path has dropout zero and no causal/padding mask.

Heatmaps have exact-value tables, hover/keyboard readouts, optional ordered textures, and selected light/dark palettes. Probability colors use a fixed 0–1 scale; signed tensors use a symmetric zero-centered scale based on full-matrix statistics. Every returned matrix has full shape/statistics but at most a 16 × 16 display window. Query/Key starting offsets are explicit, truncation is labelled, and changing selectors invalidates old observations until rerun.

The optional `/api/infer` field `attention` maps selected node IDs to `{branch, head, queryStart, keyStart}` (zero-based, defaults zero). A request allows at most one head/branch per selected node and eight selected nodes. The **full** selected-head score matrix is capped at 262,144 elements even if its displayed window is small; all selected inspection scratch estimates are added to the existing 64 MiB activation budget before model allocation. Other existing inference limits remain in force. Graphs can therefore run ordinary layer inference but exceed the stricter attention inspection budget. Invalid selectors/offsets or exceeded budgets fail explicitly, not as a partial or random-weight fallback.

Python generation is also on demand: opening the Python export preview generates code for the current graph; a closed export dialog or a JSON-only preview does not generate Python.

## About the warnings

Architecture-time: shape errors, cycles, unconnected layers, spatial downsampling truncation, hidden linear layers losing more than 90% of their information at once, eight consecutive parameterised layers without a residual, and a single layer holding more than 70% of the parameters. Training-time: non-finite loss or gradients, exploding/vanishing gradients, per-layer anomalies, ReLU zero activation above 90%, training loss persistently failing to improve, diverging loss, and a persistently widening train/validation gap.

These are heuristic risk signals; overfitting cannot be determined from static structure alone, nor can convergence be guaranteed. Overfitting requires validation-set metrics. Demo scenarios are explicitly labelled "demo data" and are kept distinct from real training and imported metrics.

JSON metric arrays or CSV metrics can be imported:

```csv
epoch,trainLoss,valLoss,accuracy,gradNorm
1,1.8,1.9,0.30,0.8
2,1.5,1.6,0.40,0.7
```

## Directory

`src/analysis.ts` shape/warning rules; `src/Scene.tsx` GPU 3D scene and export; `src/AttentionModule.tsx` attention polygons and internals; `src/Flow.tsx` GPU particle flows; `src/Topology.tsx` node editing; `src/Connections.tsx` input/output connection objects; `src/export.ts` PyTorch generation; `backend/graph.py` pre-allocation validation; `backend/training.py` real training; `backend/app.py` HTTP/WebSocket; `run.py` launch entry point.

This version supports basic CNN/MLP/residual and Transformer encoder computation graphs. RNNs, live bidirectional Python sync, ONNX import, distributed training and a full performance profiler are not supported yet.

## Building models from imported PyTorch code

"Import PyTorch code" in the top toolbar accepts pasted PyTorch source or a local `.py` file. You first choose the model object and input shapes, then submit to the local service for static AST parsing; the preview shows nodes, connections, parameter counts and diagnostics. On confirmation, the Input, layers, Add/Multiply/Concat, the Q/K/V ports of Attention and the Output are written into the current project, where you can keep dragging, editing parameters, training, viewing the 3D data flow and exporting code.

The current graph editor and real training path cover most common `torch.nn` layers: Conv/ConvTranspose 1D, 2D and 3D, Linear, Bilinear, BatchNorm/InstanceNorm/GroupNorm/LayerNorm, MaxPool/AvgPool/AdaptivePool, Flatten, the Dropout family, Embedding, Upsample, ReLU/GELU/Sigmoid/Tanh/SiLU/LeakyReLU/ELU/SELU/Softplus/PReLU/Hardsigmoid/Hardswish/Mish/Softsign/Softmax/LogSoftmax/Identity, plus MultiheadAttention, Transformer and custom MQA/GQA/Cross-Attention layers. Every layer's parameters take part in shape derivation, parameter counting, PyTorch training and Python export.

Example request:

```powershell
$body = @{
  source = "from torch import nn`nmodel = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))"
  model_name = "model"
  input_shapes = @{ x = @(1, 4) }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod -Uri http://127.0.0.1:8765/api/import/pytorch -Method Post -ContentType 'application/json' -Body $body
```

See the [source import support matrix](pytorch-import-support.md) for constructor/forward restrictions, known limitations and diagnostics, and the [five reproducible examples](../examples/import_models/README.md) for CNN, residual CNN, MQA, GQA and cross-attention inputs and run/import commands. Source-import support is narrower than layer-editor/runtime support; a matching layer name does not guarantee every PyTorch option is preserved.

The parser only reads the Python AST; it does not execute uploaded code, import modules from the source or load weights, optimizers or training scripts. It supports static `nn.Module`, `nn.Sequential`, nested modules, `ModuleList`, `ModuleDict`, residual addition, `torch.cat`, `flatten/view/reshape`, the common layers listed above, `unsqueeze/squeeze`, bounded tensor indexing, `MultiheadAttention`, `TransformerEncoderLayer`/`TransformerEncoder` and the custom Attention/Transformer/ConstantAdd classes. Query/Context and Q/K/V port connections for Cross-Attention and MQA/GQA are supported.

Static float32 buffers used in addition are evaluated by a bounded, torch-free interpreter and saved as `ConstantAdd` parameters, not dropped as passthrough operations. Supported construction includes `torch.zeros/ones/arange`, static arithmetic, selected `math` functions, and elementwise `exp/sin/cos/sqrt/abs`. Supported indexing includes integer selection, positive-step slices, negative bounds/indices, one ellipsis and inserted singleton axes. Examples such as `x + self.pe[:, :x.size(1)]` retain the buffer's real values and use the current input sequence length, within the stored capacity; `x[:, -1]` becomes `Select`. Batch-axis indexing, advanced/boolean indexing, negative steps, unsupported dynamic bounds, uninitialized `torch.empty` buffers and non-float32 buffers are rejected with diagnostics. Static tensors have at most 5 dimensions and 65,536 elements, evaluation is limited to 1,000,000 charged work units, and graph constant buffers total at most 65,536 elements. The supported operations are available in editing, shape analysis, backend execution and standalone Python export; float32 transcendental values are compared with numerical tolerances, not a bitwise equivalence guarantee.

What is imported is the model structure, its constructor arguments and these supported static constants, not the original trainable weights; after importing you can choose the data and the CPU/CUDA device in the training panel. Dynamic control flow, loops that depend on runtime data, arbitrary custom operators, weight sharing, complex masks and tensor operations that cannot be statically inferred are reported as errors or warnings. Source is limited to 512 KB, 30,000 AST nodes, and a generated graph of at most 128 nodes and 512 edges; up to 8 input shapes are allowed. When shapes are missing, conservative inference is used with a warning, and you can edit them in the window and re-parse.

The HTTP endpoint is `POST /api/import/pytorch`, with request fields `source`, optional `model_name` and `input_shapes`. The response contains `graph`, `models`, `model`, `inputs`, `diagnostics` and `analysis`, and can be integrated into other editors or automated pipelines.

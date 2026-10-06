# TensorLab 3D

A locally-run workbench for building deep learning models and visualising them in 3D. It is a runnable MVP based on a supplied Gemini conversation: a React / TypeScript / Three.js / React Flow frontend with a FastAPI / PyTorch training service.

## Running on Windows

Requires Node.js 20.19+ (22/24 recommended) and Python 3.10+ (3.12/3.13 recommended).

```powershell
# First-time install, including real training
powershell -ExecutionPolicy Bypass -File .\install.ps1 -Training

# Start, opening the browser on a free local port
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

Dependencies are already installed in the current directory and a compiled `dist` is present, so you can also just double-click `start.cmd`. The default address is http://127.0.0.1:8765, and it listens on localhost only. Closing the terminal stops the service.

If you don't need real training, `npm install`, `npm run build` and `python run.py` are enough. The standard-library server still provides the 3D view, model editing, warning demos and export.

## Development, testing and packaging

```powershell
# Terminal 1: real training backend
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8765

# Terminal 2: dev frontend, proxying /api to the service above
npm run dev

npm test
.\.venv\Scripts\python.exe -m unittest backend.test_training
# With the service running on 8765, run the browser flow tests using the installed Microsoft Edge
npm run test:e2e
npm run build
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

`build.ps1` produces `release/TensorLab-3D.zip`, containing source, build output, the backend and the launch scripts, but not `node_modules` or the virtual environment. Recipients need Python to run it, and real training additionally needs the backend dependencies installed; this is not a runtime-free Windows EXE.

## The workbench

- Ships with Compact CNN, Feature MLP, Residual CNN, Transformer Encoder, MQA, GQA, Cross-Attention and a blank template. The layer library supports both clicking and dragging into the scene. New layers connect to the currently selected layer and are inserted before the final output.
- The 3D view generates solid layers from each tensor's spatial dimensions and channels; geometric sizes use a logarithmic mapping so extreme sizes don't occlude each other.
- Clicking any 3D layer centres rotation and zoom on that layer; attention heads and their QKV, weight matrices, LN, FFN, merge and add modules can each be clicked and centred individually. Add-focused framing keeps the full residual bypass intact while centring on the Add node.
- The generic Add residual and the Transformer Encoder share the pink outer bypass, directional arrows and GPU particle styling. The residual branches off at the source layer's output, bypasses the intermediate modules, enters the pink "+" node from above, and continues after being summed with the main path. Camera focus and PNG / SVG export include the full bypass; the topology graph shows the pink outer curve as well. Skip branches are identified from reachability in the graph, while independent parallel branches keep ordinary edges.
- The topology graph supports dragging nodes and pulling wires from an output port to an input port; Add / Concat accept multiple inputs. Press Delete after clicking an edge to remove it, or remove the connection from the properties panel.
- The topology toolbar can add standalone input/output objects and delete the selected object; the properties panel's "connected objects" section lets you pick upstream/downstream objects and add or remove input/output connections individually. Q/K/V objects have their own ports, so input sources and output targets can be configured per vertex. Invalid connections stay in the graph with the reason shown; cycles, shape mismatches and illegal multi-inputs block training and Python export. Computation graphs support 1–8 model Inputs and one Output.
- Editing layer properties re-derives tensor shapes and checks for cycles, rank-4/rank-2 mismatches, residual shapes, concatenation dimensions, pooling parameters and resource limits. Projects are saved automatically in local browser storage, with undo/redo and JSON import/export.
- Exports a PyTorch `nn.Module`, including a runnable random-input test, with logits as output. Python code uploaded from the browser is never executed.
- PNG offers 1920 / 3840 / 7680 pixel widths, the current viewpoint and an optional transparent background; SVG export produces a simplified projection of the 3D geometry with editable layer labels.

### Attention and Transformer

`MultiHeadAttention` and `Transformer` use a shared PyTorch `scaled_dot_product_attention` runtime. `Transformer` is a pre-norm encoder layer (LN → Attention → residual → LN → GELU FFN → residual). Inputs and outputs use `[B,S,E]`. `attention_type` supports `self`, `multi_query` (MQA), `grouped_query` (GQA), `cross` and `multi_branch`; `num_heads` (1–16) is the number of queries and defaults to 1; `kv_heads` is the number of K/V groups; `branches` (1–8) is the number of independently computed branches that are then averaged. Embedding must equal the last input dimension and be divisible by the query count, and the query count must be divisible by the number of K/V groups. MQA uses a single K/V group, while standard self-attention uses one K/V group per query. Cross-Attention takes two inputs, Query and Context, and allows the two sequence lengths to differ. Every parameter changes the real computation, the parameter count and the exported code.

One polygon is drawn per shared K/V group, with all Q vertices that use that K/V placed in the same face; K and V are each drawn once, with no extra floating shared layer. The number of sides equals the number of queries in the group plus 2: a single query gives a triangle; MQA's 4 queries sharing one K/V give a hexagon; GQA's 8 queries over 2 K/V groups give two hexagons. `qkv_count` has been removed from the parameters panel, and the field is ignored in older projects. Groups are stacked along the data-flow X axis in the same direction, with attention heads computing in parallel; multiple branches are displayed separately per branch. Each query has its own softmax and V-weighting path, and the shared K/V connect directly to those compute nodes. The Transformer continues to show LN, FFN and the two pink residual paths. Rotation stops on selection or hover; once per-vertex connections are configured, the polygon orientation is fixed so that external particles stay aligned with the vertices. PNG / SVG share the same geometry and port layout as the scene.

Clicking a 3D Q/K/V cube centres it and opens that object's connection configuration, and it can also be selected from the "Q/K/V objects" selector in the properties panel. Each vertex inherits the model layer's inputs by default (for Cross-Attention, Q inherits Query and K/V inherit Context); specifying an upstream object overrides the corresponding projection source, and deleting the override connection restores the default source. A vertex outputs a real projected tensor `[B,S,E/num_heads]` and can connect to other model layers or to the input ports of other Q/K/V objects. All vertex inputs require `[B,S,E]`; a query's sequence length must match the layer's Query, and within a branch all K/V must share a sequence length. The Transformer's Q and self-attention overrides also pass through LN. Each projection port takes at most one specified input, while outputs can fan out to multiple objects. A complete model still needs a base input, and the Transformer's residual comes from the base Query.

Project connections store vertex ports via `sourcePort` / `targetPort`, for example `{"source":"memory","target":"attention","targetPort":"b0:k0"}` specifies the input of K1 in the first branch; `{"source":"attention","sourcePort":"b0:q0","target":"flatten"}` passes the projected Q1 of the first branch to Flatten. The same pair of model layers can be connected multiple times through different ports. Training, shape validation and standalone Python export all use the same connection semantics.

The colours and particles of the 3D weight matrices are still a **simulated illustration**; real attention weights, Q/K/V activations and token-level gradients are not collected, and the properties panel labels this "weight schematic". Loss and per-layer gradients in real training monitoring come from actual PyTorch training. The example input is a token feature vector; there is no text tokenizer, positional encoding or causal/padding mask yet.

## GPU animation and training

Three.js uses WebGL, GPU instancing and custom vertex shaders. Each particle's Bézier path is computed on the GPU, and the CPU only updates the time/direction uniforms per frame rather than rewriting positions per particle. Ordinary connections are cyan forward and coral backward; residual bypasses stay pink in both directions, with arrows and particle direction reversed when running backward. Pause, speed adjustment, orbit rotation, zoom, pan and expanded spacing are supported. The animation represents simulated propagation, not actual tensor values.

The browser needs hardware acceleration enabled. A software renderer is reported as "software rendering"; the app cannot force a browser or operating system to provide a physical GPU. WebGL GPU acceleration and PyTorch CUDA training are two separate device mechanisms.

Real training uses Adam + CrossEntropyLoss, with CPU / CUDA / automatic selection, over synthetic classification data or a local CSV. CSVs support an optional header row, with the last column as a zero-based integer class and the preceding columns as numeric features; the feature count must equal the number of elements in the input tensor excluding batch, with images flattened in C/H/W order. CSV values are standardised and then reshaped to the model's input shape. The validation split is a fixed random split. Epochs, learning rate, batch size, validation fraction and the early-stopping window are configurable.

Training pushes measured train/val loss, accuracy, global and per-layer gradient norms, and the ReLU zero-activation fraction over WebSocket. Gradient clipping is at a threshold of 100; early stopping triggers on a run of unimproved validation loss windows. Only one local training job runs at a time, and disconnecting the page or pressing stop halts it on the next batch. Inputs, data and activations have resource limits, and complex models need to run in a professional training environment.

PyTorch from the default PyPI index may be a CPU build. If you need CUDA, get the install command matching your GPU driver from https://pytorch.org/get-started/locally/, run it in this project's `.venv`, and restart the service; the interface will then show the CUDA device.

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

"Import PyTorch code" in the top toolbar accepts pasted PyTorch source or a local `.py` file. You first choose the model object and input shapes, then submit to the local service for static AST parsing; the preview shows nodes, connections, parameter counts and diagnostics. On confirmation, the Input, layers, Add/Concat, the Q/K/V ports of Attention and the Output are written into the current project, where you can keep dragging, editing parameters, training, viewing the 3D data flow and exporting code.

The current graph editor and real training path cover most common `torch.nn` layers: Conv/ConvTranspose 1D, 2D and 3D, Linear, Bilinear, BatchNorm/InstanceNorm/GroupNorm/LayerNorm, MaxPool/AvgPool/AdaptivePool, Flatten, the Dropout family, Embedding, Upsample, ReLU/GELU/Sigmoid/Tanh/SiLU/LeakyReLU/ELU/SELU/Softplus/PReLU/Hardsigmoid/Hardswish/Mish/Softsign/Softmax/LogSoftmax/Identity, plus MultiheadAttention, Transformer and TensorLab's MQA/GQA/Cross-Attention. Every layer's parameters take part in shape derivation, parameter counting, PyTorch training and Python export.

Example request:

```powershell
$body = @{
  source = "from torch import nn`nmodel = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))"
  model_name = "MLP"
  input_shapes = @{ x = @(1, 4) }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod -Uri http://127.0.0.1:8765/api/import/pytorch -Method Post -ContentType 'application/json' -Body $body
```

The parser only reads the Python AST; it does not execute uploaded code, import modules from the source or load weights, optimizers or training scripts. It supports static `nn.Module`, `nn.Sequential`, nested modules, `ModuleList`, `ModuleDict`, residual addition, `torch.cat`, `flatten/view/reshape`, the common layers listed above, `MultiheadAttention`, `TransformerEncoderLayer`/`TransformerEncoder` and TensorLab Attention/Transformer. Query/Context and Q/K/V port connections for Cross-Attention and MQA/GQA are supported.

What is imported is the model structure and its constructor arguments, not the original weights; after importing you can choose the data and the CPU/CUDA device in the training panel. Dynamic control flow, loops that depend on runtime data, arbitrary custom operators, weight sharing, complex masks and tensor operations that cannot be statically inferred are reported as errors or warnings. Source is limited to 512 KB, 30,000 AST nodes, and a generated graph of at most 128 nodes and 512 edges; up to 8 input shapes are allowed. When shapes are missing, conservative inference is used with a warning, and you can edit them in the window and re-parse.

The HTTP endpoint is `POST /api/import/pytorch`, with request fields `source`, optional `model_name` and `input_shapes`. The response contains `graph`, `models`, `model`, `inputs`, `diagnostics` and `analysis`, and can be integrated into other editors or automated pipelines.

# CLAUDE.md

## Project

TensorCraft3D (Build and Explore Neural Networks in 3D) is a localhost-only model workbench: React 19 / TypeScript / Vite, Three.js / React Three Fiber, React Flow, and a FastAPI backend with optional PyTorch training. UI labels are primarily Chinese.

A version-1 `Graph` is the single source of truth for editing, shape analysis, visualization, Python export, and training. Supports CNN/MLP/residual and Transformer/MQA/GQA/cross-/multi-branch attention models. Animated particles and weight schematics are illustrative, not measured activations; WebGL and PyTorch CUDA are independent.

## Commands

Run from the repository root. Windows is primary; use the existing `.venv` for Python. Requires Node.js 20.19+ and Python 3.10+.

| Task | Command |
| --- | --- |
| Build | `npm run build` |
| Typecheck | `npm exec --no -- tsc --noEmit` |
| Frontend dev | `npm run dev` |
| Backend dev | `.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8765` |
| Integrated app | `.\.venv\Scripts\python.exe run.py --no-browser` |
| One source test file | `npm exec --no -- vitest run src/imageLayout.test.ts` |
| One backend test | `.\.venv\Scripts\python.exe -m unittest backend.test_training.TrainingTests.test_shape_and_parameter_contract` |
| Focused browser tests | `npm run test:e2e -- e2e/runtime.spec.ts` (or `e2e/imageExport.spec.ts`) |
| Explicit PyTorch setup | `.\setup-training.ps1 -Variant cu128` (or `-Variant cpu`) |
| Release archive | `powershell -ExecutionPolicy Bypass -File .\build.ps1` |

- Dev: Vite proxies `/api` and WebSockets to backend port 8765; keep the frontend on a different port.
- `run.py` serves `dist`: build after source changes. It tries ports 8765–8784 by default; `--port` selects the starting port. Without FastAPI/Uvicorn it falls back to static-only serving (no Python API or training).
- `start.ps1` builds and prefers `.venv`; `start.cmd` uses the existing build. `npm start` uses the shell's `python`.
- Playwright requires an already-running server, defaults to `http://127.0.0.1:8765`, and uses installed Microsoft Edge. For another port, set `$env:TENSORLAB_TEST_URL = 'http://127.0.0.1:8766'`.
- Prefer targeted checks; full suites only when requested/required: `npm test`, `npm run test:e2e`, or `.\.venv\Scripts\python.exe -m unittest backend.test_training backend.test_pytorch_import backend.test_tensor_ops backend.test_cuda_environment`. Some source tests and real-training browser tests require Python/PyTorch. No lint script.
- Do not install/change dependencies without authorization. `setup-training.ps1` changes project `.venv` only and refuses PyTorch replacement while project Python processes run; it does not change drivers/environment variables. `install.ps1` is for explicitly requested setup. Release output is `release/TensorCraft3D.zip`, not a self-contained executable.

## Key Files & Contracts

| Area | Files / responsibility |
| --- | --- |
| Graph state | `src/types.ts`: shared contract; `src/App.tsx`: graph history, selection, import/export/training UI, persistence (`tensorlab-project`) |
| Validation | `src/analysis.ts` and `backend/graph.py`: independent graph/shape/resource validation |
| Attention | `src/attentionConfig.ts`: configuration/ports; `src/attentionLayout.ts`: geometry/endpoints/routes; `src/AttentionModule.tsx`, `src/Scene.tsx`, `src/Flow.tsx`: rendering; `src/graphRoutes.ts`: reachability-based Add shortcuts |
| Image export | `src/sceneImageExport.ts`: PNG/SVG preparation/routing/camera; `src/imageLayout.ts`: compact placement/scaling/labels; `src/attentionExport.ts`: flat attention geometry |
| Python | `src/export.ts`: standalone `VisualModel`; `backend/pytorch_import.py`: static AST import; `backend/static_tensors.py`: torch-free constant evaluation |
| Runtime | `backend/training.py`, `backend/attention.py`, `backend/tensor_ops.py`: validated execution shared with generated Python; `backend/cuda_environment.py`: local CUDA diagnostics |
| API | `backend/app.py`: health/analyze/PyTorch import, read-only environment diagnostics, explicit smoke probe, training WebSocket, built frontend |

### Graph, import & training

- Graphs allow 1–8 Inputs, one Output, up to 128 nodes / 512 edges. Invalid editable connections may remain with diagnostics; execution and Python export reject invalid graphs. Validate before allocation.
- Operation/connection changes must stay consistent across frontend/backend analysis, training, Python export, and static import. Attention projection IDs (e.g. `b0:q0`) are edge `sourcePort`/`targetPort`; cross-attention uses `query`/`context`. Different ports between the same nodes are distinct connections.
- Import never executes uploaded source, imports its modules, or loads weights. Limits: 512,000 UTF-8 source bytes / 30,000 AST nodes. Unsupported dynamic behavior produces diagnostics.
- Static float32 constants are bounded to 5 dimensions / 65,536 elements, a 1,000,000-unit evaluation budget, and 65,536 total graph-buffer elements. Preserve `ConstantAdd` buffer values and `Slice`/`Select` indexing semantics across all paths. Reject unsupported dtype/uninitialized buffers, batch-axis or advanced indexing, negative steps, and dynamic bounds.
- Real training is local classification with logits `[B,2..256]`, synthetic/CSV data, CPU/CUDA. This is not arbitrary Python execution, distributed training, ONNX import, live Python synchronization, or a full profiler. See `README.md` for operation support and user-facing limits.
- CUDA diagnostics load only when the panel expands. Opening/refreshing never installs anything or runs a GPU probe; smoke testing is explicit. Training and full diagnostics/probes share one local lock. Block training while any diagnostic/probe is in flight, including across panel folding and dialog close/reopen.

### Image export

- Keep compact placement separate from rendered size: placement uses ordinary-block scale 1.6 and unscaled flat attention; render targets are 2.0 / 1.12, capped by `imageNodeScale()` within existing space. Preserve accepted node centers, stages, lanes, and wrap decisions.
- Use effective rendered bounds for endpoints, camera fitting, captions, residual clearance, and cross-band offsets. PNG/SVG share preparation and label collision handling. Export must not mutate interactive scene/camera or shared geometry.
- Shared-input heads may stack; independently wired faces and separate branches remain separate. Preserve full face size and real projection-port associations. Do not change attention/flow colors for size/layout tasks.
- Regression tests: `src/imageLayout.test.ts`, `src/attentionLayout.test.ts`, `e2e/imageExport.spec.ts`. Dense fixture: `artifacts/DualBranchCrossAttentionTransformer.json` via `e2e/imageGraph.ts`. Browser tests regenerate files under `artifacts`; inspect the worktree and resulting images.

## Improvement Priorities

Product direction: an editable 3D PyTorch model workbench for learners, teachers, and small-model experiments. Prioritize the workflow **import structure → edit and validate → inspect a run → export code and figures**. The items below are a proposed backlog, not implemented capabilities or an instruction to execute every item during unrelated work. Research and rationale: [competitive analysis](docs/research/tensor-visualization-landscape-2026-10-07.md).

| Priority | Improvement | Acceptance target |
| --- | --- | --- |
| P0 | Clear project description, short demo, and guided examples | Explain the workbench in one sentence; CNN, residual, and MQA/GQA examples demonstrate import → edit → shape validation → export. Offer structure exploration before optional training setup. |
| P1 | Single-sample inference and Tensor Inspector | Selecting a layer shows measured shape/dtype, min/max/mean/std, a histogram, and a tensor slice, with the originating run/sample identified. |
| P1 | Small-model Attention Inspector | Select a branch/head and inspect actual Q/K/V, scaled QKᵀ scores, softmax probabilities, and output, with correct tensor dimensions and MQA/GQA sharing. |
| P2 | Module folding, synchronized navigation, and figure annotations | Switch between graph overview and module internals; synchronize 2D/3D selection and retain readable PNG/SVG exports without altering graph execution. |
| P2 | Import support matrix and reproducible examples | Document supported layers/operations and limitations; provide CNN, residual, and Transformer source examples with input shapes and expected import results. |
| P3 | External Python observation interface, then evaluate ONNX | Start with offline structure and sampled-tensor snapshots; assess live synchronization and additional formats after this workflow works. |

### Observation implementation constraints

- Keep structural animation/weight schematics distinguishable from measured observation. Real training metrics do not make the existing particle animation or weight colors measured data.
- Start with one sample from a batch; bound captured layers, channels, sequence lengths, payload sizes, and sampling frequency. Send statistics/downsampled values rather than every activation every frame; do not retain autograd graphs for display.
- Use channel feature maps for CNNs, position/token × feature heatmaps for sequences, and query × key heatmaps for attention. Label sequence positions as tokens only when actual token metadata is available. Let 3D provide navigation and the inspector provide precise values.
- `scaled_dot_product_attention` currently does not return attention probabilities. If an explicit diagnostic mode reconstructs them from captured Q/K, preserve scaling, KV grouping, and any supported mask semantics; bound the quadratic matrix size. Check deterministic eval output against the normal execution path within numerical tolerance, and keep ordinary training behavior unchanged.
- Distinguish learned projection matrices Wq/Wk/Wv from input-dependent attention probabilities A. Diagnostic snapshots must identify their source and must not present simulated values as observations.
- Describe import as supported static structure plus supported constants; it does not load the original trainable weights or support arbitrary Python behavior. Do not advertise arbitrary-model import or full GPT debugging without implementing and verifying the missing capabilities.

### Branding considerations

- The adopted product name is TensorCraft3D with the subtitle “Build and Explore Neural Networks in 3D” (Chinese UI: “神经网络三维构建与探索”). Repository description/topics/name changes belong to the publication workflow.
- Preserve compatibility for `tensorlab-project`, `application/tensorlab-layer`, `TENSORLAB_TEST_URL`, `TensorLabAttention`/`TensorLabTransformer`, and `_TENSORLAB_INPUT_SHAPES`. Introduce migrations or parsing aliases before replacing persisted or exported identifiers; avoid a global search-and-replace that breaks saved projects or existing Python exports.

## Conventions & Status

- TypeScript: strict, ES2022, functional React/hooks, two spaces, single quotes, semicolons, type-only imports. Python: four spaces, established type hints, explicit validation. Match surrounding style; avoid unrelated refactors.
- Reuse shared attention/layout/routing helpers. Preserve Chinese control labels and `aria-label` semantics used by browser tests.
- Completed: compact-export module enlargement without reflow; static tensor/indexing round-trip and execution; local CUDA diagnostics and explicit probes. Check Git for current branch/commit and `README.md` for detailed behavior; historical test runs are not verification of new changes.
- Keep this guide focused on durable contracts and commands. Report only checks actually run and unresolved limitations; do not accumulate session logs, screenshots, or obsolete branch/environment details here.

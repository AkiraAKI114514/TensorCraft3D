# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TensorLab 3D is a localhost-only deep learning model workbench. A React 19 / TypeScript / Vite frontend combines a Three.js / React Three Fiber scene with a React Flow topology editor. A FastAPI backend provides static PyTorch source import, graph analysis, and optional real PyTorch training.

The canonical project is a version-1 `Graph` containing layers, parameters, editor positions, and edges. The same graph drives editing, shape diagnostics, visualization, standalone Python generation, and training. Supported model families include CNN, MLP, residual networks, Transformer encoders, MQA, GQA, cross-attention, and multi-branch attention.

The browser's animated particles and weight schematics are illustrative, not measured tensor activations. Real training metrics arrive separately over WebSocket. WebGL acceleration and PyTorch CUDA support are independent.

## Build & Run Commands

Run commands from the repository root. Windows is the primary environment; use the existing `.venv` interpreter for Python commands. Prerequisites documented in `README.md` are Node.js 20.19+ and Python 3.10+.

| Task | Command |
| --- | --- |
| Production build (TypeScript + Vite) | `npm run build` |
| Typecheck without building | `npm exec --no -- tsc --noEmit` |
| Frontend development server | `npm run dev` |
| Preview the production frontend | `npm run preview` |
| Integrated built app, without opening a browser | `.\.venv\Scripts\python.exe run.py --no-browser` |
| Start with a fresh build and open the browser | `powershell -ExecutionPolicy Bypass -File .\start.ps1` |
| Backend for frontend development | `.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8765` |
| All source tests | `npm test` |
| One source test file | `npm exec --no -- vitest run src/imageLayout.test.ts` |
| One source test by name | `npm exec --no -- vitest run src/imageLayout.test.ts -t "caps wide blocks"` |
| Backend tests | `.\.venv\Scripts\python.exe -m unittest backend.test_training backend.test_pytorch_import` |
| One backend test | `.\.venv\Scripts\python.exe -m unittest backend.test_training.TrainingTests.test_shape_and_parameter_contract` |
| All browser tests | `npm run test:e2e` |
| Focused image-export browser tests | `npm run test:e2e -- e2e/imageExport.spec.ts` |
| Build the Windows release archive | `powershell -ExecutionPolicy Bypass -File .\build.ps1` |

There is no lint script configured in `package.json`.

### Startup Details

- Development uses two terminals: backend on `127.0.0.1:8765`, then `npm run dev` on Vite's frontend port. `vite.config.ts` proxies `/api`, including WebSockets, to port 8765. Do not put the dev frontend on 8765 when using that backend.
- `run.py` serves `dist`, so run `npm run build` after source changes before using the integrated app. It defaults to port 8765, tries the next 19 ports if occupied, and prints the selected URL. `--port` selects the starting port; `--no-browser` suppresses browser launch.
- With FastAPI and Uvicorn installed, `run.py` launches `backend.app:app`. Without them, it serves a browser-only static fallback; Python API import and training are unavailable in that mode.
- `start.ps1` builds and prefers the virtual-environment interpreter. `start.cmd` launches the existing build without rebuilding. `npm start` uses the current shell's `python`, not explicitly `.venv`.
- `playwright.config.ts` expects a server already running at `http://127.0.0.1:8765`; it does not start one. Tests use installed Microsoft Edge (`msedge`), headless WebGL, one worker, and a 1440x900 default viewport. Verify the actual port if using `run.py`.
- Full browser tests include backend requests and real training. Full source tests also require Python/PyTorch: several Vitest files invoke `.venv/Scripts/python.exe` to compare generated models with backend execution. The image-layout/attention-layout tests do not require the backend.
- For explicitly requested environment setup, `install.ps1` installs npm dependencies, builds, creates `.venv`, and installs base backend requirements. `install.ps1 -Training` additionally installs PyTorch. Base requirements are in `backend/requirements.txt`; training requirements add `torch>=2.6,<3`. Do not install or change dependencies merely to run a documentation or narrow frontend task.
- `build.ps1` creates `release/TensorLab-3D.zip`. The release includes sources and `dist`, but not `node_modules` or `.venv`; it is not a self-contained Windows executable.

## Current Status & Progress

Last updated: 2026-10-07.

### Completed Image-Export Work

The latest requested change enlarges modules while preserving the previous compact arrangement shown in `artifacts/compact-model-before-size.png`.

- Placement and render dimensions are separate. Compact placement still uses the original ordinary-block scale of 1.6 and unscaled flat-attention dimensions, so stage widths, lane ordering, and wrap decisions remain unchanged.
- Render-only target scales are 2.0 for ordinary blocks and 1.12 for attention modules. `imageNodeScale()` caps growth within existing column/lane space rather than reflowing the model.
- Attention geometry and external query/context/Q/K/V endpoints share the effective per-node scale. Cross-band entry/exit offsets account for enlarged boundaries; residual clearance uses rendered dimensions.
- PNG and SVG share preparation, camera fitting, and label collision handling. The interactive scene and current-view export are not resized by compact-export scaling.
- The dense fixture retains two compact bands and the reference node-center arrangement, allowing only uniform camera zoom/translation. Measured projected bounding-box area increased approximately 51% for ordinary sample modules and 20-22% for attention sample modules; these percentages are fixture-specific.
- Updated outputs are `artifacts/compact-model-1920.png`, `artifacts/compact-model-square.png`, and `artifacts/compact-model.svg`. Original `compact-model-before-size.png` and `.svg` references are retained alongside the updated samples.
- Delivery branch: `feat/enlarge-export-modules`. Code, regression tests, export samples, baseline references, and this project guide are included in the same change. Check Git for the current commit and remote synchronization status.

### Verification At Handoff

- `npm exec --no -- vitest run src/imageLayout.test.ts src/attentionLayout.test.ts`: 24 tests passed.
- `npx --no-install playwright test e2e/imageExport.spec.ts`: 2 tests passed. Coverage includes legacy normalized node positions, module size, node/label overlap, PNG dimensions/transparency/frame fill, and query/context/Q/K/V edge alignment in a 16-head fixture.
- `npm exec --no -- tsc --noEmit`: passed.
- `git diff --check`: passed; Git reported only LF-to-CRLF conversion warnings.
- Actual PNG output was rendered and visually inspected. Comparison against the original SVG confirmed unchanged normalized node centers.
- The full Vitest suite, full browser suite, backend suites, and production build were not run for this export change. Do not describe those as verified.

### Existing Boundaries

Real training is local classification with logits `[B,2..256]`, synthetic or CSV data, and CPU/CUDA execution. Python import reads model structure and constructor arguments, not weights or training scripts. This is not a general Python execution environment, distributed trainer, ONNX importer, live Python synchronization system, or full profiler. See `README.md` for the supported operation set and user-facing limitations; verify implementation before extending its claims.

## Architecture

### Graph State & Validation

`src/types.ts` defines the shared frontend graph/edge/parameter contract. `src/App.tsx` owns graph history, selection, import/export UI, and training state, and persists the graph under the browser-storage key `tensorlab-project`. The scene and topology editor are views over this state, not separate model representations.

`src/analysis.ts` validates client graphs, computes topological order, derives shapes and parameter/activation estimates, and emits diagnostics. `backend/graph.py` independently validates the graph before model allocation and derives backend shapes/limits. Operation or connection changes often require coordinated updates to both analyzers, training construction, Python generation, and static import mappings.

Graphs allow 1-8 Inputs and one Output, up to 128 nodes and 512 edges. Shapes, parameters, and attention allocations have bounded resource checks. Invalid editable connections may remain in the project with diagnostics, but execution and standalone Python export must reject invalid graphs.

### Attention & Connections

`src/attentionConfig.ts` centralizes attention configuration and port semantics. Projection IDs such as `b0:q0`, `b0:k0`, and `b0:v0` are stored in edge `sourcePort`/`targetPort`; cross-attention uses base `query`/`context` inputs. Connections between the same nodes through different ports are distinct.

`src/attentionLayout.ts` is the shared geometry/port/route description. Interactive attention rendering consumes it in `src/AttentionModule.tsx`; `src/Scene.tsx` composes nodes and connections, while GPU flow rendering uses `src/Flow.tsx` and shared routing geometry. `src/graphRoutes.ts` detects Add shortcuts from reachability rather than naming them as residuals.

Image export builds flat attention geometry in `src/attentionExport.ts`. Heads with a shared input can form stepped stacks; independently wired faces keep separate columns, and distinct branches do not stack together. Preserve full face size and the association between displayed vertices and real projection ports.

### Image Export

`src/sceneImageExport.ts` clones the scene, prepares compact/current layouts, rebuilds export attention geometry, routes edges, fits the camera, and exports PNG/SVG. `src/imageLayout.ts` handles stage-preserving compact placement, bounded render scaling, caption wrapping, and shared screen-space label packing.

Keep layout dimensions distinct from rendered bounds. Module-size changes must not silently alter the accepted compact arrangement. Use rendered geometry for visible endpoints, fitting, and caption obstacles; do not mutate the interactive scene/camera or shared geometry during export.

`e2e/imageGraph.ts` loads the dense fixture from `artifacts/DualBranchCrossAttentionTransformer.json`. `e2e/imageExport.spec.ts` checks actual downloaded PNG/SVG rather than only internal layout calculations. Browser tests write samples under `artifacts`; inspect the resulting worktree so regenerated samples are not mistaken for unrelated user edits.

### Python Import, Export & Training

`src/export.ts` generates standalone PyTorch `VisualModel` code and embeds `backend/attention.py` as raw source for attention graphs. Backend training and generated Python must agree on graph order, layer configuration, projection shapes, and attention behavior.

`backend/pytorch_import.py` is a bounded static AST parser. It must never execute uploaded source, import its modules, or load its weights. Import limits include 512,000 UTF-8 source bytes, 30,000 AST nodes, and 128 nodes/512 edges; unsupported dynamic behavior produces diagnostics. The frontend import preview confirms a parsed graph before replacing the current project.

`backend/app.py` exposes `/api/health`, `/api/analyze`, `/api/import/pytorch`, and WebSocket `/api/train`, and serves built frontend files. A lock permits one local training job. `backend/training.py` constructs validated, whitelisted PyTorch layers and trains with Adam/CrossEntropyLoss, validation metrics, gradient clipping, early stopping, and cooperative stop/disconnect handling. `backend/attention.py` supplies the attention runtime used by both paths.

## Style & Implementation Guidelines

- TypeScript uses strict checking, ES2022, React functional components/hooks, two-space indentation, single-quoted imports/strings, semicolons, and type-only imports. Follow the surrounding compact style without reformatting unrelated code.
- Python uses four-space indentation, type hints where established, and explicit validation. Preserve the static-import boundary and validation-before-allocation behavior.
- The UI is primarily Chinese, with technical identifiers and operation names in English. Keep existing control labels and `aria-label` semantics consistent; Playwright locates many controls by their accessible Chinese names.
- Shape analysis and execution rules are cross-language contracts. Reuse existing attention configuration, layout, and route helpers rather than implementing parallel interpretations.
- For export-only visual edits, preserve the user's accepted compact topology and protect layout positions as well as geometry size in tests. Updating a camera or passing enlarged bounds into the placement optimizer is not equivalent to enlarging module area.
- Avoid changing attention/flow colors or claiming the animation represents measured activations when the task concerns only size or layout.
- Keep `Current Status & Progress` synchronized with completed work and actual verification. Record unresolved limitations without marking unrun tests as passed.

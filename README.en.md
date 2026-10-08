# TensorCraft3D

[简体中文](README.md) | **English**

### Build it. Understand it. Run it.

A local **3D neural network workbench**: drag and drop layers or import PyTorch code, explore your model, run real training, then export code and model diagrams. The interface supports Chinese and English.

## ✨ What can you do?

| Your goal | How TensorCraft3D helps |
| --- | --- |
| **Build models visually** | Drag in layers, connect nodes and edit parameters. See tensor shapes and parameter counts update, with checks for shape conflicts and invalid connections. |
| **Understand the architecture** | Explore CNNs, MLPs, residual networks and Transformers in 3D and topology views. Look inside Attention to see Q/K/V, multiple heads and shared projections. |
| **Keep using your code** | Import supported PyTorch source, preview its structure and keep editing. Export a runnable `nn.Module`. |
| **Train and spot problems** | Train on synthetic data or a local CSV using CPU / CUDA. Track real loss, accuracy, gradients and risk warnings. |
| **Inspect real computation** | Run single-sample inference to inspect tensor values, statistics and distributions. Attention Inspector shows Q/K/V, attention scores and probability heatmaps. |
| **Save and share your work** | Autosave, undo/redo, project JSON import/export, plus high-resolution PNG and editable SVG model diagrams. |

**Built-in templates:** CNN · MLP · Residual CNN · Transformer Encoder · MQA · GQA · Cross-Attention. Or start with a blank canvas.

## 🚀 Quick start (Windows)

Install **Node.js 20.19+** and **Python 3.10+**, then run from the project root:

```powershell
# First-time setup, including PyTorch training dependencies
powershell -ExecutionPolicy Bypass -File .\install.ps1 -Training

# Start and open the browser automatically
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

The default address is **http://127.0.0.1:8765**; an available port is selected if it is busy. After installation, you can also double-click `start.cmd`. Close the terminal to stop the service.

Only want to build models and explore 3D? Omit `-Training` during setup. CUDA training requires a compatible NVIDIA driver and a CUDA build of PyTorch; see the [setup guide](docs/workbench-guide.md#local-cuda-diagnostics-and-setup).

## 🎯 Try it in four steps

1. **Pick a template:** open Residual CNN or Transformer, rotate and zoom, then click a layer to explore its internals.
2. **Make it yours:** add layers, change parameters and connect nodes. Watch the shapes and structure update.
3. **Run real computation:** start training to track metrics, or open Tensor Inspector and run single-sample inference.
4. **Take it with you:** export PyTorch code for further development, or PNG / SVG diagrams for presentations.

## A few things to know

- **Animation is illustrative:** particles and weight colors are schematics. Training metrics and explicitly requested tensor observations come from real PyTorch computation.
- **Source import does not load weights:** it parses supported static structures and constants without executing uploaded code or importing original trainable weights.
- **Trained weights stay in memory:** only the latest successfully trained model is retained. Restarting the backend clears it; exported code and project JSON do not include trained weights.
- **Built for learning and small experiments:** training currently targets classification. Use professional tools for large models, distributed training and full performance profiling. Risk warnings do not guarantee training results.

## Learn more

- [Usage, development and technical guide](docs/workbench-guide.md) — training, CUDA, inspection, export, testing and packaging
- [PyTorch import support](docs/pytorch-import-support.md) — supported syntax, limitations and diagnostics
- [5 reproducible import examples](examples/import_models/README.md) — CNN, residual, MQA, GQA and Cross-Attention

**Built with:** React · TypeScript · Three.js · React Flow · FastAPI · PyTorch

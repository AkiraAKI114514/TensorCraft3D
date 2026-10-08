# TensorCraft3D

### 把神经网络搭出来、看明白、跑起来。
### Build it. Understand it. Run it.

一个在本地运行的 **3D 神经网络工作台**：拖拽搭建或导入 PyTorch 代码，探索模型结构、运行真实训练，再导出代码与模型图。支持中英文界面。

A local **3D neural network workbench**: build visually or import PyTorch code, explore your model, run real training, then export code and diagrams. Chinese and English UI supported.

## ✨ 你可以做什么？ · What can you do?

| 功能 · Feature | 你能获得什么 · What you get |
| --- | --- |
| **直观搭建 · Build visually** | 拖拽图层、连接节点、修改参数；实时查看形状与参数量，检查无效连接。<br>Drag in layers, wire nodes and edit parameters. See shapes and parameter counts update, with checks for invalid connections. |
| **看懂结构 · Explore in 3D** | 在 3D 与拓扑图中探索 CNN、MLP、残差和 Transformer，查看 Attention 的 Q/K/V、多头与共享关系。<br>Explore CNNs, MLPs, residual networks and Transformers in 3D and topology views, including Attention's Q/K/V, heads and shared projections. |
| **复用代码 · Use your code** | 导入支持的 PyTorch 源码，预览后继续编辑；导出可运行的 `nn.Module`。<br>Import supported PyTorch source, preview and edit its structure, then export a runnable `nn.Module`. |
| **真实训练 · Train for real** | 用合成数据或本地 CSV 在 CPU / CUDA 上训练，查看 loss、准确率、梯度与风险提示。<br>Train on synthetic data or a local CSV using CPU / CUDA. Track real loss, accuracy, gradients and risk warnings. |
| **观测计算 · Inspect tensors** | 单样本推理查看数值、统计与分布；Attention Inspector 展示 Q/K/V、注意力分数与概率热图。<br>Inspect tensor values, statistics and distributions with single-sample inference; view Q/K/V, scores and probability heatmaps in Attention Inspector. |
| **保存分享 · Save and share** | 自动保存、撤销/重做、项目 JSON 导入导出，以及高清 PNG / 可编辑 SVG 模型图。<br>Autosave, undo/redo, project JSON import/export, plus high-resolution PNG and editable SVG diagrams. |

**内置模板 · Templates：** CNN · MLP · Residual CNN（残差）· Transformer Encoder · MQA · GQA · Cross-Attention
也可以从空白画布开始。You can also start with a blank canvas.

## 🚀 快速开始 · Quick start (Windows)

准备 **Node.js 20.19+** 和 **Python 3.10+**，在项目根目录运行：<br>Install **Node.js 20.19+** and **Python 3.10+**, then run from the project root:

```powershell
# 首次安装（含 PyTorch 训练依赖）· First-time setup with training dependencies
powershell -ExecutionPolicy Bypass -File .\install.ps1 -Training

# 启动并打开浏览器 · Start and open the browser
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

默认地址 **http://127.0.0.1:8765**，占用时自动换端口。安装后可双击 `start.cmd`；关闭终端停止服务。<br>Opens at **http://127.0.0.1:8765**, or an available port if busy. After setup, double-click `start.cmd` to launch; close the terminal to stop.

只想搭模型和看 3D？去掉 `-Training`。CUDA 训练需要兼容的 NVIDIA 驱动与 CUDA 版 PyTorch。<br>For editing and 3D only, omit `-Training`. CUDA training needs a compatible NVIDIA driver and CUDA-enabled PyTorch. [配置说明 · CUDA setup](docs/workbench-guide.md#local-cuda-diagnostics-and-setup)

## 🎯 四步上手 · Try it in four steps

1. **选模板 · Pick a template**：打开残差 CNN 或 Transformer，旋转、缩放、点击图层。<br>Open Residual CNN or Transformer, rotate and zoom, then click a layer to explore.
2. **改模型 · Make it yours**：拖入图层、调整参数、连接节点，观察结构与形状变化。<br>Add layers, edit parameters and wire nodes. Watch shapes and structure update.
3. **跑计算 · Run computation**：开启训练看指标，或打开「张量观测」运行单样本推理。<br>Start training to track metrics, or open Tensor Inspector for single-sample inference.
4. **导出成果 · Take it with you**：导出 PyTorch 代码继续开发，或导出 PNG / SVG 用于讲解。<br>Export PyTorch code for further development, or PNG / SVG diagrams for presentations.

## 使用须知 · Good to know

- **动画是示意 · Animation is illustrative**：粒子与权重配色不是测量值；训练指标和显式运行的张量观测来自真实 PyTorch 计算。<br>Particles and weight colors are schematics; training metrics and explicitly requested tensor observations come from real PyTorch computation.
- **导入结构，不导入权重 · Structure, not weights**：仅解析支持的静态结构与常量，不执行上传源码，也不导入原始可训练权重。<br>Source import parses supported static structures and constants, without executing uploaded code or loading original trainable weights.
- **训练权重仅存内存 · Weights stay in memory**：仅保留最近一次成功训练的模型，重启后端会清除；导出的代码与 JSON 均不含训练权重。<br>Only the latest successfully trained model is retained; restarting the backend clears it. Exported code and JSON do not include trained weights.
- **学习与小规模实验 · Learning and small experiments**：目前面向分类训练；大型模型、分布式训练和完整性能分析请使用专业工具，风险提示不保证训练效果。<br>Training currently targets classification. Use professional tools for large models, distributed training and full profiling; warnings do not guarantee training results.

## 深入了解 · Learn more

- [使用与技术说明 · Workbench guide](docs/workbench-guide.md) — 训练、CUDA、观测、导出、测试与打包 · Training, CUDA, inspection, export, testing and packaging
- [PyTorch 导入支持 · Import support](docs/pytorch-import-support.md) — 语法、限制与诊断 · Syntax, limitations and diagnostics
- [5 个可复现示例 · 5 reproducible examples](examples/import_models/README.md) — CNN、残差 / residual、MQA、GQA、Cross-Attention

**技术栈 · Built with：** React · TypeScript · Three.js · React Flow · FastAPI · PyTorch

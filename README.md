# TensorCraft3D

### 把神经网络搭出来、看明白、跑起来。

一个在本地运行的 **3D 神经网络工作台**：从拖拽搭建或导入 PyTorch 代码开始，探索模型结构、运行真实训练，再导出代码与模型图。支持中英文界面。

## ✨ 你可以做什么？

| 想做的事 | TensorCraft3D 帮你完成 |
| --- | --- |
| **直观搭建模型** | 拖拽图层、连接节点、修改参数，实时查看张量形状与参数量；自动检查形状冲突和无效连接。 |
| **看懂网络结构** | 在 3D 场景与拓扑图中探索 CNN、MLP、残差和 Transformer，展开 Attention 的 Q/K/V、多头与共享关系。 |
| **接着用已有代码** | 导入支持的 PyTorch 源码，预览结构后继续编辑；导出可运行的 `nn.Module` 代码。 |
| **训练并发现问题** | 使用合成数据或本地 CSV，在 CPU / CUDA 上训练；查看真实 loss、准确率、梯度与风险提示。 |
| **观察真实计算** | 单样本推理查看张量数值、统计与分布；Attention Inspector 展示 Q/K/V、注意力分数与概率热图。 |
| **保存与分享成果** | 自动保存、撤销/重做、项目 JSON 导入导出，以及高清 PNG / 可编辑 SVG 模型图导出。 |

**内置模板：** CNN · MLP · 残差 CNN · Transformer Encoder · MQA · GQA · Cross-Attention，也可以从空白画布开始。

## 🚀 快速开始（Windows）

准备 **Node.js 20.19+** 和 **Python 3.10+**，在项目根目录运行：

```powershell
# 首次安装，包含 PyTorch 训练依赖
powershell -ExecutionPolicy Bypass -File .\install.ps1 -Training

# 启动并自动打开浏览器
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

默认访问 **http://127.0.0.1:8765**，端口占用时自动换用空闲端口。安装完成后，也可以双击 `start.cmd` 启动；关闭终端即可停止服务。

只想搭模型和看 3D？首次安装去掉 `-Training` 即可。CUDA 训练需要兼容的 NVIDIA 驱动和 CUDA 版 PyTorch，[查看配置说明](docs/workbench-guide.md#local-cuda-diagnostics-and-setup)。

## 🎯 第一次怎么玩？

1. **选个模板**：先打开残差 CNN 或 Transformer，旋转、缩放，点击图层看内部结构。
2. **动手改一改**：拖入新图层、调整参数、连接节点，观察形状与结构如何变化。
3. **跑一次真实计算**：开启训练看指标，或打开「张量观测」运行单样本推理。
4. **带走你的模型**：导出 PyTorch 代码继续开发，或导出 PNG / SVG 用于讲解。

## 使用前知道这几点

- **动画不等于测量值**：粒子流与权重配色是示意；训练指标与显式运行的张量观测才来自真实 PyTorch 计算。
- **源码导入不是加载权重**：仅解析受支持的静态结构与常量，不执行上传代码，也不导入原始可训练权重。
- **训练权重暂存在内存**：仅保留最近一次成功训练的模型；重启后端会清除，导出的代码与项目 JSON 均不包含训练权重。
- **面向学习与小规模实验**：训练目前针对分类任务；大型模型、分布式训练和完整性能分析请使用专业工具。风险提示不代表训练效果保证。

## 深入了解

- [使用、开发与技术说明](docs/workbench-guide.md) — 训练、CUDA、观测、导出、测试与打包
- [PyTorch 导入支持范围](docs/pytorch-import-support.md) — 支持的语法、限制与诊断
- [5 个可复现导入示例](examples/import_models/README.md) — CNN、残差、MQA、GQA、Cross-Attention

**技术栈：** React · TypeScript · Three.js · React Flow · FastAPI · PyTorch

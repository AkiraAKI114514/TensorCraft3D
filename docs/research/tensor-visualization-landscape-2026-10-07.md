# Tensor 可视化竞品、项目定位与命名建议

调研日期：2026-10-07（Asia/Kuala_Lumpur）。项目：AkiraAKI114514/Tensor_lab，当前产品名 TensorLab 3D；本地代码基线 `e4ed5fb`。

## 建议

推荐将产品定位收敛为 **可编辑的 PyTorch 模型三维工作台**，主打“导入结构 → 编辑与检查 → 观察小模型运行 → 导出代码和插图”。优先补真实张量观测，再扩充模型类型。品牌首选 **TensorScene**，备选 **TensorStage**；现阶段名称只是提案。

我们已有完整的工作台雏形，但“3D”不是独有优势。TensorSpace、Zetane Viewer、LLM Visualization，以及较新的 modelviz 都覆盖部分相似场景。更有价值的组合是：结构编辑、张量形状校验、细粒度 Attention 接线、受支持代码的导入导出，以及本地真实训练。

## 方法与证据边界

- 使用 GitHub 公共 REST API 搜索仓库，读取入选项目的仓库元数据和 README；对照本项目 README 与实现。没有安装或实测竞品。
- 搜索包括 `tensorlab in:name`、`"tensor lab" in:name,description`、`tensor visualization in:name,description`、`tensor visualizer in:name,description`、`neural network 3D visualization in:description`，并直接核验重要项目。
- 星数是当天快照，只表示 GitHub 关注程度。`pushed_at` 只是最近推送，不证明核心功能持续维护，也不证明兼容当前依赖。
- 宽泛搜索包含无关结果，已经按产品任务筛选；列表不是穷尽清单。品牌搜索只覆盖 GitHub，未核验域名、软件包注册表或商标。
- 原始仓库元数据与候选名查询结果保存在同目录的 `tensor-visualization-sources-2026-10-07.json`。

## 与我们有关的项目

| 项目 | Stars | README 描述的能力 | 对我们的启示 |
| --- | ---: | --- | --- |
| [Netron](https://github.com/lutzroeder/netron) | 33,551 | 多种模型文件格式的结构查看，包括 ONNX、PyTorch、TensorFlow、Core ML 等 | 格式覆盖是成熟竞争领域；我们优先做可编辑工作流和受支持结构的准确映射 |
| [TensorSpace](https://github.com/tensorspace-team/tensorspace) | 5,199 | Three.js / TensorFlow.js 交互式 3D 网络；预训练模型和中间推理可视化 | 最直接的 3D 参照；三维层块和旋转交互本身不足以区分我们 |
| [Zetane Viewer](https://github.com/zetane/viewer) | 1,846 | 3D 模型与内部张量；特征图、权重、偏置、输出、均值和直方图；ONNX/ZTN | 用户会期待“点击层能看到真实值”。仓库提供免费查看器说明与示例，不能据此认定完整引擎开源 |
| [LLM Visualization](https://github.com/bbycroft/llm-viz) | 5,552 | GPT 风格网络的三维交互解释；小型排序模型具有工作权重，大尺寸展示不一定加载权重 | 学习体验要把结构、数值和计算步骤连起来；不能笼统把它视为任意大模型实测工具 |
| [modelviz](https://github.com/shreyanshjain05/modelviz) | 25 | PyTorch/Keras 模型对象生成 2D/3D 图；残差、模块组合、Notebook、图片与 HTML 导出 | 新项目也在做 Three.js + PyTorch + 图像导出；我们需要突出编辑、接线和执行能力 |
| [Torchview](https://github.com/mert-kurttutan/torchview) | 1,081 | PyTorch 模块、函数、张量形状的图；层级展开/折叠及 meta tensor 支持 | 专业用户先需要准确的结构和形状；大图折叠值得优先借鉴 |
| [PyTorchViz / torchviz](https://github.com/szagoruyko/pytorchviz) | 3,506 | PyTorch 执行与 autograd 图，可显示 backward 保存的信息 | 结构图与反向计算图要区分；目前我们的反向粒子动画只是示意 |
| [TensorBoard](https://github.com/tensorflow/tensorboard) | 7,231 | 训练运行、图、标量、直方图等观测 | 专业实验监控已有成熟方案；我们的指标应直接对应可点击的模型层 |
| [TensorWatch](https://github.com/microsoft/tensorwatch) | 3,474 | Notebook 内实时训练可视化、调试与流式查询 | 未来可考虑用户 Python 进程的观测接口，先做好自己的本地模型观测 |
| [TorchShow](https://github.com/xwying/torchshow) | 713 | 一行代码查看图片、掩码、视频、光流等张量数据 | 张量查看器需要按语义选视图，不能把所有高维数组都画成方块 |
| [BertViz](https://github.com/jessevig/bertviz) | 8,195 | Hugging Face 模型的真实注意力；head/model/neuron 视图和 Q/K 计算解释 | 我们的 QKV 接线较有特点，但 token-to-token 数值观测存在缺口 |
| [Transformer Explainer](https://github.com/poloclub/transformer-explainer) | 8,828 | 浏览器内运行 GPT-2，让用户输入文本并观察下一个 token 的计算 | “输入 → 真实计算 → 解释”比循环播放的数据流动画更有说服力 |
| [CNN Explainer](https://github.com/poloclub/cnn-explainer) | 9,058 | 面向非专家的交互式 CNN 学习 | 教学模式应有小样本、分步解释和具体问题，而不仅是层库 |
| [NN-SVG](https://github.com/alexlenail/NN-SVG) | 5,689 | 参数化生成神经网络架构插图，导出 SVG | 高质量插图是独立需求；我们已有紧凑布局和 PNG/SVG，可作为明确卖点 |
| [Visualkeras](https://github.com/paulgavrikov/visualkeras) | 661 | Keras/TensorFlow 模型的 layered、graph、LeNet 等架构图 | 一键从模型得到清楚的图，比更多视觉特效更有价值 |
| [Deep Playground](https://github.com/tensorflow/playground) | 13,046 | 浏览器交互式神经网络可视化 | 入门流程应让用户很快完成一次可理解的实验 |

TensorSpace 和 Zetane Viewer 的最近推送分别在 2022 年，但这不代表项目不可用，也不能据此断言市场空白。相反，modelviz 说明同类工具仍在出现。

## Tensor Lab 名称已经被哪些项目使用

| 项目 | Stars | 用途 | 冲突性质 |
| --- | ---: | --- | --- |
| [Texas Instruments / edgeai-tensorlab](https://github.com/TexasInstruments/edgeai-tensorlab) | 101 | Edge AI 训练、量化、编译和 benchmark 工具集 | 同一机器学习领域，品牌主体辨识度高 |
| [Texas Instruments / tinyml-tensorlab](https://github.com/TexasInstruments/tinyml-tensorlab) | 52 | 面向 TI MCU 的 TinyML 工具链 | 同名系列，可能让用户误以为存在关联 |
| [haotong-Duan / tensor-lab](https://github.com/haotong-Duan/tensor-lab) | 4 | 张量分解交互学习；README 列出双语与 3D viewer | 教育、张量、3D 都相近，搜索语义冲突尤其明显 |
| [fangde / TensorLab](https://github.com/fangde/TensorLab) | 5 | 基于 TensorLayer 的云端机器学习平台 | 较早使用该名；即使用途不同，也降低品牌唯一性 |

我们的远端名 `Tensor_lab`、界面 `TensorLab 3D`、npm 名 `tensorlab-3d` 尚不统一；GitHub 仓库 description 当前为空。搜索可发现性的问题不只来自重名，也来自缺少一句明确描述。

## 我们实际站在哪里

### 已经成立的优势

1. **结构可编辑并影响执行。** 2D 拓扑与 3D 场景共享 Graph；调整层参数和连接会改变形状分析、导出代码和本地模型计算。
2. **Attention 拆分有表达力。** MHA/MQA/GQA/Cross-Attention、Q/K/V 投影端口、残差和多分支能被单独查看与配置。应把这些展示成清楚的案例。
3. **受支持结构可以形成工作流。** PyTorch 静态 AST 导入、图编辑、代码导出、小模型训练和插图导出已经连接起来。
4. **本地运行与中文界面。** Windows 启动、CPU/CUDA 训练与诊断适合中文学习者和需要本地实验的人。
5. **结构图可交付。** PNG/SVG 共用布局，支持紧凑排布、残差外绕线和注意力分组展示。

这些是从代码确认的产品能力组合，不构成“行业唯一”的结论。我们当前只有 2 stars，也没有本次调研可用的用户留存或使用量，仍应按早期 MVP 判断。

### 最重要的缺口

- **真实张量观测不足。** 当前真实训练发回 loss、accuracy、梯度范数和 ReLU 零激活比例；没有把层输出、Q/K/V、attention probabilities 或 token 梯度送进三维查看器。粒子和权重配色是示意。
- **导入的是受支持结构，不是原模型的完整状态。** 不加载原始可训练权重；不覆盖任意控制流、自定义算子、权重共享和复杂 mask。宣传“任意 PyTorch 可视化”会造成预期落差。
- **不是成熟 LLM 调试器。** 当前 Attention 为非 causal 的 BSE 计算；没有文本 tokenizer 和 causal/padding mask，不能把 Transformer 模板直接宣传为完整 GPT 工具。
- **上手和传播仍有空间。** 阅读安装文档、安装依赖与配置训练后才进入核心价值；应让用户先体验无需训练的结构演示，再按需启用训练。

推荐主用户：学习模型结构的 PyTorch 用户、教师、需要模型插图与小规模结构实验的研究者。大模型生产调试、全格式模型查看与分布式实验管理会显著扩大成本，应放在后续。

## 优化顺序与验收目标

| 顺序 | 工作 | 为什么先做 | 可验收的结果 |
| --- | --- | --- | --- |
| P0 | 项目说明、短演示与示例流程 | 成本较低，直接解释我们与查看器的区别 | 用户用 CNN/残差/MQA 三个例子完成“导入 → 编辑 → 导出”，首页明确支持边界 |
| P1 | 单样本真实推理和 Tensor Inspector | 将结构可视化扩展到真正的张量观测 | 点选层查看实测 shape/dtype、min/max/mean/std、直方图与一张切片；标注运行来源 |
| P1 | 小型 Attention 诊断视图 | 最能深化已有 QKV/MQA/GQA 特色 | 对指定分支和 head 查看 Q/K/V、QKᵀ、softmax 热图和输出，显示实际 tensor 维度 |
| P2 | 大图阅读、模块折叠与图像标注 | 支持规模增长并保留现有插图优势 | 同一模型可切换结构概览/模块内部，选中对象与 2D 图同步；导出保持可读 |
| P2 | 受支持导入的覆盖说明和实例库 | 减少第一次导入失败和错误预期 | 常见 CNN、Residual、Transformer 例子附支持说明、输入 shape 和可复现导入结果 |
| P3 | 外部 Python 观测接口，再评估 ONNX | 接触用户实际模型，而非只在内置训练器内工作 | 先以离线快照传入结构和采样张量，再决定是否需要实时连接和新格式 |

实施 P1 时，先采样一个 batch 中的一个样本，并限制层、通道和采样频率；统计或降采样后传输，不逐帧发送全部激活。现有 `scaled_dot_product_attention` 不直接返回完整注意力矩阵，可在显式诊断模式下用捕获的 Q/K 重算，并对序列长度设限；确定性 eval 对照应验证诊断结果与实际输出一致。图中务必区分投影参数矩阵 Wq/Wk/Wv 与 attention probabilities A，它们不是同一个对象。

建议结构模式先保留当前动画，观测模式由真实采样驱动，用户始终能知道正在看哪种信息。CNN 用通道特征图，序列用 token/position × feature 热图，Attention 用 query × key 热图；三维场景承担导航，数值面板承担精确读取。

## 候选名字与初筛

| 名字 | GitHub 初筛 | 与产品的关系 | 建议 |
| --- | --- | --- | --- |
| **TensorScene** | `TensorScene in:name` 返回 0；`tensor-scene in:name` 返回 12 个宽泛匹配，返回页主要为 TensorFlow 场景识别等项目 | 把模型变成可导航、可编辑的场景；贴合 3D 工作台 | **首选**；正式采用前补齐名称核验 |
| **TensorStage** | `TensorStage in:name` 返回 0 | 层级与计算阶段，也有实验舞台的含义 | 备选；产品含义需要副标题帮助解释 |
| TensorLab 3D | 已有多个 TensorLab / tensor-lab 项目 | 直白，但 Lab 很泛，增加 3D 不解决主名称重名 | 可作过渡名称 |
| TensorLoom | `TensorLoom in:name` 返回 2，含 [techxsarwar/tensorloom](https://github.com/techxsarwar/tensorloom) | 接线和数据流的比喻不错 | 不作为首选 |
| TensorScope | `TensorScope in:name` 返回 8，包含模型性能分析工具 | 更偏观测与调试 | 重名且容易弱化编辑定位 |
| TensorPrism | 连写查询返回 0，但分隔变体找到 [Arctenox/Tensor_Prism](https://github.com/Arctenox/Tensor_Prism)，是 ComfyUI 节点项目 | 视觉、空间含义不错 | 邻近 AI 领域已有使用，不作为首选 |

查询没有命中不能证明名称可注册。这里推荐 TensorScene 是基于语义贴合与当前 GitHub 可发现性；未声称其域名、npm/PyPI 名或商标可用。

### 推荐的品牌表达

```text
TensorScene
Interactive 3D PyTorch Model Workbench
PyTorch 模型三维工作台

Import model structure. Edit connections. Inspect runs. Export code and figures.
导入模型结构，编辑连接，观察运行，导出代码与插图。
```

可直接使用的 GitHub description 提案：

> A local 3D PyTorch model workbench for editing computation graphs, exploring MHA/MQA/GQA, running small training experiments, and exporting Python, PNG and SVG.

仓库名提案：`tensorscene`。README 首屏用一段短动图演示“导入一段 CNN → 改一个连接 → 检查 shape → 导出 SVG”，另放一个 MQA/GQA 对比案例；支持限制移到有链接的具体说明中，首屏仍明确动画为示意、训练指标为实测。

### 采用新名字时的迁移范围

先统一可见品牌：README、HTML title、顶部栏/状态栏/对话框、启动提示、FastAPI title、打包文件名和 package 元数据。GitHub 仓库重命名及 description/topics 属于发布层面的后续动作。

兼容性命名单独处理：`tensorlab-project` 本地存储键、拖拽 MIME、`TENSORLAB_TEST_URL`、`TensorLabAttention`/`TensorLabTransformer` 和 `_TENSORLAB_INPUT_SHAPES` 等导入导出标记不宜直接全局替换。先保留读取与解析兼容，再迁移；可见品牌更新不需要让旧项目或已导出的 Python 失效。

当前建议先确定产品定位与品牌方向；真实张量观测完成后，再考虑将宣传扩展为 Tensor Inspector / Attention Inspector。即使暂不改名，补 GitHub description、示例动图和支持矩阵也应立即进入下一次产品迭代。

## 本项目依据

- `README.md`：工作台、Attention、实测指标、静态导入与功能限制。
- `src/App.tsx`、`src/types.ts`：可见品牌、本地存储、Graph 与编辑流程。
- `backend/training.py`：真实训练及 WebSocket 指标字段。
- `backend/attention.py`：实际 QKV 投影与 SDPA 计算，目前不回传可视化数值。
- `src/sceneImageExport.ts`、`src/imageLayout.ts`：现有紧凑图像导出基础。
- 所有竞品链接指向此次核验的 GitHub 项目；详细 README URL 与查询快照见配套 JSON。

## 2026-10-08 品牌决策

产品名更新为 **TensorCraft3D**，副标题为 **Build and Explore Neural Networks in 3D**；中文界面使用“神经网络三维构建与探索”。GitHub 初筛发现过多个 TensorCraft 同名项目，见上方既有查询记录；采用名称是产品决策，不代表已完成域名、包注册表或商标核验。应用的可见文案、元数据、启动信息和发布包已按该品牌更新，`TensorLab*` Python 运行时类名以及 `tensorlab-project` 等持久化标识保留兼容。

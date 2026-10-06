# TensorLab 3D

本地运行的深度学习模型搭建与三维可视化工作台。根据提供的 Gemini 对话实现可运行 MVP：React / TypeScript / Three.js / React Flow 前端，FastAPI / PyTorch 训练服务。

## 在 Windows 上运行

需要 Node.js 20.19+（推荐 22/24）和 Python 3.10+（推荐 3.12/3.13）。

```powershell
# 首次安装，包含真实训练
powershell -ExecutionPolicy Bypass -File .\install.ps1 -Training

# 启动，自动打开浏览器并选择可用的本地端口
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

当前目录已经安装依赖并提供编译后的 `dist`，可直接双击 `start.cmd`。默认地址为 http://127.0.0.1:8765。默认只监听本机。关闭终端结束服务。

不需要真实训练时，只需 `npm install`、`npm run build`、`python run.py`。标准库服务器也可以提供三维视图、模型编辑、预警演示和导出。

## 开发、测试与打包

```powershell
# 终端 1，真实训练后端
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8765

# 终端 2，开发前端，代理 /api 到上述服务
npm run dev

npm test
.\.venv\Scripts\python.exe -m unittest backend.test_training
# 启动 8765 服务后，使用已安装的 Microsoft Edge 跑浏览器流程测试
npm run test:e2e
npm run build
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

`build.ps1` 生成 `release/TensorLab-3D.zip`，包含源代码、编译结果、后端和启动脚本，不包含 `node_modules` 或虚拟环境。接收者需要 Python 才能运行，真实训练需要安装后端依赖；这不是无需运行时的 Windows EXE。

## 工作台

- 内置 Compact CNN、Feature MLP、Residual CNN、Transformer Encoder、MQA、GQA、Cross-Attention 与空白模板。层库支持点击与拖入场景。新层连接到当前选中层，在最终输出前插入。
- 三维视图根据张量空间维度和通道生成立体层；几何尺寸采用对数映射，避免极端尺寸遮挡。
- 所有三维层点击后以该层为旋转和缩放中心；注意力头及其 QKV、权重矩阵、LN、FFN、汇流和相加模块也可以单独点击居中。Add 聚焦保留完整残差旁路，同时以 Add 节点为中心。
- 通用 Add 残差与 Transformer Encoder 共用粉色外部旁路、方向箭头和 GPU 粒子样式。残差从源层输出处分流，绕过中间模块，从上方进入粉色“+”节点，与主支路相加后继续传递。相机聚焦和 PNG / SVG 导出包含完整旁路；拓扑图同步显示粉色外部曲线。根据图中的可达关系识别跳跃分支，独立并行分支保持普通连线。
- 拓扑图支持拖动节点、从输出端口拉线至输入端口，Add / Concat 可接多条输入。点击连线后 Delete 删除，属性面板也可以移除连接。
- 拓扑工具条可添加独立输入/输出对象、删除选中对象；属性面板“连接对象”可以选择上游/下游对象，分别添加或取消输入/输出连线。Q/K/V 对象有独立端口，可以分别配置输入源和输出目标。错误连接保留在图中并提示原因；循环、形状不匹配、非法多输入等会阻止训练和 Python 导出。计算图支持 1–8 个模型 Input 和一个 Output。
- 层属性修改会重新推导张量形状，检查循环、四维/二维不匹配、残差形状、拼接维度、池化参数和资源限制。项目自动保存在本地浏览器，支持撤销/重做以及 JSON 导入导出。
- 导出 PyTorch `nn.Module`，含可执行的随机输入测试，输出为 logits。不会执行浏览器上传的 Python 代码。
- PNG 提供 1920 / 3840 / 7680 像素宽度、当前视角、可选透明背景；SVG 导出简化的三维几何投影与可编辑层标签。

### 注意力与 Transformer

`MultiHeadAttention` 与 `Transformer` 使用共享的 PyTorch `scaled_dot_product_attention` 运行时。`Transformer` 为 pre-norm 编码器层（LN → Attention → 残差 → LN → GELU FFN → 残差）。输入输出采用 `[B,S,E]`。`attention_type` 支持 `self`、`multi_query`（MQA）、`grouped_query`（GQA）、`cross` 和 `multi_branch`；`num_heads`（1–16）是 Q 数量，默认 1；`kv_heads` 是 K/V 组数；`branches`（1–8）是独立计算后取平均的分支数。Embedding 必须等于输入最后一维并能被 Q 数量整除，Q 数量也必须能被 K/V 组数整除。MQA 使用一个 K/V 组，标准自注意力使用每 Q 一组 K/V。Cross-Attention 用 Query 和 Context 两个输入，允许两者序列长度不同。各参数改变真实计算、参数统计和导出代码。

每个共享 K/V 组绘制一个多边形，所有使用该 K/V 的 Q 顶点放在同一面内，K 与 V 各画一次，无额外悬浮共享层。边数自动等于本组 Q 数量 + 2：一个 Q 为三角形；MQA 的 4 个 Q 共用 K/V 为六边形；GQA 的 8 个 Q、2 个 K/V 组为两个六边形。`qkv_count` 已从参数面板移除，旧项目中的该字段会被忽略。各组沿数据流 X 轴同向叠放，注意力头并行计算；多分支按分支分开显示。每个 Q 有自己的 Softmax 和 V 加权路径，共享 K/V 直接连到这些计算节点。Transformer 继续显示 LN、FFN 和两条粉色残差路径。选中或悬停时停止转动；配置独立顶点连线后固定多边形朝向，保证外部粒子与顶点贴合。PNG / SVG 与场景共用同一几何和端口布局。

点击三维 Q/K/V 立方体会居中并打开该对象的连接配置，也可在属性面板的“Q/K/V 对象”选择器中选择。每个顶点默认继承模型层输入（Cross-Attention 的 Q 继承 Query，K/V 继承 Context），指定上游对象后覆盖对应投影来源；删除覆盖连线后恢复默认来源。顶点输出为真实投影张量 `[B,S,E/num_heads]`，可连接其他模型层或其他 Q/K/V 的输入端口。所有顶点输入要求 `[B,S,E]`，Q 的序列长度须与本层 Query 一致，同分支所有 K/V 的序列长度须一致。Transformer 的 Q 和自注意力覆盖输入也经过 LN。每个投影端口最多一个指定输入，输出可扇出到多个对象。完整模型仍需要基础输入，Transformer 的残差来自基础 Query。

项目连线通过 `sourcePort` / `targetPort` 存储顶点端口，例如 `{"source":"memory","target":"attention","targetPort":"b0:k0"}` 指定第一分支 K1 的输入；`{"source":"attention","sourcePort":"b0:q0","target":"flatten"}` 将第一分支 Q1 的投影传到 Flatten。同一对模型层可通过不同端口多次连接。训练、形状校验和独立 Python 导出使用相同连接语义。

三维权重矩阵的颜色与粒子目前仍是**模拟示意**，未采集真实 attention weights、Q/K/V 激活或 token 级梯度；属性面板标记“权重示意”。真实训练监控中的损失与逐层梯度来自实际 PyTorch 训练。示例输入为 token 特征向量，暂不包含文本 tokenizer、位置编码或 causal/padding mask。

## GPU 动画与训练

Three.js 使用 WebGL、GPU 实例化和自定义顶点着色器。每个粒子的贝塞尔路径在 GPU 上计算，CPU 每帧仅更新时间/方向 uniform，不逐粒子改写位置。普通连线前向青色、反向珊瑚色；残差旁路前后向均保留粉色，反向时箭头和粒子方向反转。支持暂停、速度调节、轨道旋转、缩放、平移和展开间距。动画表示模拟传播，不表示实际张量数值。

浏览器需要开启硬件加速。软件渲染器会显示“软件渲染”；应用不能强制浏览器或操作系统提供物理 GPU。WebGL GPU 加速与 PyTorch CUDA 训练是两套独立设备机制。

真实训练使用 Adam + CrossEntropyLoss，可选择 CPU / CUDA / 自动；合成分类数据或本地 CSV。CSV 支持可选首行表头，最后一列为从 0 开始的整数类别，前面为数值特征；特征数必须等于输入张量除 batch 外的元素数，图像按 C/H/W 顺序展平。CSV 数值标准化后按模型输入形状重构。验证集固定随机切分。训练轮次、学习率、batch size、验证比例和早停窗口可配置。

训练通过 WebSocket 推送实测 train/val loss、准确率、全局/逐层梯度范数、ReLU 零激活比例。梯度裁剪阈值 100；早停按验证损失连续未改善窗口触发。每次只允许一个本地训练任务，断开页面或点击停止会在下一批次停止。输入、数据和激活有资源上限，复杂模型需在专业训练环境中运行。

默认 PyPI 的 PyTorch 可能是 CPU 版本。需要 CUDA 时，从 https://pytorch.org/get-started/locally/ 获取与你的 GPU 驱动匹配的安装命令，在本项目 `.venv` 中执行，重启服务后界面会显示 CUDA 设备。

## 预警说明

架构期：形状错误、循环、未连接层、空间下采样截断、隐藏线性层 >90% 信息骤缩、连续 8 个无残差参数层、单层占 >70% 参数。训练期：损失或梯度非有限、梯度爆炸/消失、逐层异常、ReLU 零激活 >90%、训练损失持续不改善、损失发散、训练/验证差距持续扩大。

这些是启发式风险提示，不能仅凭静态结构判定是否过拟合或保证收敛。过拟合需要验证集指标支持。演示场景明确标记为“演示数据”，与真实训练和导入指标区分。

可导入 JSON 指标数组，或 CSV 指标：

```csv
epoch,trainLoss,valLoss,accuracy,gradNorm
1,1.8,1.9,0.30,0.8
2,1.5,1.6,0.40,0.7
```

## 目录

`src/analysis.ts` 形状/预警规则；`src/Scene.tsx` GPU 三维场景与导出；`src/AttentionModule.tsx` 注意力多边形与内部结构；`src/Flow.tsx` GPU 粒子流；`src/Topology.tsx` 节点编辑；`src/Connections.tsx` 输入/输出连接对象；`src/export.ts` PyTorch 生成；`backend/graph.py` 分配前校验；`backend/training.py` 真实训练；`backend/app.py` HTTP/WebSocket；`run.py` 启动入口。

本版本支持基础 CNN/MLP/残差和 Transformer 编码器计算图。暂不支持 RNN、动态图 Python 双向同步、ONNX 导入、分布式训练或完整性能 profiler。

## 导入 PyTorch 代码自动建模

顶部工具栏的“导入 PyTorch 代码”支持粘贴 PyTorch 源码或选择本地 `.py` 文件。程序先选择模型对象和输入形状，再提交到本地服务进行静态 AST 解析；预览会显示节点、连接、参数量和诊断信息。确认后，Input、层、Add/Concat、Attention 的 Q/K/V 端口和 Output 会写入当前项目，可继续拖动、修改参数、训练、查看三维数据流和导出代码。

当前图编辑器和真实训练链路覆盖大部分常用 `torch.nn` 层：Conv/ConvTranspose 1D、2D、3D，Linear、Bilinear，BatchNorm/InstanceNorm/GroupNorm/LayerNorm，MaxPool/AvgPool/AdaptivePool，Flatten，Dropout 系列、Embedding、Upsample，ReLU/GELU/Sigmoid/Tanh/SiLU/LeakyReLU/ELU/SELU/Softplus/PReLU/Hardsigmoid/Hardswish/Mish/Softsign/Softmax/LogSoftmax/Identity，以及 MultiheadAttention、Transformer 和 TensorLab 的 MQA/GQA/Cross-Attention。每个层的参数会参与形状推导、参数统计、PyTorch 训练和 Python 导出。

示例请求：

```powershell
$body = @{
  source = "from torch import nn`nmodel = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2))"
  model_name = "MLP"
  input_shapes = @{ x = @(1, 4) }
} | ConvertTo-Json -Depth 8

Invoke-RestMethod -Uri http://127.0.0.1:8765/api/import/pytorch -Method Post -ContentType 'application/json' -Body $body
```

解析器只读取 Python AST，不执行上传代码，也不会导入源码中的模块或加载权重、优化器和训练脚本。支持静态 `nn.Module`、`nn.Sequential`、嵌套模块、`ModuleList`、`ModuleDict`、残差加法、`torch.cat`、`flatten/view/reshape`，以及上面列出的常用层、`MultiheadAttention`、`TransformerEncoderLayer`/`TransformerEncoder` 和 TensorLab Attention/Transformer。支持 Cross-Attention、MQA/GQA 的 Query/Context 与 Q/K/V 端口连接。

导入的是模型结构和构造参数，不包含原始权重；导入后可在训练面板选择数据和 CPU/CUDA 设备。动态控制流、依赖运行时数据的循环、任意自定义算子、权重共享、复杂 mask 和无法静态推断的张量操作会报告错误或警告。源码限制为 512 KB、AST 节点最多 30,000 个、生成图最多 128 个节点和 512 条边；输入形状最多 8 个。缺少形状时会使用保守推断并给出 warning，可在窗口中修改后重新解析。

HTTP 接口为 `POST /api/import/pytorch`，请求字段为 `source`、可选 `model_name` 和 `input_shapes`。响应包含 `graph`、`models`、`model`、`inputs`、`diagnostics` 与 `analysis`，可用于集成到其他编辑器或自动化流程。

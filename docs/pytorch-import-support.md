# PyTorch source import support

This matrix describes **Python source import**, not everything available through graph JSON or the layer editor. A supported layer name does not mean every PyTorch constructor option or forward expression is preserved.

The importer reads the Python AST without executing uploaded imports, constructors, `forward`, or training scripts. It imports a bounded computation graph, supported constructor arguments, and evaluated float32 constants. **Original trainable weights, checkpoints, optimizers and training state are not imported.** Retained weights come from a separate successful local training run.

## Import contract

Upload/paste source in **导入 PyTorch 代码**, choose its actual model name, enter input shapes, and parse again after changing shapes. Candidates include classes directly inheriting `nn.Module` and top-level `nn.Sequential` variables. A Sequential variable named `model` must be selected as `model`, not a descriptive name such as `MLP`.

The local endpoint is `POST /api/import/pytorch`:

```json
{
  "source": "from torch import nn\nmodel = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))",
  "model_name": "model",
  "input_shapes": { "x": [2, 4] }
}
```

A successful response contains `graph`, `models`, `model`, `inputs`, `diagnostics`, and `analysis`. Parser failures return `graph: null`: inspect diagnostics, rather than treating HTTP success as an accepted graph. Request-schema errors can instead return HTTP 422. Success includes `STRUCTURE_ONLY`; missing shapes or unused modules can produce `INFERRED_SHAPE` or `UNUSED_MODULES`. Diagnostics usually identify a source line/column, while global limits and some shape failures have no source location.

## Supported source patterns

These patterns are supported **within the restrictions below**, not as arbitrary Python equivalents.

| Category | Accepted patterns | Restrictions |
| --- | --- | --- |
| Static modules and containers | `nn.Module`, `nn.Sequential`, nested custom modules, literal `ModuleList`/`ModuleDict`, static indexing, supported static module factories | No runtime-sized containers or runtime-dependent branches/loops. A factory returning modules is not an arbitrary forward helper. |
| Residual addition | `x + residual` | Both runtime tensors must have identical shapes; general broadcasting and inplace `+=` are unsupported. |
| Concatenation | `torch.cat([a, b], dim=1)`, `torch.concat(...)` | A static list/tuple and a positive non-batch axis; other dimensions must match. Specify `dim`: default batch concatenation is unsupported. |
| Convolution | Conv1d/2d/3d, ConvTranspose1d/2d/3d | Bias enabled, `groups=1`, `dilation=1`, zero padding mode, uniform spatial tuples. Ordinary Conv accepts the PyTorch default `padding=0`. |
| Dense layers | Linear, Bilinear | Bias enabled. Linear acts on the last dimension. Bilinear takes two `[B,F]` inputs with the declared second-input width. |
| BatchNorm | BatchNorm1d/2d/3d | Default epsilon/momentum, affine and running statistics enabled. BN1d accepts `[B,C]` and `[B,C,L]`; BN2d/3d require ranks 4/5. |
| Other normalization | LayerNorm, InstanceNorm1d/2d/3d, GroupNorm | LayerNorm supports static matching suffix dimensions, `eps` and `elementwise_affine`, with bias enabled. InstanceNorm source import uses default epsilon/momentum, affine disabled and no running statistics. GroupNorm uses default epsilon and affine enabled. Channel-first ranks remain required. |
| MaxPool | MaxPool1d/2d/3d | Uniform spatial parameters, dilation 1, `return_indices=False`, `ceil_mode=False`. |
| Avg/Adaptive pools | AvgPool1d/2d/3d, AdaptiveAvgPool1d/2d/3d, AdaptiveMaxPool1d/2d/3d | AvgPool options must remain at defaults (`ceil_mode=False`, `count_include_pad=True`, `divisor_override=None`); AdaptiveMaxPool requires `return_indices=False`. Non-default options are rejected before graph creation. |
| Activations | ReLU, GELU, Sigmoid, Tanh, SiLU, LeakyReLU, ELU, SELU, Softplus, PReLU, Hardsigmoid, Hardswish, Mish, Softsign, Softmax, LogSoftmax, Identity | No inplace variants; GELU uses `approximate="none"`. Specify an explicit dimension for Softmax/LogSoftmax. Supported slope/alpha/beta/threshold/PReLU options are preserved. |
| Dropout | Dropout, Dropout1d/2d/3d, AlphaDropout | Static probability and no inplace mutation. Functional `F.dropout` must preserve `training=self.training`. |
| Embedding | `nn.Embedding(num_embeddings, embedding_dim)` | Direct Input with integer IDs. No padding index, max norm, sparse/frequency-scaled weights, supplied weights or freezing. |
| Upsample | Static `scale_factor`, `mode` | No explicit size, align-corners, recompute-scale-factor or antialias. Mode must still match rank during execution. |
| Flatten and reshape spellings | `nn.Flatten(1)`, `torch.flatten(x, 1)`, `x.flatten(1)`, `x.view(x.size(0), -1)`, `x.reshape(x.shape[0], -1)` | Preserve batch and flatten remaining dimensions only; no general reshape, transpose, permute or custom head-splitting expressions. |
| Singleton axes and indexing | Static `unsqueeze`/`squeeze`, integer selection, positive-step slices, negative bounds/indices, one ellipsis, inserted singleton axes, `select` | Known upstream shapes; no batch-axis modification, advanced/boolean indexing or arbitrary dynamic bounds. |
| Static float32 buffers | Supported `zeros`/`ones`/`arange`, arithmetic, selected math functions, exp/sin/cos/sqrt/abs, bounded indexing/assignments, `register_buffer` | Actual values become ConstantAdd parameters. No uninitialized `empty`, non-float32/device-specific constructors, non-finite constants or arbitrary constant execution. |
| Positional addition | `x + self.pe[:, :x.size(1)]` and supported equivalent shape spelling | One sequence prefix from the added tensor; explicit input shape and sequence length within buffer capacity. |

Functional operations use a whitelist. A module equivalent being supported does not imply support for functional convolution, linear, normalization, interpolation, PReLU, SDPA or arbitrary tensor methods.

## Attention and Transformer

| Source | Accepted configuration | Not supported |
| --- | --- | --- |
| `nn.MultiheadAttention` | `batch_first=True`, bias enabled, matching embedding widths; output tensor unpacked as `x, _ = ...` or selected with `[0]`; different Query/Context lengths | Masks, causal mode, `add_bias_kv`, `add_zero_attn`, distinct kdim/vdim, or importing attention-weight outputs. Prefer `need_weights=False`. |
| `nn.TransformerEncoderLayer` / `nn.TransformerEncoder` | Batch-first encoder, bias enabled, default LayerNorm epsilon, activation string `relu`/`gelu`, pre/post-norm, static encoder layer count | Masks, causal flags, custom activation callables, final encoder norm, decoder/full Transformer stacks. |
| `TensorLabAttention` / `TensorLabTransformer` | Recognized repository helper descriptors for self, cross, MQA, GQA, multi-branch attention and projection ports | Arbitrary hand-written MQA/GQA or SDPA/projection code. A differently implemented class using the same helper name is not validated as equivalent. |
| TensorCraft3D Python export | Generated VisualModel, embedded input shapes, recognized input guard and Q/K/V port wiring | Arbitrarily changed control flow or unsupported new operations. |

Embedding must match the input's last dimension and be divisible by the Q-head count; Q heads must divide evenly by KV heads. MQA uses one KV group. Cross-attention needs Query and Context inputs with matching batch/embedding dimensions. All K/V sources within a branch must agree in length, including overrides.

The MQA/GQA examples import `backend.attention` and run **from the repository root**. They are not standalone exports. The workbench's Python export embeds the helper runtime for standalone use. Neither source import route loads original trainable weights.

## Known semantic limitations

- **Unsupported pool options are rejected, not discarded.** AvgPool requires `ceil_mode=False`, `count_include_pad=True`, and `divisor_override=None`; AdaptiveMaxPool requires `return_indices=False`. The importer reports the source location and returns no graph for non-default values.
- **Softmax/LogSoftmax need explicit `dim`.** An unspecified dimension maps to `-1`, not every PyTorch implicit-axis behavior.
- Repeated parameter-bearing layer calls are rejected rather than treated as supported weight sharing.
- Missing input shapes may be conservatively inferred with a warning. Indexing/positional buffers need sufficiently explicit shapes. Import success does not guarantee every runtime value or unrepresented PyTorch option is valid.
- Float32 transcendental constants are compared with numerical tolerances, not a cross-platform bitwise guarantee.

More generally unsupported: runtime-dependent control flow, inplace runtime operations, arbitrary tensor arithmetic/reductions, custom operators, checkpoint/ONNX import, RNN/LSTM/GRU and live bidirectional Python synchronization. Uploading a training script does not run it.

## Bounds

| Area | Limit |
| --- | --- |
| Source / AST | 512,000 UTF-8 bytes; 30,000 AST nodes |
| Graph | 128 nodes; 512 edges; 1–8 Inputs and exactly one tensor Output |
| Input shapes | Rank 2–5; positive dimensions at most 65,536; matching input batches |
| Static tensors | At most rank 5 and 65,536 elements per tensor; graph constant buffers total at most 65,536 elements |
| Static evaluation | 1,000,000 charged work units |
| Graph resources | 50M total parameters; 16M elements in a layer tensor or attention score estimate |
| Attention | Up to 16 Q/KV heads, 8 branches, embedding width 4,096 |
| Training | Classification logits `[B,2..256]`, trainable parameters and batch activation estimate below 256 MB |
| Single-sample inference | Separate stricter limits: 5M parameters / 20 MiB parameter cap, 4M elements per input, 64 MiB activation/inspection budget; see the main README |

Import, execution, and training are different checks. Non-classification outputs can be imported and inspected/exported but are not accepted by the classification trainer. Training never starts automatically after import.

## Reproducible examples

[Six examples and their manifest](../examples/import_models/README.md) cover CNN, residual CNN, MQA, GQA, cross-attention and repeated-block folding. All are small two-logit classifiers with explicit model names, input shapes, expected operations and parameter counts; test-batch output is `[2,2]`, while the interactive inspector uses batch 1.

From the repository root with the project environment active:

```text
python -m examples.import_models.cnn
python -m examples.import_models.residual
python -m examples.import_models.mqa
python -m examples.import_models.gqa
python -m examples.import_models.cross_attention
python -m examples.import_models.grouped_stack
```

Each seeds PyTorch, constructs a CPU model, performs one eval forward pass and prints shape/parameter count. They do not train or access the network. Cross-attention uses `query: [2,4,16]` and `context: [2,6,16]`; repeated-block folding uses `x: [2,8,4]` and imports to a few grouped nodes instead of the flat operator cascade.

Focused checks from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest backend.test_import_examples
npm exec --no -- vitest run src/importExamples.test.ts src/pytorchImport.test.ts
npm run test:e2e -- e2e/importExamples.spec.ts
```

The backend contract statically parses all six sources without executing them, then compares **trusted checked-in classes** and imported graphs with compatible weights, matching parameter counts, forward values and input gradients when PyTorch is installed. The source tests additionally execute standalone exports and reimports with common weights. This trusted fixture execution is not the browser upload path.

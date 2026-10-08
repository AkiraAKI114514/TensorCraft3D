# PyTorch import examples

These five tiny models are reproducible fixtures for the bounded static importer. Each model is a two-logit classifier: the expected output shape is explicitly `[2, 2]` for the manifest test batch. Run commands from the repository root so the `backend` package and TensorLab attention helpers resolve.

## Model matrix

| File / model class | Input shape(s) | Expected output | Import coverage |
| --- | --- | --- | --- |
| `cnn.py` / `CNNClassifier` | `x: [2, 3, 8, 8]` | `[2, 2]` | `Sequential`, `Conv2d`, `ReLU`, `AdaptiveAvgPool2d`, `Flatten`, `Linear` |
| `residual.py` / `ResidualClassifier` | `x: [2, 3, 8, 8]` | `[2, 2]` | nested modules, same-shape residual `Add`, `BatchNorm2d` |
| `mqa.py` / `MQAClassifier` | `x: [2, 4, 16]` | `[2, 2]` | `TensorLabAttention` multi-query attention (`kv_heads=1`) |
| `gqa.py` / `GQAClassifier` | `x: [2, 4, 16]` | `[2, 2]` | `TensorLabTransformer` grouped-query attention (`kv_heads=2`) |
| `cross_attention.py` / `CrossAttentionClassifier` | `query: [2, 4, 16]`; `context: [2, 6, 16]` | `[2, 2]` | `nn.MultiheadAttention(batch_first=True)` with separate Query and Context |

The complete support matrix, parser flags, unsupported semantics, and resource limits are documented in [`docs/pytorch-import-support.md`](../../docs/pytorch-import-support.md). `manifest.json` is the machine-readable version of this table and is consumed by `backend/test_import_examples.py`.

## Run the reference models

From the repository root, each module seeds PyTorch, constructs the named model on CPU, switches to evaluation mode, performs one forward pass, and prints `Output shape: (2, 2)` plus the parameter count. No training or network access occurs.

```text
python -m examples.import_models.cnn            # CNNClassifier: [2,3,8,8] -> [2,2]
python -m examples.import_models.residual        # ResidualClassifier: [2,3,8,8] -> [2,2]
python -m examples.import_models.mqa             # MQAClassifier: [2,4,16] -> [2,2]
python -m examples.import_models.gqa             # GQAClassifier: [2,4,16] -> [2,2]
python -m examples.import_models.cross_attention # CrossAttentionClassifier: query [2,4,16], context [2,6,16] -> [2,2]
```

## Parse a model source

The import endpoint is structure-only. To reproduce a fixture import in Python, read the source and pass the manifest's model name and named input shapes to `import_pytorch`:

```python
import json
from pathlib import Path
from backend.pytorch_import import import_pytorch

root = Path("examples/import_models")
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
example = next(item for item in manifest["examples"] if item["filename"] == "mqa.py")
source = (root / example["filename"]).read_text(encoding="utf-8")
result = import_pytorch(
    source,
    model_name=example["model_name"],
    input_shapes=example["input_shapes"],
)
assert result["analysis"]["shapes"][result["analysis"]["output"]] == [2, 2]
```

The parser does not execute constructors or `forward`; `result["diagnostics"]` reports the structure-only boundary.

## Focused verification

Run the focused importer contract test from the repository root:

```text
.venv/Scripts/python -m unittest backend.test_import_examples -v
```

The test statically imports every manifest entry without executing uploaded source, then compares the checked-in trusted reference model and imported graph on compatible weights, outputs, parameter counts, and finite input/parameter gradients when PyTorch is available.

## Support boundary

The importer accepts bounded, static `nn.Module` definitions and recognized containers (`Sequential`, nested `ModuleList`/`ModuleDict`), common convolution, linear, normalization, pooling, activation, dropout, embedding, and upsampling layers, plus explicit residual adds and positive-step concatenations/slices. TensorLab attention and Transformer helper descriptors support self, cross, MQA, GQA, and multi-branch forms within the graph limits.

Imports are structure-only: source constructors and `forward` methods are parsed, never executed. Original trainable weights, dynamic runtime control flow, arbitrary reshapes/permutations, custom scaled-dot-product attention, and runtime parameter sharing are outside this contract. Keep convolution padding explicit and use the defaults documented by the importer for pooling and attention flags.

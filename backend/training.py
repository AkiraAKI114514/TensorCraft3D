"""Train whitelisted graph operations; graph input is never evaluated as code."""
import csv
import io
import math
import uuid
from .graph import analyze_graph


def build_model(graph):
    import torch
    from torch import nn
    info = analyze_graph(graph)
    from .attention import TensorLabAttention, TensorLabTransformer
    from .tensor_ops import TensorLabConstantAdd
    class GraphModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleDict()
            self.dead_relu = {}
            for node in graph["nodes"]:
                key, op, p = node["id"], node["op"], node["params"]
                repeat = int(node.get("repeat", 1) or 1)
                edge = info["baseEdges"][key][0] if info["baseEdges"][key] else None
                in_shape = (info["portShapes"][edge["source"]][edge["sourcePort"]] if edge.get("sourcePort") else info["shapes"][edge["source"]]) if edge else []
                def build():
                    if op in ("Conv1d", "Conv2d", "Conv3d"):
                        cls = getattr(nn, op); return cls(in_shape[1], int(p.get("out_channels", 16)), int(p.get("kernel_size", 3)), int(p.get("stride", 1)), int(p.get("padding", 1)), groups=int(p.get("groups", 1)))
                    elif op in ("ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"):
                        cls = getattr(nn, op); return cls(in_shape[1], int(p.get("out_channels", 16)), int(p.get("kernel_size", 4)), int(p.get("stride", 2)), int(p.get("padding", 1)), int(p.get("output_padding", 0)), groups=int(p.get("groups", 1)))
                    elif op == "Linear": return nn.Linear(in_shape[-1], int(p.get("out_features", 10)))
                    elif op == "Bilinear": return nn.Bilinear(in_shape[1], int(p.get("in2_features", 16)), int(p.get("out_features", 10)))
                    elif op in ("BatchNorm1d", "BatchNorm2d", "BatchNorm3d"):
                        return getattr(nn, op)(in_shape[1])
                    elif op in ("InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d"):
                        return getattr(nn, op)(in_shape[1], affine=bool(p.get("affine", 1)), track_running_stats=bool(p.get("track_running_stats", 0)))
                    elif op == "LayerNorm":
                        normalized = p.get("normalized_shape", in_shape[1:])
                        normalized = tuple(int(v) for v in normalized) if isinstance(normalized, list) else int(normalized)
                        return nn.LayerNorm(normalized, eps=float(p.get("eps", 1e-5)), elementwise_affine=bool(p.get("elementwise_affine", 1)))
                    elif op == "GroupNorm": return nn.GroupNorm(int(p.get("num_groups", 1)), in_shape[1], affine=bool(p.get("affine", 1)))
                    elif op in ("MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d", "AvgPool2d", "AvgPool3d"):
                        cls = getattr(nn, op); return cls(int(p.get("kernel_size", 2)), int(p.get("stride", 2)), int(p.get("padding", 0)))
                    elif op in ("AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d", "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d"):
                        size = p.get("output_size", 1); size = tuple(size) if isinstance(size, list) else size; return getattr(nn, op)(size)
                    elif op == "Flatten": return nn.Flatten(1)
                    elif op in ("Unsqueeze", "Squeeze", "Slice", "Select"): return None
                    elif op == "ConstantAdd": return TensorLabConstantAdd(p["shape"], p["values"], p.get("sequence_dim"))
                    elif op in ("Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "AlphaDropout"): return getattr(nn, op)(float(p.get("p", 0.3)))
                    elif op == "Embedding": return nn.Embedding(int(p.get("num_embeddings", 100)), int(p.get("embedding_dim", 32)))
                    elif op == "Upsample":
                        scale = p.get("scale_factor", 2); scale = tuple(scale) if isinstance(scale, list) else scale; return nn.Upsample(scale_factor=scale, mode=str(p.get("mode", "nearest")))
                    elif op in ("ReLU", "GELU", "Sigmoid", "Tanh", "SiLU", "SELU", "Hardsigmoid", "Hardswish", "Mish", "Softsign", "Identity"): return getattr(nn, op)()
                    elif op == "PReLU": return nn.PReLU(int(p.get("num_parameters", 1)), float(p.get("init", 0.25)))
                    elif op == "LeakyReLU": return nn.LeakyReLU(float(p.get("negative_slope", 0.01)))
                    elif op == "ELU": return nn.ELU(float(p.get("alpha", 1.0)))
                    elif op == "Softplus": return nn.Softplus(float(p.get("beta", 1.0)), float(p.get("threshold", 20.0)))
                    elif op in ("Softmax", "LogSoftmax"): return getattr(nn, op)(dim=int(p.get("dim", -1)))
                    elif op in ("MultiHeadAttention", "Transformer"):
                        args = {"embed_dim": int(p.get("embed_dim", 64)), "num_heads": int(p.get("num_heads", 1)), "kv_heads": int(p["kv_heads"]) if "kv_heads" in p else None, "dropout": float(p.get("dropout", 0.1)), "attention_type": p.get("attention_type", "self"), "branches": int(p.get("branches", 1))}
                        return TensorLabTransformer(**args, ff_dim=int(p.get("ff_dim", 128)), norm_first=bool(p.get("norm_first", 1)), activation=p.get("activation", "gelu")) if op == "Transformer" else TensorLabAttention(**args)
                    else: return None
                try:
                    module = build()
                    if module is None: continue
                    # A folded node owns `repeat` fully independent instances (never one shared module),
                    # so sum(model.parameters()) equals analyze_graph's count * repeat.
                    self.layers[key] = nn.ModuleList([module, *(build() for _ in range(repeat - 1))]) if repeat > 1 else module
                except Exception as exc:
                    raise ValueError(f"{key}: cannot build {op} with repeat={repeat}: {exc}") from exc
        def forward(self, x, observer=None, attention_observer=None):
            if isinstance(x, dict):
                if set(x) != set(info["inputs"]): raise ValueError("Input dictionary must contain exactly the model's Input node IDs")
            elif len(info["inputs"]) != 1:
                raise ValueError("Multi-input models require a dictionary keyed by Input node ID")
            values, ports = {}, {}
            def edge_value(edge):
                return ports[edge["source"]][edge["sourcePort"]] if edge.get("sourcePort") else values[edge["source"]]
            def observe(key, value):
                if observer is not None:
                    observer(key, value)
            by_id = {node["id"]: node for node in graph["nodes"]}
            for key in info["order"]:
                node = by_id[key]; op = node["op"]; args = [edge_value(e) for e in info["baseEdges"][key]]
                if op == "Input": values[key] = x[key] if isinstance(x, dict) else x
                elif op == "Output": values[key] = args[0]
                elif op == "Add": values[key] = sum(args)
                elif op == "Concat": values[key] = torch.cat(args, dim=int(node["params"].get("dim", 1)))
                elif op == "Unsqueeze": values[key] = torch.unsqueeze(args[0], int(node["params"].get("dim", 0)))
                elif op == "Squeeze":
                    dim = node["params"].get("dim", "all")
                    values[key] = torch.squeeze(args[0]) if dim == "all" else torch.squeeze(args[0], dim=tuple(dim) if isinstance(dim, list) else dim)
                elif op == "Slice":
                    p = node["params"]; dim = int(p.get("dim", 1)) % args[0].ndim
                    start = None if p.get("start", "none") == "none" else int(p["start"]); end = None if p.get("end", "none") == "none" else int(p["end"])
                    indices = [slice(None)] * args[0].ndim; indices[dim] = slice(start, end, int(p.get("step", 1))); values[key] = args[0][tuple(indices)]
                elif op == "Select": values[key] = torch.select(args[0], int(node["params"].get("dim", 1)), int(node["params"].get("index", -1)))
                elif op in ("Transformer", "MultiHeadAttention"):
                    overrides = {e["targetPort"]: edge_value(e) for e in info["overrideEdges"][key]}
                    attention_callback = None
                    if attention_observer is not None:
                        attention_callback = lambda stage, branch, tensors, node_id=key: attention_observer(node_id, stage, branch, tensors)
                    instances = self.layers[key]
                    query, context = args[0], (args[1] if len(args) > 1 else None)
                    for instance in (instances if isinstance(instances, nn.ModuleList) else (instances,)):
                        value, ports[key] = instance.forward_with_ports(query, context, overrides, observer=attention_callback)
                        query = value
                    values[key] = query
                elif key in self.layers:
                    instances = self.layers[key]
                    # Folded nodes run their independent instances in series; shapes are preserved.
                    dead = 0.0; value = None
                    for index, instance in enumerate(instances if isinstance(instances, nn.ModuleList) else (instances,)):
                        # Later instances consume the previous output as their first operand but still
                        # need the node's remaining inputs (Bilinear takes two). This must mirror the
                        # generated export, which chains `instance(value, *other_inputs)`.
                        value = instance(*args) if index == 0 else instance(value, *args[1:])
                        if op in ("ReLU", "LeakyReLU"): dead += float((value.detach() == 0).float().mean().item())
                    values[key] = value
                    if op in ("ReLU", "LeakyReLU"): self.dead_relu[key] = dead
                observe(key, values[key])
            return values[info["output"]]
    return GraphModel(), info


def load_data(info, config, classes):
    import torch
    input_ids = info["inputs"]
    shapes = {key: info["shapes"][key][1:] for key in input_ids}
    # Embedding consumes integer token IDs.  Keep this information with the
    # dataset loader so synthetic data, CSV data and normalization all agree
    # on the dtype expected by the real PyTorch module.
    by_id = {node["id"]: node for node in info.get("nodes", [])}
    embedding_inputs = {}
    for key, edges in info.get("baseEdges", {}).items():
        node = by_id.get(key)
        if node and node.get("op") == "Embedding" and edges:
            source = edges[0]["source"]
            if by_id.get(source, {}).get("op") == "Input":
                embedding_inputs[source] = node.get("params", {})
    features = sum(math.prod(shape) for shape in shapes.values())
    if config["dataset"] == "csv":
        content = config.get("csv", "")
        if not content or len(content.encode("utf-8")) > 8_000_000: raise ValueError("CSV is empty or exceeds 8 MB")
        rows = list(csv.reader(io.StringIO(content)))
        if rows:
            try: [float(value) for value in rows[0]]
            except ValueError: rows = rows[1:]
        if not 16 <= len(rows) <= 4096 or any(len(row) != features + 1 for row in rows): raise ValueError(f"CSV needs 16-4096 rows, {features} features and a final integer class column")
        try: data = torch.tensor([[float(v) for v in row] for row in rows], dtype=torch.float32)
        except ValueError as exc: raise ValueError("CSV contains nonnumeric values") from exc
        if not torch.isfinite(data).all(): raise ValueError("CSV contains NaN or Inf")
        labels = data[:, -1]
        if not ((labels == labels.long()) & (labels >= 0) & (labels < classes)).all(): raise ValueError(f"Class labels must be integers in [0,{classes - 1}]")
        values, offset = {}, 0
        for key, shape in shapes.items():
            width = math.prod(shape)
            value = data[:, offset:offset + width].reshape(-1, *shape)
            if key in embedding_inputs:
                limit = int(embedding_inputs[key].get("num_embeddings", 100))
                if not torch.isfinite(value).all() or not ((value == value.long()) & (value >= 0) & (value < limit)).all():
                    raise ValueError(f"Embedding input {key} must contain integer IDs in [0,{limit - 1}]")
                value = value.long()
            values[key] = value
            offset += width
        return values, labels.long()
    samples = config["samples"]
    if samples * features * 4 > 256_000_000: raise ValueError("Dataset exceeds 256 MB; reduce samples or input size")
    generator = torch.Generator().manual_seed(42)
    labels = torch.randint(classes, (samples,), generator=generator)
    values = {}
    for key, shape in shapes.items():
        if key in embedding_inputs:
            limit = int(embedding_inputs[key].get("num_embeddings", 100))
            values[key] = torch.randint(limit, (samples, *shape), generator=generator, dtype=torch.long)
            continue
        x = torch.randn(samples, *shape, generator=generator) * 0.25
        if len(shape) == 1:
            x += torch.nn.functional.one_hot(labels % shape[0], shape[0]).float() * 2
        elif len(shape) == 2:
            signal = torch.nn.functional.one_hot(labels % shape[1], shape[1]).float()
            x += signal[:, None, :] * 1.5
        else:
            for category in range(classes):
                mask = labels == category
                row = category * max(1, shape[1] // classes) % shape[1]
                channel = category % shape[0]
                x[mask, channel, row:min(row + 3, shape[1]), :] += 1.5
        values[key] = x
    return values, labels


def train(graph, config, emit, stop, retain_model=None):
    import torch
    from torch import nn
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(42)
    info = analyze_graph(graph)
    output_shape = info["shapes"][info["output"]]
    if len(output_shape) != 2 or not 2 <= output_shape[1] <= 256: raise ValueError("Training requires classification logits [B,2..256]")
    if info["activationBytesPerSample"] * config["batchSize"] > 256_000_000: raise ValueError("Batch activations exceed 256 MB; reduce batch size")
    if config["device"] == "cuda" and not torch.cuda.is_available(): raise ValueError("CUDA is unavailable in this PyTorch installation")
    device = "cuda" if config["device"] != "cpu" and torch.cuda.is_available() else "cpu"
    model, info = build_model(graph); model.to(device)
    if not list(model.parameters()): raise ValueError("Model has no trainable parameters")
    x, labels = load_data(info, config, output_shape[1])
    permutation = torch.randperm(len(labels), generator=torch.Generator().manual_seed(7))
    x, labels = {key: value[permutation] for key, value in x.items()}, labels[permutation]
    count = max(2, int(len(labels) * config["validationFraction"]))
    val_x, val_y = {key: value[:count] for key, value in x.items()}, labels[:count]
    train_x, train_y = {key: value[count:] for key, value in x.items()}, labels[count:]
    preprocessing = {}
    if config["dataset"] == "csv":
        # Fit normalization on the training split only, avoiding validation leakage.
        for key in train_x:
            if not train_x[key].is_floating_point():
                continue
            mean = train_x[key].mean(0); scale = train_x[key].std(0).clamp_min(1e-6)
            preprocessing[key] = {"mean": mean, "scale": scale}
            train_x[key], val_x[key] = (train_x[key] - mean) / scale, (val_x[key] - mean) / scale
    criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.Adam(model.parameters(), lr=config["learningRate"])
    best_loss, stale, best_state, best_epoch = float("inf"), 0, None, 0
    training_run_id = str(uuid.uuid4())
    def finish(reason, epoch):
        if stop.is_set():
            emit({"type": "done", "reason": "stopped"}); return
        message = {"type": "done", "reason": reason}
        if retain_model is not None:
            metadata = retain_model(graph, model, preprocessing, {
                "trainingRunId": training_run_id, "device": device.upper(), "dataset": config["dataset"],
                "seed": 42, "epochsCompleted": epoch, "weightsEpoch": best_epoch if reason == "early_stopping" else epoch, "reason": reason,
            })
            message["model"] = metadata
            if metadata is None:
                message["retentionWarning"] = "训练完成，但模型状态与预处理参数超过 64 MiB，未保留本次权重。"
        emit(message)
    batch = config["batchSize"]
    for epoch in range(1, config["epochs"] + 1):
        if stop.is_set(): emit({"type": "done", "reason": "stopped"}); return
        model.train(); total_loss = 0.0; seen = 0; norms = {}; norm_count = 0; dead_sums = {}; grad_total = 0.0
        index = torch.randperm(len(train_y))
        for start in range(0, len(train_y), batch):
            if stop.is_set(): emit({"type": "done", "reason": "stopped"}); return
            indices = index[start:start + batch]
            # BatchNorm cannot estimate variance for a singleton batch at a 1x1 spatial layer.
            if len(indices) == 1 and any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) for m in model.modules()): continue
            bx, by = {key: value[indices].to(device) for key, value in train_x.items()}, train_y[indices].to(device)
            optimizer.zero_grad(set_to_none=True); prediction = model(bx); loss = criterion(prediction, by)
            if not torch.isfinite(loss): raise ValueError("Loss became NaN/Inf. Reduce learning rate and inspect inputs")
            loss.backward(); square = 0.0
            for key, layer in model.layers.items():
                value = sum(float(p.grad.detach().square().sum().item()) for p in layer.parameters() if p.grad is not None)
                norm = math.sqrt(value)
                if not math.isfinite(norm): raise ValueError(f"Layer {key} gradient became NaN/Inf")
                norms[key] = norms.get(key, 0) + norm; square += value
            grad_total += math.sqrt(square); norm_count += 1
            for key, value in model.dead_relu.items(): dead_sums[key] = dead_sums.get(key, 0) + value
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step(); total_loss += loss.item() * len(indices); seen += len(indices)
        model.eval(); val_loss = 0.0; correct = 0
        with torch.no_grad():
            for start in range(0, len(val_y), batch):
                bx, by = {key: value[start:start + batch].to(device) for key, value in val_x.items()}, val_y[start:start + batch].to(device)
                logits = model(bx); value = criterion(logits, by).item()
                if not math.isfinite(value): raise ValueError("Validation loss became NaN/Inf")
                val_loss += value * len(by); correct += int((logits.argmax(1) == by).sum().item())
        val_loss /= len(val_y)
        emit({"type": "metric", "device": device.upper(), "metric": {"epoch": epoch, "trainLoss": total_loss / max(1, seen), "valLoss": val_loss, "accuracy": correct / len(val_y), "gradNorm": grad_total / max(1, norm_count), "layerGradients": {key: value / max(1, norm_count) for key, value in norms.items()}, "deadRelu": {key: value / max(1, norm_count) for key, value in dead_sums.items()}, "source": "training"}})
        if val_loss < best_loss - 1e-4:
            best_loss, stale, best_epoch = val_loss, 0, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else: stale += 1
        if config["earlyStopping"] and stale >= config["patience"]:
            if best_state: model.load_state_dict(best_state)
            finish("early_stopping", epoch); return
    finish("completed", config["epochs"])

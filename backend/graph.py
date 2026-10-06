"""Validate the graph before allocating PyTorch tensors or model parameters."""
from math import prod, isfinite

OPS = {
    "Input", "Output", "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "Linear", "Bilinear",
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LayerNorm", "GroupNorm", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d",
    "ReLU", "GELU", "Sigmoid", "Tanh", "SiLU", "LeakyReLU", "ELU", "SELU", "Softplus", "Softmax", "LogSoftmax", "PReLU", "Hardsigmoid", "Hardswish", "Mish", "Softsign", "Identity",
    "MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d", "AvgPool2d", "AvgPool3d",
    "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d", "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d",
    "Flatten", "Unsqueeze", "Squeeze", "Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "AlphaDropout", "Embedding", "Upsample",
    "Add", "Concat", "MultiHeadAttention", "Transformer"
}

_ACTIVATIONS = {"ReLU", "GELU", "Sigmoid", "Tanh", "SiLU", "LeakyReLU", "ELU", "SELU", "Softplus", "Softmax", "LogSoftmax", "PReLU", "Hardsigmoid", "Hardswish", "Mish", "Softsign", "Identity"}
_DROPOUTS = {"Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "AlphaDropout"}
_CONVS = {"Conv1d": 3, "Conv2d": 4, "Conv3d": 5}
_TRANSPOSE_CONVS = {"ConvTranspose1d": 3, "ConvTranspose2d": 4, "ConvTranspose3d": 5}
_POOLS = {"MaxPool1d": 3, "MaxPool2d": 4, "MaxPool3d": 5, "AvgPool1d": 3, "AvgPool2d": 4, "AvgPool3d": 5}
_ADAPTIVE_POOLS = {
    "AdaptiveAvgPool1d": 3, "AdaptiveAvgPool2d": 4, "AdaptiveAvgPool3d": 5,
    "AdaptiveMaxPool1d": 3, "AdaptiveMaxPool2d": 4, "AdaptiveMaxPool3d": 5,
}


def analyze_graph(graph):
    import re
    if not isinstance(graph, dict) or graph.get("version") != 1:
        raise ValueError("Unsupported graph version")
    nodes, edges = graph.get("nodes"), graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list) or not 2 <= len(nodes) <= 128 or len(edges) > 512:
        raise ValueError("Graph must have 2-128 nodes and at most 512 edges")
    by_id = {}
    for node in nodes:
        if not isinstance(node, dict) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,63}", str(node.get("id", ""))) or node["id"] in by_id or node.get("op") not in OPS or not isinstance(node.get("params"), dict):
            raise ValueError("Invalid node, operation or duplicate ID")
        by_id[node["id"]] = node
    incoming = {key: [] for key in by_id}
    outgoing = {key: [] for key in by_id}
    pairs = set()
    port_pattern = r"b\d+:[qkv]\d+"
    def projection_ports(node):
        if node["op"] not in ("Transformer", "MultiHeadAttention"): return []
        p = node["params"]; heads = p.get("num_heads", 1); kind = p.get("attention_type", "self")
        kv = p.get("kv_heads", 1 if kind == "multi_query" else heads); branches = p.get("branches", 1)
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or int(v) != v or not 1 <= v <= limit for v, limit in ((heads, 16), (kv, 16), (branches, 8))):
            raise ValueError("Invalid projection head/branch count")
        return [f"b{b}:{role}{i}" for b in range(int(branches)) for role in "qkv" for i in range(int(heads if role == "q" else kv))]
    for edge in edges:
        a, b = edge.get("source"), edge.get("target")
        pair = (a, edge.get("sourcePort"), b, edge.get("targetPort"))
        if a not in by_id or b not in by_id or a == b or pair in pairs:
            raise ValueError("Invalid or duplicate edge")
        if edge.get("targetPort") not in (None, "query", "context") and edge.get("targetPort") not in projection_ports(by_id[b]):
            raise ValueError("Invalid attention input port")
        if edge.get("sourcePort") is not None and edge["sourcePort"] not in projection_ports(by_id[a]): raise ValueError("Invalid attention output port")
        pairs.add(pair); incoming[b].append(a); outgoing[a].append(b)
    base_edges, override_edges = {}, {}
    for key, node in by_id.items():
        node_edges = [e for e in edges if e["target"] == key]
        override_edges[key] = [e for e in node_edges if re.fullmatch(port_pattern, e.get("targetPort") or "")]
        node_edges = [e for e in node_edges if e not in override_edges[key]]
        if len({e["targetPort"] for e in override_edges[key]}) != len(override_edges[key]): raise ValueError("Each Q/K/V port accepts one input")
        cross = node["op"] in ("Transformer", "MultiHeadAttention") and node["params"].get("attention_type") == "cross"
        if cross:
            roles = [e.get("targetPort") or ("query" if i == 0 else "context") for i, e in enumerate(node_edges)]
            if len(roles) != 2 or set(roles) != {"query", "context"}:
                raise ValueError(f"{key}: Cross-Attention requires one Query and one Context input")
            node_edges = [e for _, e in sorted(zip(roles, node_edges), key=lambda item: item[0] == "context")]
        elif any(e.get("targetPort") for e in node_edges):
            raise ValueError(f"{key}: Query/Context ports require Cross-Attention")
        base_edges[key] = node_edges
        incoming[key] = [e["source"] for e in node_edges + override_edges[key]]
    inputs = [n for n in nodes if n["op"] == "Input"]
    outputs = [n for n in nodes if n["op"] == "Output"]
    if not 1 <= len(inputs) <= 8 or len(outputs) != 1:
        raise ValueError("1-8 Inputs and exactly one Output are required")
    degree = {key: len(value) for key, value in incoming.items()}
    queue = [key for key, d in degree.items() if d == 0]
    order = []
    while queue:
        key = queue.pop(0); order.append(key)
        for target in outgoing[key]:
            degree[target] -= 1
            if degree[target] == 0: queue.append(target)
    if len(order) != len(nodes): raise ValueError("Graph contains a cycle")
    ancestors = set()
    def visit(key):
        if key in ancestors: return
        ancestors.add(key)
        for parent in incoming[key]: visit(parent)
    visit(outputs[0]["id"])
    if any(n["id"] not in ancestors for n in inputs): raise ValueError("Every Input must be connected to the model Output")
    shapes, port_shapes, parameters, total, activation = {}, {}, {}, 0, 0
    def edge_shape(edge):
        return port_shapes[edge["source"]][edge["sourcePort"]] if edge.get("sourcePort") else shapes[edge["source"]]
    for key in order:
        node = by_id[key]; op = node["op"]; p = node["params"]; parents = [e["source"] for e in base_edges[key]]
        in_shapes = [edge_shape(e) for e in base_edges[key]]
        def integer(name, default, minimum=1, maximum=65536):
            value = p.get(name, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value != int(value) or not minimum <= value <= maximum:
                raise ValueError(f"{key}: invalid {name}")
            return int(value)
        count = 0
        if op == "Input":
            shape = p.get("shape")
            if incoming[key] or not isinstance(shape, list) or len(shape) not in (2, 3, 4, 5) or not all(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 65536 for v in shape):
                raise ValueError("Input requires [B,F], [B,S,E], [B,C,H,W] or [B,C,D,H,W]")
            shape = list(shape)
        else:
            cross = op in ("Transformer", "MultiHeadAttention") and p.get("attention_type") == "cross"
            required_inputs = 2 if cross or op == "Bilinear" else 1
            if not parents or (op not in ("Add", "Concat") and len(parents) != required_inputs): raise ValueError(f"{key}: invalid input count")
            shape = list(in_shapes[0])
            if op in _TRANSPOSE_CONVS:
                rank = _TRANSPOSE_CONVS[op]
                if len(shape) != rank: raise ValueError(f"{key}: requires {rank}D channel input")
                k = integer("kernel_size", 4, maximum=64); s = integer("stride", 2, maximum=64); pad = integer("padding", 1, 0, 64); output_pad = integer("output_padding", 0, 0, 64)
                if output_pad >= s: raise ValueError(f"{key}: output_padding must be smaller than stride")
                channels = integer("out_channels", 16, maximum=4096); groups = integer("groups", 1, maximum=shape[1])
                if shape[1] % groups or channels % groups: raise ValueError(f"{key}: channels must be divisible by groups")
                spatial = [(size - 1) * s - 2 * pad + k + output_pad for size in shape[2:]]
                if min(spatial) < 1: raise ValueError(f"{key}: spatial output is empty")
                count = shape[1] * (channels // groups) * (k ** (rank - 2)) + channels; shape = [shape[0], channels, *spatial]
            elif op in _CONVS:
                rank = _CONVS[op]
                if len(shape) != rank: raise ValueError(f"{key}: requires {rank}D channel input")
                k = integer("kernel_size", 3, maximum=64); s = integer("stride", 1, maximum=64); pad = integer("padding", 1, 0, 64)
                spatial = [((size + 2 * pad - k) // s + 1) for size in shape[2:]]
                if min(spatial) < 1: raise ValueError(f"{key}: spatial output is empty")
                channels = integer("out_channels", 16, maximum=4096)
                groups = integer("groups", 1, maximum=shape[1])
                if shape[1] % groups or channels % groups: raise ValueError(f"{key}: channels must be divisible by groups")
                count = channels * (shape[1] // groups) * (k ** (rank - 2)) + channels
                shape = [shape[0], channels, *spatial]
            elif op in _POOLS:
                rank = _POOLS[op]
                if len(shape) != rank: raise ValueError(f"{key}: requires {rank}D channel input")
                k = integer("kernel_size", 2, maximum=64); s = integer("stride", 2, maximum=64); pad = integer("padding", 0, 0, 64)
                spatial = [((size + 2 * pad - k) // s + 1) for size in shape[2:]]
                if min(spatial) < 1: raise ValueError(f"{key}: pooling output is empty")
                shape = [shape[0], shape[1], *spatial]
            elif op == "Linear":
                # PyTorch Linear applies to the final dimension and preserves
                # leading batch/sequence dimensions (e.g. [B,S,F] -> [B,S,O]).
                if len(shape) < 2: raise ValueError(f"{key}: Linear requires at least [B,F]")
                features = integer("out_features", 10)
                count = (shape[-1] + 1) * features; shape = [*shape[:-1], features]
            elif op in ("BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d"):
                expected_rank = int(op[-2]) + 2
                if len(shape) != expected_rank: raise ValueError(f"{key}: {op} requires {expected_rank}D channel input")
                if op.startswith("BatchNorm"): count = shape[1] * 2
                elif p.get("affine", 1): count = shape[1] * 2
            elif op == "LayerNorm":
                normalized = p.get("normalized_shape")
                normalized = normalized if isinstance(normalized, list) else [normalized]
                if not normalized or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in normalized): raise ValueError(f"{key}: normalized_shape must be positive integers")
                if len(normalized) > len(shape) - 1 or shape[-len(normalized):] != normalized: raise ValueError(f"{key}: normalized_shape must match the input suffix")
                if int(p.get("elementwise_affine", 1)): count = 2 * prod(normalized)
            elif op == "GroupNorm":
                if len(shape) < 3: raise ValueError(f"{key}: GroupNorm requires NCHW-like input")
                groups = integer("num_groups", 1, maximum=shape[1])
                if shape[1] % groups: raise ValueError(f"{key}: num_channels must be divisible by num_groups")
                if int(p.get("affine", 1)): count = shape[1] * 2
            elif op in _ADAPTIVE_POOLS:
                rank = _ADAPTIVE_POOLS[op]
                if len(shape) != rank: raise ValueError(f"{key}: requires {rank}D channel input")
                size = p.get("output_size", 1)
                sizes = size if isinstance(size, list) else [size] * (rank - 2)
                if len(sizes) != rank - 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 or v > 256 for v in sizes): raise ValueError(f"{key}: invalid output_size")
                shape = [shape[0], shape[1], *sizes]
            elif op == "Flatten": shape = [shape[0], prod(shape[1:])]
            elif op == "Unsqueeze":
                dim = integer("dim", 0, minimum=-len(shape) - 1, maximum=len(shape))
                if dim < 0: dim += len(shape) + 1
                shape = [*shape[:dim], 1, *shape[dim:]]
            elif op == "Squeeze":
                raw = p.get("dim", "all")
                if raw == "all": shape = [size for size in shape if size != 1]
                else:
                    dims = raw if isinstance(raw, list) else [raw]
                    rank = max(len(shape), 1)
                    if any(type(dim) is not int or not -rank <= dim < rank for dim in dims):
                        raise ValueError(f"{key}: squeeze dim must be an integer in [{-rank}, {rank - 1}]")
                    indices = [dim % rank for dim in dims]
                    if len(set(indices)) != len(indices): raise ValueError(f"{key}: duplicate squeeze dimensions")
                    shape = [size for index, size in enumerate(shape) if index not in indices or size != 1]
            elif op == "Bilinear":
                if len(shape) != 2: raise ValueError(f"{key}: Bilinear requires [B,F1]")
                other = integer("in2_features", shape[1]); features = integer("out_features", 10)
                count = shape[1] * other * features + features; shape = [shape[0], features]
            elif op == "Embedding":
                if len(shape) < 2: raise ValueError(f"{key}: Embedding requires integer index input")
                if by_id[parents[0]]["op"] != "Input": raise ValueError(f"{key}: Embedding input must come directly from an Input layer")
                features = integer("embedding_dim", 32, maximum=4096); count = integer("num_embeddings", 100, maximum=1_000_000) * features
                shape = [*shape, features]
            elif op == "Upsample":
                if len(shape) < 3: raise ValueError(f"{key}: Upsample requires channel and spatial dimensions")
                scale = p.get("scale_factor", 2); scale = scale if isinstance(scale, list) else [scale] * (len(shape) - 2)
                if len(scale) != len(shape) - 2 or any(not isinstance(v, (int, float)) or v <= 0 or not isfinite(v) for v in scale): raise ValueError(f"{key}: invalid scale_factor")
                shape = [shape[0], shape[1], *[max(1, int(size * factor)) for size, factor in zip(shape[2:], scale)]]
            elif op in ("Transformer", "MultiHeadAttention"):
                if len(shape) != 3: raise ValueError(f"{key}: attention requires [B,S,E]")
                embed = integer("embed_dim", 64, maximum=4096); heads = integer("num_heads", 1, maximum=16)
                kind = p.get("attention_type", "self")
                if kind not in ("self", "multi_query", "grouped_query", "cross", "multi_branch"): raise ValueError(f"{key}: invalid attention_type")
                kv_heads = integer("kv_heads", 1 if kind == "multi_query" else heads, maximum=16)
                branches = integer("branches", 1, maximum=8)
                if kv_heads > heads or heads % kv_heads: raise ValueError(f"{key}: num_heads must be divisible by kv_heads")
                if kind in ("self", "multi_branch") and kv_heads != heads: raise ValueError(f"{key}: self attention requires kv_heads = num_heads")
                if kind == "multi_query" and kv_heads != 1: raise ValueError(f"{key}: MQA requires kv_heads = 1")
                if kind == "multi_branch" and branches < 2: raise ValueError(f"{key}: multi_branch requires branches >= 2")
                if shape[2] != embed or embed % heads: raise ValueError(f"{key}: embed_dim must match input and be divisible by num_heads")
                context = in_shapes[1] if cross else shape
                if len(context) != 3 or context[0] != shape[0] or context[2] != embed: raise ValueError(f"{key}: Context must be BSE with matching batch and embed_dim")
                dropout = p.get("dropout", 0.1)
                if not isinstance(dropout, (int, float)) or not isfinite(dropout) or not 0 <= dropout < 1: raise ValueError("Attention dropout must be in [0,1)")
                sources = {}
                for port in projection_ports(node):
                    override = next((e for e in override_edges[key] if e["targetPort"] == port), None)
                    source_shape = edge_shape(override) if override else (shape if ":q" in port else context)
                    if len(source_shape) != 3 or source_shape[0] != shape[0] or source_shape[2] != embed or (":q" in port and source_shape[1] != shape[1]): raise ValueError(f"{key}:{port}: requires BSE input with matching batch/embed and Q length")
                    sources[port] = source_shape
                for branch in range(branches):
                    lengths = {s[1] for port, s in sources.items() if port.startswith(f"b{branch}:") and ":q" not in port}
                    if len(lengths) != 1: raise ValueError("All K/V lengths within a branch must match")
                elements = shape[0] * heads * shape[1] * sum(sources[f"b{b}:k0"][1] for b in range(branches))
                if elements > 16_000_000: raise ValueError("Attention matrix exceeds local limits")
                activation += elements // shape[0] * 4
                port_shapes[key] = {port: [s[0], s[1], embed // heads] for port, s in sources.items()}
                kv_dim = kv_heads * (embed // heads)
                count = branches * (2 * embed * embed + 2 * embed * kv_dim + 2 * embed + 2 * kv_dim)
                if op == "Transformer":
                    integer("norm_first", 1, 0, 1)
                    if p.get("activation", "gelu") not in ("relu", "gelu"): raise ValueError("Encoder activation must be relu or gelu")
                    ff = integer("ff_dim", 128, maximum=16384)
                    if shape[0] * shape[1] * ff > 16_000_000: raise ValueError("FFN activation exceeds local limits")
                    activation += shape[1] * ff * 4
                    count += 2 * embed * ff + ff + 5 * embed
            elif op in _DROPOUTS:
                rate = p.get("p", 0.3)
                if not isinstance(rate, (int, float)) or not isfinite(rate) or not 0 <= rate < 1: raise ValueError("Dropout must be in [0,1)")
            elif op in _ACTIVATIONS:
                if op in ("Softmax", "LogSoftmax"):
                    dim = p.get("dim", -1)
                    if not isinstance(dim, int) or not -len(shape) <= dim < len(shape): raise ValueError(f"{key}: invalid dim")
                elif op == "PReLU":
                    num_parameters = integer("num_parameters", 1, maximum=4096)
                    if num_parameters not in (1, shape[1] if len(shape) > 1 else 1): raise ValueError(f"{key}: PReLU num_parameters must be 1 or the channel count")
                    count = num_parameters
            elif op == "Add":
                if len(parents) < 2 or any(s != shape for s in in_shapes): raise ValueError(f"{key}: Add input shapes must match")
            elif op == "Concat":
                dim = integer("dim", 1, maximum=len(shape) - 1)
                if len(parents) < 2 or any(len(s) != len(shape) or any(v != shape[i] for i, v in enumerate(s) if i != dim) for s in in_shapes): raise ValueError(f"{key}: Concat dimensions must match")
                shape[dim] = sum(s[dim] for s in in_shapes)
            elif op == "Output" and outgoing[key]: raise ValueError("Output cannot have outgoing edges")
        if prod(shape) > 16_000_000 or count > 50_000_000: raise ValueError(f"{key}: layer exceeds local memory limits")
        shapes[key] = shape; parameters[key] = count; total += count; activation += prod(shape[1:]) * 4
    if total > 50_000_000: raise ValueError("Model exceeds 50M parameters")
    if len({shapes[n["id"]][0] for n in inputs}) != 1: raise ValueError("All model inputs must have the same batch dimension")
    return {"order": order, "shapes": shapes, "portShapes": port_shapes, "baseEdges": base_edges, "overrideEdges": override_edges, "incoming": incoming, "parameters": parameters, "totalParameters": total, "activationBytesPerSample": activation, "input": inputs[0]["id"], "inputs": [n["id"] for n in inputs], "output": outputs[0]["id"], "nodes": nodes}

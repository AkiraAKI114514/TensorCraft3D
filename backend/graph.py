"""Validate the graph before allocating PyTorch tensors or model parameters."""
from math import prod, isfinite

OPS = {
    "Input", "Output", "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "Linear", "Bilinear",
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LayerNorm", "RMSNorm", "GroupNorm", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d",
    "ReLU", "GELU", "Sigmoid", "Tanh", "SiLU", "LeakyReLU", "ELU", "SELU", "Softplus", "Softmax", "LogSoftmax", "PReLU", "Hardsigmoid", "Hardswish", "Mish", "Softsign", "Identity",
    "MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d", "AvgPool2d", "AvgPool3d",
    "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d", "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d",
    "Flatten", "Unsqueeze", "Squeeze", "Slice", "Select", "ConstantAdd", "Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "AlphaDropout", "Embedding", "Upsample",
    "Add", "Multiply", "Concat", "MultiHeadAttention", "Transformer", "Group"
}

_ACTIVATIONS = {"ReLU", "GELU", "Sigmoid", "Tanh", "SiLU", "LeakyReLU", "ELU", "SELU", "Softplus", "Softmax", "LogSoftmax", "PReLU", "Hardsigmoid", "Hardswish", "Mish", "Softsign", "Identity"}
_DROPOUTS = {"Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "AlphaDropout"}
_CONVS = {"Conv1d": 3, "Conv2d": 4, "Conv3d": 5}
_TRANSPOSE_CONVS = {"ConvTranspose1d": 3, "ConvTranspose2d": 4, "ConvTranspose3d": 5}
MAX_REPEAT = 1024
MAX_SUBGRAPH_DEPTH = 16
EXPAND_SEP = "/"
# Backstop for the expanded graph only; the authored graph keeps the 128-node budget.
EXPANDED_NODE_LIMIT = 20_000
EXPANDED_EDGE_LIMIT = 60_000
_POOLS = {"MaxPool1d": 3, "MaxPool2d": 4, "MaxPool3d": 5, "AvgPool1d": 3, "AvgPool2d": 4, "AvgPool3d": 5}
_ADAPTIVE_POOLS = {
    "AdaptiveAvgPool1d": 3, "AdaptiveAvgPool2d": 4, "AdaptiveAvgPool3d": 5,
    "AdaptiveMaxPool1d": 3, "AdaptiveMaxPool2d": 4, "AdaptiveMaxPool3d": 5,
}


def expand_subgraphs(graph):
    """Expand subgraph instances into an equivalent flat operator graph.

    Everything downstream then walks the existing validation and shape-derivation
    path, so training, inference and snapshots never see a Group node. Expanded ids
    look like ``g1/0/attn``, matching src/subgraph.ts on the frontend.
    """
    subgraphs = graph.get("subgraphs")
    if not subgraphs or not any(node.get("op") == "Group" for node in graph["nodes"]):
        return graph
    nodes, edges = [], []

    def endpoints(definition, name):
        inner = {node["id"] for node in definition["nodes"]}
        for edge in definition["edges"]:
            if edge.get("source") not in inner or edge.get("target") not in inner:
                raise ValueError(f"Subgraph {name} has an edge to a missing node")
        if not definition["nodes"]:
            raise ValueError(f"Subgraph {name} is empty")
        targets = {edge["target"] for edge in definition["edges"]}
        sources = {edge["source"] for edge in definition["edges"]}
        entry = [node["id"] for node in definition["nodes"] if node["id"] not in targets]
        exit_ = [node["id"] for node in definition["nodes"] if node["id"] not in sources]
        if len(entry) != 1:
            raise ValueError(f"Subgraph {name} needs exactly one entry, found {len(entry)}")
        if len(exit_) != 1:
            raise ValueError(f"Subgraph {name} needs exactly one exit, found {len(exit_)}")
        return entry[0], exit_[0]

    def walk(items, prefix, depth, ports):
        if depth > MAX_SUBGRAPH_DEPTH:
            raise ValueError(f"Subgraph nesting exceeds {MAX_SUBGRAPH_DEPTH} levels")
        for node in items:
            key = f"{prefix}{EXPAND_SEP}{node['id']}" if prefix else node["id"]
            if node.get("op") != "Group":
                nodes.append({**{k: v for k, v in node.items() if k != "subgraph"}, "id": key})
                ports[key] = (key, key)
                continue
            name = node.get("subgraph")
            definition = subgraphs.get(name)
            if definition is None:
                raise ValueError(f"Node {node['id']} references missing subgraph {name}")
            entry, exit_ = endpoints(definition, name)
            count = int(node.get("repeat", 1) or 1)
            for index in range(count):
                scope = f"{key}{EXPAND_SEP}{index}"
                nested = {}
                walk(definition["nodes"], scope, depth + 1, nested)
                for edge in definition["edges"]:
                    edges.append({
                        **edge,
                        "id": f"{scope}{EXPAND_SEP}{edge['id']}",
                        "source": nested[f"{scope}{EXPAND_SEP}{edge['source']}"][1],
                        "target": nested[f"{scope}{EXPAND_SEP}{edge['target']}"][0],
                    })
                # Instances are chained, matching the single-operator repeat semantics.
                if index > 0:
                    edges.append({
                        "id": f"{scope}{EXPAND_SEP}chain",
                        "source": ports[f"{key}{EXPAND_SEP}{index - 1}"][1],
                        "target": nested[f"{scope}{EXPAND_SEP}{entry}"][0],
                    })
                ports[f"{key}{EXPAND_SEP}{index}"] = (
                    nested[f"{scope}{EXPAND_SEP}{entry}"][0],
                    nested[f"{scope}{EXPAND_SEP}{exit_}"][1],
                )
            ports[key] = (ports[f"{key}{EXPAND_SEP}0"][0], ports[f"{key}{EXPAND_SEP}{count - 1}"][1])

    ports = {}
    walk(graph["nodes"], "", 0, ports)
    for edge in graph["edges"]:
        if edge["source"] not in ports or edge["target"] not in ports:
            raise ValueError(f"Edge {edge['id']} references a missing node")
        edges.append({**edge, "source": ports[edge["source"]][1], "target": ports[edge["target"]][0]})
    return {"version": 1, "name": graph.get("name", ""), "nodes": nodes, "edges": edges}


def analyze_expanded_graph(graph):
    """Validate and account for a graph whose subgraph instances are already expanded."""
    import re
    if not isinstance(graph, dict) or graph.get("version") != 1:
        raise ValueError("Unsupported graph version")
    nodes, edges = graph.get("nodes"), graph.get("edges")
    # The 128-node budget applies to the authored graph, which is all a user ever edits. A
    # folded block expands back into one copy of its body per instance, so the expanded graph
    # is legitimately larger than what was authored; holding it to the authored budget would
    # make folding useless for exactly the models it exists for. This pair is a backstop
    # against pathological repeats, not a user-facing limit.
    if not isinstance(nodes, list) or not isinstance(edges, list) or not 2 <= len(nodes) <= EXPANDED_NODE_LIMIT or len(edges) > EXPANDED_EDGE_LIMIT:
        raise ValueError(f"Graph must have 2-{EXPANDED_NODE_LIMIT} expanded nodes and at most {EXPANDED_EDGE_LIMIT} edges")
    by_id = {}
    for node in nodes:
        # Expanded ids look like `import_1/0/n0`: the separator is derived, not authored, so the
        # strict authored-id pattern deliberately does not apply here (analyze_graph checks it).
        if not isinstance(node, dict) or node["id"] in by_id or node.get("op") not in OPS - {"Group"} or not isinstance(node.get("params"), dict):
            raise ValueError("Invalid node, operation or duplicate ID")
        by_id[node["id"]] = node
    seen = set()
    for edge in edges:
        if not isinstance(edge, dict) or edge.get("source") not in by_id or edge.get("target") not in by_id or edge["source"] == edge["target"]:
            raise ValueError("Invalid or duplicate edge")
        pair = (edge["source"], edge["target"], edge.get("sourcePort"), edge.get("targetPort"))
        if pair in seen: raise ValueError("Invalid or duplicate edge")
        seen.add(pair)

    for node in nodes:
        repeat = node.get("repeat")
        # JavaScript's JSON.parse cannot distinguish 2 from 2.0, while Python's json.loads
        # yields a float for 2.0. Since the wire format is produced by the frontend, accept
        # any integral number and reject only genuine non-integers.
        if repeat is not None and (isinstance(repeat, bool) or not isinstance(repeat, (int, float)) or not isfinite(repeat) or repeat != int(repeat) or not 1 <= repeat <= MAX_REPEAT):
            raise ValueError(f"{node['id']}: repeat must be an integer in 1-{MAX_REPEAT}")
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
    constant_elements = 0
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
            if not parents or (op not in ("Add", "Multiply", "Concat") and len(parents) != required_inputs): raise ValueError(f"{key}: invalid input count")
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
                dimensions = int(op[-2])
                expected_ranks = (2, 3) if op == "BatchNorm1d" else (dimensions + 2,)
                if len(shape) not in expected_ranks: raise ValueError(f"{key}: {op} requires {expected_ranks}D channel input")
                if op.startswith("BatchNorm"): count = shape[1] * 2
                elif p.get("affine", 1): count = shape[1] * 2
            elif op in ("LayerNorm", "RMSNorm"):
                normalized = p.get("normalized_shape")
                normalized = normalized if isinstance(normalized, list) else [normalized]
                if not normalized or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in normalized): raise ValueError(f"{key}: normalized_shape must be positive integers")
                if len(normalized) > len(shape) - 1 or shape[-len(normalized):] != normalized: raise ValueError(f"{key}: normalized_shape must match the input suffix")
                # RMSNorm has no bias and no mean centering, so it owns a single weight per
                # normalized element; LayerNorm owns weight and bias. Must mirror src/analysis.ts.
                if int(p.get("elementwise_affine", 1)): count = (2 if op == "LayerNorm" else 1) * prod(normalized)
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
            elif op in ("Slice", "Select"):
                dim = integer("dim", 1, -len(shape), len(shape) - 1) % len(shape)
                if dim == 0: raise ValueError(f"{key}: slicing/selecting the batch axis is not supported")
                if op == "Select":
                    index = integer("index", -1, -shape[dim], shape[dim] - 1)
                    shape.pop(dim)
                else:
                    bounds = []
                    for name in ("start", "end"):
                        raw = p.get(name, "none")
                        bounds.append(None if raw == "none" else integer(name, 0, -65536, 65536))
                    step = integer("step", 1)
                    start, end, step = slice(*bounds, step).indices(shape[dim])
                    size = len(range(start, end, step))
                    if size < 1: raise ValueError(f"{key}: slice output is empty")
                    shape[dim] = size
            elif op == "ConstantAdd":
                buffer_shape, values = p.get("shape"), p.get("values")
                if not shape or not isinstance(buffer_shape, list) or len(buffer_shape) > min(5, len(shape)) or any(type(v) is not int or not 1 <= v <= 65536 for v in buffer_shape):
                    raise ValueError(f"{key}: invalid constant shape")
                if not isinstance(values, list) or not 1 <= len(values) <= 65536 or prod(buffer_shape) != len(values) or any(type(v) not in (int, float) or not isfinite(v) or abs(v) > 3.4028234663852886e38 for v in values):
                    raise ValueError(f"{key}: invalid constant values")
                constant_elements += len(values)
                if constant_elements > 65536: raise ValueError(f"{key}: graph constant buffers exceed 65536 elements")
                offset = len(shape) - len(buffer_shape)
                aligned = [1] * offset + buffer_shape
                if aligned[0] != 1: raise ValueError(f"{key}: constants must broadcast across the batch axis")
                sequence_dim = integer("sequence_dim", 1, 1, len(shape) - 1) if "sequence_dim" in p else None
                if sequence_dim is not None and sequence_dim < offset: raise ValueError(f"{key}: sequence_dim is not present in the constant")
                for dim, (size, actual) in enumerate(zip(aligned, shape)):
                    if dim == sequence_dim:
                        if actual > size: raise ValueError(f"{key}: input sequence exceeds constant buffer capacity")
                    elif size not in (1, actual): raise ValueError(f"{key}: constant shape does not broadcast to input")
            elif op == "Bilinear":
                # in2_features is a weight dimension: it must equal the second
                # input's last dim, and it is also what nn.Bilinear is built
                # with, so inferring it from either side alone drifts.
                if len(shape) != 2: raise ValueError(f"{key}: Bilinear requires [B,F1]")
                if len(in_shapes[1]) != 2: raise ValueError(f"{key}: Bilinear requires two 2D inputs")
                if shape[0] != in_shapes[1][0]: raise ValueError(f"{key}: Bilinear inputs must share the batch dimension")
                other = integer("in2_features", 16); features = integer("out_features", 10)
                if in_shapes[1][1] != other: raise ValueError(f"{key}: in2_features must equal the second input's last dimension {in_shapes[1][1]}")
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
            elif op == "Multiply":
                if len(parents) < 2 or any(s != shape for s in in_shapes): raise ValueError(f"{key}: Multiply input shapes must match")
            elif op == "Concat":
                dim = integer("dim", 1, maximum=len(shape) - 1)
                if len(parents) < 2 or any(len(s) != len(shape) or any(v != shape[i] for i, v in enumerate(s) if i != dim) for s in in_shapes): raise ValueError(f"{key}: Concat dimensions must match")
                shape[dim] = sum(s[dim] for s in in_shapes)
            elif op == "Output" and outgoing[key]: raise ValueError("Output cannot have outgoing edges")
        # A folded layer stands for `repeat` isomorphic instances: shapes are unchanged, but
        # parameters and activation memory scale with the instance count. Chaining N instances
        # only executes when the op maps its input shape to itself.
        repeat = int(node.get("repeat", 1))
        if repeat > 1 and in_shapes and list(shape) != list(in_shapes[0]):
            raise ValueError(f"{key}: repeat > 1 requires a shape-preserving op")
        if prod(shape) > 16_000_000 or count * repeat > 50_000_000: raise ValueError(f"{key}: layer exceeds local memory limits")
        shapes[key] = shape; parameters[key] = count * repeat
        total += count * repeat; activation += prod(shape[1:]) * 4 * repeat
    if total > 50_000_000: raise ValueError("Model exceeds 50M parameters")
    if len({shapes[n["id"]][0] for n in inputs}) != 1: raise ValueError("All model inputs must have the same batch dimension")
    return {"order": order, "shapes": shapes, "portShapes": port_shapes, "baseEdges": base_edges, "overrideEdges": override_edges, "incoming": incoming, "parameters": parameters, "totalParameters": total, "activationBytesPerSample": activation, "input": inputs[0]["id"], "inputs": [n["id"] for n in inputs], "output": outputs[0]["id"], "nodes": nodes}


def analyze_graph(graph):
    import re
    if not isinstance(graph, dict) or graph.get("version") != 1:
        raise ValueError("Unsupported graph version")
    # The authored graph carries the strict contract: real node ids, real ops, and a
    # declared subgraph on every Group node. Expansion below derives ids such as
    # `g/0/attn`, which the id pattern deliberately does not admit, so the strict pass
    # has to happen on the authored graph rather than on the expanded one.
    authored_nodes, authored_edges = graph.get("nodes"), graph.get("edges")
    if not isinstance(authored_nodes, list) or not isinstance(authored_edges, list) or not 2 <= len(authored_nodes) <= 128 or len(authored_edges) > 512:
        raise ValueError("Graph must have 2-128 nodes and at most 512 edges")
    authored = {}
    for node in authored_nodes:
        if not isinstance(node, dict) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,63}", str(node.get("id", ""))) or node["id"] in authored or node.get("op") not in OPS or not isinstance(node.get("params"), dict):
            raise ValueError("Invalid node, operation or duplicate ID")
        if node["op"] == "Group":
            if not isinstance(node.get("subgraph"), str) or not node["subgraph"]:
                raise ValueError(f"{node['id']}: Group requires a subgraph name")
        elif node.get("subgraph") is not None:
            raise ValueError(f"{node['id']}: only Group nodes may declare a subgraph")
        authored[node["id"]] = node
    for edge in authored_edges:
        if not isinstance(edge, dict) or edge.get("source") not in authored or edge.get("target") not in authored:
            raise ValueError("Invalid or duplicate edge")

    return analyze_expanded_graph(expand_subgraphs(graph))

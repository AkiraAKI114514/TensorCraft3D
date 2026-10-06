"""Translate a bounded, static subset of PyTorch source into editable graphs.

No uploaded imports, constructors or forward methods are executed. Unsupported
semantics fail with a source location instead of being dropped from the graph.
"""
import ast
import copy
import math
import operator
from dataclasses import dataclass, field
from .graph import analyze_graph


class ImportIssue(ValueError):
    def __init__(self, message, node=None, code="UNSUPPORTED"):
        super().__init__(message)
        self.diagnostic = {"level": "error", "code": code, "message": message,
                           "line": getattr(node, "lineno", None), "column": getattr(node, "col_offset", 0) + 1}


@dataclass
class Module:
    kind: str
    params: dict = field(default_factory=dict)
    children: dict = field(default_factory=dict)
    expected: int | None = None
    node: object = None
    attrs: dict = field(default_factory=dict)
    constants: dict = field(default_factory=dict)
    forward: object = None


@dataclass(frozen=True)
class Tensor:
    node: str
    port: str | None = None


class Parser:
    def __init__(self, tree, input_shapes):
        self.tree = tree
        self.aliases = {"TensorLabAttention": "TensorLabAttention", "TensorLabTransformer": "TensorLabTransformer"}
        for statement in tree.body:
            if isinstance(statement, ast.Import):
                for name in statement.names:
                    self.aliases[name.asname or name.name] = name.name
            elif isinstance(statement, ast.ImportFrom) and statement.module:
                for name in statement.names:
                    self.aliases[name.asname or name.name] = f"{statement.module}.{name.name}"
        self.classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)
                        and any(self.path(b) in ("torch.nn.Module", "nn.Module") for b in n.bases)}
        # The embedded runtime in TensorLab exports is provided by our backend.
        for name in ("AttentionBranch", "TensorLabAttention", "TensorLabTransformer"):
            self.classes.pop(name, None)
        self.constants = {}
        for n in tree.body:
            if isinstance(n, (ast.Assign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        try: self.constants[target.id] = self.literal(n.value, self.constants)
                        except ImportIssue: pass
        self.input_shapes = input_shapes
        self.graph = {"version": 1, "name": "Imported PyTorch", "nodes": [], "edges": []}
        self.inputs = []
        self.node_sources = {}
        # Keep functional-operation descriptors alive too; object IDs can be reused.
        self.used_modules = {}
        self.constraints = []
        self.warnings = []
        self.calls = 0

    def path(self, node):
        if isinstance(node, ast.Name): return self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute): return f"{self.path(node.value)}.{node.attr}"
        return ""

    def literal(self, node, env):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, bool, type(None))): return node.value
        if isinstance(node, ast.Name) and node.id in env: return env[node.id]
        if isinstance(node, ast.Attribute) and self.path(node) in env: return env[self.path(node)]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.literal(n, env) for n in node.elts]
        if isinstance(node, ast.Dict):
            if any(k is None for k in node.keys): raise ImportIssue("不支持字典展开", node)
            return {self.literal(k, env): self.literal(v, env) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.literal(node.operand, env)
            if not isinstance(value, (int, float)): raise ImportIssue("需要数值常量", node)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and type(node.op) in (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Div):
            a, b = self.literal(node.left, env), self.literal(node.right, env)
            if type(a) not in (int, float) or type(b) not in (int, float): raise ImportIssue("参数运算只支持数值常量", node)
            try: value = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv, ast.Div: operator.truediv}[type(node.op)](a, b)
            except (ZeroDivisionError, OverflowError) as error: raise ImportIssue("参数运算无效", node) from error
            if not math.isfinite(value) or abs(value) > 1e9: raise ImportIssue("参数运算超过范围", node)
            return value
        raise ImportIssue("参数必须是常量、默认参数或可静态计算的数值表达式", node, "DYNAMIC_PARAMETER")

    def arguments(self, call, names, defaults, env):
        if len(call.args) > len(names) or any(isinstance(a, ast.Starred) for a in call.args):
            raise ImportIssue("不支持展开参数或过多位置参数", call)
        values = dict(defaults)
        given = set()
        for key, value in zip(names, call.args): values[key] = self.literal(value, env); given.add(key)
        for kw in call.keywords:
            if kw.arg not in names or kw.arg in given: raise ImportIssue(f"不支持或重复的参数：{kw.arg}", kw.value)
            values[kw.arg] = self.literal(kw.value, env); given.add(kw.arg)
        for key in names:
            if key not in values: raise ImportIssue(f"缺少构造参数 {key}", call)
        return values

    def require(self, args, expected, node):
        for key, value in expected.items():
            if args.get(key) != value: raise ImportIssue(f"当前模型不支持 {key}={args.get(key)!r}（需要 {value!r}）", node)

    def square(self, value, node):
        if isinstance(value, list):
            if len(value) != 2 or value[0] != value[1]: raise ImportIssue("目前仅支持两个方向相同的尺寸", node)
            value = value[0]
        return value

    def spatial_value(self, value, dimensions, node, default=None, allow_zero=False):
        if value is None and default is not None: value = default
        if isinstance(value, list):
            if len(value) != dimensions or any(type(v) is not int or v < (0 if allow_zero else 1) for v in value): raise ImportIssue("空间参数维度或数值无效", node)
            if len(set(value)) != 1: raise ImportIssue("目前仅支持各空间方向相同的尺寸", node)
            return value[0]
        if type(value) is not int or value < (0 if allow_zero else 1): raise ImportIssue("空间参数需要正整数", node)
        return value

    def module(self, expression, env, attrs=None, depth=0):
        if depth > 16: raise ImportIssue("模块嵌套超过 16 层", expression, "LIMIT")
        if attrs is not None:
            try:
                value = self.resolve_module(expression, attrs, env)
                if isinstance(value, Module): return value
            except ImportIssue: pass
        if not isinstance(expression, ast.Call): raise ImportIssue("需要 PyTorch 层构造表达式", expression)
        path = self.path(expression.func)
        kind = path.split(".")[-1]
        if path in self.classes:
            cls = self.classes[path]
            init = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
            forward = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"), None)
            if not init or not forward: raise ImportIssue("nn.Module 需要 __init__ 和 forward", cls)
            if init.args.vararg or init.args.kwarg or init.args.kwonlyargs: raise ImportIssue("构造函数暂不支持可变或仅关键字参数", init)
            names = [a.arg for a in init.args.args[1:]]
            defaults = {key: self.literal(value, env) for key, value in zip(names[len(names)-len(init.args.defaults):], init.args.defaults)}
            local = {**env, **self.arguments(expression, names, defaults, env)}
            module = Module("custom", node=cls, forward=forward, constants=local)
            for statement in init.body:
                if isinstance(statement, (ast.Pass, ast.Expr)):
                    if isinstance(statement, ast.Pass) or isinstance(statement.value, ast.Constant): continue
                    call = statement.value
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "__init__":
                        base = call.func.value
                        if isinstance(base, ast.Call) and self.path(base.func) == "super" and not base.args and not base.keywords and not call.args and not call.keywords: continue
                        if self.path(call.func) in ("torch.nn.Module.__init__", "nn.Module.__init__") and len(call.args) == 1 and isinstance(call.args[0], ast.Name) and call.args[0].id == "self" and not call.keywords: continue
                    raise ImportIssue("构造函数含有不支持的语句", statement)
                if not isinstance(statement, (ast.Assign, ast.AnnAssign)): raise ImportIssue("构造函数仅支持静态层和常量赋值", statement)
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                for target in targets:
                    key = self.path(target)
                    try: value = self.literal(statement.value, local)
                    except ImportIssue: value = self.module(statement.value, local, module.attrs, depth + 1)
                    if key.startswith("self."):
                        module.attrs[key] = value
                        if not isinstance(value, Module): local[key] = value
                    elif isinstance(target, ast.Name): local[key] = value
                    else: raise ImportIssue("不支持此赋值对象", target)
            return module
        if not path.startswith(("torch.nn.", "nn.")) and kind not in ("TensorLabAttention", "TensorLabTransformer"):
            raise ImportIssue(f"不支持自定义层 {path or ast.unparse(expression.func)}", expression)
        if kind in ("Sequential", "ModuleList", "ModuleDict"):
            if expression.keywords: raise ImportIssue("容器不支持关键字参数", expression)
            children = {}
            elements = expression.args
            if kind == "ModuleDict":
                if len(elements) != 1 or not isinstance(elements[0], ast.Dict): raise ImportIssue("ModuleDict 需要字面量字典", expression)
                for key, value in zip(elements[0].keys, elements[0].values):
                    children[str(self.literal(key, env))] = self.module(value, env, attrs, depth + 1)
            else:
                if kind == "ModuleList":
                    if len(elements) != 1 or not isinstance(elements[0], (ast.List, ast.Tuple)): raise ImportIssue("ModuleList 需要静态列表", expression)
                    elements = elements[0].elts
                if len(elements) == 1 and isinstance(elements[0], ast.Call) and self.path(elements[0].func) == "collections.OrderedDict":
                    ordered = elements[0].args
                    if len(ordered) != 1 or not isinstance(ordered[0], (ast.List, ast.Tuple)): raise ImportIssue("OrderedDict 需要静态层列表", expression)
                    for entry in ordered[0].elts:
                        if not isinstance(entry, (ast.List, ast.Tuple)) or len(entry.elts) != 2: raise ImportIssue("OrderedDict 需要 (名称, 层)", entry)
                        children[str(self.literal(entry.elts[0], env))] = self.module(entry.elts[1], env, attrs, depth + 1)
                else:
                    children = {str(i): self.module(value, env, attrs, depth + 1) for i, value in enumerate(elements)}
            return Module(kind, children=children, node=expression)
        specs = {
            "Conv1d": (["in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias", "padding_mode", "device", "dtype"], {"stride": 1, "padding": 0, "dilation": 1, "groups": 1, "bias": True, "padding_mode": "zeros", "device": None, "dtype": None}),
            "Conv2d": (["in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias", "padding_mode", "device", "dtype"], {"stride": 1, "padding": 0, "dilation": 1, "groups": 1, "bias": True, "padding_mode": "zeros", "device": None, "dtype": None}),
            "Conv3d": (["in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "bias", "padding_mode", "device", "dtype"], {"stride": 1, "padding": 0, "dilation": 1, "groups": 1, "bias": True, "padding_mode": "zeros", "device": None, "dtype": None}),
            "ConvTranspose1d": (["in_channels", "out_channels", "kernel_size", "stride", "padding", "output_padding", "groups", "bias", "dilation", "padding_mode", "device", "dtype"], {"stride": 1, "padding": 0, "output_padding": 0, "groups": 1, "bias": True, "dilation": 1, "padding_mode": "zeros", "device": None, "dtype": None}),
            "ConvTranspose2d": (["in_channels", "out_channels", "kernel_size", "stride", "padding", "output_padding", "groups", "bias", "dilation", "padding_mode", "device", "dtype"], {"stride": 1, "padding": 0, "output_padding": 0, "groups": 1, "bias": True, "dilation": 1, "padding_mode": "zeros", "device": None, "dtype": None}),
            "ConvTranspose3d": (["in_channels", "out_channels", "kernel_size", "stride", "padding", "output_padding", "groups", "bias", "dilation", "padding_mode", "device", "dtype"], {"stride": 1, "padding": 0, "output_padding": 0, "groups": 1, "bias": True, "dilation": 1, "padding_mode": "zeros", "device": None, "dtype": None}),
            "Linear": (["in_features", "out_features", "bias", "device", "dtype"], {"bias": True, "device": None, "dtype": None}),
            "Bilinear": (["in1_features", "in2_features", "out_features", "bias", "device", "dtype"], {"bias": True, "device": None, "dtype": None}),
            "LayerNorm": (["normalized_shape", "eps", "elementwise_affine", "bias", "device", "dtype"], {"eps": 1e-5, "elementwise_affine": True, "bias": True, "device": None, "dtype": None}),
            "GroupNorm": (["num_groups", "num_channels", "eps", "affine", "device", "dtype"], {"eps": 1e-5, "affine": True, "device": None, "dtype": None}),
            "InstanceNorm1d": (["num_features", "eps", "momentum", "affine", "track_running_stats", "device", "dtype"], {"eps": 1e-5, "momentum": 0.1, "affine": False, "track_running_stats": False, "device": None, "dtype": None}),
            "BatchNorm2d": (["num_features", "eps", "momentum", "affine", "track_running_stats", "device", "dtype"], {"eps": 1e-5, "momentum": 0.1, "affine": True, "track_running_stats": True, "device": None, "dtype": None}),
            "BatchNorm1d": (["num_features", "eps", "momentum", "affine", "track_running_stats", "device", "dtype"], {"eps": 1e-5, "momentum": 0.1, "affine": True, "track_running_stats": True, "device": None, "dtype": None}),
            "BatchNorm3d": (["num_features", "eps", "momentum", "affine", "track_running_stats", "device", "dtype"], {"eps": 1e-5, "momentum": 0.1, "affine": True, "track_running_stats": True, "device": None, "dtype": None}),
            "InstanceNorm2d": (["num_features", "eps", "momentum", "affine", "track_running_stats", "device", "dtype"], {"eps": 1e-5, "momentum": 0.1, "affine": False, "track_running_stats": False, "device": None, "dtype": None}),
            "InstanceNorm3d": (["num_features", "eps", "momentum", "affine", "track_running_stats", "device", "dtype"], {"eps": 1e-5, "momentum": 0.1, "affine": False, "track_running_stats": False, "device": None, "dtype": None}),
            "MaxPool1d": (["kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"], {"stride": None, "padding": 0, "dilation": 1, "return_indices": False, "ceil_mode": False}),
            "MaxPool2d": (["kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"], {"stride": None, "padding": 0, "dilation": 1, "return_indices": False, "ceil_mode": False}),
            "MaxPool3d": (["kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"], {"stride": None, "padding": 0, "dilation": 1, "return_indices": False, "ceil_mode": False}),
            "AvgPool1d": (["kernel_size", "stride", "padding", "ceil_mode", "count_include_pad"], {"stride": None, "padding": 0, "ceil_mode": False, "count_include_pad": True}),
            "AvgPool2d": (["kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"], {"stride": None, "padding": 0, "ceil_mode": False, "count_include_pad": True, "divisor_override": None}),
            "AvgPool3d": (["kernel_size", "stride", "padding", "ceil_mode", "count_include_pad"], {"stride": None, "padding": 0, "ceil_mode": False, "count_include_pad": True}),
            "AdaptiveAvgPool1d": (["output_size"], {}), "AdaptiveAvgPool2d": (["output_size"], {}), "AdaptiveAvgPool3d": (["output_size"], {}),
            "AdaptiveMaxPool1d": (["output_size", "return_indices"], {"return_indices": False}), "AdaptiveMaxPool2d": (["output_size", "return_indices"], {"return_indices": False}), "AdaptiveMaxPool3d": (["output_size", "return_indices"], {"return_indices": False}),
            "Flatten": (["start_dim", "end_dim"], {"start_dim": 1, "end_dim": -1}),
            "Dropout": (["p", "inplace"], {"p": 0.5, "inplace": False}), "Dropout1d": (["p", "inplace"], {"p": 0.5, "inplace": False}), "Dropout2d": (["p", "inplace"], {"p": 0.5, "inplace": False}), "Dropout3d": (["p", "inplace"], {"p": 0.5, "inplace": False}), "AlphaDropout": (["p", "inplace"], {"p": 0.5, "inplace": False}),
            "ReLU": (["inplace"], {"inplace": False}),
            "GELU": (["approximate"], {"approximate": "none"}),
            "Sigmoid": ([], {}), "Tanh": ([], {}), "SiLU": (["inplace"], {"inplace": False}), "SELU": (["inplace"], {"inplace": False}), "Hardsigmoid": (["inplace"], {"inplace": False}), "Hardswish": (["inplace"], {"inplace": False}), "Mish": (["inplace"], {"inplace": False}), "Softsign": ([], {}), "PReLU": (["num_parameters", "init"], {"num_parameters": 1, "init": 0.25}), "LeakyReLU": (["negative_slope", "inplace"], {"negative_slope": 0.01, "inplace": False}), "ELU": (["alpha", "inplace"], {"alpha": 1.0, "inplace": False}), "Softplus": (["beta", "threshold"], {"beta": 1, "threshold": 20}), "Softmax": (["dim"], {"dim": None}), "LogSoftmax": (["dim"], {"dim": None}), "Identity": ([], {}),
            "Embedding": (["num_embeddings", "embedding_dim", "padding_idx", "max_norm", "norm_type", "scale_grad_by_freq", "sparse", "_weight", "_freeze", "device", "dtype"], {"padding_idx": None, "max_norm": None, "norm_type": 2.0, "scale_grad_by_freq": False, "sparse": False, "_weight": None, "_freeze": False, "device": None, "dtype": None}),
            "Upsample": (["size", "scale_factor", "mode", "align_corners", "recompute_scale_factor", "antialias"], {"size": None, "scale_factor": None, "mode": "nearest", "align_corners": None, "recompute_scale_factor": None, "antialias": False}),
            "MultiheadAttention": (["embed_dim", "num_heads", "dropout", "bias", "add_bias_kv", "add_zero_attn", "kdim", "vdim", "batch_first", "device", "dtype"], {"dropout": 0.0, "bias": True, "add_bias_kv": False, "add_zero_attn": False, "kdim": None, "vdim": None, "batch_first": False, "device": None, "dtype": None}),
            "TransformerEncoderLayer": (["d_model", "nhead", "dim_feedforward", "dropout", "activation", "layer_norm_eps", "batch_first", "norm_first", "bias", "device", "dtype"], {"dim_feedforward": 2048, "dropout": 0.1, "activation": "relu", "layer_norm_eps": 1e-5, "batch_first": False, "norm_first": False, "bias": True, "device": None, "dtype": None}),
            "TensorLabAttention": (["embed_dim", "num_heads", "kv_heads", "dropout", "attention_type", "branches"], {"num_heads": 1, "kv_heads": None, "dropout": 0.1, "attention_type": "self", "branches": 1}),
            "TensorLabTransformer": (["embed_dim", "num_heads", "kv_heads", "ff_dim", "dropout", "attention_type", "branches", "norm_first", "activation"], {"num_heads": 1, "kv_heads": None, "ff_dim": 128, "dropout": 0.1, "attention_type": "self", "branches": 1, "norm_first": True, "activation": "gelu"}),
        }
        if kind == "TransformerEncoder":
            if not expression.args: raise ImportIssue("TransformerEncoder 缺少编码器层", expression)
            layer = self.module(expression.args[0], env, attrs, depth + 1)
            rest = ast.Call(func=expression.func, args=expression.args[1:], keywords=expression.keywords)
            args = self.arguments(rest, ["num_layers", "norm", "enable_nested_tensor", "mask_check"], {"norm": None, "enable_nested_tensor": True, "mask_check": True}, env)
            self.require(args, {"norm": None}, expression)
            count = args["num_layers"]
            if type(count) is not int or not 1 <= count <= 64: raise ImportIssue("编码器层数需要在 1–64 之间", expression)
            if layer.kind != "Transformer": raise ImportIssue("TransformerEncoder 需要 TransformerEncoderLayer", expression)
            return Module("Sequential", children={str(i): copy.deepcopy(layer) for i in range(count)}, node=expression)
        if kind not in specs: raise ImportIssue(f"当前不支持层 {kind}", expression, "UNSUPPORTED_LAYER")
        names, defaults = specs[kind]
        args = self.arguments(expression, names, defaults, env)
        self.require(args, {k: None for k in ("device", "dtype") if k in args}, expression)
        params, expected = {}, None
        if kind in ("Conv1d", "Conv2d", "Conv3d"):
            self.require(args, {"dilation": 1, "groups": 1, "bias": True, "padding_mode": "zeros"}, expression)
            expected = args["in_channels"]
            dims = int(kind[-2]); params = {key: self.spatial_value(args[key], dims, expression, args["kernel_size"] if key == "stride" and args[key] is None else None) for key in ("out_channels", "kernel_size", "stride", "padding")}
        elif kind in ("ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"):
            self.require(args, {"dilation": 1, "groups": 1, "bias": True, "padding_mode": "zeros"}, expression)
            expected = args["in_channels"]
            dims = int(kind[-2]); params = {key: self.spatial_value(args[key], dims, expression, args["kernel_size"] if key == "stride" and args[key] is None else None, key in ("padding", "output_padding")) for key in ("out_channels", "kernel_size", "stride", "padding", "output_padding")}
        elif kind == "Linear":
            self.require(args, {"bias": True}, expression); expected = args["in_features"]; params = {"out_features": args["out_features"]}
        elif kind == "Bilinear":
            self.require(args, {"bias": True}, expression); expected = args["in1_features"]; params = {"in2_features": args["in2_features"], "out_features": args["out_features"]}
        elif kind == "LayerNorm":
            self.require(args, {"bias": True}, expression)
            shape = args["normalized_shape"]
            if type(shape) is int: normalized = shape
            elif isinstance(shape, list) and shape and all(type(v) is int and v > 0 for v in shape): normalized = shape
            else: raise ImportIssue("LayerNorm normalized_shape 必须是正整数或静态列表", expression)
            params = {"normalized_shape": normalized, "eps": args["eps"], "elementwise_affine": int(bool(args["elementwise_affine"]))}; expected = normalized[-1] if isinstance(normalized, list) else normalized
        elif kind == "GroupNorm":
            self.require(args, {"eps": 1e-5, "affine": True}, expression); expected = args["num_channels"]; params = {"num_groups": args["num_groups"]}
        elif kind.startswith("BatchNorm") or kind.startswith("InstanceNorm"):
            self.require(args, {"eps": 1e-5, "momentum": 0.1, "affine": True if kind.startswith("BatchNorm") else False, "track_running_stats": True if kind.startswith("BatchNorm") else False}, expression); expected = args["num_features"]; params = {"affine": int(bool(args["affine"]))}
        elif kind.startswith("MaxPool"):
            self.require(args, {"dilation": 1, "return_indices": False, "ceil_mode": False}, expression)
            dims = int(kind[-2]); params = {"kernel_size": self.spatial_value(args["kernel_size"], dims, expression), "stride": self.spatial_value(args["stride"], dims, expression, args["kernel_size"]), "padding": self.spatial_value(args["padding"], dims, expression, 0, True)}
        elif kind.startswith("AvgPool"):
            dims = int(kind[-2]); params = {"kernel_size": self.spatial_value(args["kernel_size"], dims, expression), "stride": self.spatial_value(args["stride"], dims, expression, args["kernel_size"]), "padding": self.spatial_value(args["padding"], dims, expression, 0, True)}
        elif kind.startswith("Adaptive"):
            dims = int(kind[-2]); params = {"output_size": args["output_size"] if isinstance(args["output_size"], int) else [self.spatial_value(args["output_size"], dims, expression)] * dims}
        elif kind == "Flatten": self.require(args, {"start_dim": 1, "end_dim": -1}, expression)
        elif kind in ("ReLU", "SiLU", "SELU", "Hardsigmoid", "Hardswish", "Mish", "LeakyReLU", "ELU", "Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "AlphaDropout"):
            self.require(args, {"inplace": False}, expression)
            if kind in ("Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "AlphaDropout"): params = {"p": args["p"]}
            elif kind == "LeakyReLU": params = {"negative_slope": args["negative_slope"]}
            elif kind == "ELU": params = {"alpha": args["alpha"]}
        elif kind == "GELU": self.require(args, {"approximate": "none"}, expression)
        elif kind in ("Sigmoid", "Tanh", "Softsign", "Identity"): pass
        elif kind == "PReLU":
            params = {"num_parameters": args["num_parameters"], "init": args["init"]}
        elif kind == "Softplus": params = {"beta": args["beta"], "threshold": args["threshold"]}
        elif kind in ("Softmax", "LogSoftmax"): params = {"dim": args["dim"] if args["dim"] is not None else -1}
        elif kind == "Embedding":
            self.require(args, {"padding_idx": None, "max_norm": None, "scale_grad_by_freq": False, "sparse": False, "_weight": None, "_freeze": False, "device": None, "dtype": None}, expression); params = {"num_embeddings": args["num_embeddings"], "embedding_dim": args["embedding_dim"]}; expected = None
        elif kind == "Upsample":
            self.require(args, {"size": None, "align_corners": None, "recompute_scale_factor": None, "antialias": False}, expression)
            if args["scale_factor"] is None: raise ImportIssue("Upsample 目前需要静态 scale_factor", expression)
            params = {"scale_factor": args["scale_factor"] if isinstance(args["scale_factor"], list) else args["scale_factor"], "mode": args["mode"]}
        elif kind in ("MultiheadAttention", "TransformerEncoderLayer"):
            self.require(args, {"batch_first": True, "bias": True}, expression)
            if kind == "MultiheadAttention":
                self.require(args, {"add_bias_kv": False, "add_zero_attn": False}, expression)
                for key in ("kdim", "vdim"):
                    if args[key] not in (None, args["embed_dim"]): raise ImportIssue("K/V 的维度需要等于 embed_dim", expression)
                expected = args["embed_dim"]; heads = args["num_heads"]; kind = "MultiHeadAttention"
            else:
                self.require(args, {"layer_norm_eps": 1e-5}, expression)
                if args["activation"] not in ("relu", "gelu"): raise ImportIssue("编码器激活只支持 relu/gelu", expression)
                expected = args["d_model"]; heads = args["nhead"]; kind = "Transformer"
                params.update(ff_dim=args["dim_feedforward"], norm_first=int(bool(args["norm_first"])), activation=args["activation"])
            params.update(embed_dim=expected, num_heads=heads, kv_heads=heads, attention_type="self", branches=1, dropout=args["dropout"])
        elif kind in ("TensorLabAttention", "TensorLabTransformer"):
            expected = args["embed_dim"]; params = args.copy()
            if params["kv_heads"] is None: params["kv_heads"] = 1 if params["attention_type"] == "multi_query" else params["num_heads"]
            if kind == "TensorLabTransformer": params["norm_first"] = int(bool(params["norm_first"]))
            kind = "Transformer" if kind == "TensorLabTransformer" else "MultiHeadAttention"
        return Module(kind, params=params, expected=expected, node=expression)

    def resolve_module(self, expression, attrs, env):
        if isinstance(expression, ast.Name) and isinstance(env.get(expression.id), Module): return env[expression.id]
        if self.path(expression) in attrs: return attrs[self.path(expression)]
        if isinstance(expression, ast.Subscript):
            parent = self.resolve_module(expression.value, attrs, env)
            key = str(self.literal(expression.slice, env))
            if isinstance(parent, Module) and key in parent.children: return parent.children[key]
        raise ImportIssue("无法识别调用的模块", expression)

    def add(self, kind, params, sources, name, node=None):
        if len(self.graph["nodes"]) >= 128: raise ImportIssue("导入模型超过 128 个节点", node, "LIMIT")
        key = f"import_{len(self.graph['nodes'])}"
        self.node_sources[key] = node
        self.graph["nodes"].append({"id": key, "name": name[:120], "op": kind, "params": params.copy(), "position": {"x": 0, "y": 0}})
        for source, target_port in sources:
            if not isinstance(source, Tensor): raise ImportIssue("此处需要张量", node)
            edge = {"id": f"edge_{len(self.graph['edges'])}", "source": source.node, "target": key}
            if source.port: edge["sourcePort"] = source.port
            if target_port: edge["targetPort"] = target_port
            if len(self.graph["edges"]) >= 512: raise ImportIssue("导入模型超过 512 条连线", node, "LIMIT")
            self.graph["edges"].append(edge)
        return Tensor(key)

    def input(self, name):
        shape = self.input_shapes.get(name)
        if shape is not None and (not isinstance(shape, list) or len(shape) not in (2, 3, 4, 5) or any(type(v) is not int or not 1 <= v <= 65536 for v in shape)):
            raise ImportIssue(f"输入 {name} 需要 [B,F]、[B,S,E]、[B,C,H,W] 或 [B,C,D,H,W]", code="INPUT_SHAPE")
        tensor = self.add("Input", {"shape": shape or []}, [], name)
        self.inputs.append({"name": name, "id": tensor.node, "inferred": shape is None})
        return tensor

    def hint(self, tensor, module):
        if not module.expected or not isinstance(tensor, Tensor) or tensor.port: return
        node = next(n for n in self.graph["nodes"] if n["id"] == tensor.node)
        if node["op"] in ("ReLU", "GELU", "Sigmoid", "Tanh", "SiLU", "LeakyReLU", "ELU", "SELU", "Softplus", "Softmax", "LogSoftmax", "PReLU", "Hardsigmoid", "Hardswish", "Mish", "Softsign", "Identity", "Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "AlphaDropout", "Add"):
            for edge in self.graph["edges"]:
                if edge["target"] == node["id"]: self.hint(Tensor(edge["source"], edge.get("sourcePort")), module)
        if node["op"] != "Input" or node["params"]["shape"]: return
        if module.kind in ("Conv1d", "ConvTranspose1d", "BatchNorm1d", "InstanceNorm1d"): shape = [1, module.expected, 32]
        elif module.kind in ("Conv2d", "ConvTranspose2d", "BatchNorm2d", "InstanceNorm2d"): shape = [1, module.expected, 32, 32]
        elif module.kind in ("Conv3d", "ConvTranspose3d", "BatchNorm3d", "InstanceNorm3d"): shape = [1, module.expected, 8, 16, 16]
        elif module.kind in ("Transformer", "MultiHeadAttention"): shape = [1, 8, module.expected]
        else: shape = [1, module.expected]
        node["params"]["shape"] = shape

    def apply(self, module, args, keywords, name, call):
        self.calls += 1
        if self.calls > 512: raise ImportIssue("模型调用次数超过限制", call, "LIMIT")
        if module.kind == "custom":
            method = module.forward
            if method.args.vararg or method.args.kwarg or method.args.kwonlyargs: raise ImportIssue("forward 暂不支持可变或仅关键字参数", method)
            names = [a.arg for a in method.args.args[1:]]
            local = module.constants.copy()
            if len(args) != len(names) or keywords: raise ImportIssue("forward 输入数量不匹配", call)
            local.update(zip(names, args))
            return self.statements(method.body, local, module.attrs)
        if module.kind == "Sequential":
            if len(args) != 1 or keywords: raise ImportIssue("Sequential 需要单个输入", call)
            value = args[0]
            for key, child in module.children.items(): value = self.apply(child, [value], {}, f"{name}.{key}", call)
            return value
        if module.kind in ("ModuleDict", "ModuleList"): raise ImportIssue("容器需选取具体层或静态遍历", call)
        if id(module) in self.used_modules and module.kind in ("Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "Linear", "Bilinear", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LayerNorm", "GroupNorm", "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d", "Embedding", "PReLU", "Transformer", "MultiHeadAttention"):
            raise ImportIssue("同一参数层被多次调用涉及权重共享，目前无法忠实转换", call, "SHARED_WEIGHTS")
        self.used_modules[id(module)] = module
        params = module.params.copy()
        standard = False
        sources = []
        if module.kind in ("MultiHeadAttention", "Transformer"):
            standard = module.node and self.path(module.node.func).endswith("MultiheadAttention")
            if standard:
                names = ["query", "key", "value", "key_padding_mask", "need_weights", "attn_mask", "average_attn_weights", "is_causal"]
                if len(args) > len(names): raise ImportIssue("注意力参数过多", call)
                opts = dict(zip(names, args)); opts.update(keywords)
                if any(k not in names for k in opts): raise ImportIssue("未知注意力参数", call)
                for key, default in (("key_padding_mask", None), ("attn_mask", None), ("is_causal", False)):
                    if opts.get(key, default) != default: raise ImportIssue(f"目前不支持注意力 {key}", call)
                if not all(isinstance(opts.get(k), Tensor) for k in ("query", "key", "value")): raise ImportIssue("注意力需要 Q、K、V 三个张量", call)
                q, k, v = (opts[n] for n in ("query", "key", "value"))
                params["attention_type"] = "self" if q == k == v else "cross"
                sources = [(q, "query" if params["attention_type"] == "cross" else None)]
                if params["attention_type"] == "cross": sources.append((k, "context"))
                if k != v:
                    sources += [(v, f"b0:v{i}") for i in range(params["kv_heads"])]
                for t in (q, k, v): self.hint(t, module)
            else:
                permitted = {"overrides"} if self.path(getattr(call, "func", None)).endswith("forward_with_ports") else set()
                if set(keywords) - permitted: raise ImportIssue("不支持掩码或额外编码器参数", call)
                base = [a for a in args if a is not None]
                cross = params.get("attention_type") == "cross"
                if len(base) != (2 if cross else 1): raise ImportIssue("注意力基础输入数量不匹配", call)
                sources = [(base[0], "query" if cross else None)] + ([(base[1], "context")] if cross else [])
                for t in base: self.hint(t, module)
                overrides = keywords.get("overrides", {})
                if not isinstance(overrides, dict): raise ImportIssue("overrides 需要静态字典", call)
                sources += [(value, key) for key, value in overrides.items()]
                for t in overrides.values(): self.hint(t, module)
        else:
            required = 2 if module.kind == "Bilinear" else 1
            if len(args) != required or keywords: raise ImportIssue(f"{module.kind} 需要 {required} 个张量输入", call)
            for tensor in args: self.hint(tensor, module)
            sources = [(tensor, None) for tensor in args]
        result = self.add(module.kind, params, sources, name, call)
        if module.expected is not None: self.constraints.append((result.node, module.expected, module.kind, call))
        if standard:
            return (result, "attention_weights_unsupported")
        if self.path(getattr(call, "func", None)).endswith("forward_with_ports"):
            ports = {f"b{b}:{role}{i}": Tensor(result.node, f"b{b}:{role}{i}") for b in range(params.get("branches", 1)) for role in "qkv" for i in range(params["num_heads"] if role == "q" else params["kv_heads"])}
            return result, ports
        return result

    def shape_operation(self, call, env, attrs, method=False):
        kind = call.func.attr if method else self.path(call.func).split(".")[-1]
        names = ["dim"] if method else ["input", "dim"]
        if len(call.args) > len(names) or any(isinstance(arg, ast.Starred) for arg in call.args):
            raise ImportIssue(f"{kind} 参数数量无效或包含展开参数", call)
        arguments = dict(zip(names, call.args))
        for keyword in call.keywords:
            if keyword.arg not in names or keyword.arg in arguments:
                raise ImportIssue(f"{kind} 不支持或重复的参数：{keyword.arg}", keyword.value)
            arguments[keyword.arg] = keyword.value
        if not method and "input" not in arguments:
            raise ImportIssue(f"{kind} 缺少输入张量", call)
        if kind == "unsqueeze" and "dim" not in arguments:
            raise ImportIssue("unsqueeze 缺少 dim 参数", call)
        dim = self.literal(arguments["dim"], env) if "dim" in arguments else "all"
        if kind == "unsqueeze":
            if type(dim) is not int: raise ImportIssue("unsqueeze 的 dim 必须是整数常量", arguments["dim"])
        elif "dim" in arguments and not (type(dim) is int or isinstance(dim, list) and all(type(value) is int for value in dim)):
            raise ImportIssue("squeeze 的 dim 必须是整数或整数元组/列表", arguments["dim"])
        source = self.expr(call.func.value if method else arguments["input"], env, attrs)
        op = "Unsqueeze" if kind == "unsqueeze" else "Squeeze"
        return self.add(op, {"dim": dim}, [(source, None)], op, call)

    def expr(self, expression, env, attrs):
        if isinstance(expression, ast.Name) and expression.id in env: return env[expression.id]
        if isinstance(expression, ast.Constant): return self.literal(expression, env)
        if isinstance(expression, (ast.List, ast.Tuple)): return [self.expr(n, env, attrs) for n in expression.elts]
        if isinstance(expression, ast.Dict):
            if any(k is None for k in expression.keys): raise ImportIssue("不支持字典展开", expression)
            return {self.literal(k, env): self.expr(v, env, attrs) for k, v in zip(expression.keys, expression.values)}
        if isinstance(expression, ast.Subscript):
            parent = self.expr(expression.value, env, attrs); key = self.literal(expression.slice, env)
            try: value = parent[key]
            except (TypeError, KeyError, IndexError) as error: raise ImportIssue("不支持张量切片或未知索引", expression) from error
            if value == "attention_weights_unsupported": raise ImportIssue("注意力权重输出暂不支持", expression)
            return value
        if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
            return self.add("Add", {}, [(self.expr(expression.left, env, attrs), None), (self.expr(expression.right, env, attrs), None)], "残差相加", expression)
        if isinstance(expression, ast.Call):
            function = expression.func
            module_expression = function.value if isinstance(function, ast.Attribute) and function.attr in ("forward", "forward_with_ports") else function
            try: module = self.resolve_module(module_expression, attrs, env)
            except ImportIssue: module = None
            if isinstance(module, Module):
                args = [self.expr(n, env, attrs) for n in expression.args]
                if any(k.arg is None for k in expression.keywords): raise ImportIssue("不支持参数展开", expression)
                keywords = {k.arg: self.expr(k.value, env, attrs) for k in expression.keywords}
                return self.apply(module, args, keywords, ast.unparse(module_expression), expression)
            path = self.path(function)
            if path in ("torch.cat", "torch.concat"):
                if not expression.args: raise ImportIssue("cat 缺少张量列表", expression)
                tensors = self.expr(expression.args[0], env, attrs)
                if not isinstance(tensors, (list, tuple)): raise ImportIssue("cat 需要静态张量列表", expression)
                rest = ast.Call(func=function, args=expression.args[1:], keywords=expression.keywords)
                args = self.arguments(rest, ["dim"], {"dim": 0}, env)
                return self.add("Concat", args, [(t, None) for t in tensors], "拼接", expression)
            functional_paths = {
                "torch.flatten": "Flatten", "torch.relu": "ReLU", "torch.sigmoid": "Sigmoid", "torch.tanh": "Tanh",
                "torch.nn.functional.relu": "ReLU", "torch.nn.functional.gelu": "GELU", "torch.nn.functional.silu": "SiLU", "torch.nn.functional.sigmoid": "Sigmoid", "torch.nn.functional.tanh": "Tanh", "torch.nn.functional.hardsigmoid": "Hardsigmoid", "torch.nn.functional.hardswish": "Hardswish", "torch.nn.functional.mish": "Mish", "torch.nn.functional.softsign": "Softsign", "torch.nn.functional.leaky_relu": "LeakyReLU", "torch.nn.functional.elu": "ELU", "torch.nn.functional.selu": "SELU", "torch.nn.functional.softplus": "Softplus", "torch.nn.functional.softmax": "Softmax", "torch.nn.functional.log_softmax": "LogSoftmax",
                "torch.nn.functional.dropout": "Dropout", "torch.nn.functional.max_pool1d": "MaxPool1d", "torch.nn.functional.max_pool2d": "MaxPool2d", "torch.nn.functional.max_pool3d": "MaxPool3d", "torch.nn.functional.avg_pool1d": "AvgPool1d", "torch.nn.functional.avg_pool2d": "AvgPool2d", "torch.nn.functional.avg_pool3d": "AvgPool3d", "torch.nn.functional.adaptive_avg_pool1d": "AdaptiveAvgPool1d", "torch.nn.functional.adaptive_avg_pool2d": "AdaptiveAvgPool2d", "torch.nn.functional.adaptive_avg_pool3d": "AdaptiveAvgPool3d", "torch.nn.functional.adaptive_max_pool1d": "AdaptiveMaxPool1d", "torch.nn.functional.adaptive_max_pool2d": "AdaptiveMaxPool2d", "torch.nn.functional.adaptive_max_pool3d": "AdaptiveMaxPool3d",
            }
            if path in ("torch.unsqueeze", "torch.squeeze"):
                return self.shape_operation(expression, env, attrs)
            if path in functional_paths:
                if not expression.args: raise ImportIssue("函数缺少输入", expression)
                source = self.expr(expression.args[0], env, attrs)
                kind = functional_paths[path]
                kwargs = list(expression.keywords)
                if kind == "Flatten" and len(expression.args) == 1 and not any(k.arg == "start_dim" for k in kwargs):
                    raise ImportIssue("torch.flatten 默认会展平 batch，请使用 torch.flatten(x, 1)", expression)
                if kind == "Dropout":
                    training = next((k for k in kwargs if k.arg == "training"), None)
                    if training is None or self.path(training.value) != "self.training": raise ImportIssue("F.dropout 需要 training=self.training，才能保持训练/推理语义", expression)
                    kwargs.remove(training)
                fake = ast.Call(func=ast.Attribute(value=ast.Name(id="nn"), attr=kind), args=expression.args[1:], keywords=kwargs)
                ast.copy_location(fake, expression)
                return self.apply(self.module(fake, env), [source], {}, kind, expression)
            if isinstance(function, ast.Attribute) and function.attr == "flatten":
                source = self.expr(function.value, env, attrs)
                fake = ast.Call(func=ast.Attribute(value=ast.Name(id="nn"), attr="Flatten"), args=expression.args, keywords=expression.keywords)
                ast.copy_location(fake, expression)
                # Tensor.flatten starts at dimension zero by default.
                if not fake.args and not any(k.arg == "start_dim" for k in fake.keywords): raise ImportIssue("请使用 flatten(1)，保留批次维度", expression)
                return self.apply(self.module(fake, env), [source], {}, "Flatten", expression)
            if isinstance(function, ast.Attribute) and function.attr in ("unsqueeze", "squeeze"):
                return self.shape_operation(expression, env, attrs, method=True)
            if isinstance(function, ast.Attribute) and function.attr in ("view", "reshape"):
                # A very common flatten spelling; arbitrary reshapes require a new op.
                dimensions = expression.args
                if len(dimensions) == 1 and isinstance(dimensions[0], (ast.List, ast.Tuple)): dimensions = dimensions[0].elts
                batch = dimensions[0] if len(dimensions) == 2 else None
                batch_ok = isinstance(batch, ast.Call) and isinstance(batch.func, ast.Attribute) and batch.func.attr == "size" and ast.dump(batch.func.value) == ast.dump(function.value) and len(batch.args) == 1 and self.literal(batch.args[0], env) == 0
                batch_ok = batch_ok or isinstance(batch, ast.Subscript) and isinstance(batch.value, ast.Attribute) and batch.value.attr == "shape" and ast.dump(batch.value.value) == ast.dump(function.value) and self.literal(batch.slice, env) == 0
                if expression.keywords or not batch_ok or self.literal(dimensions[1], env) != -1: raise ImportIssue("reshape/view 目前仅支持保留 batch 的展平，如 x.view(x.size(0), -1)", expression)
                return self.add("Flatten", {}, [(self.expr(function.value, env, attrs), None)], "Flatten", expression)
            raise ImportIssue(f"不支持操作 {path or ast.unparse(function)}", expression, "UNSUPPORTED_OPERATION")
        try: return self.literal(expression, env)
        except ImportIssue: raise ImportIssue(f"不支持表达式 {ast.unparse(expression)[:100]}", expression)

    def assign(self, target, value, env):
        if isinstance(target, ast.Name): env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (tuple, list)) and len(target.elts) == len(value):
            for item, child in zip(target.elts, value): self.assign(item, child, env)
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and isinstance(env.get(target.value.id), dict):
            env[target.value.id][self.literal(target.slice, env)] = value
        else: raise ImportIssue("不支持此赋值（只支持变量、解包或中间字典）", target)

    def statements(self, statements, env, attrs, exported=False):
        for statement in statements:
            if isinstance(statement, (ast.Pass, ast.Expr)) and (isinstance(statement, ast.Pass) or isinstance(statement.value, ast.Constant)): continue
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                if exported and isinstance(statement.value, ast.IfExp):
                    conditional = statement.value
                    # Only the known generated dictionary/tensor input guard is accepted.
                    if ast.unparse(conditional.test) != "isinstance(x, dict)": raise ImportIssue("不支持动态条件表达式", conditional)
                    value = self.expr(conditional.body, env, attrs)
                else: value = self.expr(statement.value, env, attrs)
                for target in targets: self.assign(target, value, env)
            elif isinstance(statement, ast.Return): return self.expr(statement.value, env, attrs)
            elif isinstance(statement, ast.For):
                if statement.orelse: raise ImportIssue("不支持 for/else", statement)
                container = self.resolve_module(statement.iter, attrs, env)
                if not isinstance(container, Module) or container.kind not in ("ModuleList", "Sequential"): raise ImportIssue("循环只支持静态层容器", statement)
                for module in container.children.values():
                    self.assign(statement.target, module, env)
                    if any(isinstance(s, ast.Return) for s in ast.walk(statement)): raise ImportIssue("循环内不支持 return", statement)
                    self.statements(statement.body, env, attrs)
            elif exported and isinstance(statement, ast.If):
                # Export input validation is checked separately against our template.
                expected = 'if isinstance(x, dict):\n    if set(x) != set(input_ids):\n        raise ValueError("Input dictionary must contain exactly the model\'s Input node IDs")\nelif len(input_ids) != 1:\n    raise ValueError("Multi-input models require a dictionary keyed by Input node ID")'
                if ast.dump(statement, include_attributes=False) != ast.dump(ast.parse(expected).body[0], include_attributes=False): raise ImportIssue("导出代码输入检查已改变，不支持此控制流", statement)
            else: raise ImportIssue("不支持动态控制流、原地操作或此语句", statement, "UNSUPPORTED_STATEMENT")
        return None

    def candidates(self):
        names = list(self.classes)
        for n in self.tree.body:
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and self.path(n.value.func).endswith("Sequential"):
                names += [t.id for t in n.targets if isinstance(t, ast.Name)]
        return names

    def build(self, selected):
        candidates = self.candidates()
        selected = selected or self.default_model(candidates)
        if selected not in candidates: raise ImportIssue("未找到可导入的 nn.Module 类或 nn.Sequential 对象", code="MODEL_NOT_FOUND")
        self.graph["name"] = selected
        if selected in self.classes:
            expression = ast.Call(func=ast.Name(id=selected), args=[], keywords=[])
            # Literal constructor arguments on top-level model instantiation override defaults.
            for n in self.tree.body:
                if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and self.path(n.value.func) == selected: expression = n.value; break
            module = self.module(expression, self.constants)
            args = [a.arg for a in module.forward.args.args[1:]]
            exported = selected == "VisualModel" and "self.layers" in module.attrs and module.attrs["self.layers"].kind == "ModuleDict" and any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "values" for t in n.targets) for n in module.forward.body)
            if exported:
                shapes = self.constants.get("_TENSORLAB_INPUT_SHAPES", {})
                if not shapes:
                    # Earlier TensorLab exports include an explicit input_ids list.
                    assignment = next((n for n in module.forward.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "input_ids" for t in n.targets)), None)
                    if assignment: shapes = {key: None for key in self.literal(assignment.value, {})}
                if not isinstance(shapes, dict) or not 1 <= len(shapes) <= 8: raise ImportIssue("无法确定导出模型的输入，请提供输入形状", code="INPUT_SHAPE")
                for key, shape in shapes.items():
                    if key not in self.input_shapes and shape: self.input_shapes[key] = shape
                inputs = {key: self.input(key) for key in shapes}
                env = {**module.constants, "x": inputs}
                output = self.statements(module.forward.body, env, module.attrs, exported=True)
            else:
                if not 1 <= len(args) <= 8: raise ImportIssue("需要 1–8 个 forward 输入", module.forward)
                inputs = [self.input(name) for name in args]
                output = self.apply(module, inputs, {}, selected, module.forward)
        else:
            assignment = next(n for n in self.tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == selected for t in n.targets))
            module = self.module(assignment.value, self.constants)
            output = self.apply(module, [self.input("x")], {}, selected, assignment)
        if not isinstance(output, Tensor): raise ImportIssue("当前需要单个张量输出；注意力返回值请解包为 output, _", module.node)
        self.add("Output", {}, [(output, None)], "输出")
        registered = []
        def collect(current):
            if current.kind == "custom":
                for child in current.attrs.values():
                    if isinstance(child, Module): collect(child)
            elif current.children:
                for child in current.children.values(): collect(child)
            else: registered.append(current)
        collect(module)
        unused = {id(m) for m in registered if id(m) not in self.used_modules}
        if unused: self.warnings.append({"level": "warning", "code": "UNUSED_MODULES", "message": f"有 {len(unused)} 个声明的层未在 forward 中调用，没有加入计算图。"})
        for input_info in self.inputs:
            node = next(n for n in self.graph["nodes"] if n["id"] == input_info["id"])
            if not node["params"]["shape"]: node["params"]["shape"] = [1, 4]
            input_info["shape"] = node["params"]["shape"]
            if input_info["inferred"]:
                self.warnings.append({"level": "warning", "code": "INFERRED_SHAPE", "message": f"输入 {input_info['name']} 使用推断形状 {input_info['shape']}，请确认批次、序列长度或图像尺寸。"})
        try: analysis = analyze_graph(self.graph)
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            source = self.node_sources.get(str(error).split(":", 1)[0])
            raise ImportIssue(f"结构或输入形状错误：{error}。请修改输入形状后重新解析。", source, code="STRUCTURE") from error
        for key, expected, kind, source in self.constraints:
            edge = analysis["baseEdges"][key][0]
            shape = analysis["portShapes"][edge["source"]][edge["sourcePort"]] if edge.get("sourcePort") else analysis["shapes"][edge["source"]]
            actual = shape[-1] if kind in ("Transformer", "MultiHeadAttention", "LayerNorm") else shape[1]
            if actual != expected: raise ImportIssue(f"{kind} 声明输入维度 {expected}，实际为 {actual}；请检查输入形状或代码", source, "SHAPE_MISMATCH")
        depths, rows = {}, {}
        for key in analysis["order"]:
            depths[key] = max([depths[e["source"]] + 1 for e in self.graph["edges"] if e["target"] == key] or [0])
            row = rows.get(depths[key], 0); rows[depths[key]] = row + 1
            next(n for n in self.graph["nodes"] if n["id"] == key)["position"] = {"x": 210 * depths[key], "y": 100 + 170 * row}
        self.warnings.append({"level": "info", "code": "STRUCTURE_ONLY", "message": "已导入模型结构和参数配置；原模型权重、训练脚本及优化器不随源码导入。"})
        return selected, analysis

    def default_model(self, candidates):
        if "VisualModel" in candidates: return "VisualModel"
        for n in self.tree.body:
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) and self.path(n.value.func) in candidates: return self.path(n.value.func)
        return candidates[-1] if candidates else None


def import_pytorch(source, model_name=None, input_shapes=None):
    result = {"graph": None, "models": [], "model": model_name, "inputs": [], "diagnostics": [], "analysis": None}
    parser = None
    try:
        if not isinstance(source, str) or len(source.encode("utf-8")) > 512_000: raise ImportIssue("代码不能超过 512 KB", code="LIMIT")
        if not source.strip(): raise ImportIssue("请粘贴或上传 PyTorch 代码", code="EMPTY_SOURCE")
        tree = ast.parse(source)
        nodes = list(ast.walk(tree))
        if len(nodes) > 30000: raise ImportIssue("源码语法节点超过 30000，请仅导入模型定义", code="LIMIT")
        parser = Parser(tree, dict(input_shapes or {}))
        result["models"] = parser.candidates()
        result["model"] = model_name or parser.default_model(result["models"])
        selected, analysis = parser.build(model_name)
        result.update(graph=parser.graph, model=selected, analysis=analysis, diagnostics=parser.warnings)
    except SyntaxError as error:
        result["diagnostics"] = [{"level": "error", "code": "SYNTAX", "message": f"Python 语法错误：{error.msg}", "line": error.lineno, "column": error.offset}]
    except ImportIssue as error: result["diagnostics"] = [error.diagnostic]
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, RecursionError, OverflowError) as error:
        result["diagnostics"] = [{"level": "error", "code": "PARSE", "message": f"无法静态解析：{type(error).__name__}；请简化为常量参数与静态 forward。"}]
    if parser:
        result["inputs"] = [{**i, "shape": next(n["params"].get("shape", []) for n in parser.graph["nodes"] if n["id"] == i["id"])} for i in parser.inputs]
    return result

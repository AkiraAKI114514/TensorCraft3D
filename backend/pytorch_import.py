"""Translate a bounded, static subset of PyTorch source into editable graphs.

No uploaded imports, constructors or forward methods are executed. Unsupported
semantics fail with a source location instead of being dropped from the graph.
"""
import ast
import copy
import math
import operator
from dataclasses import dataclass, field
from .graph import MAX_REPEAT, EXPAND_SEP, analyze_graph, einsum_shape
from .static_tensors import DynamicStaticSlice, StaticTensor, StaticTensorError, f32, new_budget


def unparse(node):
    return ast.unparse(node) if node is not None else ""


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
    # A container whose children are repeat `count` independent, chained, isomorphic
    # instances. This is exactly the `repeat` fold the graph IR already understands:
    # walking the container once produces one `Group` node + one `SubgraphDef` carrying
    # `count`, instead of `count` flattened copies. That is what lets a 48-block model
    # fit the 128-node budget. Set only when every instance really is identical and the
    # instances chain into each other (see Parser.repeat_container).
    count: int | None = None


@dataclass(frozen=True)
class Tensor:
    node: str
    port: str | None = None


class Parser:
    def __init__(self, tree, input_shapes):
        self.tree = tree
        self.aliases = {name: name for name in ("TensorLabAttention", "TensorLabTransformer", "TensorLabConstantAdd")}
        for statement in tree.body:
            if isinstance(statement, ast.Import):
                for name in statement.names:
                    self.aliases[name.asname or name.name] = name.name
            elif isinstance(statement, ast.ImportFrom) and statement.module:
                for name in statement.names:
                    self.aliases[name.asname or name.name] = f"{statement.module}.{name.name}"
        self.classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)
                        and any(self.path(b) in ("torch.nn.Module", "nn.Module") for b in n.bases)}
        self.functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        # The embedded runtime in TensorLab exports is provided by our backend.
        for name in ("AttentionBranch", "TensorLabAttention", "TensorLabTransformer", "TensorLabConstantAdd"):
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
        # A Group node stands for a whole chain, so the body's final node is not reachable
        # through `self.graph["edges"]`; these pairs carry it to the fold in `build`.
        self.group_counts = {}
        # ids of the nodes each Group application built; the fold moves them into a subgraph.
        self.group_bodies = {}
        # flat node id -> expanded id the analyzer reports, for constraint checks after folding.
        self.group_map = {}
        # Counts Group applications, so a loop can tell whether index 0 covered all instances.
        self.group_applies = 0
        self.constraints = []
        self.warnings = []
        self.calls = 0
        self.static_budget = new_budget()

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

    def comprehension_count(self, node, env):
        """A `[layer for _ in range(N)]` list comprehension reduces to just its count.

        Only the generated shape this importer itself emits is accepted: a bound variable,
        an unused element expression, `range(N)` and no conditionals. The element is parsed
        by the caller, which is why it is returned rather than built here.
        """
        if len(node.generators) != 1: raise ImportIssue("列表推导式只支持单个 for 子句", node)
        generator = node.generators[0]
        if generator.is_async or generator.ifs: raise ImportIssue("列表推导式不支持 async 或 if 子句", node)
        if not isinstance(generator.target, ast.Name): raise ImportIssue("列表推导式的循环变量必须是简单名称", node)
        # Every instance has to be identical, so the element may not depend on the loop
        # variable. This is the check that makes `[Block(8) for _ in range(N)]` safe to fold.
        if any(isinstance(inner, ast.Name) and inner.id == generator.target.id for inner in ast.walk(node.elt)):
            raise ImportIssue("列表推导式的元素不能依赖循环变量，每个实例必须完全一致", node)
        iterator = generator.iter
        if not isinstance(iterator, ast.Call) or self.path(iterator.func) != "range" or len(iterator.args) not in (1, 2) or iterator.keywords:
            raise ImportIssue("列表推导式只支持 range(N) 或 range(a, b)", iterator)
        bounds = [self.literal(argument, env) for argument in iterator.args]
        if any(type(value) is not int for value in bounds): raise ImportIssue("range 的边界必须是整数常量", iterator)
        start, stop = (0, bounds[0]) if len(bounds) == 1 else (bounds[0], bounds[1])
        return stop - start

    def repeat_container(self, kind, expression, env, attrs, elements, count):
        """Parse one written instance and, only if it is safely repeatable, mark it as `count`."""
        if count < 1 or count > MAX_REPEAT: raise ImportIssue(f"层实例数量需要在 1–{MAX_REPEAT} 之间", expression)
        if count == 1: return Module(kind, children={"0": self.module(elements[0], env, attrs, 1)}, node=expression)
        try: unit = self.module(elements[0], env, attrs, 1)
        except ImportIssue: return Module(kind, children={str(i): self.module(element, env, attrs, 1) for i, element in enumerate(elements)}, node=expression)
        if self.repeatable(unit): return Module("Group", count=count, children={"0": unit}, node=expression)
        # Not provably identical, so the instances stay distinct modules. When the source only
        # wrote one instance there is nothing to fall back to; say why instead of dropping it.
        if len(elements) == 1:
            raise ImportIssue("该层无法确认 N 个实例完全一致（可能依赖 __init__ 中的可变状态或按序号生成），无法折叠为结构块；请显式写出每个实例或改为固定配置", expression)
        return Module(kind, children={str(i): self.module(element, env, attrs, 1) for i, element in enumerate(elements)}, node=expression)

    def repeatable(self, module):
        """True when one written instance stands in for N isomorphic copies.

        Custom modules carry their own `__init__`/`forward` source, so their internal
        qualifiers (the `_` of `for _ in range(N)`) are skipped and every child must itself
        be repeatable. Built-in constructors always build the same layer for the same
        arguments, so only the container case needs recursion.
        """
        if module.kind == "custom":
            return all(self.repeatable(child) for child in module.attrs.values() if isinstance(child, Module))
        if module.kind in ("Sequential", "ModuleList", "ModuleDict"):
            if module.kind == "ModuleDict": return False
            return all(self.repeatable(child) for child in module.children.values())
        if module.kind == "Group": return bool(module.children) and all(self.repeatable(child) for child in module.children.values())
        return True

    def same_module(self, left, right):
        """Structural equality of two parsed modules: same kind, params, and children.

        Used to confirm that written-out repeats really are identical copies. AST nodes are
        ignored because positions differ; a mismatch only costs the fold (the instances stay
        separate), never a wrong grouping.
        """
        if left.kind != right.kind or left.params != right.params or left.expected != right.expected:
            return False
        if left.kind == "custom":
            if unparse(left.forward) != unparse(right.forward): return False
            if set(left.attrs) != set(right.attrs): return False
            for key, value in left.attrs.items():
                other = right.attrs[key]
                if isinstance(value, Module) or isinstance(other, Module):
                    if not isinstance(value, Module) or not isinstance(other, Module) or not self.same_module(value, other): return False
                elif value != other: return False
            return True
        if set(left.children) != set(right.children): return False
        return all(self.same_module(left.children[key], right.children[key]) for key in left.children)

    def referenced(self, statements, target):
        """How many times a `for …` body mentions its loop variable.

        Both `for blk in blocks: x = blk(x)` and `for i in range(N): x = blocks[i](x)` must
        resolve to exactly one node when the container is folded, so the count has to be 1;
        a body that would need several distinct instances is rejected rather than misread.
        """
        return sum(isinstance(node, ast.Name) and node.id == target.id for statement in statements for node in ast.walk(statement))

    def range_count(self, node, env):
        """`range(N)` / `range(a, b)` as a positive instance count, or None."""
        if not isinstance(node, ast.Call) or self.path(node.func) != "range" or len(node.args) not in (1, 2) or node.keywords: return None
        try: bounds = [self.literal(argument, env) for argument in node.args]
        except ImportIssue: return None
        if any(type(value) is not int for value in bounds): return None
        start, stop = (0, bounds[0]) if len(bounds) == 1 else (bounds[0], bounds[1])
        count = stop - start
        return count if 1 <= count <= MAX_REPEAT else None

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
        if path in self.functions:
            function = self.functions[path]
            if function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
                raise ImportIssue("辅助函数暂不支持可变或仅关键字参数", function)
            names = [a.arg for a in function.args.args]
            defaults = {key: self.literal(value, env) for key, value in zip(names[len(names)-len(function.args.defaults):], function.args.defaults)}
            local = {**env, **self.arguments(expression, names, defaults, env)}
            for statement in function.body:
                if isinstance(statement, (ast.Pass, ast.Expr)) and (isinstance(statement, ast.Pass) or isinstance(statement.value, ast.Constant)):
                    continue
                if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                    try:
                        value = self.expr(statement.value, local, attrs)
                    except ImportIssue:
                        value = self.module(statement.value, local, attrs, depth + 1)
                    for target in targets:
                        if not isinstance(target, ast.Name): raise ImportIssue("辅助函数只支持局部变量赋值", target)
                        local[target.id] = value
                    continue
                if isinstance(statement, ast.Return):
                    if statement.value is None: raise ImportIssue("辅助函数必须返回模块", statement)
                    return self.module(statement.value, local, attrs, depth + 1)
                raise ImportIssue("辅助函数只支持静态层赋值和 return", statement)
            raise ImportIssue("辅助函数缺少 return", function)
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
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "register_buffer":
                        if not isinstance(call.func.value, ast.Name) or call.func.value.id != "self" or len(call.args) not in (2, 3) or any(k.arg not in ("persistent",) for k in call.keywords):
                            raise ImportIssue("register_buffer 只支持 self、名称、静态缓冲区和 persistent", statement)
                        name = self.literal(call.args[0], local)
                        if not isinstance(name, str) or not name or any(k.arg == "persistent" for k in call.keywords) and len(call.args) > 2:
                            raise ImportIssue("register_buffer 名称或 persistent 参数无效", statement)
                        value = self.expr(call.args[1], local, module.attrs)
                        if isinstance(value, DynamicStaticSlice): raise ImportIssue("register_buffer 不支持运行时切片", call.args[1])
                        if not isinstance(value, StaticTensor): raise ImportIssue("register_buffer 需要静态缓冲区", call.args[1])
                        module.attrs[f"self.{name}"] = value
                        local[f"self.{name}"] = value
                        continue
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "__init__":
                        base = call.func.value
                        if isinstance(base, ast.Call) and self.path(base.func) == "super" and not base.args and not base.keywords and not call.args and not call.keywords: continue
                        if self.path(call.func) in ("torch.nn.Module.__init__", "nn.Module.__init__") and len(call.args) == 1 and isinstance(call.args[0], ast.Name) and call.args[0].id == "self" and not call.keywords: continue
                    raise ImportIssue("构造函数含有不支持的语句", statement)
                if not isinstance(statement, (ast.Assign, ast.AnnAssign)): raise ImportIssue("构造函数仅支持静态层和常量赋值", statement)
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                for target in targets:
                    if isinstance(target, ast.Subscript):
                        base = self.expr(target.value, local, module.attrs)
                        if isinstance(base, StaticTensor):
                            try:
                                indices = self.static_indices(target.slice, local, base.shape, target)
                                value = self.expr(statement.value, local, module.attrs)
                                if not isinstance(value, StaticTensor): raise ImportIssue("静态缓冲区赋值需要静态张量", statement.value)
                                base.assign(indices, value)
                            except StaticTensorError as error: raise ImportIssue(str(error), target) from error
                            continue
                        raise ImportIssue("只支持对静态缓冲区做索引赋值", target)
                    key = self.path(target)
                    try: value = self.literal(statement.value, local)
                    except ImportIssue:
                        try: value = self.expr(statement.value, local, module.attrs)
                        except ImportIssue: value = self.module(statement.value, local, module.attrs, depth + 1)
                    if key.startswith("self."):
                        module.attrs[key] = value
                        if not isinstance(value, Module): local[key] = value
                    elif isinstance(target, ast.Name): local[key] = value
                    else: raise ImportIssue("不支持此赋值对象", target)
            return module
        if path == "TensorLabConstantAdd":
            args = self.arguments(expression, ["shape", "values", "sequence_dim"], {"sequence_dim": None}, env)
            params = {"shape": args["shape"], "values": args["values"]}
            if args["sequence_dim"] is not None: params["sequence_dim"] = args["sequence_dim"]
            return Module("ConstantAdd", params=params, node=expression)
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
                    if len(elements) == 1 and isinstance(elements[0], ast.ListComp):
                        # `[Block(...) for _ in range(N)]` is the idiomatic way to write N
                        # identical layers; it carries N by construction.
                        count = self.comprehension_count(elements[0], env)
                        return self.repeat_container(kind, expression, env, attrs, [elements[0].elt], count)
                    if len(elements) != 1 or not isinstance(elements[0], (ast.List, ast.Tuple)): raise ImportIssue("ModuleList 需要静态列表", expression)
                    elements = elements[0].elts
                if len(elements) == 1 and isinstance(elements[0], ast.Call) and self.path(elements[0].func) == "collections.OrderedDict":
                    ordered = elements[0].args
                    if len(ordered) != 1 or not isinstance(ordered[0], (ast.List, ast.Tuple)): raise ImportIssue("OrderedDict 需要静态层列表", expression)
                    for entry in ordered[0].elts:
                        if not isinstance(entry, (ast.List, ast.Tuple)) or len(entry.elts) != 2: raise ImportIssue("OrderedDict 需要 (名称, 层)", entry)
                        children[str(self.literal(entry.elts[0], env))] = self.module(entry.elts[1], env, attrs, depth + 1)
                else:
                    # Written-out instances are folded when they are provably identical copies,
                    # by parsing the first and checking it against the rest. This is what turns
                    # `ModuleList([Block(), Block(), Block()])` into one instance with repeat=3,
                    # which the 128-node import budget depends on.
                    if len(elements) > 1:
                        try: unit = self.module(elements[0], env, attrs, depth + 1)
                        except ImportIssue: unit = None
                        if unit is not None and self.repeatable(unit):
                            clones = True
                            for element in elements[1:]:
                                try: other = self.module(element, env, attrs, depth + 1)
                                except ImportIssue: clones = False; break
                                if not self.same_module(unit, other): clones = False; break
                            if clones: return Module("Group", count=len(elements), children={"0": unit}, node=expression)
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
            # nn.RMSNorm needs torch >= 2.4 (the training requirements pin >= 2.6). It has no
            # bias and does not center the mean, so `bias` is absent here and the analyzer
            # counts a single weight per normalized element.
            "RMSNorm": (["normalized_shape", "eps", "elementwise_affine", "device", "dtype"], {"eps": None, "elementwise_affine": True, "device": None, "dtype": None}),
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
            dims = int(kind[-2]); params = {key: self.spatial_value(args[key], dims, expression, args["kernel_size"] if key == "stride" and args[key] is None else None, key == "padding") for key in ("out_channels", "kernel_size", "stride", "padding")}
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
        elif kind == "RMSNorm":
            # Same normalized_shape contract as LayerNorm; the difference is only the parameter
            # set (weight with no bias). nn.RMSNorm's default eps is dtype-dependent (1e-5 for
            # fp16, 1e-6 otherwise), so it is resolved against the recorded dtype here rather
            # than guessed, keeping the graph's parameter count and behavior faithful.
            shape = args["normalized_shape"]
            if type(shape) is int: normalized = shape
            elif isinstance(shape, list) and shape and all(type(v) is int and v > 0 for v in shape): normalized = shape
            else: raise ImportIssue("RMSNorm normalized_shape 必须是正整数或静态列表", expression)
            eps = args["eps"]
            if eps is None: eps = 1e-5 if "float16" in str(args["dtype"]) else 1e-6
            params = {"normalized_shape": normalized, "eps": eps, "elementwise_affine": int(bool(args["elementwise_affine"]))}; expected = normalized[-1] if isinstance(normalized, list) else normalized
        elif kind == "GroupNorm":
            self.require(args, {"eps": 1e-5, "affine": True}, expression); expected = args["num_channels"]; params = {"num_groups": args["num_groups"]}
        elif kind.startswith("BatchNorm") or kind.startswith("InstanceNorm"):
            self.require(args, {"eps": 1e-5, "momentum": 0.1, "affine": True if kind.startswith("BatchNorm") else False, "track_running_stats": True if kind.startswith("BatchNorm") else False}, expression); expected = args["num_features"]; params = {"affine": int(bool(args["affine"]))}
        elif kind.startswith("MaxPool"):
            self.require(args, {"dilation": 1, "return_indices": False, "ceil_mode": False}, expression)
            dims = int(kind[-2]); params = {"kernel_size": self.spatial_value(args["kernel_size"], dims, expression), "stride": self.spatial_value(args["stride"], dims, expression, args["kernel_size"]), "padding": self.spatial_value(args["padding"], dims, expression, 0, True)}
        elif kind.startswith("AvgPool"):
            self.require(args, {"ceil_mode": False, "count_include_pad": True, "divisor_override": None}, expression)
            dims = int(kind[-2]); params = {"kernel_size": self.spatial_value(args["kernel_size"], dims, expression), "stride": self.spatial_value(args["stride"], dims, expression, args["kernel_size"]), "padding": self.spatial_value(args["padding"], dims, expression, 0, True)}
        elif kind.startswith("Adaptive"):
            if kind.startswith("AdaptiveMaxPool"): self.require(args, {"return_indices": False}, expression)
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
            if isinstance(parent, Module) and parent.kind == "Group" and parent.count:
                # `blocks[index]` selects one of `count` identical instances. The Group node
                # stands for the whole chain, so the index only has to be in range.
                ordinal = self.literal(expression.slice, env)
                if type(ordinal) is not int or not 0 <= ordinal < parent.count:
                    raise ImportIssue("结构块实例下标越界", expression)
                return parent
            key = self.literal(expression.slice, env)
            if isinstance(parent, Module) and key in parent.children: return parent.children[key]
            # `for i in range(n): self.blocks[i]` yields an integer, not a module; the call
            # site treats it as index 0 of the container, which the fold stands for.
            if isinstance(parent, Module) and type(key) is int and parent.kind in ("ModuleList", "Sequential"): return 0
            raise ImportIssue(f"容器下标 {key!r} 不在 {parent.kind} 中", expression)
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
        # Shape-preserving ops and the folded-block instance pass their declared dimension
        # upstream; a `Group` node is transparent, so a body's expectation reaches the Input.
        if node["op"] in ("ReLU", "GELU", "Sigmoid", "Tanh", "SiLU", "LeakyReLU", "ELU", "SELU", "Softplus", "Softmax", "LogSoftmax", "PReLU", "Hardsigmoid", "Hardswish", "Mish", "Softsign", "Identity", "Dropout", "Dropout1d", "Dropout2d", "Dropout3d", "AlphaDropout", "Add", "Multiply", "Group"):
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
        if module.kind == "Group":
            # A container marked as `count` identical instances. The body is replayed once per
            # instance to keep forward semantics exact, but every replay resolves to this one
            # node, so the chain is emitted once and split afterwards.
            if len(args) != 1 or keywords: raise ImportIssue("结构块实例需要单个输入", call)
            identifier = self.add("Group", {}, [(args[0], None)], name, call)
            self.group_counts[identifier.node] = module.count
            self.group_applies += 1
            # The body is exactly what the replay appended; recording it here avoids having to
            # infer the body from the finished graph.
            before = len(self.graph["nodes"])
            unit = module.children["0"]
            try: value = self.apply(unit, [Tensor(identifier.node)], {}, f"{name}.0", call)
            except ImportIssue as error:
                error.diagnostic.setdefault("occurrences", module.count)
                raise
            self.group_bodies[identifier.node] = [candidate["id"] for candidate in self.graph["nodes"][before:]]
            return identifier
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
            for i, tensor in enumerate(args):
                self.hint(tensor, Module("Bilinear", expected=params["in2_features"]) if module.kind == "Bilinear" and i == 1 else module)
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
        # Shape operations on compile-time buffers stay compile-time.
        if isinstance(source, StaticTensor):
            try: return source.unsqueeze(dim) if kind == "unsqueeze" else source.squeeze(dim)
            except StaticTensorError as error: raise ImportIssue(str(error), call) from error
        op = "Unsqueeze" if kind == "unsqueeze" else "Squeeze"
        return self.add(op, {"dim": dim}, [(source, None)], op, call)

    def index_items(self, node, rank, source, allow_newaxis=False):
        items = list(node.elts) if isinstance(node, ast.Tuple) else [node]
        ellipses = [i for i, item in enumerate(items) if isinstance(item, ast.Constant) and item.value is Ellipsis]
        newaxis = lambda item: isinstance(item, ast.Constant) and item.value is None
        if len(items) > 10 or len(ellipses) > 1 or any(isinstance(item, ast.Starred) for item in items):
            raise ImportIssue("不支持索引展开或多个 Ellipsis", source)
        if not allow_newaxis and any(newaxis(item) for item in items): raise ImportIssue("静态缓冲区索引不支持 None", source)
        consumed = sum(not newaxis(item) and not (isinstance(item, ast.Constant) and item.value is Ellipsis) for item in items)
        if consumed > rank: raise ImportIssue("张量索引维度不匹配", source)
        full = lambda: ast.Slice(lower=None, upper=None, step=None)
        if ellipses:
            pos = ellipses[0]; items = items[:pos] + [full() for _ in range(rank - consumed)] + items[pos + 1:]
        else: items += [full() for _ in range(rank - consumed)]
        return items

    def size_reference(self, node, env, attrs):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "size" and len(node.args) == 1 and not node.keywords:
            target, dimension = node.func.value, node.args[0]
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "shape":
            target, dimension = node.value.value, node.slice
        else: return None
        try:
            tensor = self.expr(target, env, attrs)
            dim = self.literal(dimension, env)
        except ImportIssue: return None
        if not isinstance(tensor, Tensor) or type(dim) is not int: return None
        shape = self.tensor_shape(tensor, node)
        if not -len(shape) <= dim < len(shape): raise ImportIssue("size/shape 维度越界", node)
        return tensor, dim % len(shape)

    def dynamic_static_slice(self, parent, node, env, attrs, source):
        items = self.index_items(node, len(parent.shape), source)
        references = [(dim, self.size_reference(item.upper, env, attrs)) for dim, item in enumerate(items) if isinstance(item, ast.Slice)]
        references = [(dim, ref) for dim, ref in references if ref is not None]
        if not references: return None
        if len(references) != 1: raise ImportIssue("常量缓冲区只支持一个动态序列前缀", source)
        buffer_dim, (tensor, sequence_dim) = references[0]
        for dim, item in enumerate(items):
            if not isinstance(item, ast.Slice): raise ImportIssue("动态常量前缀不能混用整数索引", source)
            start = None if item.lower is None else self.literal(item.lower, env)
            step = 1 if item.step is None else self.literal(item.step, env)
            if start not in (None, 0) or type(step) is not int or step != 1 or (dim != buffer_dim and item.upper is not None):
                raise ImportIssue("动态常量仅支持完整前缀，其余维度必须完整保留", source)
        if sequence_dim == 0: raise ImportIssue("禁止将常量序列切片映射到 batch 维", source)
        return DynamicStaticSlice(parent, buffer_dim, sequence_dim, tensor)

    def static_indices(self, node, env, shape, source):
        result = []
        for item in self.index_items(node, len(shape), source):
            if isinstance(item, ast.Slice):
                bounds = [None if value is None else self.expr(value, env, {}) for value in (item.lower, item.upper, item.step)]
                if any(value is not None and type(value) is not int for value in bounds) or bounds[2] is not None and bounds[2] <= 0:
                    raise ImportIssue("静态切片需要整数边界和正整数步长", source)
                result.append(slice(*bounds))
            else:
                value = self.literal(item, env)
                if type(value) is not int: raise ImportIssue("静态索引必须是整数或切片", source)
                result.append(value)
        return result

    def dynamic_index(self, parent, node, env, attrs, source):
        rank = len(self.tensor_shape(parent, source))
        items = self.index_items(node, rank, source, allow_newaxis=True)
        result, dim = parent, 0
        for item in items:
            if isinstance(item, ast.Constant) and item.value is None:
                if dim == 0: raise ImportIssue("不能在 batch 维度前插入轴", source)
                result = self.add("Unsqueeze", {"dim": dim}, [(result, None)], "Unsqueeze", source); dim += 1
            elif isinstance(item, ast.Slice):
                start = None if item.lower is None else self.literal(item.lower, env)
                step = 1 if item.step is None else self.literal(item.step, env)
                reference = self.size_reference(item.upper, env, attrs)
                if reference is not None:
                    if reference != (parent, dim) or start not in (None, 0) or step != 1:
                        raise ImportIssue("张量动态切片只支持同一张量同一维度的完整前缀", source)
                    end = None
                else: end = None if item.upper is None else self.literal(item.upper, env)
                if any(value is not None and type(value) is not int for value in (start, end)) or type(step) is not int or step <= 0:
                    raise ImportIssue("切片需要整数边界和正整数步长", source)
                if dim == 0 and (start is not None or end is not None or step != 1): raise ImportIssue("禁止修改 batch 维度", source)
                if start is not None or end is not None or step != 1:
                    result = self.add("Slice", {"dim": dim, "start": "none" if start is None else start, "end": "none" if end is None else end, "step": step}, [(result, None)], "Slice", source)
                dim += 1
            else:
                index = self.literal(item, env)
                if type(index) is not int: raise ImportIssue("不支持高级索引；索引必须是整数常量", source)
                if dim == 0: raise ImportIssue("禁止修改 batch 维度", source)
                result = self.add("Select", {"dim": dim, "index": index}, [(result, None)], "Select", source)
        return result

    def tensor_shape(self, tensor, source=None):
        if not isinstance(tensor, Tensor): raise ImportIssue("需要运行时张量", source)
        ancestors, pending = set(), [tensor.node]
        while pending:
            key = pending.pop()
            if key in ancestors: continue
            ancestors.add(key); pending.extend(e["source"] for e in self.graph["edges"] if e["target"] == key)
        nodes = [n for n in self.graph["nodes"] if n["id"] in ancestors]
        if any(n["op"] == "Input" and not n["params"].get("shape") for n in nodes):
            raise ImportIssue("索引或位置编码需要明确的输入形状，请填写后重新解析", source, "INPUT_SHAPE")
        edges = [e for e in self.graph["edges"] if e["target"] in ancestors]
        output = {"id": "import_shape_output", "op": "Output", "params": {}}
        edge = {"source": tensor.node, "target": output["id"]}
        if tensor.port: edge["sourcePort"] = tensor.port
        try:
            info = analyze_graph({"version": 1, "nodes": nodes + [output], "edges": edges + [edge]})
            return info["shapes"][output["id"]]
        except (ValueError, TypeError, KeyError) as error:
            raise ImportIssue(f"无法推断张量形状：{error}", source, "STRUCTURE") from error

    def expr(self, expression, env, attrs):
        if isinstance(expression, ast.Name) and expression.id in env: return env[expression.id]
        if isinstance(expression, ast.Attribute):
            path = self.path(expression)
            if path in (attrs or {}): return attrs[path]
            if path in ("torch.float32", "torch.float"):
                return "torch.float32"
            source = self.expr(expression.value, env, attrs)
            if isinstance(source, StaticTensor) and expression.attr == "shape": return list(source.shape)
            raise ImportIssue(f"不支持属性 {expression.attr}", expression)
        if isinstance(expression, ast.Constant): return self.literal(expression, env)
        if isinstance(expression, (ast.List, ast.Tuple)): return [self.expr(n, env, attrs) for n in expression.elts]
        if isinstance(expression, ast.Dict):
            if any(k is None for k in expression.keys): raise ImportIssue("不支持字典展开", expression)
            return {self.literal(k, env): self.expr(v, env, attrs) for k, v in zip(expression.keys, expression.values)}
        if isinstance(expression, ast.Subscript):
            parent = self.expr(expression.value, env, attrs)
            if isinstance(parent, StaticTensor):
                dynamic = self.dynamic_static_slice(parent, expression.slice, env, attrs, expression)
                if dynamic is not None: return dynamic
                try: return parent.index(self.static_indices(expression.slice, env, parent.shape, expression))
                except StaticTensorError as error: raise ImportIssue(str(error), expression) from error
            if isinstance(parent, Tensor):
                return self.dynamic_index(parent, expression.slice, env, attrs, expression)
            key = self.literal(expression.slice, env)
            try: value = parent[key]
            except (TypeError, KeyError, IndexError) as error: raise ImportIssue("未知索引或不支持此切片", expression) from error
            if value == "attention_weights_unsupported": raise ImportIssue("注意力权重输出暂不支持", expression)
            return value
        if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, (ast.USub, ast.UAdd)):
            value = self.expr(expression.operand, env, attrs)
            if isinstance(value, StaticTensor):
                try: return value.unary(lambda v: -v if isinstance(expression.op, ast.USub) else v)
                except StaticTensorError as error: raise ImportIssue(str(error), expression) from error
            if isinstance(value, DynamicStaticSlice):
                raise ImportIssue("动态静态缓冲区不支持一元运算", expression)
            if type(value) in (int, float): return -value if isinstance(expression.op, ast.USub) else value
            raise ImportIssue("静态一元运算需要数值或缓冲区", expression)
        if isinstance(expression, ast.BinOp):
            left, right = self.expr(expression.left, env, attrs), self.expr(expression.right, env, attrs)
            if isinstance(expression.op, ast.Add) and ((isinstance(left, (StaticTensor, DynamicStaticSlice)) and isinstance(right, Tensor)) or (isinstance(right, (StaticTensor, DynamicStaticSlice)) and isinstance(left, Tensor))):
                constant = left if isinstance(left, (StaticTensor, DynamicStaticSlice)) else right
                tensor = right if constant is left else left
                buffer = constant.tensor if isinstance(constant, DynamicStaticSlice) else constant
                params = {"shape": list(buffer.shape), "values": list(buffer.values)}
                if isinstance(constant, DynamicStaticSlice):
                    input_shape = self.tensor_shape(tensor, expression)
                    if constant.source != tensor or constant.buffer_dim + len(input_shape) - len(buffer.shape) != constant.sequence_dim:
                        raise ImportIssue("动态常量前缀必须引用相加输入的同一序列维度", expression)
                    params["sequence_dim"] = constant.sequence_dim
                return self.add("ConstantAdd", params, [(tensor, None)], "ConstantAdd", expression)
            if isinstance(left, DynamicStaticSlice) or isinstance(right, DynamicStaticSlice):
                raise ImportIssue("动态常量前缀只支持与对应运行时张量相加", expression)
            if isinstance(left, StaticTensor) or isinstance(right, StaticTensor):
                operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv}
                fn = operations.get(type(expression.op))
                if fn is None: raise ImportIssue("不支持此静态张量运算", expression)
                try:
                    return left.binary(right, fn) if isinstance(left, StaticTensor) else right.binary(left, lambda a, b: fn(b, a))
                except (StaticTensorError, ValueError, ZeroDivisionError, OverflowError) as error:
                    raise ImportIssue(f"静态张量运算无效：{error}", expression) from error
            if type(left) in (int, float) and type(right) in (int, float):
                try:
                    value = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv, ast.Div: operator.truediv}[type(expression.op)](left, right)
                    if not math.isfinite(value) or abs(value) > 1e9: raise ValueError
                    return value
                except (KeyError, ZeroDivisionError, OverflowError, ValueError) as error:
                    raise ImportIssue("静态参数运算无效", expression) from error
            if isinstance(expression.op, ast.Add) and isinstance(left, Tensor) and isinstance(right, Tensor):
                return self.add("Add", {}, [(left, None), (right, None)], "残差相加", expression)
            if isinstance(expression.op, ast.Mult) and isinstance(left, Tensor) and isinstance(right, Tensor):
                # Element-wise gating (`gate * value`), the shape of a SwiGLU/GLU feed-forward.
                return self.add("Multiply", {}, [(left, None), (right, None)], "逐元素相乘", expression)
            raise ImportIssue("不支持此张量运算", expression)
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
            if path in {"math.log", "math.exp", "math.sqrt", "math.sin", "math.cos"}:
                values = [self.expr(arg, env, attrs) for arg in expression.args]
                if expression.keywords or len(values) != 1 or type(values[0]) not in (int, float): raise ImportIssue("math 函数需要单个静态数值", expression)
                fn = {"math.log": math.log, "math.exp": math.exp, "math.sqrt": math.sqrt, "math.sin": math.sin, "math.cos": math.cos}[path]
                try:
                    value = fn(values[0])
                    if not math.isfinite(value) or abs(value) > 1e9: raise ValueError("数值超过范围")
                    return value
                except (ValueError, OverflowError) as error: raise ImportIssue(f"math 参数无效：{error}", expression) from error
            if path in {"torch.arange", "torch.zeros", "torch.ones", "torch.empty", "torch.exp", "torch.sin", "torch.cos", "torch.sqrt", "torch.abs"}:
                if path == "torch.empty": raise ImportIssue("torch.empty 未初始化，不能忠实导入", expression)
                names = [k.arg for k in expression.keywords]
                if any(name not in ("dtype",) for name in names) or len(set(names)) != len(names):
                    raise ImportIssue("静态构造仅支持 dtype，不支持 device/out/未知参数", expression)
                if path not in ("torch.arange", "torch.zeros", "torch.ones") and names:
                    raise ImportIssue("逐元素静态函数不支持额外参数", expression)
                try:
                    dtype = next((self.path(k.value) if isinstance(k.value, ast.Attribute) else self.literal(k.value, env) for k in expression.keywords if k.arg == "dtype"), "float32")
                    if dtype not in ("float32", "torch.float32", "torch.float", None): raise ImportIssue("静态张量仅支持 float32", expression)
                    args = [self.expr(a, env, attrs) for a in expression.args]
                    if path == "torch.arange":
                        if any(type(v) not in (int, float) for v in args): raise ImportIssue("arange 参数必须是数值常量", expression)
                        if len(args) == 1: start, end, step = 0, args[0], 1
                        elif len(args) == 2: start, end, step = args[0], args[1], 1
                        elif len(args) == 3: start, end, step = args
                        else: raise ImportIssue("arange 参数数量无效", expression)
                        return StaticTensor.arange(start, end, step, self.static_budget)
                    if path in ("torch.zeros", "torch.ones"):
                        if len(args) == 1 and isinstance(args[0], (list, tuple)):
                            shape = args[0]
                        elif args and all(type(v) is int and v >= 0 for v in args):
                            shape = args
                        else:
                            raise ImportIssue("zeros/ones 需要静态 shape", expression)
                        return StaticTensor.filled(shape, 0 if path.endswith("zeros") else 1, self.static_budget)
                    if len(args) != 1 or not isinstance(args[0], StaticTensor): raise ImportIssue("静态逐元素函数需要静态张量", expression)
                    return args[0].unary({"torch.exp": math.exp, "torch.sin": math.sin, "torch.cos": math.cos, "torch.sqrt": math.sqrt, "torch.abs": abs}[path])
                except (StaticTensorError, ValueError, OverflowError) as error: raise ImportIssue(str(error), expression) from error
            if path in ("torch.select", "torch.Tensor.select") or isinstance(function, ast.Attribute) and function.attr == "select":
                method = path not in ("torch.select", "torch.Tensor.select")
                names = ["dim", "index"] if method else ["input", "dim", "index"]
                if len(expression.args) > len(names): raise ImportIssue("select 参数过多", expression)
                arguments = dict(zip(names, expression.args))
                for keyword in expression.keywords:
                    if keyword.arg not in names or keyword.arg in arguments: raise ImportIssue("select 参数无效或重复", expression)
                    arguments[keyword.arg] = keyword.value
                if set(arguments) != set(names): raise ImportIssue("select 缺少参数", expression)
                tensor = self.expr(function.value if method else arguments["input"], env, attrs)
                dim, index = (self.literal(arguments[key], env) for key in ("dim", "index"))
                if type(dim) is not int or type(index) is not int: raise ImportIssue("select 需要整数常量 dim/index", expression)
                rank = len(self.tensor_shape(tensor, expression))
                if not -rank <= dim < rank or dim % rank == 0: raise ImportIssue("select 不支持 batch 维度或越界维度", expression)
                return self.add("Select", {"dim": dim, "index": index}, [(tensor, None)], "Select", expression)
            if path in ("torch.einsum", "torch.functional.einsum"):
                # The equation is the first positional argument and must be a literal
                # string. Its output shape is resolved here rather than deferred, so a
                # mismatched contraction is reported at the call site.
                if not expression.args or expression.keywords: raise ImportIssue("einsum 需要位置参数形式的方程与张量", expression)
                equation = self.literal(expression.args[0], env)
                if not isinstance(equation, str): raise ImportIssue("einsum 方程必须是字符串常量", expression)
                operands = [self.expr(argument, env, attrs) for argument in expression.args[1:]]
                if not operands: raise ImportIssue("einsum 至少需要一个张量", expression)
                for operand in operands:
                    if not isinstance(operand, Tensor): raise ImportIssue("einsum 只接受运行时张量", expression)
                shapes = [self.tensor_shape(operand, expression) for operand in operands]
                try:
                    einsum_shape(equation, shapes)
                except ValueError as error:
                    raise ImportIssue(f"einsum 方程无效：{error}", expression) from error
                return self.add("Einsum", {"equation": equation}, [(operand, None) for operand in operands], "张量收缩", expression)
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
            if path in ("torch.unsqueeze", "torch.squeeze", "torch.functional.unsqueeze", "torch.functional.squeeze", "torch.Tensor.unsqueeze", "torch.Tensor.squeeze"):
                # torch.Tensor.unsqueeze(x, dim) is the explicit class-method form;
                # the other spellings are functional calls with input as arg 0.
                return self.shape_operation(expression, env, attrs, method=False)
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
                if any(isinstance(s, ast.Return) for s in ast.walk(statement)): raise ImportIssue("循环内不支持 return", statement)
                container, iterator = None, None
                try: iterator = self.resolve_module(statement.iter, attrs, env)
                except ImportIssue: pass
                # `self.blocks[i]` selects one of the container's instances, so the container
                if isinstance(iterator, Module) and iterator.kind in ("ModuleList", "Sequential", "Group"): container = iterator
                if container is not None and container.count:
                    # A folded container stands for `count` instances; one replay covers them
                    # all, since every index resolves to the same node (see Module.count).
                    if self.referenced(statement.body, statement.target) != 1: raise ImportIssue("折叠结构块的循环体需要恰好调用一次该块", statement)
                    self.assign(statement.target, container, env)
                    self.statements(statement.body, env, attrs)
                elif container is not None:
                    for module in container.children.values():
                        self.assign(statement.target, module, env)
                        self.statements(statement.body, env, attrs)
                else:
                    # `for i in range(n): x = self.blocks[i](x)`, the other idiomatic spelling.
                    # Index 0 stands in for the whole chain when the container is folded; if it
                    # is not, the remaining instances are replayed normally.
                    count = self.range_count(statement.iter, env)
                    if count is None: raise ImportIssue("循环只支持静态层容器", statement)
                    if self.referenced(statement.body, statement.target) != 1: raise ImportIssue("循环体需要恰好使用一次循环变量", statement)
                    before = self.group_applies
                    self.assign(statement.target, 0, env)
                    self.statements(statement.body, env, attrs)
                    if self.group_applies == before:
                        for ordinal in range(1, count):
                            self.assign(statement.target, ordinal, env)
                            self.statements(statement.body, env, attrs)
            elif exported and isinstance(statement, ast.If):
                # Export input validation is checked separately against our template.
                expected = 'if isinstance(x, dict):\n    if set(x) != set(input_ids):\n        raise ValueError("Input dictionary must contain exactly the model\'s Input node IDs")\nelif len(input_ids) != 1:\n    raise ValueError("Multi-input models require a dictionary keyed by Input node ID")'
                if ast.dump(statement, include_attributes=False) != ast.dump(ast.parse(expected).body[0], include_attributes=False): raise ImportIssue("导出代码输入检查已改变，不支持此控制流", statement)
            else: raise ImportIssue("不支持动态控制流、原地操作或此语句", statement, "UNSUPPORTED_STATEMENT")
        return None

    def fold_containers(self):
        """Turn each Group node into one subgraph definition plus one folded instance.

        The Group node was added as a plain op to hold a position in the edge stream; the body
        built through it is moved into a subgraph and the instance becomes the Group node. Both
        ids and subgraph names stay inside `[a-zA-Z][a-zA-Z0-9_]{0,63}`, matching the graph IR
        contract, and the result is what `backend.graph.expand_subgraphs` rebuilds.
        """
        groups = [node for node in self.graph["nodes"] if node["op"] == "Group"]
        if not groups: return
        # 由深到浅折叠：一个 Group 节点必然在它父块 replay 时被创建，所以 id 更大。反过来处理
        # 外层块会先把它内部那个 Group 节点一起搬进子图并移出主图，内层就再也没有机会折叠。
        for node in reversed(groups):
            identifier = node["id"]
            count = self.group_counts[identifier]
            descendants = set(self.group_bodies[identifier])
            if not descendants: raise ImportIssue(f"折叠结构块 {node['name']} 没有内部节点", self.node_sources.get(identifier))
            children = [candidate for candidate in self.graph["nodes"] if candidate["id"] in descendants]
            # 内层块先折时会被移出主图，外层记录的 body 里就留下了已不存在的 id。把它们剔除，
            # 否则入口/出口会被算成多个，外层块再也折不了。
            live = {candidate["id"] for candidate in children}
            inner = [edge for edge in self.graph["edges"] if edge["source"] in live and edge["target"] in live]
            entries = live - {edge["target"] for edge in inner}
            exits = live - {edge["source"] for edge in inner}
            if len(entries) != 1 or len(exits) != 1:
                raise ImportIssue(f"折叠结构块 {node['name']} 的内部连线不构成唯一入口和出口", self.node_sources.get(identifier))
            entry, departure = entries.pop(), exits.pop()
            name = f"group_{identifier}"
            position = {child["id"]: index for index, child in enumerate(children)}
            self.graph.setdefault("subgraphs", {})[name] = {
                "id": name, "name": node["name"][:120], "origin": "auto",
                "nodes": [{**child, "id": f"n{index}", "position": {"x": index * 210, "y": 100}} for index, child in enumerate(children)],
                "edges": [{**edge, "id": f"e_{edge['id']}", "source": f"n{position[edge['source']]}", "target": f"n{position[edge['target']]}"} for edge in inner],
            }
            # Constraints recorded while the body was built name the flat node ids, but the
            # analyzer reports expanded ids; keep the mapping so both can be checked.
            for index, child in enumerate(children): self.group_map[child["id"]] = f"{identifier}{EXPAND_SEP}0{EXPAND_SEP}n{index}"
            self.group_map[identifier] = f"{identifier}{EXPAND_SEP}0{EXPAND_SEP}n{position[entry]}"
            node.update({"op": "Group", "params": {}, "subgraph": name, "name": f"{node['name'][:110]} ×{count}"})
            if count > 1: node["repeat"] = count
            # Move the body out. Outside edges now address the instance, exactly the boundary
            # `expand_subgraphs` rebuilds: entering the subgraph goes through its entry, leaving
            # it through its exit. The body was built through an instance-to-entry edge, which is
            # scaffolding rather than a connection, so it is dropped along with the moved edges.
            kept = []
            for edge in self.graph["edges"]:
                source, target = edge["source"], edge["target"]
                if source in descendants and target in descendants: continue
                if source == identifier and target in descendants: continue
                if target in descendants: target = identifier
                if source in descendants: source = identifier
                if source == identifier and target == identifier: continue
                kept.append({**edge, "source": source, "target": target})
            self.graph["edges"] = kept
            self.graph["nodes"] = [candidate for candidate in self.graph["nodes"] if candidate["id"] not in descendants]

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
        self.fold_containers()
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
            # A folded body reports expanded ids such as `import_1/0/n0`; map the recorded flat
            # id back so the declared dimension is still checked.
            key = self.group_map.get(key, key)
            incoming = analysis["baseEdges"].get(key) or []
            if not incoming: continue
            edge = incoming[0]
            shape = analysis["portShapes"][edge["source"]][edge["sourcePort"]] if edge.get("sourcePort") else analysis["shapes"][edge["source"]]
            actual = shape[-1] if kind in ("Transformer", "MultiHeadAttention", "LayerNorm", "RMSNorm", "Linear", "Bilinear") else shape[1]
            if actual != expected: raise ImportIssue(f"{kind} 声明输入维度 {expected}，实际为 {actual}；请检查输入形状或代码", source, "SHAPE_MISMATCH")
        # Only top-level nodes have a position; a folded body is laid out inside its subgraph.
        depths, rows = {}, {}
        for key in analysis["order"]:
            top = key.split(EXPAND_SEP)[0]
            if top in depths: continue
            parents = [edge["source"].split(EXPAND_SEP)[0] for edge in self.graph["edges"] if edge["target"] == top]
            depths[top] = max([depths[parent] + 1 for parent in parents if parent in depths] or [0])
            row = rows.get(depths[top], 0); rows[depths[top]] = row + 1
            next(n for n in self.graph["nodes"] if n["id"] == top)["position"] = {"x": 210 * depths[top], "y": 100 + 170 * row}
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

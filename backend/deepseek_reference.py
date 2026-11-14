"""Recognise DeepSeek-V3's official `inference/model.py` and import its `Transformer`.

The reference file cannot be parsed line by line: its forward passes run KV caches, complex-number
rotary encoding, distributed collectives and FP8 kernels. It is a fixed file, though, so the importer
recognises it as a whole. Every class and function that defines the computation must match the
pinned commit below exactly, docstrings aside, and `Transformer(ModelArgs(...))` then maps onto the
graph operators that compute the same thing: `LatentAttention` for MLA, `MoE` for MoE, and the dense
SwiGLU MLP as Linear, SiLU and Multiply layers.

What this reproduces is the single-device bf16 forward from position 0. Its KV cache then holds just
the current sequence, so it equals the cache-free computation the graph runs, in float32. Both of
the reference's attention implementations ("naive" and "absorb") compute the same function.
"""
import ast
import copy
import hashlib

OFFICIAL_COMMIT = "b15f0dbbbe6a4bc403306175698439ef380f5fb5"
SOURCE_URL = f"https://github.com/deepseek-ai/DeepSeek-V3/blob/{OFFICIAL_COMMIT}/inference/model.py"

# sha256 of each definition's AST with docstrings removed, at OFFICIAL_COMMIT. `ModelArgs` is data
# and is read rather than fingerprinted; everything that computes is here.
DIGESTS = {
    "ParallelEmbedding": "51007c41b8dda539e62cc57b766df2182e2e63a9b74fda981030800df33620d4",
    "linear": "bd241349cbd1d988d5df46a3671664cfa5fa3a88e42dbd193df17c5c3e05c15a",
    "Linear": "2b184ada610e6fae6f816ac6a0177eaf4950f326cd74433c5b8bb17819babdcd",
    "ColumnParallelLinear": "d8d62101a5620485aff547488d497051a065cc645812867fe17b4bfe89921c25",
    "RowParallelLinear": "6a350385e893d7c65ffdffbbedb6ae1e929d3ea5903d94c61f48acfe9a699c76",
    "RMSNorm": "3faf9992cf7ed796f8623271b19a05be20576a1cc42ed92b1a5abf4668e96e27",
    "precompute_freqs_cis": "af63de40cef67bd4d7c1db6b0cf3d1183b8d8e0c1e0d3486a861c330f05aedf1",
    "apply_rotary_emb": "7caf1e14dc011c9ccb4f53a89ccb950c5ef405204e55fdd3f2a710c5ff58defb",
    "MLA": "ca22a3fa30bd9ccd2391520d1fa461d3c3b399f59e6d22348a391332a7a9d935",
    "MLP": "342e9da9c7c8947f90bb87175e891d429a2f7ce2d5096d200f648498fa6a9ae7",
    "Gate": "6fc14e4b9846e51ad4890424be04dc236f235a8e5afbbd90436ec545228df13a",
    "Expert": "1708748a697dcb09a659813c705f11d1f8b78ff356e6250dfdb5b69d5393582c",
    "MoE": "1f3229593ab29ea9f2d48befde076bc6b8d8ded5d83234340d3e81a432538a73",
    "Block": "74e2b4ae8fad9ef59c5458e12370366920d58216174c5b02b958aeb979a295b4",
    "Transformer": "adc475771edc2f4518f73591acf5a89ef223ed250a38cfe673f88f4ac41fb4fc",
}

# Names that together say a file means to be the reference; if they are all present, any
# difference in DIGESTS is reported instead of falling through to the line-by-line importer.
_MARKERS = ("ModelArgs", "ParallelEmbedding", "ColumnParallelLinear", "RowParallelLinear", "MLA", "Gate", "Transformer")
_MAX_NODES, _MAX_PARAMETERS = 128, 50_000_000


class ReferenceIssue(ValueError):
    """An error the importer reports at `node`; kept separate so this module does not import it."""
    def __init__(self, message, node=None, code="DEEPSEEK_REFERENCE"):
        super().__init__(message)
        self.node, self.code = node, code


def _without_docstrings(node):
    node = copy.deepcopy(node)
    for inner in ast.walk(node):
        body = getattr(inner, "body", None)
        if isinstance(inner, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
            inner.body = body[1:] or [ast.Pass()]
    return node


# Fields that vary between Python versions (3.12 added `type_params`, `kind` marks u"" strings) or
# carry no meaning for the computation; `ast.dump` itself also changes form between versions.
_VOLATILE_FIELDS = {"ctx", "kind", "type_comment", "type_params", "type_ignores"}


def _canonical(value):
    if isinstance(value, ast.AST):
        fields = [f"{name}={_canonical(field)}" for name, field in ast.iter_fields(value) if name not in _VOLATILE_FIELDS and field is not None and field != []]
        return f"{type(value).__name__}({','.join(fields)})"
    if isinstance(value, list): return "[" + ",".join(_canonical(item) for item in value) + "]"
    return repr(value)


def fingerprint(node):
    """sha256 of a definition's AST, docstrings removed, in a form every supported Python agrees on."""
    return hashlib.sha256(_canonical(_without_docstrings(node)).encode("utf-8")).hexdigest()


def definitions(tree):
    """Top-level classes and functions by name; a later definition replaces an earlier one, as in Python."""
    return {node.name: node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}


def recognise(tree):
    """The reference's definitions when `tree` is the pinned file, None when it is not that file at all."""
    found = definitions(tree)
    if not all(name in found for name in _MARKERS): return None
    differing = [name for name in DIGESTS if name not in found or fingerprint(found[name]) != DIGESTS[name]]
    if differing:
        raise ReferenceIssue(f"这像是 DeepSeek-V3 的 inference/model.py，但 {', '.join(differing)} 与固定版本 {OFFICIAL_COMMIT[:7]} 不一致；只识别未经修改的官方文件", found.get(differing[0]))
    return found


def model_args(found, call, literal):
    """ModelArgs defaults from the dataclass, overridden by the literal arguments of `call`."""
    cls = found["ModelArgs"]
    fields = {}
    for statement in cls.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name) and statement.value is not None:
            fields[statement.target.id] = literal(statement.value)
    config = dict(fields)
    if call is not None:
        if len(call.args) > len(fields) or any(isinstance(argument, ast.Starred) for argument in call.args) or any(keyword.arg is None for keyword in call.keywords):
            raise ReferenceIssue("ModelArgs 只接受逐个写出的常量参数", call)
        for name, argument in zip(fields, call.args): config[name] = literal(argument)
        for keyword in call.keywords:
            if keyword.arg not in fields: raise ReferenceIssue(f"ModelArgs 没有参数 {keyword.arg}", keyword.value)
            config[keyword.arg] = literal(keyword.value)
    integers = ("vocab_size", "dim", "inter_dim", "moe_inter_dim", "n_layers", "n_dense_layers", "n_heads", "n_routed_experts", "n_shared_experts", "n_activated_experts", "n_expert_groups", "n_limited_groups", "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "max_seq_len", "original_seq_len")
    for name in integers:
        if type(config.get(name)) is not int or config[name] < 0: raise ReferenceIssue(f"ModelArgs.{name} 需要非负整数", call)
    for name in ("route_scale", "rope_theta"):
        if type(config.get(name)) not in (int, float): raise ReferenceIssue(f"ModelArgs.{name} 需要数值", call)
    if config.get("dtype") != "bf16":
        raise ReferenceIssue('ModelArgs.dtype 只支持 "bf16"：FP8 权重需要量化内核和缩放参数，图中无法表示', call)
    if config["max_seq_len"] > config["original_seq_len"]:
        raise ReferenceIssue("max_seq_len 大于 original_seq_len 时官方代码启用 YaRN（改写 RoPE 频率与 softmax 缩放），图中不包含；请让 max_seq_len 不超过 original_seq_len", call)
    if config["n_layers"] < 1: raise ReferenceIssue("ModelArgs.n_layers 至少为 1", call)
    return config


def check_size(config, latent_parameters, moe_parameters, node):
    """Refuse a configuration the workbench cannot hold before emitting a hundred nodes of it."""
    dim, layers, dense = config["dim"], config["n_layers"], min(config["n_dense_layers"], config["n_layers"])
    nodes = 2 + dense * 10 + (layers - dense) * 6 + 4
    attention = latent_parameters(dim, config["n_heads"], config["q_lora_rank"], config["kv_lora_rank"], config["qk_nope_head_dim"], config["qk_rope_head_dim"], config["v_head_dim"])
    feed_forward = dense * 3 * dim * config["inter_dim"] + (layers - dense) * moe_parameters(dim, config["n_routed_experts"], config["n_shared_experts"], config["moe_inter_dim"])
    parameters = 2 * config["vocab_size"] * dim + layers * (attention + 2 * dim) + feed_forward + dim
    if nodes > _MAX_NODES or parameters > _MAX_PARAMETERS:
        raise ReferenceIssue(f"这组 ModelArgs 约有 {parameters:,} 个参数、{nodes} 个节点，超出工作台的 5000 万参数 / 128 节点；请在文件末尾写一组缩小的 args = ModelArgs(...)，再写 model = Transformer(args)", node, "LIMIT")


def emit(config, add, tokens, node):
    """Append the reference Transformer after `tokens`, naming each node by its module path."""
    dim = config["dim"]
    norm = {"normalized_shape": dim, "eps": 1e-6, "elementwise_affine": 1}
    attention = {"embed_dim": dim, "num_heads": config["n_heads"], "q_lora_rank": config["q_lora_rank"], "kv_lora_rank": config["kv_lora_rank"], "qk_nope_head_dim": config["qk_nope_head_dim"], "qk_rope_head_dim": config["qk_rope_head_dim"], "v_head_dim": config["v_head_dim"], "rope_theta": config["rope_theta"], "causal": 1, "eps": 1e-6}
    # The reference gives its gate a bias only at DeepSeek-V3's full width.
    experts = {"experts": config["n_routed_experts"], "top_k": config["n_activated_experts"], "shared_experts": config["n_shared_experts"], "expert_dim": config["moe_inter_dim"], "score_func": config["score_func"], "route_scale": config["route_scale"], "groups": config["n_expert_groups"], "topk_groups": config["n_limited_groups"], "gate_bias": int(dim == 7168)}
    h = add("Embedding", {"num_embeddings": config["vocab_size"], "embedding_dim": dim}, [(tokens, None)], "embed", node)
    for layer in range(config["n_layers"]):
        prefix = f"layers.{layer}"
        x = add("RMSNorm", norm, [(h, None)], f"{prefix}.attn_norm", node)
        x = add("LatentAttention", attention, [(x, None)], f"{prefix}.attn", node)
        h = add("Add", {}, [(h, None), (x, None)], f"{prefix} 残差（attn）", node)
        x = add("RMSNorm", norm, [(h, None)], f"{prefix}.ffn_norm", node)
        if layer < config["n_dense_layers"]:
            gate = add("Linear", {"out_features": config["inter_dim"], "bias": 0}, [(x, None)], f"{prefix}.ffn.w1", node)
            value = add("Linear", {"out_features": config["inter_dim"], "bias": 0}, [(x, None)], f"{prefix}.ffn.w3", node)
            gated = add("Multiply", {}, [(add("SiLU", {}, [(gate, None)], f"{prefix}.ffn.silu", node), None), (value, None)], f"{prefix}.ffn 门控", node)
            x = add("Linear", {"out_features": dim, "bias": 0}, [(gated, None)], f"{prefix}.ffn.w2", node)
        else:
            x = add("MoE", experts, [(x, None)], f"{prefix}.ffn", node)
        h = add("Add", {}, [(h, None), (x, None)], f"{prefix} 残差（ffn）", node)
    h = add("RMSNorm", norm, [(h, None)], "norm", node)
    h = add("Select", {"dim": 1, "index": -1}, [(h, None)], "取最后位置", node)
    return add("Linear", {"out_features": config["vocab_size"], "bias": 0}, [(h, None)], "head", node)

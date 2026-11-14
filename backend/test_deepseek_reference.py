"""Recognising DeepSeek-V3's official inference/model.py, and computing what it computes."""
import hashlib
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from . import deepseek_reference
from .pytorch_import import import_pytorch
from .test_latent_attention import deepseek_naive_mla
from .test_moe import deepseek_moe
from .training import build_model

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "deepseek_v3"
SOURCE = (FIXTURE / "model.py").read_text(encoding="utf-8").replace("\r\n", "\n")
# A scaled-down configuration: one dense block, then MoE blocks, as in DeepSeek-V3.
CONFIG = dict(vocab_size=128, dim=64, inter_dim=128, moe_inter_dim=32, n_layers=3, n_dense_layers=1, n_heads=4,
              n_routed_experts=8, n_shared_experts=1, n_activated_experts=2, score_func="sigmoid",
              q_lora_rank=32, kv_lora_rank=16, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
              max_seq_len=32, original_seq_len=32)


def with_config(source=SOURCE, **overrides):
    """The official file plus what a user appends to import it: a small ModelArgs and the model."""
    arguments = ", ".join(f"{key}={value!r}" for key, value in {**CONFIG, **overrides}.items())
    return f"{source}\n\nargs = ModelArgs({arguments})\nmodel = Transformer(args)\n"


def official_forward(layers, tokens):
    """`Transformer.forward` and `Block.forward` of the reference (start_pos 0, world_size 1),
    written out over `layers` -- the imported modules by official path -- with MLA and MoE from
    their transcriptions in test_latent_attention / test_moe. The vendored file itself is never run."""
    norm = lambda name, x: F.rms_norm(x, (x.size(-1),), layers[name].weight, 1e-6)
    h = F.embedding(tokens, layers["embed"].weight)
    for index in range(CONFIG["n_layers"]):
        prefix = f"layers.{index}"
        h = h + deepseek_naive_mla(layers[f"{prefix}.attn"], norm(f"{prefix}.attn_norm", h), causal=True)
        x = norm(f"{prefix}.ffn_norm", h)
        if index < CONFIG["n_dense_layers"]:
            w1, w2, w3 = (layers[f"{prefix}.ffn.{name}"].weight for name in ("w1", "w2", "w3"))
            h = h + F.linear(F.silu(F.linear(x, w1)) * F.linear(x, w3), w2)
        else:
            h = h + deepseek_moe(layers[f"{prefix}.ffn"], x)
    return F.linear(norm("norm", h)[:, -1], layers["head"].weight)


class DeepSeekReferenceTests(unittest.TestCase):
    def imported(self, source=None, model="Transformer", shapes=None):
        result = import_pytorch(source or with_config(), model, shapes or {"tokens": [2, 12]})
        self.assertIsNotNone(result["graph"], result["diagnostics"])
        return result

    def rejected(self, source, fragment, model="Transformer", shapes=None):
        result = import_pytorch(source, model, shapes or {"tokens": [2, 12]})
        self.assertIsNone(result["graph"], fragment)
        self.assertIn(fragment, result["diagnostics"][0]["message"])
        return result

    def test_the_fixture_is_the_pinned_official_file(self):
        data = SOURCE.encode("utf-8")
        self.assertEqual(hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest(), "886849975d3b146bfb84f8ea5fdff265c210fcec")
        self.assertIn("MIT License", (FIXTURE / "LICENSE-CODE").read_text(encoding="utf-8"))

    def test_the_transformer_maps_onto_latent_attention_and_moe(self):
        result = self.imported()
        nodes = result["graph"]["nodes"]
        self.assertEqual([node["op"] for node in nodes], [
            "Input", "Embedding",
            "RMSNorm", "LatentAttention", "Add", "RMSNorm", "Linear", "Linear", "SiLU", "Multiply", "Linear", "Add",
            "RMSNorm", "LatentAttention", "Add", "RMSNorm", "MoE", "Add",
            "RMSNorm", "LatentAttention", "Add", "RMSNorm", "MoE", "Add",
            "RMSNorm", "Select", "Linear", "Output"])
        attention = next(node for node in nodes if node["op"] == "LatentAttention")
        self.assertEqual(attention["name"], "layers.0.attn")
        self.assertEqual(attention["params"], {"embed_dim": 64, "num_heads": 4, "q_lora_rank": 32, "kv_lora_rank": 16, "qk_nope_head_dim": 16, "qk_rope_head_dim": 8, "v_head_dim": 16, "rope_theta": 10000.0, "causal": 1, "eps": 1e-6})
        self.assertEqual(next(node for node in nodes if node["op"] == "MoE")["params"], {"experts": 8, "top_k": 2, "shared_experts": 1, "expert_dim": 32, "score_func": "sigmoid", "route_scale": 1.0, "groups": 1, "topk_groups": 1, "gate_bias": 0})
        self.assertEqual(result["inputs"], [{"name": "tokens", "id": nodes[0]["id"], "inferred": False, "shape": [2, 12]}])
        self.assertIn("DEEPSEEK_REFERENCE", [diagnostic["code"] for diagnostic in result["diagnostics"]])
        self.assertEqual(result["analysis"]["shapes"][nodes[-1]["id"]], [2, 128])

    def test_the_imported_graph_computes_the_reference_forward(self):
        result = self.imported()
        model = build_model(result["graph"])[0].eval()
        # Every parameter-owning node is named by the official module path it stands for.
        layers = {node["name"]: model.layers[node["id"]] for node in result["graph"]["nodes"] if node["id"] in model.layers and model.layers[node["id"]].state_dict()}
        self.assertEqual(sorted(layers), sorted(["embed", "norm", "head", "layers.0.ffn.w1", "layers.0.ffn.w2", "layers.0.ffn.w3", "layers.1.ffn", "layers.2.ffn"] + [f"layers.{i}.{name}" for i in range(3) for name in ("attn", "attn_norm", "ffn_norm")]))
        tokens = torch.randint(0, CONFIG["vocab_size"], (2, 12))
        with torch.no_grad():
            torch.testing.assert_close(model(tokens), official_forward(layers, tokens), rtol=1e-4, atol=1e-5)

    def test_a_docstring_edit_is_still_the_reference(self):
        edited = SOURCE.replace("Multi-Head Latent Attention (MLA) Layer.", "Our copy of the attention layer.")
        self.assertNotEqual(edited, SOURCE)
        self.imported(with_config(edited))

    def test_configurations_and_copies_it_cannot_represent_are_refused(self):
        self.rejected(with_config(dtype="fp8"), '只支持 "bf16"')
        self.rejected(with_config(max_seq_len=64), "YaRN")
        self.rejected(with_config(foo=1), "没有参数 foo")
        self.rejected(SOURCE, "缩小的 args = ModelArgs")
        self.rejected(with_config(), "请选择 Transformer", model="MLA")
        self.rejected(with_config(), "超过 ModelArgs.max_seq_len", shapes={"tokens": [1, 40]})
        self.rejected(with_config(), "[B,S]", shapes={"tokens": [1, 12, 4]})
        changed = SOURCE.replace("self.softmax_scale = self.qk_head_dim ** -0.5", "self.softmax_scale = self.qk_head_dim ** -0.25")
        self.assertNotEqual(changed, SOURCE)
        self.rejected(with_config(changed), "MLA 与固定版本 b15f0db 不一致")
        self.rejected(with_config(SOURCE.replace('attn_impl: Literal["naive", "absorb"] = "absorb"', 'attn_impl = "fast"')), "attn_impl")

    def test_the_full_width_gate_keeps_its_bias(self):
        emitted = []
        def add(kind, params, sources, name, node=None):
            emitted.append((kind, params)); return name
        deepseek_reference.emit({**CONFIG, "dim": 7168, "n_layers": 2, "rope_theta": 10000.0, "route_scale": 1.0, "n_expert_groups": 1, "n_limited_groups": 1}, add, "tokens", None)
        self.assertEqual([params["gate_bias"] for kind, params in emitted if kind == "MoE"], [1])


if __name__ == "__main__":
    unittest.main()

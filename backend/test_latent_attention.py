"""Multi-head Latent Attention (MLA): parity with DeepSeek-V3's reference computation, analysis,
training, inspection, snapshots and source import.

The tests are skipped when the optional PyTorch training extra is unavailable.
"""
import unittest

from .graph import analyze_graph, latent_attention_parameters
from .inference import TensorInferenceRequest, run_inference
from .pytorch_import import import_pytorch
from .training import build_model

try:
    import torch
    from torch.nn import functional as F

    from .attention import TensorLabLatentAttention
    from .observation_snapshot import capture_snapshot
except ImportError:  # pragma: no cover - deployment without training extra
    torch = None

SIZES = {"embed_dim": 32, "num_heads": 4, "kv_lora_rank": 8, "qk_nope_head_dim": 8, "qk_rope_head_dim": 4, "v_head_dim": 8}


def deepseek_naive_mla(module, x, causal):
    """DeepSeek-V3 `inference/model.py` `MLA.forward` with attn_impl="naive", start_pos=0,
    world_size=1, float32 and max_seq_len <= original_seq_len (so no YaRN), reading `module`'s
    weights. RoPE is `precompute_freqs_cis` / `apply_rotary_emb`, with complex numbers."""
    bsz, seqlen, _ = x.size()
    heads, nope, rope, value = module.num_heads, module.qk_nope_head_dim, module.qk_rope_head_dim, module.v_head_dim
    if module.q_lora_rank == 0:
        q = F.linear(x, module.wq.weight)
    else:
        q = F.linear(F.rms_norm(F.linear(x, module.wq_a.weight), (module.q_lora_rank,), module.q_norm.weight, module.q_norm.eps), module.wq_b.weight)
    q = q.view(bsz, seqlen, heads, nope + rope)
    q_nope, q_pe = torch.split(q, [nope, rope], dim=-1)
    freqs = 1.0 / (module.rope_theta ** (torch.arange(0, rope, 2, dtype=torch.float32) / rope))
    freqs = torch.outer(torch.arange(seqlen), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    def apply_rotary_emb(t, cis):
        dtype = t.dtype
        t = torch.view_as_complex(t.float().view(*t.shape[:-1], -1, 2))
        cis = cis.view(1, t.size(1), 1, t.size(-1))
        return torch.view_as_real(t * cis).flatten(3).to(dtype)

    q_pe = apply_rotary_emb(q_pe, freqs_cis)
    kv = F.linear(x, module.wkv_a.weight)
    kv, k_pe = torch.split(kv, [module.kv_lora_rank, rope], dim=-1)
    k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
    q = torch.cat([q_nope, q_pe], dim=-1)
    kv = F.linear(F.rms_norm(kv, (module.kv_lora_rank,), module.kv_norm.weight, module.kv_norm.eps), module.wkv_b.weight)
    kv = kv.view(bsz, seqlen, heads, nope + value)
    k_nope, v = torch.split(kv, [nope, value], dim=-1)
    k = torch.cat([k_nope, k_pe.expand(-1, -1, heads, -1)], dim=-1)
    scores = torch.einsum("bshd,bthd->bsht", q, k) * module.softmax_scale
    if causal:
        mask = torch.full((seqlen, seqlen), float("-inf")).triu_(1)
        scores += mask.unsqueeze(1)
    scores = scores.softmax(dim=-1, dtype=torch.float32)
    out = torch.einsum("bsht,bthd->bshd", scores, v)
    return F.linear(out.flatten(2), module.wo.weight)


def latent_graph(params=None, *, length=6):
    layer = {**SIZES, "q_lora_rank": 16, "causal": 1, **(params or {})}
    return {"version": 1, "name": "MLA", "nodes": [
        {"id": "tokens", "name": "tokens", "op": "Input", "params": {"shape": [1, length, 32]}, "position": {"x": 0, "y": 0}},
        {"id": "mla", "name": "mla", "op": "LatentAttention", "params": layer, "position": {"x": 1, "y": 0}},
        {"id": "flatten", "name": "flatten", "op": "Flatten", "params": {}, "position": {"x": 2, "y": 0}},
        {"id": "logits", "name": "logits", "op": "Linear", "params": {"out_features": 3}, "position": {"x": 3, "y": 0}},
        {"id": "output", "name": "output", "op": "Output", "params": {}, "position": {"x": 4, "y": 0}},
    ], "edges": [{"id": f"e{i}", "source": a, "target": b} for i, (a, b) in enumerate((("tokens", "mla"), ("mla", "flatten"), ("flatten", "logits"), ("logits", "output")))]}


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class LatentAttentionTests(unittest.TestCase):
    def module(self, **overrides):
        torch.manual_seed(0)
        module = TensorLabLatentAttention(**{**SIZES, "q_lora_rank": 16, **overrides})
        # Random norm weights too, so a swapped or skipped RMSNorm cannot hide behind its ones.
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.normal_(0, 0.3)
        return module

    def test_matches_deepseek_v3_naive_mla(self):
        for q_lora_rank in (16, 0):
            for causal in (True, False):
                with self.subTest(q_lora_rank=q_lora_rank, causal=causal):
                    module = self.module(q_lora_rank=q_lora_rank, causal=causal)
                    x = torch.randn(2, 6, 32)
                    with torch.no_grad():
                        torch.testing.assert_close(module(x), deepseek_naive_mla(module, x, causal), rtol=1e-5, atol=1e-5)

    def test_a_causal_position_never_reads_a_later_one(self):
        module = self.module()
        x = torch.randn(1, 6, 32)
        changed = x.clone()
        changed[:, 3:] = torch.randn(1, 3, 32)
        with torch.no_grad():
            torch.testing.assert_close(module(x)[:, :3], module(changed)[:, :3])
            self.assertFalse(torch.allclose(module(x)[:, 3:], module(changed)[:, 3:]))

    def test_parameter_count_agrees_with_both_analyzers(self):
        for q_lora_rank in (16, 0):
            with self.subTest(q_lora_rank=q_lora_rank):
                module = self.module(q_lora_rank=q_lora_rank)
                expected = latent_attention_parameters(32, 4, q_lora_rank, 8, 8, 4, 8)
                self.assertEqual(sum(p.numel() for p in module.parameters()), expected)
                info = analyze_graph(latent_graph({"q_lora_rank": q_lora_rank}))
                self.assertEqual(info["parameters"]["mla"], expected)

    def test_trains_through_the_graph_runtime(self):
        model, _ = build_model(latent_graph())
        loss = F.cross_entropy(model(torch.randn(4, 6, 32)), torch.tensor([0, 1, 2, 1]))
        loss.backward()
        latent = model.layers["mla"]
        self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0 for p in latent.parameters()))

    def test_the_attention_inspector_reads_every_latent_head(self):
        request = TensorInferenceRequest(graph=latent_graph(), nodeIds=["mla"], attention={"mla": {"head": 2}})
        item = run_inference(request)["attentions"][0]
        self.assertEqual((item["numHeads"], item["kvHeads"], item["branches"]), (4, 4, 1))
        self.assertEqual(item["headDim"], 12)
        self.assertEqual(item["mask"], "causal")
        self.assertEqual([entry["head"] for entry in item["overview"]], [0, 1, 2, 3])
        probabilities = torch.tensor(item["tensors"]["probabilities"]["slice"]["values"])
        values = torch.tensor(item["tensors"]["v"]["slice"]["values"])
        heads = torch.tensor(item["tensors"]["headOutput"]["slice"]["values"])
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(6), rtol=1e-5, atol=1e-5)
        self.assertTrue(torch.all(probabilities.triu(1) == 0))
        # The recomputed probabilities explain the head output the layer actually produced.
        torch.testing.assert_close(probabilities @ values, heads, rtol=1e-4, atol=1e-5)

    def test_a_snapshot_captures_latent_attention(self):
        snapshot = capture_snapshot(latent_graph())
        self.assertEqual([entry["nodeId"] for entry in snapshot["attentions"]], ["mla"])
        self.assertEqual(snapshot["attentions"][0]["mask"], "causal")

    def test_imports_the_helper_descriptor(self):
        source = """
from torch import nn
from backend.attention import TensorLabLatentAttention

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mla = TensorLabLatentAttention(32, num_heads=4, q_lora_rank=0, kv_lora_rank=8, qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8, causal=False)
        self.head = nn.Linear(32, 3)

    def forward(self, x):
        return self.head(self.mla(x))
"""
        result = import_pytorch(source, "Model", {"x": [1, 6, 32]})
        self.assertIsNotNone(result["graph"], result["diagnostics"])
        layer = next(node for node in result["graph"]["nodes"] if node["op"] == "LatentAttention")
        self.assertEqual(layer["params"], {**SIZES, "q_lora_rank": 0, "rope_theta": 10000.0, "causal": 0, "eps": 1e-6})
        self.assertEqual(result["analysis"]["totalParameters"], latent_attention_parameters(32, 4, 0, 8, 8, 4, 8) + 33 * 3)
        rejected = import_pytorch(source.replace("causal=False", "causal=0"), "Model", {"x": [1, 6, 32]})
        self.assertIsNone(rejected["graph"])


if __name__ == "__main__":
    unittest.main()

"""Repo-local MQA classifier using the importer's recognized attention helper."""
import torch
from torch import nn
from backend.attention import TensorLabAttention


class MQAClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = TensorLabAttention(
            embed_dim=16, num_heads=4, kv_heads=1,
            attention_type="multi_query", dropout=0,
        )
        self.classifier = nn.Linear(16, 2)

    def forward(self, x):
        x = self.attention(x)
        return self.classifier(x[:, -1])


if __name__ == "__main__":
    torch.manual_seed(42)
    model = MQAClassifier().eval()
    sample = torch.randn(2, 4, 16)
    with torch.inference_mode():
        output = model(sample)
    print("Output shape:", tuple(output.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))

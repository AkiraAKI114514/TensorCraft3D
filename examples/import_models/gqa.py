"""Repo-local GQA encoder classifier using the recognized Transformer helper."""
import torch
from torch import nn
from backend.attention import TensorLabTransformer


class GQAClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TensorLabTransformer(
            embed_dim=16, num_heads=4, kv_heads=2, ff_dim=32,
            attention_type="grouped_query", dropout=0,
            norm_first=True, activation="gelu",
        )
        self.classifier = nn.Linear(16, 2)

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x[:, -1])


if __name__ == "__main__":
    torch.manual_seed(42)
    model = GQAClassifier().eval()
    sample = torch.randn(2, 4, 16)
    with torch.inference_mode():
        output = model(sample)
    print("Output shape:", tuple(output.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))

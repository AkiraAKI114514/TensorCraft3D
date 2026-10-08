"""Cross-attention classifier with different Query and Context lengths."""
import torch
from torch import nn


class CrossAttentionClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=16, num_heads=4, dropout=0, batch_first=True,
        )
        self.classifier = nn.Linear(16, 2)

    def forward(self, query, context):
        x, _ = self.attention(query, context, context, need_weights=False)
        return self.classifier(x[:, -1])


if __name__ == "__main__":
    torch.manual_seed(42)
    model = CrossAttentionClassifier().eval()
    query = torch.randn(2, 4, 16)
    context = torch.randn(2, 6, 16)
    with torch.inference_mode():
        output = model(query, context)
    print("Output shape:", tuple(output.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))

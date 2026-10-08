"""Residual CNN classifier with an explicit, same-shape skip connection."""
import torch
from torch import nn


class ResidualClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.block = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.relu(self.stem(x))
        residual = x
        x = torch.relu(self.block(x) + residual)
        return self.classifier(torch.flatten(self.pool(x), 1))


if __name__ == "__main__":
    torch.manual_seed(42)
    model = ResidualClassifier().eval()
    sample = torch.randn(2, 3, 8, 8)
    with torch.inference_mode():
        output = model(sample)
    print("Output shape:", tuple(output.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))

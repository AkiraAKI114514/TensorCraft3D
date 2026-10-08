"""Small CNN classifier accepted by the static PyTorch importer."""
import torch
from torch import nn


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )
        self.classifier = nn.Linear(8, 2)

    def forward(self, x):
        return self.classifier(self.features(x))


if __name__ == "__main__":
    torch.manual_seed(42)
    model = CNNClassifier().eval()
    sample = torch.randn(2, 3, 8, 8)
    with torch.inference_mode():
        output = model(sample)
    print("Output shape:", tuple(output.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))

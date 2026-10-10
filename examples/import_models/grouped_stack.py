"""Repeated-block classifier whose grouped containers import as a handful of nodes.

`self.blocks` is the idiomatic `nn.ModuleList([Block(d) for _ in range(N)])`, and every
`Block` in turn holds its own `nn.ModuleList` of sub-blocks. The importer recognizes both
comprehensions as N identical instances and folds each into one `Group` node backed by a
reusable subgraph, so the imported graph stays a handful of nodes instead of exploding into
N*K flat operators: importing this file yields seven top-level nodes and two nested subgraphs.

The repeated body is a fixed normalization-and-activation cascade without trainable weights on
purpose. The reproducible-fixture tests map weights onto top-level nodes only, so a folded body
carrying trainable layers cannot be checked by them; the case where a folded block does carry
weights is covered by the export round-trip contract in `src/exportBlock.test.ts`. Keeping the
learnable layers (the stem convolution and the classifier) outside the repeated containers lets
this fixture prove the grouping is lossless, including gradients back to the input.
"""
import torch
from torch import nn


class NormBlock(nn.Module):
    """One fixed normalization-and-activation step; several make up a single `Block`."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.InstanceNorm1d(channels, affine=False)

    def forward(self, x):
        return torch.relu(self.norm(x))


class Block(nn.Module):
    """A block that itself holds a repeated `ModuleList` of normalization steps."""

    def __init__(self, channels):
        super().__init__()
        self.steps = nn.ModuleList([NormBlock(channels) for _ in range(3)])

    def forward(self, x):
        for step in self.steps:
            x = step(x)
        return x


class GroupedStackClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Conv1d(8, 16, kernel_size=1)
        self.blocks = nn.ModuleList([Block(16) for _ in range(4)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(16, 2)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(torch.flatten(self.pool(x), 1))


if __name__ == "__main__":
    torch.manual_seed(42)
    model = GroupedStackClassifier().eval()
    sample = torch.randn(2, 8, 4)
    with torch.inference_mode():
        output = model(sample)
    print("Output shape:", tuple(output.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))

import torch
from torch import nn

from nnll.dissect import Dissector

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


if __name__ == "__main__":
    with torch.device("meta"):
        model = ResidualBlock(3, 64)
    dissect = Dissector(model)
    g, t = dissect()
    # print(g)

    # print(ast_to_string(t))
    print(t)

    import networkx as nx
    import matplotlib.pyplot as plt

    # print(g.nodes.data())
    # print(g.edges.data())

    nx.draw(g, with_labels=True)
    plt.show()
import torch
import torch.nn as nn
from .layers import SpectralConv2d, FNOBlock

class FNO(nn.Module):
    def __init__(self, in_channels, out_channels, width, modes1, modes2, layers):
        super(FNO, self).__init__()

        self.fc0 = nn.Linear(in_channels, width)

        self.fno_blocks = nn.ModuleList()
        for _ in range(layers):
            self.fno_blocks.append(FNOBlock(width, modes1, modes2))

        self.fc1 = nn.Linear(width, out_channels)

    def forward(self, x):
        x = self.fc0(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (B, width, H, W)

        for fno in self.fno_blocks:
            x = fno(x)

        x = self.fc1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (B, out_channels, H, W)
        return x

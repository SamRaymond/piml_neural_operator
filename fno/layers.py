# fno/layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(out_channels, in_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(out_channels, in_channels, self.modes1, self.modes2, dtype=torch.cfloat)
        )

    def forward(self, x):
        """
        Forward pass for the SpectralConv2d layer.

        :param x: Input tensor of shape (batchsize, in_channels, height, width)
        :return: Output tensor of shape (batchsize, out_channels, height, width)
        """
        batchsize, in_channels, height, width = x.shape
        # print(f"Input x shape: {x.shape}")

        # Perform FFT
        x_ft = torch.fft.rfft2(x)  # [B, C, H, W_freq]
        # print(f"x_ft shape after FFT: {x_ft.shape}")

        # Slice the Fourier coefficients to retain only the top modes
        x_ft_slice = x_ft[:, :, :self.modes1, :self.modes2]  # [B, C, modes1, modes2]
        # print(f"x_ft_slice shape: {x_ft_slice.shape}")

        # Initialize output in Fourier space
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.size(-2), x_ft.size(-1), 
                             dtype=torch.cfloat, device=x.device)

        # Perform complex multiplication
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft_slice, self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft_slice, self.weights2
        )

        # Inverse FFT
        x = torch.fft.irfft2(out_ft, s=(height, width))

        return x

class FNOBlock(nn.Module):
    def __init__(self, width, modes1, modes2):
        super(FNOBlock, self).__init__()
        self.conv = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)
        self.bn = nn.BatchNorm2d(width)
        self.activation = nn.SiLU()

    def forward(self, x):
        # print("----- FNOBlock Forward Pass -----")
        x = self.conv(x)  # [B, C, H, W]
        x = self.bn(x)
        x = self.activation(x)
        x = self.w(x)
        return x
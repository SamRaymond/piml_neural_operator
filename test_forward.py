# test_forward_pass.py

import torch
from fno.model import FNO

def test_forward_pass():
    # Define model parameters based on grid_size=(64, 64)
    in_channels = 1 
    out_channels = 1
    width = 64
    modes1 = 12
    modes2 = 12  # Updated based on grid_size=64
    layers = 4

    # Initialize the model
    model = FNO(in_channels, out_channels, width, modes1, modes2, layers)

    # Create a dummy input tensor (batchsize, channels, height, width)
    batchsize = 2
    height = 64 
    width_dim = 64
    x = torch.randn(batchsize, in_channels, height, width_dim)

    # Forward pass
    output = model(x)

    # Check output shape
    assert output.shape == (batchsize, out_channels, height, width_dim), f"Unexpected output shape: {output.shape}"
    print("Forward pass successful. Output shape:", output.shape)

if __name__ == "__main__":
    test_forward_pass()
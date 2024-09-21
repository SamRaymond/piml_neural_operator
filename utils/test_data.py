import torch
import matplotlib.pyplot as plt
from utils.poisson_solver import poisson_solver
import numpy as np

# Define grid size
grid_size = (64, 64)

# Create a sample f with a single Gaussian blob
f = np.zeros((1, *grid_size), dtype=np.float32)
amplitude = 5.0
sigma = 5.0
x_center, y_center = 32, 32
x = np.linspace(0, grid_size[1] - 1, grid_size[1])
y = np.linspace(0, grid_size[0] - 1, grid_size[0])
X, Y = np.meshgrid(x, y)
f[0] = amplitude * np.exp(-((X - x_center) ** 2 + (Y - y_center) ** 2) / (2 * sigma ** 2))
f_tensor = torch.tensor(f).unsqueeze(0)  # Shape: (1, 1, H, W)

# Solve for u
u = poisson_solver(f_tensor, iterations=500, omega=1.0).squeeze(0).squeeze(0).numpy()

# Plot f and u
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.title('Source Term (f)')
plt.imshow(f[0], cmap='viridis')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title('Solution (u)')
plt.imshow(u, cmap='viridis')
plt.colorbar()

plt.tight_layout()
plt.show()
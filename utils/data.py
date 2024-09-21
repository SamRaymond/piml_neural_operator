import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .poisson_solver import poisson_solver
from tqdm import tqdm
import platform

def normalize(tensor):
    """
    Normalize a tensor to have zero mean and unit variance.

    :param tensor: Input tensor
    :return: Normalized tensor
    """
    mean = tensor.mean()
    std = tensor.std()
    return (tensor - mean) / std

class TimeIntegrationDataset(Dataset):
    def __init__(self, num_samples=1000, grid_size=(64, 64), time_steps=2):
        """
        Initialize the dataset by generating dynamical system samples for time integration.

        :param num_samples: Number of samples to generate
        :param grid_size: Tuple indicating the grid size (height, width)
        :param time_steps: Number of consecutive time steps to include
        """
        self.num_samples = num_samples
        self.grid_size = grid_size
        self.time_steps = time_steps
        self.data = []
        self.generate_data()

    def generate_data(self):
        # Add tqdm progress bar
        for _ in tqdm(range(self.num_samples), desc="Generating data"):
            states = []
            for t in range(self.time_steps):
                # Generate non-trivial source term f
                f = self.generate_non_trivial_f()
                f = torch.tensor(f).unsqueeze(0)  # Shape: (1, 1, H, W)

                # Solve for u
                with torch.no_grad():
                    u = poisson_solver(f, iterations=500, omega=1.0)

                u_normalized = normalize(u.squeeze(0))  # Normalize each state
                states.append(u_normalized)  # Shape: (1, H, W)

            # Create input-output pair: (state_t, state_t+1)
            for t in range(self.time_steps - 1):
                input_state = states[t]
                output_state = states[t + 1]
                self.data.append((input_state, output_state))

    def generate_non_trivial_f(self):
        height, width = self.grid_size
        f = np.zeros((1, height, width), dtype=np.float32)

        num_blobs = np.random.randint(1, 4)  # 1 to 3 blobs
        for _ in range(num_blobs):
            x_center = np.random.uniform(0.2, 0.8) * width
            y_center = np.random.uniform(0.2, 0.8) * height
            amplitude = np.random.uniform(1.0, 5.0)
            sigma = np.random.uniform(3.0, 10.0)

            x = np.linspace(0, width - 1, width)
            y = np.linspace(0, height - 1, height)
            X, Y = np.meshgrid(x, y)
            blob = amplitude * np.exp(-((X - x_center) ** 2 + (Y - y_center) ** 2) / (2 * sigma ** 2))
            f[0] += blob

        # Optionally add some noise
        noise = np.random.normal(0, 0.1, size=f.shape).astype(np.float32)
        f += noise

        return f

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_state, output_state = self.data[idx]
        return input_state, output_state

def get_time_integration_data_loaders(batch_size, grid_size, num_workers):
    # Determine if we're running on macOS
    is_macos = platform.system() == 'Darwin'

    # If on macOS, set num_workers to 0 to disable multiprocessing
    if is_macos:
        num_workers = 0

    train_dataset = TimeIntegrationDataset(num_samples=1000, grid_size=grid_size, time_steps=10)
    val_dataset = TimeIntegrationDataset(num_samples=200, grid_size=grid_size, time_steps=10)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader

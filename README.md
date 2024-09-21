# piml_neural_operator
A **2D Fourier Neural Operator (FNO)** implementation, designed to integrate seamlessly with particle-based methods like **Material Point Method (MPM)** and **Smoothed Particle Hydrodynamics (SPH)**.
## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [Training the Model](#training-the-model)
- [Using the Trained Model](#using-the-trained-model)
- [Interfacing with Particle Methods (MPM/SPH)](#interfacing-with-particle-methods-mpmsph)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [License](#license)
- [Contributing](#contributing)

## Overview

The `piml_neural_operator` repository provides a comprehensive implementation of a **2D Fourier Neural Operator (FNO)**. FNOs are specialized neural network architectures adept at learning mappings between function spaces, making them highly effective for solving **partial differential equations (PDEs)** and other intricate numerical problems.
## Features

- **2D Fourier Neural Operator**: Captures global dependencies in spatial data with high efficiency.
- **Modular Structure**: Enables easy extension and modification of components.
- **Mesh-Independent**: Generalizes across varying spatial resolutions without necessitating retraining.
- **Poisson Solver**: Incorporates a utility for solving Poisson equations using conventional numerical methods.
- **Seamless Integration with MPM/SPH**: Designed to interface smoothly with particle-based simulation methods.

## Installation

### Prerequisites

- **Python 3.8+**
- **pip** package manager

### Setup Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/SamRaymond/piml_neural_operator.git
   cd piml_neural_operator
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   *If `requirements.txt` is not provided, install the necessary packages manually:*

   ```bash
   pip install torch numpy matplotlib
   ```

## Data Requirements

### Expected Data Format

The FNO model expects input and output data in the form of **2D grids**. When interfacing with particle-based methods like **MPM** or **SPH**, data must be appropriately transformed between particle representations and grid formats.

- **Input (`f`)**:
  - **Shape**: `(batch_size, in_channels, height, width)`
  - **Type**: Numerical data representing source terms, boundary conditions, or other relevant physical quantities.

- **Output (`u`)**:
  - **Shape**: `(batch_size, out_channels, height, width)`
  - **Type**: Numerical data representing the solution to the PDE, such as velocity fields, pressure distributions, or other derived quantities.

### Preparing Your Data

1. **From Particles to Grid**

   - **Mapping Particle Data**: Convert particle-based data (from MPM/SPH simulations) to a grid-based representation. This typically involves interpolating particle properties onto a fixed grid.
   - **Tools & Techniques**: Utilize methods like **Particle-In-Cell (PIC)** or **Fluid-Immovable-PIC (FLIP)** for accurate mapping.

2. **Grid Representation**

   - Ensure your data is discretized on a 2D grid of size `(height, width)`. Common applications include simulations in fluid dynamics, heat distribution, and electromagnetic fields.

3. **Batching**

   - Organize your data into batches for efficient training. Each batch should maintain consistent grid sizes to facilitate seamless training.

4. **Normalization (Optional but Recommended)**

   - Normalize your input and output data to enhance training stability and convergence.

### Example Data Loading

- **Location**: `utils/data.py`
- **Function**: `get_data_loaders`

## Training the Model

### Steps to Train

1. **Configure Training Parameters**

   Modify the training parameters directly in `main.py` or make them configurable via command-line arguments or configuration files.

   - **Batch Size**: Number of samples per batch (default: `16`)
   - **Grid Size**: Spatial dimensions of the input/output data (default: `(64, 64)`)
   - **Number of Epochs**: Training duration (default: `100`)
   - **Learning Rate**: Optimization speed (default: `1e-3`)
   - **Device**: `cuda`, `mps` (Apple Metal), or `cpu` (automatically detected)

2. **Run the Training Script**

   Execute the main training pipeline:

   ```bash
   python main.py
   ```

   **Output Logs:**

   The script provides detailed logs of each stage:

   - **Stage 1: Data Preparation**
   - **Stage 2: Model Definition**
   - **Stage 3: Training**
     - Epoch-wise training and validation loss
   - **Stage 4: Evaluation**
     - Final validation loss and sample visualizations

3. **Monitor Training Progress**

   - **Loss Metrics**: Observe the Mean Squared Error (MSE) loss for both training and validation sets.
   - **Visualizations**: After training, sample predictions are visualized to assess model performance qualitatively.

### Example Output

```
Stage 1: Data Preparation
Data Preparation completed.

Stage 2: Model Definition
Model Definition completed.

Stage 3: Training
Using MPS (Metal) for training.
Starting training...
Epoch 1/100 - Train Loss: 0.123456 - Val Loss: 0.234567
...
Epoch 100/100 - Train Loss: 0.012345 - Val Loss: 0.023456
Training completed.

Stage 4: Evaluation
Validation MSE Loss: 0.023456
Evaluation completed.
```

## Using the Trained Model

Once the model is trained, you can employ it to make predictions on new data derived from particle-based simulations. Below outlines the steps to utilize the trained FNO model effectively.

### 1. **Loading the Trained Model**

Ensure that your trained model's weights are saved (modify `train_model` to save if not already). For simplicity, assume the model is saved as `model.pth`.

```python
import torch
from fno.model import FNO

def load_model(model_path, device='cpu'):
    """
    Load the trained FNO model.

    :param model_path: Path to the saved model weights.
    :param device: Device to load the model on ('cuda', 'mps', 'cpu').
    :return: Loaded FNO model.
    """
    # Initialize the model architecture with appropriate parameters
    model = FNO(in_channels=1, out_channels=1, width=64, modes1=12, modes2=12, layers=4)
    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
```

### 2. **Making Predictions**

Prepare your input data in the same format as the training data and pass it through the model.

```python
import torch
from utils.data import preprocess_input  # Define as needed
from utils.visualization import plot_output  # Define as needed

def predict(model, input_data, device='cpu'):
    """
    Make predictions using the trained FNO model.

    :param model: Loaded FNO model.
    :param input_data: Input tensor of shape (batch_size, in_channels, height, width).
    :param device: Device to perform computation on.
    :return: Predicted output tensor.
    """
    input_tensor = preprocess_input(input_data).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    return output.cpu()

# Example Usage
if __name__ == "__main__":
    model_path = 'path/to/model.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = load_model(model_path, device=device)
    
    # Replace with your actual input data
    # Example: Convert particle data from MPM/SPH to grid format
    new_particle_data = ...  # Your particle data here
    grid_input = particle_to_grid(new_particle_data)  # Implement this function as needed
    new_input = torch.tensor(grid_input, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 1, height, width)
    
    prediction = predict(model, new_input, device=device)
    
    # Convert the grid prediction back to particle format if necessary
    predicted_particle_data = grid_to_particle(prediction.numpy())  # Implement this function as needed
    
    # Visualize the prediction
    plot_output(prediction)
```

### 3. **Visualizing Results**

Visualizing the model's predictions is essential to assess their accuracy and reliability.

- **Function**: Implement visualization utilities in `utils/functions.py` as needed.
- **Example**: Use Matplotlib to plot input vs. predicted output.

```python
import matplotlib.pyplot as plt

def plot_output(prediction, ground_truth=None):
    """
    Visualize the model's prediction.

    :param prediction: Output tensor of shape (batch_size, out_channels, height, width).
    :param ground_truth: (Optional) Ground truth tensor for comparison.
    """
    for i in range(prediction.shape[0]):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.title('Prediction')
        plt.imshow(prediction[i, 0].numpy(), cmap='viridis')
        plt.colorbar()
        
        if ground_truth is not None:
            plt.subplot(1, 2, 2)
            plt.title('Ground Truth')
            plt.imshow(ground_truth[i, 0].numpy(), cmap='viridis')
            plt.colorbar()
        
        plt.show()
```

## Interfacing with Particle Methods (MPM/SPH)

Integrating the FNO model with particle-based methods like **Material Point Method (MPM)** or **Smoothed Particle Hydrodynamics (SPH)** involves specific data handling and transformation steps. Below outlines the recommended approach to ensure seamless interoperability.

### 1. **Data Transformation**

#### From Particles to Grid

- **Purpose**: Since FNO operates on grid-based data, particle data from MPM/SPH must be mapped onto a fixed grid.
  
- **Techniques**:
  
  - **Particle-In-Cell (PIC)/FLIP**: Methods to transfer particle properties to the grid.
  
  - **Nearest Neighbor Interpolation**: Assign particle values to the nearest grid points.
  
  - **Tetrahedral or Voronoi Mapping**: For more accurate spatial representations.

- **Implementation Steps**:
  
  1. **Define Grid Resolution**: Determine the spatial resolution `(height, width)` consistent with the FNO model.
  
  2. **Attribute Mapping**: Map relevant particle attributes (e.g., velocity, pressure) to grid cells.
  
  3. **Handling Multiple Particles per Cell**: Aggregate values (e.g., average, sum) when multiple particles map to the same grid cell.

```python
def particle_to_grid(particles, grid_size=(64, 64)):
    """
    Convert particle-based data to grid-based representation.

    :param particles: Particle data containing positions and attributes.
    :param grid_size: Tuple indicating the grid dimensions (height, width).
    :return: Grid-based numpy array.
    """
    height, width = grid_size
    grid = np.zeros((height, width))
    
    # Example: Assign velocity magnitude to grid
    for particle in particles:
        x, y, velocity = particle['x'], particle['y'], particle['velocity']
        grid_x = int(x * width)
        grid_y = int(y * height)
        grid[grid_y, grid_x] += np.linalg.norm(velocity)
    
    # Normalize or apply other preprocessing as needed
    return grid
```

#### From Grid to Particles

- **Purpose**: After obtaining predictions from the FNO model, map grid-based outputs back to particle-based representations.
  
- **Techniques**:
  
  - **Interpolation**: Assign grid values to particle positions using interpolation methods.
  
  - **Reverse Mapping**: Utilize scattering techniques to distribute grid outputs to particles.

- **Implementation Steps**:
  
  1. **Define Particle Positions**: Ensure particle positions are known and correspond to the grid.
  
  2. **Interpolating Grid Outputs**: Assign grid predictions to particles based on their spatial locations.
  
  3. **Handling Particle Attributes**: Update particle properties with the interpolated values.

```python
def grid_to_particle(grid_output, particles, grid_size=(64, 64)):
    """
    Convert grid-based predictions back to particle-based representation.

    :param grid_output: Grid-based prediction from FNO.
    :param particles: Original particle data containing positions.
    :param grid_size: Tuple indicating the grid dimensions (height, width).
    :return: Updated particle data with predictions.
    """
    height, width = grid_size
    
    for particle in particles:
        x, y = particle['x'], particle['y']
        grid_x = x * width
        grid_y = y * height
        
        # Bilinear interpolation for smoother assignment
        x0, y0 = int(np.floor(grid_x)), int(np.floor(grid_y))
        x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
        dx, dy = grid_x - x0, grid_y - y0
        
        value = (1 - dx) * (1 - dy) * grid_output[y0, x0] + \
                dx * (1 - dy) * grid_output[y0, x1] + \
                (1 - dx) * dy * grid_output[y1, x0] + \
                dx * dy * grid_output[y1, x1]
        
        # Update particle attribute with prediction
        particle['predicted_attribute'] = value
    
    return particles
```

### 2. **Workflow Integration**

Integrate the FNO model into your simulation pipeline as follows:

1. **Simulation Step (MPM/SPH)**:
   
   - Perform a simulation step to obtain updated particle states.

2. **Data Transformation**:
   
   - Convert the current particle states to grid-based data using the `particle_to_grid` function.

3. **FNO Prediction**:
   
   - Feed the grid data into the FNO model to obtain predictions.

4. **Map Predictions Back to Particles**:
   
   - Use the `grid_to_particle` function to update particle states based on FNO predictions.

5. **Update Particle States**:
   
   - Incorporate the predicted attributes into the simulation for the next step.

### 3. **Utilities and Helpers**

Implement helper functions to streamline data transformations and ensure consistency across different parts of the pipeline.

- **Location**: `utils/transformation.py`

```python
import numpy as np

def particle_to_grid(particles, grid_size=(64, 64)):
    # Implementation as shown above
    pass

def grid_to_particle(grid_output, particles, grid_size=(64, 64)):
    # Implementation as shown above
    pass
```

## Project Structure

```
piml_neural_operator/
├── fno/
│   ├── __init__.py
│   ├── model.py         # Defines the FNO model architecture
│   └── layers.py        # Custom Fourier layers
├── utils/
│   ├── data.py           # Data loading and preprocessing utilities
│   ├── poisson_solver.py # Utility for solving Poisson equations
│   ├── visualization.py  # Visualization utilities
│   └── transformation.py # Particle to grid and grid to particle transformations
├── main.py               # Main script to run training and evaluation
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
└── .gitignore            # Git ignore file
```
from fno.model import FNO
from utils.data import get_time_integration_data_loaders
import torch.optim as optim
import torch.nn as nn
import torch
from utils.functions import visualize_sample
def define_model(in_channels=1, out_channels=1, width=64, grid_size=(64, 64), layers=4):
    """
    Stage 2: Model Definition
    Define the Fourier Neural Operator architecture using PyTorch.
    
    :param in_channels: Number of input channels. Default is 1.
    :param out_channels: Number of output channels. Default is 1.
    :param width: Width of the network. Default is 64.
    :param grid_size: Tuple indicating the (height, width) of the grid.
    :param layers: Number of Fourier layers. Default is 4.
    """
    height, width_dim = grid_size
    modes1 = min(12, height)
    modes2 = min(12, width_dim // 2 + 1)
    model = FNO(in_channels, out_channels, width, modes1, modes2, layers)
    return model

def train_model(model, train_loader, val_loader, epochs=50, learning_rate=1e-3, device='cpu'):
    """
    Train the FNO model.

    :param model: The FNO model instance.
    :param train_loader: DataLoader for training data.
    :param val_loader: DataLoader for validation data.
    :param epochs: Number of training epochs.
    :param learning_rate: Learning rate for the optimizer.
    :param device: Device to train on ('cpu' or 'cuda').
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1}")

        # Optionally visualize just one sample
        if (epoch + 1) % 100 == 0:
            visualize_sample(inputs, targets, outputs) # visualize just one sample  
            

def evaluate_model(model, val_loader, device='cpu'):
    """
    Stage 4: Evaluation
    Evaluate the trained model's performance.

    :param model: Trained FNO model
    :param val_loader: DataLoader for validation data
    :param device: Device to run the evaluation on ('cuda' or 'cpu')
    """
    from utils.functions import visualize_sample

    model.to(device)
    model.eval()
    criterion = nn.MSELoss()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_loader.dataset)
    print(f"Validation MSE Loss: {val_loss:.6f}")

def main():
    print("Stage 1: Data Preparation")
    grid_size = (64, 64)
    batch_size = 32
    num_workers = 8  # You can keep this as 4, it will be set to 0 on macOS

    train_loader, val_loader = get_time_integration_data_loaders(
        batch_size=batch_size, grid_size=grid_size, num_workers=num_workers
    )
    print("Data Preparation completed.\n")
    
    print("Stage 2: Model Definition")
    model = define_model(grid_size=grid_size)
    print("Model Definition completed.\n")
    
    print("Stage 3: Training")
    if torch.cuda.is_available():
        device = 'cuda'
        print("Using CUDA for training.")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Metal) for training.")
    else:
        device = 'cpu'
        print("Using CPU for training.")
    
    train_model(model, train_loader, val_loader, epochs=100, learning_rate=1e-4, device=device)
    
    print("Stage 4: Evaluation")
    evaluate_model(model, val_loader, device=device)

if __name__ == "__main__":
    main()

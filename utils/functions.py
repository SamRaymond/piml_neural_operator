import matplotlib.pyplot as plt
import numpy as np

def visualize_sample(f_sample, u_sample, u_pred_sample):
    """
    Visualize the input, ground truth, and predicted outputs.

    :param f_sample: Input tensor of shape (batch_size, in_channels, height, width)
    :param u_sample: Ground truth tensor of shape (batch_size, out_channels, height, width)
    :param u_pred_sample: Predicted tensor of shape (batch_size, out_channels, height, width)
    """
    batch_size = f_sample.shape[0]

    for i in range(batch_size):
        plt.figure(figsize=(15, 5))

        # Input Visualization
        plt.subplot(1, 3, 1)
        plt.title('Input (f)')
        input_image = f_sample[i, 0].cpu().numpy()
        plt.imshow(input_image, cmap='viridis')
        plt.colorbar()

        # Ground Truth Visualization
        plt.subplot(1, 3, 2)
        plt.title('Ground Truth (u)')
        ground_truth = u_sample[i, 0].cpu().numpy()
        plt.imshow(ground_truth, cmap='viridis')
        plt.colorbar()

        # Predicted Output Visualization
        plt.subplot(1, 3, 3)
        plt.title('Predicted Output (u_pred)')
        predicted = u_pred_sample[i, 0].cpu().numpy()
        plt.imshow(predicted, cmap='viridis')
        plt.colorbar()

        plt.tight_layout()
        plt.show()
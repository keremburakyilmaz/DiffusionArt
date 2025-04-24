import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import numpy as np

def tensor_to_pil(tensor):
    """Convert a tensor to PIL Image"""
    tensor = tensor.clone().detach().cpu()
    tensor = tensor * 0.5 + 0.5  # Unnormalize from [-1, 1] to [0, 1]
    tensor = tensor.clamp(0, 1)
    tensor = tensor.numpy().transpose(1, 2, 0)
    return tensor

def save_samples(content_img, style_img, stylized_img, epoch, iteration, output_dir):
    """Save sample images to compare content, style, and stylized images"""
    # Convert tensors to PIL images or numpy arrays for plotting
    if isinstance(content_img, torch.Tensor):
        content_img = tensor_to_pil(content_img[0])  # Take first image in batch
    if isinstance(style_img, torch.Tensor):
        style_img = tensor_to_pil(style_img[0])
    if isinstance(stylized_img, torch.Tensor):
        stylized_img = tensor_to_pil(stylized_img[0])
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot images
    axes[0].imshow(content_img)
    axes[0].set_title("Content Image")
    axes[0].axis("off")
    
    axes[1].imshow(style_img)
    axes[1].set_title("Style Image")
    axes[1].axis("off")
    
    axes[2].imshow(stylized_img)
    axes[2].set_title("Stylized Image")
    axes[2].axis("off")
    
    # Save figure
    filename = f"epoch_{epoch}_iteration_{iteration}.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
    plt.close(fig)

def visualize_frequency_bands(low_freq, mid_freq, high_freq, output_dir, epoch, prefix=""):
    """Visualize the frequency bands extracted by the FFT module"""
    # Make grid of frequency band images
    low_freq_grid = make_grid(low_freq, normalize=True)
    mid_freq_grid = make_grid(mid_freq, normalize=True)
    high_freq_grid = make_grid(high_freq, normalize=True)
    
    # Save grids
    save_image(low_freq_grid, os.path.join(output_dir, f"{prefix}_low_freq_epoch_{epoch}.png"))
    save_image(mid_freq_grid, os.path.join(output_dir, f"{prefix}_mid_freq_epoch_{epoch}.png"))
    save_image(high_freq_grid, os.path.join(output_dir, f"{prefix}_high_freq_epoch_{epoch}.png"))

def plot_losses(losses, output_dir):
    """Plot training losses"""
    plt.figure(figsize=(10, 5))
    for loss_name, values in losses.items():
        plt.plot(values, label=loss_name)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Losses")
    plt.savefig(os.path.join(output_dir, "training_losses.png"))
    plt.close()
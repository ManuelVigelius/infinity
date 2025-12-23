import torch
import os
from PIL import Image
import numpy as np


def get_device():
    """Get the best available device: cuda > mps > cpu"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def print_quantized_encoding_dims(model, image, device):
    """
    Utility function to print the dimensions of the quantized encoding.

    Args:
        model: The autoencoder model
        image: Input image tensor of shape (b, c, h, w) or (c, h, w)
        device: Device to run the model on
    """
    model.eval()
    with torch.no_grad():
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Move to device
        image = image.to(device)

        # Encode the image
        encoding = model.encoder(image)
        print(f"Encoding dimensions: {encoding.shape}")

        # Quantize the encoding
        quantized_encoding, _ = model.quantizer(encoding)
        print(f"Quantized encoding dimensions: {quantized_encoding.shape}")

        # Print detailed shape information
        print(f"  Batch size: {quantized_encoding.shape[0]}")
        print(f"  Channels: {quantized_encoding.shape[1]}")
        print(f"  Height: {quantized_encoding.shape[2]}")
        print(f"  Width: {quantized_encoding.shape[3]}")
        print(f"  Total elements: {quantized_encoding.numel()}")


def save_sample_images(model, test_loader, device, epoch, output_dir='samples', n_images=8):
    """
    Save sample reconstructions for visualization.

    Args:
        model: The autoencoder model
        test_loader: DataLoader for test data
        device: Device to run the model on
        epoch: Current epoch number
        output_dir: Directory to save images
        n_images: Number of images to save (default: 8)
    """
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        # Get first batch
        data, _ = next(iter(test_loader))
        data = data.to(device)

        # Take only first n_images
        data = data[:n_images]

        # Get reconstruction
        reconstruction, _ = model(data)

        # Denormalize from [-1, 1] to [0, 1]
        data = (data + 1) / 2
        reconstruction = (reconstruction + 1) / 2

        # Clamp to valid range
        data = data.clamp(0, 1)
        reconstruction = reconstruction.clamp(0, 1)

        # Concatenate original and reconstruction side by side
        comparison = torch.cat([data, reconstruction], dim=0)

        # Determine if grayscale or RGB
        is_grayscale = data.shape[1] == 1
        img_size = data.shape[-1]

        # Create grid manually with padding
        padding = 2

        # Grid dimensions: 2 rows (original + reconstruction) x n_images columns
        grid_height = 2 * img_size + 3 * padding
        grid_width = n_images * img_size + (n_images + 1) * padding

        # Move to CPU and convert to numpy
        comparison_np = comparison.cpu().numpy()

        if is_grayscale:
            # Create white background (grayscale)
            grid = np.ones((grid_height, grid_width), dtype=np.uint8) * 255

            # Place images in grid
            for idx in range(2 * n_images):  # original + reconstructions
                row = idx // n_images
                col = idx % n_images

                # Calculate position
                y_start = padding + row * (img_size + padding)
                x_start = padding + col * (img_size + padding)

                # Get image (squeeze channel dimension)
                img = comparison_np[idx, 0]  # Take single channel
                img = (img * 255).astype(np.uint8)

                # Place in grid
                grid[y_start:y_start+img_size, x_start:x_start+img_size] = img

            # Save as grayscale PNG
            img_pil = Image.fromarray(grid, mode='L')
        else:
            # Create white background (RGB)
            grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255

            # Place images in grid
            for idx in range(2 * n_images):  # original + reconstructions
                row = idx // n_images
                col = idx % n_images

                # Calculate position
                y_start = padding + row * (img_size + padding)
                x_start = padding + col * (img_size + padding)

                # Get image (C, H, W) -> (H, W, C)
                img = comparison_np[idx].transpose(1, 2, 0)
                img = (img * 255).astype(np.uint8)

                # Place in grid
                grid[y_start:y_start+img_size, x_start:x_start+img_size] = img

            # Save as RGB PNG
            img_pil = Image.fromarray(grid, mode='RGB')

        img_pil.save(os.path.join(output_dir, f'epoch_{epoch:02d}.png'))

    print(f"Saved sample images to {output_dir}/epoch_{epoch:02d}.png")

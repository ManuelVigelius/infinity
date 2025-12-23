import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from autoencoder import AutoEncoder
from discriminator import Discriminator
from trainer import Trainer
from train_utils import get_device, print_quantized_encoding_dims, save_sample_images


def get_afhq_dataloaders(data_path, image_size=128, batch_size=32):
    """
    Create AFHQ dataloaders.

    Args:
        data_path: Path to AFHQ dataset root directory
        image_size: Size to resize images to (default: 256)
        batch_size: Batch size (default: 32)

    Returns:
        train_loader, val_loader
    """
    # Transform for AFHQ (RGB images)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    # Load full training dataset
    full_dataset = datasets.ImageFolder(
        root=f'{data_path}/train',
        transform=transform
    )

    # Split into train (95%) and validation (5%)
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for stable training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader


def main():
    # Hyperparameters
    data_path = './data/afhq'
    image_size = 128
    batch_size = 32
    learning_rate_ae = 1e-4
    learning_rate_disc = 5e-5
    num_epochs = 50
    disc_loss_weight = 0.5
    disc_start_epoch = 15  # Start discriminator after 5 warmup epochs

    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('samples_afhq', exist_ok=True)

    # Device selection
    device = get_device()
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader = get_afhq_dataloaders(data_path, image_size, batch_size)
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # Autoencoder - larger architecture for 256x256 RGB images
    autoencoder = AutoEncoder(
        in_channels=3,  # RGB
        channels=128,
        latent_channels=16,
        multipliers=[1, 2, 4],
        temperature=1.0,
        flip_prob=0.05,
        commitment_weight=1.0,
        sample_entropy_weight=0.1,
        codebook_entropy_weight=0.1
    ).to(device)

    # Discriminator - for 256x256 RGB images
    discriminator = Discriminator(
        in_channels=3,  # RGB
        base_channels=128
    ).to(device)

    # Optimizers
    optimizer_ae = torch.optim.AdamW(
        autoencoder.parameters(),
        lr=learning_rate_ae
        )

    optimizer_disc = torch.optim.AdamW(
        discriminator.parameters(),
        lr=learning_rate_disc,
        betas=(0.5, 0.9)
    )

    # Create trainer
    trainer = Trainer(
        autoencoder=autoencoder,
        discriminator=discriminator,
        optimizer_ae=optimizer_ae,
        optimizer_disc=optimizer_disc,
        device=device,
        disc_loss_weight=disc_loss_weight,
        disc_start_epoch=disc_start_epoch
    )

    print(f"\nAutoencoder parameters: {sum(p.numel() for p in autoencoder.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Print quantized encoding dimensions with a sample image
    print("\nQuantized encoding dimensions:")
    sample_data, _ = next(iter(val_loader))
    print_quantized_encoding_dims(autoencoder, sample_data[0], device)

    # Training loop
    print("\nStarting training...")
    for epoch in range(1, num_epochs + 1):
        train_losses = trainer.train_epoch(train_loader, epoch)
        val_losses = trainer.evaluate(val_loader)

        # Save sample images every epoch
        save_sample_images(autoencoder, val_loader, device, epoch, output_dir='samples_afhq')

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            trainer.save_checkpoint(f'checkpoints/afhq_epoch_{epoch:03d}.pt')
            print(f"Checkpoint saved at epoch {epoch}")

        print()

    # Save final checkpoint
    trainer.save_checkpoint('afhq_autoencoder_gan_final.pt')
    print("Final model saved to afhq_autoencoder_gan_final.pt")


if __name__ == "__main__":
    main()

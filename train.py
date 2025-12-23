import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from autoencoder import AutoEncoder
from discriminator import Discriminator
from trainer import Trainer
from train_utils import get_device, print_quantized_encoding_dims, save_sample_images


def get_mnist_dataloaders(batch_size=64):
    """Create MNIST dataloaders with 16x16 center crop"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(16),  # Center crop to 16x16
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, test_loader


def main():
    # Hyperparameters
    batch_size = 64
    learning_rate_ae = 1e-3
    learning_rate_disc = 1e-4
    num_epochs = 10
    disc_loss_weight = 0.1
    disc_start_epoch = 2  # Start discriminator after 2 epochs

    # Device selection
    device = get_device()
    print(f"Using device: {device}")

    # Data
    train_loader, test_loader = get_mnist_dataloaders(batch_size)
    print(f"Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")

    # Autoencoder - using smaller architecture for 16x16 patches
    autoencoder = AutoEncoder(
        in_channels=1,
        channels=32,
        latent_channels=32,
        multipliers=[1, 4, 4],
        temperature=1.0,
        flip_prob=0.1,
        commitment_weight=1.0,
        sample_entropy_weight=0.1,
        codebook_entropy_weight=0.1
    ).to(device)

    # Discriminator - smaller architecture for 16x16 images
    discriminator = Discriminator(
        in_channels=1,
        base_channels=32
    ).to(device)

    # Optimizers
    optimizer_ae = torch.optim.AdamW(
        autoencoder.parameters(),
        lr=learning_rate_ae
    )

    optimizer_disc = torch.optim.AdamW(
        discriminator.parameters(),
        lr=learning_rate_disc
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
    sample_data, _ = next(iter(test_loader))
    print_quantized_encoding_dims(autoencoder, sample_data[0], device)

    # Training loop
    print("\nStarting training...")
    for epoch in range(1, num_epochs + 1):
        train_losses = trainer.train_epoch(train_loader, epoch)
        test_losses = trainer.evaluate(test_loader)

        # Save sample images every epoch
        save_sample_images(autoencoder, test_loader, device, epoch)
        print()

    # Save checkpoint with both models
    trainer.save_checkpoint('mnist_autoencoder_gan.pt')
    print("Models saved to mnist_autoencoder_gan.pt")


if __name__ == "__main__":
    main()
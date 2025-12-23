import torch
import torch.nn.functional as F
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        autoencoder,
        discriminator,
        optimizer_ae,
        optimizer_disc,
        device,
        disc_loss_weight=0.1,
        disc_start_epoch=0
    ):
        """
        Trainer class for autoencoder with discriminator.

        Args:
            autoencoder: AutoEncoder model
            discriminator: Discriminator model
            optimizer_ae: Optimizer for autoencoder
            optimizer_disc: Optimizer for discriminator
            device: Device to train on
            disc_loss_weight: Weight for discriminator loss (default: 0.1)
            disc_start_epoch: Epoch to start using discriminator (default: 0)
        """
        self.autoencoder = autoencoder
        self.discriminator = discriminator
        self.optimizer_ae = optimizer_ae
        self.optimizer_disc = optimizer_disc
        self.device = device
        self.disc_loss_weight = disc_loss_weight
        self.disc_start_epoch = disc_start_epoch
        self.current_epoch = 0

    def train_discriminator(self, real_images, fake_images):
        """
        Train discriminator with hinge loss and LeCam regularization.

        Args:
            real_images: Real images from dataset
            fake_images: Reconstructed images from autoencoder

        Returns:
            Discriminator loss value
        """
        self.optimizer_disc.zero_grad()

        # Discriminator predictions (logits)
        real_logits = self.discriminator(real_images)
        fake_logits = self.discriminator(fake_images.detach())

        # Hinge loss for discriminator
        # D should output high values for real, low for fake
        real_loss = F.relu(1.0 - real_logits).mean()
        fake_loss = F.relu(1.0 + fake_logits).mean()

        # LeCam regularization
        # Encourages discriminator outputs to be diverse and prevents mode collapse
        real_mean = torch.sigmoid(real_logits).mean()
        fake_mean = torch.sigmoid(fake_logits).mean()
        lecam_loss = torch.pow(real_mean - fake_mean, 2)

        # Total discriminator loss
        disc_loss = real_loss + fake_loss + 0.001 * lecam_loss

        disc_loss.backward()
        self.optimizer_disc.step()

        return disc_loss.item()

    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number

        Returns:
            Dictionary with average losses
        """
        self.current_epoch = epoch
        self.autoencoder.train()
        self.discriminator.train()

        total_loss = 0
        total_recon_loss = 0
        total_aux_loss = 0
        total_gen_loss = 0
        total_disc_loss = 0

        use_discriminator = epoch >= self.disc_start_epoch

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for data, _ in pbar:
            data = data.to(self.device)

            # ===== Train Autoencoder =====
            self.optimizer_ae.zero_grad()

            # Forward pass through autoencoder
            reconstruction, aux_loss = self.autoencoder(data)

            # Reconstruction loss
            recon_loss = F.mse_loss(reconstruction, data)

            # Generator loss (fool discriminator)
            gen_loss = torch.tensor(0.0, device=self.device)
            if use_discriminator:
                fake_logits = self.discriminator(reconstruction)
                # Generator wants discriminator to output high values (thinks they're real)
                # Using negative hinge loss: we want to maximize fake_logits
                gen_loss = -fake_logits.mean()

            # Total autoencoder loss
            ae_loss = recon_loss + aux_loss
            if use_discriminator:
                ae_loss = ae_loss + self.disc_loss_weight * gen_loss

            # Backward pass for autoencoder
            ae_loss.backward()
            self.optimizer_ae.step()

            # ===== Train Discriminator =====
            disc_loss_val = 0.0
            if use_discriminator:
                # Get fresh reconstruction for discriminator (no gradients to autoencoder)
                with torch.no_grad():
                    reconstruction_for_disc, _ = self.autoencoder(data)
                disc_loss_val = self.train_discriminator(data, reconstruction_for_disc)

            # Accumulate losses
            total_loss += ae_loss.item()
            total_recon_loss += recon_loss.item()
            total_aux_loss += aux_loss.item()
            total_gen_loss += gen_loss.item() if use_discriminator else 0.0
            total_disc_loss += disc_loss_val

            # Update progress bar
            postfix = {
                'loss': f'{ae_loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'aux': f'{aux_loss.item():.4f}'
            }
            if use_discriminator:
                postfix['gen'] = f'{gen_loss.item():.4f}'
                postfix['disc'] = f'{disc_loss_val:.4f}'
            pbar.set_postfix(postfix)

        # Calculate averages
        n_batches = len(train_loader)
        avg_loss = total_loss / n_batches
        avg_recon_loss = total_recon_loss / n_batches
        avg_aux_loss = total_aux_loss / n_batches
        avg_gen_loss = total_gen_loss / n_batches if use_discriminator else 0.0
        avg_disc_loss = total_disc_loss / n_batches if use_discriminator else 0.0

        print(f"Epoch {epoch} Average - Loss: {avg_loss:.6f} (Recon: {avg_recon_loss:.6f}, "
              f"Aux: {avg_aux_loss:.6f}, Gen: {avg_gen_loss:.6f}, Disc: {avg_disc_loss:.6f})")

        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'aux_loss': avg_aux_loss,
            'gen_loss': avg_gen_loss,
            'disc_loss': avg_disc_loss
        }

    def evaluate(self, test_loader):
        """
        Evaluate on test set.

        Args:
            test_loader: DataLoader for test data

        Returns:
            Dictionary with average losses
        """
        self.autoencoder.eval()
        self.discriminator.eval()

        total_loss = 0
        total_recon_loss = 0
        total_aux_loss = 0

        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Evaluating")
            for data, _ in pbar:
                data = data.to(self.device)

                # Forward pass
                reconstruction, aux_loss = self.autoencoder(data)

                # Reconstruction loss
                recon_loss = F.mse_loss(reconstruction, data)
                loss = recon_loss + aux_loss

                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_aux_loss += aux_loss.item()

                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'recon': f'{recon_loss.item():.4f}'
                })

        # Calculate averages
        n_batches = len(test_loader)
        avg_loss = total_loss / n_batches
        avg_recon_loss = total_recon_loss / n_batches
        avg_aux_loss = total_aux_loss / n_batches

        print(f"Test - Loss: {avg_loss:.6f} (Recon: {avg_recon_loss:.6f}, Aux: {avg_aux_loss:.6f})")

        return {
            'loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'aux_loss': avg_aux_loss
        }

    def save_checkpoint(self, path):
        """Save checkpoint with both autoencoder and discriminator."""
        torch.save({
            'autoencoder': self.autoencoder.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_ae': self.optimizer_ae.state_dict(),
            'optimizer_disc': self.optimizer_disc.state_dict(),
            'epoch': self.current_epoch
        }, path)

    def load_checkpoint(self, path):
        """Load checkpoint with both autoencoder and discriminator."""
        checkpoint = torch.load(path, map_location=self.device)
        self.autoencoder.load_state_dict(checkpoint['autoencoder'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optimizer_ae.load_state_dict(checkpoint['optimizer_ae'])
        self.optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
        self.current_epoch = checkpoint['epoch']

import torch
import torch.nn as nn


class MultiScaleQuantizer(nn.Module):
    def __init__(
        self,
        dim: int,
        temperature: float = 1.0,
        flip_prob: float = 0.0,
        commitment_weight: float = 1.0,
        sample_entropy_weight: float = 1.0,
        codebook_entropy_weight: float = 1.0,
        downsample_mode: str = 'bilinear',
    ):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.flip_prob = flip_prob
        self.commitment_weight = commitment_weight
        self.sample_entropy_weight = sample_entropy_weight
        self.codebook_entropy_weight = codebook_entropy_weight
        self.downsample_mode = downsample_mode
        self.scale = 1.0 / torch.sqrt(torch.tensor(dim, dtype=torch.float32))
        self.schedule = [1, 2, 3, 4, 6, 8, 12, 16]

    def quantize(self, encoding: torch.Tensor) -> torch.Tensor:
        """
        Performs binary spherical quantization by converting values to +/-sqrt(dim).

        Args:
            encoding: Input tensor to quantize

        Returns:
            Quantized tensor with values in {1/-sqrt(dim), 1/+sqrt(dim)}
        """
        # Binary quantization: sign of input determines +/- 1/sqrt(dim)
        return encoding + (torch.sign(encoding) * self.scale - encoding).detach()

    def forward(self, encoding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-scale quantization: downsample to different resolutions, quantize,
        upsample, and sum.

        Args:
            encoding: Input tensor of shape (b, c, h, w)

        Returns:
            A tuple of (quantized_encoding, auxiliary_loss)
        """
        b, c, h, w = encoding.shape

        # L2 normalization
        encoding_normalized = torch.nn.functional.normalize(encoding, p=2, dim=1)

        result = torch.zeros_like(encoding_normalized)

        for scale_size in self.schedule:
            # Downsample
            downsampled = torch.nn.functional.interpolate(
                encoding_normalized,
                size=(scale_size, scale_size),
                mode=self.downsample_mode,
                align_corners=False if self.downsample_mode != 'area' else None
            )

            # Quantize
            quantized = self.quantize(downsampled)

            # Randomly flip bits during training
            if self.training and self.flip_prob > 0:
                flip_mask = torch.rand_like(quantized) < self.flip_prob
                quantized = torch.where(flip_mask, -quantized, quantized)

            # Upsample using bilinear interpolation
            upsampled = torch.nn.functional.interpolate(
                quantized,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )

            # Add to result
            result = result + upsampled

        # Calculate auxiliary loss
        aux_loss = self.calculate_loss(encoding_normalized, result)

        return result, aux_loss

    def entropy(self, encoding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the mean sample entropy and codebook entropy.

        Args:
            encoding: Input tensor of shape (b, c, h, w)

        Returns:
            A tuple of (mean_sample_entropy, codebook_entropy)
        """
        # Convert encoding to probabilities using sigmoid with temperature
        probs = torch.sigmoid(encoding / self.temperature)

        # Add epsilon to avoid log(0)
        eps = 1e-6

        # Calculate binary entropy for each element: -p*log(p) - (1-p)*log(1-p)
        sample_entropy = -(probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps))

        # Mean sample entropy: average across all samples
        mean_sample_entropy = sample_entropy.mean()

        # Codebook entropy: take mean of all probabilities, then calculate entropy
        mean_probs = probs.mean()
        codebook_entropy = -(mean_probs * torch.log(mean_probs + eps) + (1 - mean_probs) * torch.log(1 - mean_probs + eps))

        return mean_sample_entropy, codebook_entropy

    def calculate_loss(
        self, encoding_normalized: torch.Tensor, quantized: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the auxiliary loss consisting of commitment loss, sample entropy,
        and negative codebook entropy.

        Args:
            encoding_normalized: Normalized input encoding of shape (b, c, h, w)
            quantized: Quantized encoding of shape (b, c, h, w)

        Returns:
            Total auxiliary loss
        """
        # Commitment loss: MSE between normalized encoding and quantized encoding
        commitment_loss = torch.nn.functional.mse_loss(quantized, encoding_normalized)

        # Calculate entropies
        mean_sample_entropy, codebook_entropy = self.entropy(encoding_normalized)

        # Total auxiliary loss
        aux_loss = (
            self.commitment_weight * commitment_loss
            + self.sample_entropy_weight * mean_sample_entropy
            - self.codebook_entropy_weight * codebook_entropy
        )

        return aux_loss

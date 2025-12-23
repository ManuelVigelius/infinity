import torch
import torch.nn as nn
import torch.nn.functional as F

from quantizer import MultiScaleQuantizer


class ResidualBlockSwish(nn.Module):
    """
    Residual block with convolutions and Swish (SiLU) activation.

    Uses two convolutional layers with Swish activation, GroupNorm, and an optional residual connection.
    The hidden dimension is determined by out_channels.
    When in_channels == out_channels, a residual connection is used.
    Commonly used in vision models like diffusion models.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, num_groups=32):
        super().__init__()
        self.has_residual = (in_channels == out_channels)

        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)

    def forward(self, x):
        out = self.norm1(x)
        out = F.silu(out)
        out = self.conv1(out)

        out = self.norm2(out)
        out = F.silu(out)
        out = self.conv2(out)

        return out + x if self.has_residual else out


class ResidualBlockSwiGLU(nn.Module):
    """
    Residual block with convolutions and SwiGLU activation.

    SwiGLU combines Swish with a gating mechanism for enhanced expressiveness.
    The hidden dimension is determined by out_channels, then gated before being projected back.
    When in_channels == out_channels, a residual connection is used.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, num_groups=32):
        super().__init__()
        self.has_residual = (in_channels == out_channels)

        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        # First conv expands to 2x out_channels (for gate and value)
        self.conv1 = nn.Conv2d(in_channels, out_channels * 2, kernel_size, 1, padding)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        # Second conv projects to out_channels
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)

    def forward(self, x):
        # Normalize and first convolution
        out = self.norm1(x)
        out = self.conv1(out)

        # SwiGLU activation
        gate, value = out.chunk(2, dim=1)
        out = F.silu(gate) * value

        # Normalize and second convolution
        out = self.norm2(out)
        out = self.conv2(out)

        return out + x if self.has_residual else out

class Encoder(nn.Module):
    """
    Encoder that progressively downsamples the input through residual blocks.

    Architecture:
    1. Linear projection from in_channels to channels
    2. For each multiplier: residual block + downsample (2x2 conv stride 2)
    3. Two final residual blocks with last multiplier (no downsampling)
    4. Final normalization, activation, and linear projection to out_channels
    """
    def __init__(self, in_channels, channels, latent_channels, multipliers, kernel_size=3, padding=1, num_groups=32):
        super().__init__()
        self.multipliers = multipliers

        # Initial projection
        self.proj_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)

        # Residual blocks with downsampling
        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        current_channels = channels
        for mult in multipliers:
            out_channels = channels * mult
            # Residual block (transitions from current_channels to out_channels)
            self.down_blocks.append(
                ResidualBlockSwiGLU(current_channels, out_channels, kernel_size=kernel_size, padding=padding, num_groups=num_groups)
            )
            # Downsampling (2x2 conv with stride 2)
            self.downsample.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)
            )
            current_channels = out_channels

        # Two final residual blocks (no downsampling)
        final_channels = channels * multipliers[-1]
        self.mid_blocks = nn.ModuleList([
            ResidualBlockSwiGLU(final_channels, final_channels, kernel_size=kernel_size, padding=padding, num_groups=num_groups),
            ResidualBlockSwiGLU(final_channels, final_channels, kernel_size=kernel_size, padding=padding, num_groups=num_groups)
        ])

        # Final projection
        self.norm_out = nn.GroupNorm(num_groups, current_channels)
        self.proj_out = nn.Conv2d(current_channels, latent_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Initial projection
        x = self.proj_in(x)

        # Downsampling path
        for block, down in zip(self.down_blocks, self.downsample):
            x = block(x)
            x = down(x)

        # Middle blocks
        for block in self.mid_blocks:
            x = block(x)

        # Final projection
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.proj_out(x)

        return x


class Decoder(nn.Module):
    """
    Decoder that progressively upsamples the input through residual blocks.

    Architecture (mirror of Encoder):
    1. Linear projection from in_channels to channels
    2. Two initial residual blocks with first multiplier (no upsampling)
    3. For each multiplier (reversed): residual block + upsample (2x nearest neighbor + conv)
    4. Final normalization, activation, and linear projection to out_channels
    """
    def __init__(self, in_channels, channels, out_channels, multipliers, kernel_size=3, padding=1, num_groups=32):
        super().__init__()
        self.multipliers = multipliers

        # Initial projection to highest channel count
        initial_channels = channels * multipliers[-1]
        self.proj_in = nn.Conv2d(in_channels, initial_channels, kernel_size=3, padding=1)

        # Two initial residual blocks (no upsampling)
        self.mid_blocks = nn.ModuleList([
            ResidualBlockSwiGLU(initial_channels, initial_channels, kernel_size=kernel_size, padding=padding, num_groups=num_groups),
            ResidualBlockSwiGLU(initial_channels, initial_channels, kernel_size=kernel_size, padding=padding, num_groups=num_groups)
        ])

        # Residual blocks with upsampling (process multipliers in reverse)
        self.up_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()

        current_channels = initial_channels
        # Build list of target channels for each upsampling stage
        target_channels_list = [channels * mult for mult in reversed(multipliers[:-1])] + [channels]

        for target_channels in target_channels_list:
            # Upsampling (2x nearest neighbor + conv)
            self.upsample.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1)
                )
            )
            # Residual block (transitions from current_channels to target_channels)
            self.up_blocks.append(
                ResidualBlockSwiGLU(current_channels, target_channels, kernel_size=kernel_size, padding=padding, num_groups=num_groups)
            )
            current_channels = target_channels

        # Final projection
        self.norm_out = nn.GroupNorm(num_groups, current_channels)
        self.proj_out = nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Initial projection
        x = self.proj_in(x)

        # Middle blocks
        for block in self.mid_blocks:
            x = block(x)

        # Upsampling path
        for up, block in zip(self.upsample, self.up_blocks):
            x = up(x)
            x = block(x)

        # Final projection
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.proj_out(x)

        return x


class AutoEncoder(nn.Module):
    """
    Autoencoder combining Encoder and Decoder with multi-scale quantization.

    Compresses input to a latent representation, applies quantization, then reconstructs it.
    """
    def __init__(
        self,
        in_channels,
        channels,
        latent_channels,
        multipliers,
        temperature=1.0,
        flip_prob=0.0,
        commitment_weight=1.0,
        sample_entropy_weight=1.0,
        codebook_entropy_weight=1.0,
    ):
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            channels=channels,
            latent_channels=latent_channels,
            multipliers=multipliers,
            kernel_size=3,
            padding=1,
            num_groups=32
        )

        self.quantizer = MultiScaleQuantizer(
            dim=latent_channels,
            temperature=temperature,
            flip_prob=flip_prob,
            commitment_weight=commitment_weight,
            sample_entropy_weight=sample_entropy_weight,
            codebook_entropy_weight=codebook_entropy_weight,
        )

        self.decoder = Decoder(
            in_channels=latent_channels,
            channels=channels,
            out_channels=in_channels,
            multipliers=multipliers,
            kernel_size=3,
            padding=1,
            num_groups=32
        )

    def forward(self, x):
        latent = self.encoder(x)
        quantized_latent, aux_loss = self.quantizer(latent)
        reconstruction = self.decoder(quantized_latent)
        return reconstruction, aux_loss

    def encode(self, x):
        latent = self.encoder(x)
        quantized_latent, _ = self.quantizer(latent)
        return quantized_latent

    def decode(self, latent):
        return self.decoder(latent)


if __name__ == "__main__":
    # Test ResidualBlockSwish with same in/out channels (has residual)
    print("Testing ResidualBlockSwish (with residual):")
    block1 = ResidualBlockSwish(64, 64)
    x1 = torch.randn(2, 64, 32, 32)
    out1 = block1(x1)
    print(f"Input shape: {x1.shape}, Output shape: {out1.shape}")

    # Test ResidualBlockSwish with different in/out channels (no residual)
    print("\nTesting ResidualBlockSwish (no residual):")
    block1b = ResidualBlockSwish(64, 128)
    x1b = torch.randn(2, 64, 32, 32)
    out1b = block1b(x1b)
    print(f"Input shape: {x1b.shape}, Output shape: {out1b.shape}")

    # Test ResidualBlockSwiGLU with same in/out channels (has residual)
    print("\nTesting ResidualBlockSwiGLU (with residual):")
    block2 = ResidualBlockSwiGLU(64, 64)
    x2 = torch.randn(2, 64, 32, 32)
    out2 = block2(x2)
    print(f"Input shape: {x2.shape}, Output shape: {out2.shape}")

    # Test ResidualBlockSwiGLU with different in/out channels (no residual)
    print("\nTesting ResidualBlockSwiGLU (no residual):")
    block2b = ResidualBlockSwiGLU(64, 192)
    x2b = torch.randn(2, 64, 32, 32)
    out2b = block2b(x2b)
    print(f"Input shape: {x2b.shape}, Output shape: {out2b.shape}")

    # Test Encoder
    print("\nTesting Encoder:")
    encoder = Encoder(in_channels=3, channels=64, latent_channels=16, multipliers=[2, 2, 3])
    x3 = torch.randn(2, 3, 128, 128)
    latent = encoder(x3)
    print(f"Input shape: {x3.shape}, Latent shape: {latent.shape}")

    # Test Decoder
    print("\nTesting Decoder:")
    decoder = Decoder(in_channels=16, channels=64, out_channels=3, multipliers=[2, 2, 3])
    reconstructed = decoder(latent)
    print(f"Latent shape: {latent.shape}, Reconstructed shape: {reconstructed.shape}")

    # Test AutoEncoder
    print("\nTesting AutoEncoder:")
    autoencoder = AutoEncoder(in_channels=3, channels=64, latent_channels=16, multipliers=[2, 2, 3])
    x4 = torch.randn(2, 3, 128, 128)
    out4, aux_loss = autoencoder(x4)
    print(f"Input shape: {x4.shape}, Output shape: {out4.shape}")
    print(f"Auxiliary loss: {aux_loss.item()}")
    print(f"Latent shape: {autoencoder.encode(x4).shape}")

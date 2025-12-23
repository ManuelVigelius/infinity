import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        """
        Small discriminator for image discrimination.

        Args:
            in_channels: Number of input channels (default: 3 for RGB)
            base_channels: Base number of channels for the network (default: 64)
        """
        super().__init__()

        # Initial convolution
        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Three downsampling layers with exponentially increasing channels
        self.layers = nn.ModuleList([
            self._make_layer(base_channels * (2 ** i), base_channels * (2 ** (i + 1)))
            for i in range(3)
        ])

        # Final convolution to 1 channel (outputs logits, no sigmoid)
        final_channels = base_channels * (2 ** 3)  # After 3 layers: base_channels * 8
        self.final_conv = nn.Conv2d(final_channels, 1, kernel_size=3, padding=1)

    def _make_layer(self, in_channels, out_channels):
        """Create a layer with normalization, swish, convolution with stride 2"""
        return nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_channels),
            nn.SiLU(),  # Swish activation
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, 1, H//8, W//8) with logits (unbounded values)
        """
        # Initial convolution
        x = self.initial_conv(x)

        # Three downsampling layers
        for layer in self.layers:
            x = layer(x)

        # Final convolution (returns logits)
        x = self.final_conv(x)

        return x

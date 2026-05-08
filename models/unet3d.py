import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dense_block import DenseBlock


class ConvBlock(nn.Module):
    """Two 3x3x3 convolutions with BatchNorm and ReLU"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    """Encoder block: MaxPool3d(2) + ConvBlock"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    """Decoder block: Trilinear upsampling + skip connection + ConvBlock"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Trilinear upsampling to match skip connection size
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet3D(nn.Module):
    """3D U-Net with Dense Bottleneck for human body organ segmentation"""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 70,
        base_channels: int = 32,
        growth_rate: int = 32,
        dense_layers: int = 4,
        bn_size: int = 4,
    ):
        """
        Args:
            in_channels: Number of input channels (1 for binary occupancy grid)
            num_classes: Number of output classes (70 anatomical structures)
            base_channels: Base number of channels (doubled at each level)
            growth_rate: Dense block growth rate (channels added per layer)
            dense_layers: Number of layers in dense block
            bn_size: Bottleneck size multiplier for 1x1x1 compression
        """
        super().__init__()

        # Channel sizes at each level
        c1 = base_channels       # 32
        c2 = base_channels * 2   # 64
        c3 = base_channels * 4   # 128
        c4 = base_channels * 8   # 256

        # Encoder path
        self.enc1 = ConvBlock(in_channels, c1)  # 1 -> 32
        self.enc2 = Encoder(c1, c2)             # 32 -> 64
        self.enc3 = Encoder(c2, c3)             # 64 -> 128
        self.enc4 = Encoder(c3, c4)             # 128 -> 256

        # Dense Bottleneck
        self.bottleneck = DenseBlock(
            in_channels=c4,
            num_layers=dense_layers,
            growth_rate=growth_rate,
            bn_size=bn_size,
        )
        bottleneck_out = c4 + dense_layers * growth_rate  # 256 + 4*32 = 384

        # Decoder path
        self.dec4 = Decoder(bottleneck_out + c3, c3)  # 384+128=512 -> 128
        self.dec3 = Decoder(c3 + c2, c2)              # 128+64=192 -> 64
        self.dec2 = Decoder(c2 + c1, c1)              # 64+32=96 -> 32

        # Final output
        self.final = nn.Conv3d(c1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Args:
            x: Input tensor of shape (B, 1, D, H, W)
            return_features: If True, return (logits, decoder_features)

        Returns:
            If return_features=False: logits of shape (B, num_classes, D, H, W)
            If return_features=True: (logits, d2) where d2 is (B, base_channels, D, H, W)
        """
        # Encoder path with skip connections
        e1 = self.enc1(x)   # (B, 32, D, H, W)
        e2 = self.enc2(e1)  # (B, 64, D/2, H/2, W/2)
        e3 = self.enc3(e2)  # (B, 128, D/4, H/4, W/4)
        e4 = self.enc4(e3)  # (B, 256, D/8, H/8, W/8)

        # Dense Bottleneck
        b = self.bottleneck(e4)  # (B, 384, D/8, H/8, W/8)

        # Decoder path with skip connections
        d4 = self.dec4(b, e3)   # (B, 128, D/4, H/4, W/4)
        d3 = self.dec3(d4, e2)  # (B, 64, D/2, H/2, W/2)
        d2 = self.dec2(d3, e1)  # (B, 32, D, H, W)

        # Final output
        out = self.final(d2)  # (B, num_classes, D, H, W)

        if return_features:
            return out, d2
        return out

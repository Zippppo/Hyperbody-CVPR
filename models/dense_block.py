import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    """Single dense layer: BN -> ReLU -> Conv1x1x1 -> BN -> ReLU -> Conv3x3x3"""

    def __init__(self, in_channels: int, growth_rate: int, bn_size: int = 4):
        """
        Args:
            in_channels: Number of input channels
            growth_rate: Number of output channels (new features)
            bn_size: Bottleneck size multiplier for 1x1x1 compression
        """
        super().__init__()
        intermediate_channels = bn_size * growth_rate  # 128 by default

        self.block = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, intermediate_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                intermediate_channels, growth_rate, kernel_size=3, padding=1, bias=False
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            New features of shape (B, growth_rate, D, H, W)
        """
        return self.block(x)


class DenseBlock(nn.Module):
    """Dense block with multiple densely connected layers"""

    def __init__(
        self,
        in_channels: int,
        num_layers: int = 4,
        growth_rate: int = 32,
        bn_size: int = 4,
    ):
        """
        Args:
            in_channels: Number of input channels
            num_layers: Number of dense layers
            growth_rate: Number of channels added per layer
            bn_size: Bottleneck size multiplier for 1x1x1 compression
        """
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer = DenseLayer(
                in_channels=in_channels + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
            )
            self.layers.append(layer)

        self.out_channels = in_channels + num_layers * growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, D, H, W)

        Returns:
            Concatenated features of shape (B, C + num_layers * growth_rate, D, H, W)
        """
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        return torch.cat(features, dim=1)

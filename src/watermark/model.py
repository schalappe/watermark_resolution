"""Embedder and extractor networks (§3.2, Figure 2).

Both are fully convolutional with a fixed 16:1 host/WM side ratio, so the weights trained on 128/8 also
run on 64/4, 256/16, and 512/32 (§4.4.4). Kernel sizes other than the host pre-processing are not given
in the paper; 3x3 is assumed throughout.
"""

import torch
from torch import Tensor, nn


def _upsample(in_channels: int, out_channels: int, last: bool = False) -> nn.Sequential:
    """WM pre-processing block (§3.2.2): n → 2n. A 3×3 stride-2 transposed conv gives 2n+1, the 2×2
    stride-1 average pool brings it back to 2n and smooths the binary WM."""
    layers: list[nn.Module] = [nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2)]
    if not last:
        layers += [nn.BatchNorm2d(out_channels), nn.ReLU()]
    layers.append(nn.AvgPool2d(2, stride=1))
    return nn.Sequential(*layers)


def _conv_bn_relu(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class Embedder(nn.Module):
    """Host Y in [-1, 1] (N×1×M×M) + WM in {-1, +1} (N×1×M/16×M/16) → watermarked Y in [-1, 1]."""

    def __init__(self) -> None:
        super().__init__()
        self.host = nn.Conv2d(1, 64, 3, padding=1)  # §3.2.1
        # §3.2.2 text gives 512/256/128/1 filters; Figure 2 labels 512/128/64/1. The text is followed.
        self.mark = nn.Sequential(_upsample(1, 512), _upsample(512, 256), _upsample(256, 128), _upsample(128, 1, True))
        self.embed = nn.Sequential(  # §3.2.3
            _conv_bn_relu(65, 64, 1),
            _conv_bn_relu(64, 64, 1),
            _conv_bn_relu(64, 64, 1),
            _conv_bn_relu(64, 64, 1),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, host: Tensor, mark: Tensor, strength: float = 1.0) -> Tensor:
        """`strength` is the paper's s: trained at 1, swept at inference for Figure 7."""
        return self.embed(torch.cat([self.host(host), self.mark(mark) * strength], dim=1))


class Extractor(nn.Module):
    """Attacked watermarked Y (N×1×M×M) → WM estimate in [-1, 1] (N×1×M/16×M/16), §3.2.5."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _conv_bn_relu(1, 128, 2),
            _conv_bn_relu(128, 256, 2),
            _conv_bn_relu(256, 512, 2),
            nn.Conv2d(512, 1, 3, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.net(image)

"""Attack simulation between embedder and extractor (§3.2.4, Table 2) and evaluation attacks (Table 4).

Attacks work on images in [0, 1] (N×1×H×W); σ and p values of the paper are read on that scale
(Table 4's noise σ of 0.01-0.08 only makes sense there). `host` feeds the attacks that mix in host
pixels (dropout, cropout). Training attacks must stay differentiable with respect to the watermarked
image: the embedder learns robustness only through the extraction loss flowing back through them.
"""

import io
import math
from collections.abc import Callable, Sequence
from functools import partial

import kornia
import numpy as np
import torch
from PIL import Image
from torch import Tensor

Attack = Callable[[Tensor, Tensor], Tensor]

SHARPEN = {  # Laplacian sharpening stencils (Table 2)
    5: [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]],
    9: [[-1.0, -1.0, -1.0], [-1.0, 9.0, -1.0], [-1.0, -1.0, -1.0]],
}


def identity(image: Tensor, host: Tensor) -> Tensor:
    return image


def gaussian_filter(image: Tensor, host: Tensor, kernel: int) -> Tensor:
    # The paper gives kernel sizes only; σ follows OpenCV's rule for a given kernel size.
    sigma = 0.3 * ((kernel - 1) / 2 - 1) + 0.8
    return kornia.filters.gaussian_blur2d(image, (kernel, kernel), (sigma, sigma))


def average_filter(image: Tensor, host: Tensor, kernel: int) -> Tensor:
    return kornia.filters.box_blur(image, (kernel, kernel))


def median_filter(image: Tensor, host: Tensor, kernel: int) -> Tensor:
    return kornia.filters.median_blur(image, (kernel, kernel))


def salt_pepper(image: Tensor, host: Tensor, p: float) -> Tensor:
    hit = torch.rand_like(image) < p
    return torch.where(hit, (torch.rand_like(image) < 0.5).to(image.dtype), image)


def gaussian_noise(image: Tensor, host: Tensor, sigma: float) -> Tensor:
    return image + torch.randn_like(image) * sigma


def sharpen(image: Tensor, host: Tensor, points: int) -> Tensor:
    kernel = torch.tensor([SHARPEN[points]], device=image.device, dtype=image.dtype)
    return kornia.filters.filter2d(image, kernel)


def jpeg(image: Tensor, host: Tensor, quality: float) -> Tensor:
    """Differentiable JPEG approximation, for training only. Gray → RGB → gray keeps the luma path."""
    rgb = image.clamp(0, 1).expand(-1, 3, -1, -1)
    q = torch.full((len(image),), float(quality), device=image.device)
    return kornia.enhance.jpeg_codec_differentiable(rgb, q)[:, :1]


def real_jpeg(image: Tensor, host: Tensor, quality: int) -> Tensor:
    """Real JPEG encoding (Pillow), for evaluation on the quantized image."""
    out = []
    for item in image:
        pixels = (item[0] * 255).round().clamp(0, 255).byte().cpu().numpy()
        buffer = io.BytesIO()
        Image.fromarray(pixels).save(buffer, format="JPEG", quality=quality)
        decoded = np.asarray(Image.open(buffer)).astype(np.float32) / 255
        out.append(torch.from_numpy(decoded)[None])
    return torch.stack(out).to(image.device)


def rotation(image: Tensor, host: Tensor, degrees: Tensor | float) -> Tensor:
    angle = torch.as_tensor(degrees, device=image.device, dtype=image.dtype).expand(len(image))
    return kornia.geometry.transform.rotate(image, angle)


def _square_masks(image: Tensor, area: Tensor | float) -> Tensor:
    """1 inside a randomly placed square covering `area` of each image, 0 elsewhere."""
    n, _, height, width = image.shape
    area = torch.as_tensor(area, device=image.device, dtype=image.dtype).expand(n)
    side_h = (area.sqrt() * height).round().clamp(1, height)
    side_w = (area.sqrt() * width).round().clamp(1, width)
    top = (torch.rand(n, device=image.device) * (height - side_h + 1)).floor()
    left = (torch.rand(n, device=image.device) * (width - side_w + 1)).floor()
    rows = torch.arange(height, device=image.device).view(1, -1, 1)
    cols = torch.arange(width, device=image.device).view(1, 1, -1)
    inside_rows = (rows >= top.view(-1, 1, 1)) & (rows < (top + side_h).view(-1, 1, 1))
    inside_cols = (cols >= left.view(-1, 1, 1)) & (cols < (left + side_w).view(-1, 1, 1))
    return (inside_rows & inside_cols).unsqueeze(1).to(image.dtype)


def crop(image: Tensor, host: Tensor, keep: Tensor | float) -> Tensor:
    """Keep a random square of area `keep`; the rest is zeroed so the extractor input size is unchanged."""
    return image * _square_masks(image, keep)


def cropout(image: Tensor, host: Tensor, replaced: float) -> Tensor:
    """Replace a random square of area `replaced` with host pixels (Table 4: BER rises with p)."""
    mask = _square_masks(image, replaced)
    return image * (1 - mask) + host * mask


def dropout(image: Tensor, host: Tensor, keep: Tensor | float) -> Tensor:
    """Keep each watermarked pixel with probability `keep`, else take the host pixel (Table 4: BER falls
    as p rises)."""
    keep = torch.as_tensor(keep, device=image.device, dtype=image.dtype).view(-1, 1, 1, 1)
    mask = (torch.rand_like(image) < keep).to(image.dtype)
    return image * mask + host * (1 - mask)


def _uniform(low: float, high: float, image: Tensor) -> Tensor:
    return low + (high - low) * torch.rand(len(image), device=image.device, dtype=image.dtype)


def _choice(attack: Callable[..., Tensor], values: Sequence[int]) -> Attack:
    """Per-sample random discrete strength (kernel sizes, stencils)."""

    def apply(image: Tensor, host: Tensor) -> Tensor:
        pick = torch.randint(len(values), (len(image),), device=image.device)
        out = image
        for i, value in enumerate(values):
            chosen = (pick == i).view(-1, 1, 1, 1)
            out = torch.where(chosen, attack(image, host, value), out)
        return out

    return apply


# Table 2: name → (ratio in 24ths, training attack with Table 2 strengths).
TRAINING_ATTACKS: dict[str, tuple[int, Attack]] = {
    "identity": (2, identity),
    "gaussian_filter": (4, _choice(gaussian_filter, [3, 5, 7, 9])),
    "average_filter": (4, _choice(average_filter, [3, 5])),
    "median_filter": (2, _choice(median_filter, [3, 5])),
    "salt_pepper": (1, partial(salt_pepper, p=0.1)),
    "gaussian_noise": (1, partial(gaussian_noise, sigma=0.1)),
    "sharpening": (2, _choice(sharpen, [5, 9])),
    "jpeg": (2, partial(jpeg, quality=50)),
    "rotation": (2, lambda image, host: rotation(image, host, _uniform(0, 90, image))),
    "crop": (2, lambda image, host: crop(image, host, _uniform(0.5, 0.8, image))),
    "dropout": (2, lambda image, host: dropout(image, host, _uniform(0.3, 0.9, image))),
}


def allocate(batch: int, weights: Sequence[int]) -> list[int]:
    """Samples per attack for one mini-batch: largest-remainder rounding of `batch × ratio`, so every
    mini-batch keeps the Table 2 proportions (§3.2.4)."""
    shares = [batch * w / sum(weights) for w in weights]
    counts = [math.floor(s) for s in shares]
    by_remainder = sorted(range(len(shares)), key=lambda i: counts[i] - shares[i])
    for i in by_remainder[: batch - sum(counts)]:
        counts[i] += 1
    return counts


def simulate(watermarked: Tensor, host: Tensor, names: Sequence[str] = tuple(TRAINING_ATTACKS)) -> Tensor:
    """Apply the training attack mix to a batch in [-1, 1]; samples are assigned to attacks at random."""
    image, host01 = (watermarked + 1) / 2, (host + 1) / 2
    order = torch.randperm(len(image), device=image.device)
    counts = allocate(len(image), [TRAINING_ATTACKS[name][0] for name in names])
    parts, start = [], 0
    for name, count in zip(names, counts, strict=True):
        index = order[start : start + count]
        parts.append(TRAINING_ATTACKS[name][1](image[index], host01[index]))
        start += count
    attacked = torch.cat(parts)[torch.argsort(order)]
    return attacked * 2 - 1


# Table 4 evaluation grid: (attack, strength label) → attack on [0, 1] images.
EVALUATION_ATTACKS: dict[tuple[str, str], Attack] = {
    ("No attack", "-"): identity,
    **{("Gaussian filtering", f"{k}x{k}"): partial(gaussian_filter, kernel=k) for k in (3, 5, 7, 9)},
    **{("Average filtering", f"{k}x{k}"): partial(average_filter, kernel=k) for k in (3, 5)},
    **{("Median filtering", f"{k}x{k}"): partial(median_filter, kernel=k) for k in (3, 5)},
    **{("Salt and pepper", f"{p}"): partial(salt_pepper, p=p) for p in (0.01, 0.03, 0.05, 0.07, 0.09)},
    **{("Gaussian noise", f"σ={s}"): partial(gaussian_noise, sigma=s) for s in (0.01, 0.03, 0.05, 0.08)},
    **{("Sharpening", f"{n}-point"): partial(sharpen, points=n) for n in (5, 9)},
    **{("JPEG", f"{q}"): partial(real_jpeg, quality=q) for q in (90, 70, 50, 30, 10)},
    **{("Rotation", f"{a}"): partial(rotation, degrees=float(a)) for a in (15, 30, 45, 60, 75)},
    **{("Crop", f"{p}"): partial(crop, keep=p) for p in (0.9, 0.7, 0.5, 0.3, 0.1)},
    **{("Cropout", f"{p}"): partial(cropout, replaced=p) for p in (0.1, 0.3, 0.5, 0.7, 0.9)},
    **{("Dropout", f"{p}"): partial(dropout, keep=p) for p in (0.9, 0.7, 0.5, 0.3, 0.1)},
}

# Table 4, "Random (Average)" column, BER %.
PAPER_BER: dict[tuple[str, str], float] = dict(
    zip(
        EVALUATION_ATTACKS,
        [
            0.7015,
            *(1.5944, 7.5255, 11.5115, 18.463),
            *(4.273, 5.2296),
            *(8.4184, 10.5548),
            *(0.861, 1.1798, 1.4031, 1.8176, 2.5829),
            *(0.8291, 1.977, 6.0906, 11.9898),
            *(3.2844, 3.9222),
            *(0.9566, 4.2411, 8.0676, 14.8916, 31.4732),
            *(2.0727, 4.9107, 5.0383, 3.8265, 1.7857),
            *(0.7015, 2.1365, 14.6365, 24.9681, 39.6365),
            *(2.2003, 9.0561, 17.0281, 24.0434, 34.8533),
            *(0.9247, 2.4554, 6.25, 14.8916, 34.1199),
        ],
        strict=True,
    )
)

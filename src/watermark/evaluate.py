"""Metrics (§4.3) and checkpoint evaluation.

`python -m watermark.evaluate --checkpoint models/storage/<run>/last.pt` prints PSNR and BER on the
training subset and the evaluation set, as a Markdown table.
"""

from pathlib import Path

import click
import torch
from torch import Tensor

from watermark.attacks import Attack, identity
from watermark.data import load, random_marks, to_pixels, to_signal
from watermark.model import Embedder, Extractor

EVAL_SEED = 1234  # fixed WMs, so every run and checkpoint is measured on the same marks


def psnr(host_pixels: Tensor, watermarked_pixels: Tensor) -> Tensor:
    """Per-image PSNR in dB (Eq. 5) between [0, 255] images."""
    mse = (host_pixels - watermarked_pixels).pow(2).flatten(1).mean(1)
    return 10 * torch.log10(255**2 / mse)


def ber(marks: Tensor, extracted: Tensor) -> Tensor:
    """Per-image bit error ratio in % (Eq. 6). WMs are {-1, +1}; the extraction is thresholded at 0."""
    errors = (extracted >= 0) != (marks > 0)
    return 100 * errors.float().flatten(1).mean(1)


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def measure(
    embedder: Embedder,
    extractor: Extractor,
    images: Tensor,
    attack: Attack = identity,
    strength: float = 1.0,
    batch: int = 64,
) -> tuple[float, float]:
    """Mean PSNR and BER over images. The watermarked image is quantized to integers before the attack,
    as the final image in §4.3."""
    embedder.eval()
    extractor.eval()
    run_device = next(embedder.parameters()).device
    generator = torch.Generator().manual_seed(EVAL_SEED)
    marks = random_marks(len(images), images.shape[-1] // 16, generator)
    psnrs, bers = [], []
    for start in range(0, len(images), batch):
        pixels = images[start : start + batch].to(run_device).float()
        mark = marks[start : start + batch].to(run_device)
        host = to_signal(pixels)
        watermarked = to_pixels(embedder(host, mark, strength))
        extracted = extractor(attack(to_signal(watermarked), host))
        psnrs.append(psnr(pixels, watermarked).cpu())
        bers.append(ber(mark, extracted).cpu())
    return torch.cat(psnrs).mean().item(), torch.cat(bers).mean().item()


def load_checkpoint(path: Path, run_device: torch.device) -> tuple[Embedder, Extractor]:
    state = torch.load(path, map_location=run_device)
    embedder, extractor = Embedder().to(run_device), Extractor().to(run_device)
    embedder.load_state_dict(state["embedder"])
    extractor.load_state_dict(state["extractor"])
    return embedder, extractor


@click.command(help=__doc__)
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--limit", type=int, default=None, help="Training images to measure (dev subset).")
def main(checkpoint: Path, limit: int | None) -> None:
    embedder, extractor = load_checkpoint(checkpoint, device())
    print("| Split | Images | PSNR (dB) | BER identity (%) |\n|---|---|---|---|")
    for split, images in (("train", load("train", limit)), ("eval", load("eval"))):
        value_psnr, value_ber = measure(embedder, extractor, images)
        print(f"| {split} | {len(images)} | {value_psnr:.2f} | {value_ber:.2f} |")


if __name__ == "__main__":
    main()

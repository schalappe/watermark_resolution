"""Metrics (§4.3) and the evaluation report (§4.4).

`python -m watermark.evaluate --checkpoint models/storage/<run>/last.pt` prints a Markdown report:
PSNR and BER on the training subset and the evaluation set, then Table 4 next to the paper's values.
`--full` adds Table 5 / Figure 6 (host resolutions) and Figure 7 (strength factor s).
"""

from pathlib import Path

import click
import torch
from torch import Tensor

from watermark.attacks import EVALUATION_ATTACKS, PAPER_BER, Attack
from watermark.data import load, random_marks, to_pixels, to_signal
from watermark.model import Embedder, Extractor

EVAL_SEED = 1234  # fixed WMs and attack randomness, so every checkpoint is measured the same way


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
    attacks: dict[tuple[str, str], Attack],
    strength: float = 1.0,
    batch: int = 64,
) -> tuple[Tensor, dict[tuple[str, str], float]]:
    """Per-image PSNR and mean BER per attack. The watermarked image is quantized to integers before the
    attack, as the final image in §4.3."""
    embedder.eval()
    extractor.eval()
    run_device = next(embedder.parameters()).device
    torch.manual_seed(EVAL_SEED)
    marks = random_marks(len(images), images.shape[-1] // 16, torch.Generator().manual_seed(EVAL_SEED))
    psnrs, bers = [], {key: [] for key in attacks}
    for start in range(0, len(images), batch):
        pixels = images[start : start + batch].to(run_device).float()
        mark = marks[start : start + batch].to(run_device)
        watermarked = to_pixels(embedder(to_signal(pixels), mark, strength))
        psnrs.append(psnr(pixels, watermarked).cpu())
        for key, attack in attacks.items():
            attacked = attack(watermarked / 255, pixels / 255)
            bers[key].append(ber(mark, extractor(attacked * 2 - 1)).cpu())
    return torch.cat(psnrs), {key: torch.cat(values).mean().item() for key, values in bers.items()}


def load_checkpoint(path: Path, run_device: torch.device) -> tuple[Embedder, Extractor]:
    state = torch.load(path, map_location=run_device)
    embedder, extractor = Embedder().to(run_device), Extractor().to(run_device)
    embedder.load_state_dict(state["embedder"])
    extractor.load_state_dict(state["extractor"])
    return embedder, extractor


NO_ATTACK = ("No attack", "-")
PAPER_PSNR = {64: 39.97, 128: 40.58, 256: 41.23, 512: 42.35}  # Table 5


def _ber_table(title: str, columns: list[str], results: list[dict[tuple[str, str], float]], paper: bool) -> None:
    print(f"\n### {title}\n")
    print("| Attack | Strength | " + " | ".join(columns) + (" | Paper |" if paper else " |"))
    print("|---|---|" + "---|" * (len(columns) + paper))
    for key in EVALUATION_ATTACKS:
        cells = [f"{r[key]:.2f}" for r in results] + ([f"{PAPER_BER[key]:.2f}"] if paper else [])
        print(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + " |")


@click.command(help=__doc__)
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--limit", type=int, default=None, help="Training images to measure (dev subset).")
@click.option("--full", is_flag=True, help="Also report Table 5 / Figure 6 and Figure 7.")
def main(checkpoint: Path, limit: int | None, full: bool) -> None:
    embedder, extractor = load_checkpoint(checkpoint, device())
    evaluation = load("eval")
    train_psnr, train_ber = measure(embedder, extractor, load("train", limit), EVALUATION_ATTACKS)
    eval_psnr, eval_ber = measure(embedder, extractor, evaluation, EVALUATION_ATTACKS)

    print(f"## Evaluation of `{checkpoint}`\n")
    print("| Split | Images | PSNR mean / min / max (dB) | BER no attack (%) | BER Table 4 mean (%) |")
    print("|---|---|---|---|---|")
    for split, psnrs, bers in (("train", train_psnr, train_ber), ("eval", eval_psnr, eval_ber)):
        mean_ber = sum(bers.values()) / len(bers)
        print(
            f"| {split} | {len(psnrs)} | {psnrs.mean():.2f} / {psnrs.min():.2f} / {psnrs.max():.2f}"
            f" | {bers[NO_ATTACK]:.2f} | {mean_ber:.2f} |"
        )
    print("| paper eval | 49 | 40.58 / 37.46 / 42.13 | 0.70 | - |")
    _ber_table("Table 4: BER (%) on the evaluation set, s = 1", ["Measured"], [eval_ber], paper=True)
    if not full:
        return

    sizes = list(PAPER_PSNR)
    by_size = [measure(embedder, extractor, load("eval", size=size), EVALUATION_ATTACKS) for size in sizes]
    print("\n### Table 5: PSNR (dB) by host resolution, s = 1\n")
    print("| Host | WM | Measured | Paper |\n|---|---|---|---|")
    for size, (psnrs, _) in zip(sizes, by_size, strict=True):
        print(f"| {size}x{size} | {size // 16}x{size // 16} | {psnrs.mean():.2f} | {PAPER_PSNR[size]:.2f} |")
    _ber_table("Figure 6: BER (%) by host resolution", [f"{s}x{s}" for s in sizes], [b for _, b in by_size], False)

    strengths = [0.5, 1.0, 1.5, 2.0]
    by_strength = [measure(embedder, extractor, evaluation, EVALUATION_ATTACKS, s) for s in strengths]
    print("\n### Figure 7: PSNR (dB) by strength factor s\n")
    print("| s | PSNR |\n|---|---|")
    for s, (psnrs, _) in zip(strengths, by_strength, strict=True):
        print(f"| {s} | {psnrs.mean():.2f} |")
    _ber_table(
        "Figure 7: BER (%) by strength factor s", [f"s={s}" for s in strengths], [b for _, b in by_strength], False
    )


if __name__ == "__main__":
    main()

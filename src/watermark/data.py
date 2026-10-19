"""Paper datasets (§4.1): BOSSbase for training, 49 standard grayscale images for evaluation.

`python -m watermark.data` downloads both sets once and caches them as uint8 tensors:
`train.pt` (Nx1x128x128) and `eval.pt` (49x1x512x512, kept at full size for the Table 5 resolutions).
"""

import io
import ssl
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import certifi
import click
import numpy as np
import torch
from PIL import Image

BOSS_URL = "https://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip"
BOSS_COUNT = 10_000
EVAL_URL = "https://ccia.ugr.es/cvg/CG/images/base/{}.gif"
EVAL_COUNT = 49
DATA_DIR = Path("data")
HOST_SIZE = 128
# uv-managed Python builds do not see the macOS keychain; certifi gives every platform the same CA bundle.
TLS = ssl.create_default_context(cafile=certifi.where())


def _fetch(url: str, attempts: int = 3) -> bytes:
    """Download `url`; HTTP errors raise, transient network errors are retried."""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=120, context=TLS) as response:
                return response.read()
        except (TimeoutError, ConnectionError, urllib.error.URLError) as error:
            if isinstance(error, urllib.error.HTTPError) or attempt == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _gray(image: Image.Image, size: int) -> torch.Tensor:
    """Image → 1×size×size uint8 tensor. BOX resampling averages pixels when down-sampling."""
    image = image.convert("L")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BOX)
    return torch.from_numpy(np.asarray(image).copy()).unsqueeze(0)


def prepare(data_dir: Path = DATA_DIR) -> None:
    """Download and cache both datasets; fails if any image is missing."""
    data_dir.mkdir(parents=True, exist_ok=True)

    archive = data_dir / "BOSSbase_1.01.zip"
    if not archive.exists():
        partial = archive.with_suffix(".part")
        with urllib.request.urlopen(BOSS_URL, timeout=120, context=TLS) as response, partial.open("wb") as file:
            while chunk := response.read(1 << 20):
                file.write(chunk)
        partial.rename(archive)
    with zipfile.ZipFile(archive) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".pgm"))
        if len(names) != BOSS_COUNT:
            raise RuntimeError(f"{archive} holds {len(names)} images, expected {BOSS_COUNT}")
        train = torch.stack([_gray(Image.open(io.BytesIO(zf.read(n))), HOST_SIZE) for n in names])
    torch.save(train, data_dir / "train.pt")

    evaluation = torch.stack(
        [_gray(Image.open(io.BytesIO(_fetch(EVAL_URL.format(i)))), 512) for i in range(1, EVAL_COUNT + 1)]
    )
    torch.save(evaluation, data_dir / "eval.pt")


def load(split: str, limit: int | None = None, size: int = HOST_SIZE, data_dir: Path = DATA_DIR) -> torch.Tensor:
    """Cached split as uint8 N×1×size×size; `limit` keeps the first N images (dev subset)."""
    images = torch.load(data_dir / f"{split}.pt")[:limit]
    if images.shape[-1] != size:
        images = torch.stack([_gray(Image.fromarray(image[0].numpy()), size) for image in images])
    return images


def to_signal(images: torch.Tensor) -> torch.Tensor:
    """uint8 [0, 255] → float [-1, 1], the network's host domain (§3.1)."""
    return images.float() / 127.5 - 1


def to_pixels(signal: torch.Tensor) -> torch.Tensor:
    """Network output [-1, 1] → integer-valued [0, 255] final image, as evaluated in §4.3."""
    return ((signal + 1) * 127.5).round().clamp(0, 255)


def random_marks(count: int, size: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Random binary WMs (on CPU) encoded as {-1, +1}, matching the extractor's tanh range."""
    bits = torch.randint(0, 2, (count, 1, size, size), generator=generator)
    return bits.float() * 2 - 1


@click.command(help=__doc__)
@click.option("--data-dir", type=click.Path(path_type=Path), default=DATA_DIR, show_default=True)
def main(data_dir: Path) -> None:
    prepare(data_dir)


if __name__ == "__main__":
    main()

"""Attack simulation between embedder and extractor (§3.2.4).

Every training attack must stay differentiable with respect to the watermarked image: the embedder
learns robustness only through the extraction loss flowing back through the attack.
Attacks take and return images in [-1, 1]; `host` is available for attacks that mix in host pixels.
"""

from collections.abc import Callable

from torch import Tensor

Attack = Callable[[Tensor, Tensor], Tensor]


def identity(watermarked: Tensor, host: Tensor) -> Tensor:
    return watermarked


TRAINING_ATTACKS: dict[str, Attack] = {"identity": identity}


def simulate(watermarked: Tensor, host: Tensor) -> Tensor:
    """Apply the training attack mix to a batch."""
    return identity(watermarked, host)

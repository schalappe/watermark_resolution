"""Joint training of embedder and extractor with the paper's settings (Table 3, §4.2).

`python -m watermark.train --run models/storage/<name>` trains from scratch or resumes from
`<run>/last.pt`. Every epoch appends a row to `<run>/log.csv`.
"""

import csv
import json
import random
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import click
import numpy as np
import torch
from torch import nn

from watermark.attacks import TRAINING_ATTACKS, simulate
from watermark.data import load, random_marks, to_pixels, to_signal
from watermark.evaluate import ber, device, psnr
from watermark.model import Embedder, Extractor

LOG_FIELDS = ["epoch", "l1", "l2", "l_emb", "l_ext", "psnr", "ber", "seconds"]


@click.command(help=__doc__)
@click.option("--run", type=click.Path(path_type=Path), required=True, help="Run directory (checkpoints + log).")
@click.option("--epochs", default=4000, show_default=True)
@click.option("--batch", default=100, show_default=True)
@click.option("--limit", type=int, default=None, help="Train on the first N images (dev subset).")
@click.option("--lr-embedder", default=1e-4, show_default=True)
@click.option("--lr-extractor", default=1e-5, show_default=True)
# §4.2: "Adam optimizer [...] the weight decay rate was 0.01", read as Adam's coupled L2 decay.
@click.option("--weight-decay", default=0.01, show_default=True)
@click.option("--lambdas", type=float, nargs=3, default=(45.0, 0.2, 20.0), show_default=True, help="λ1 λ2 λ3")
@click.option("--seed", default=0, show_default=True)
@click.option(
    "--attacks",
    default=",".join(TRAINING_ATTACKS),
    show_default=True,
    help="Comma-separated Table 2 attacks to train with; their ratios are renormalized.",
)
def main(**options) -> None:
    attacks = options["attacks"].split(",")
    if unknown := set(attacks) - set(TRAINING_ATTACKS):
        raise click.BadParameter(f"unknown attacks {sorted(unknown)}", param_hint="--attacks")
    # Lists, not tuples: the config is compared with its JSON copy when a run resumes.
    train(SimpleNamespace(**{**options, "lambdas": list(options["lambdas"]), "attacks": attacks}))


def train(args: SimpleNamespace) -> None:
    run_device = device()
    args.run.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)  # batch order and WMs

    images = load("train", args.limit)
    embedder, extractor = Embedder().to(run_device), Extractor().to(run_device)
    opt_emb = torch.optim.Adam(embedder.parameters(), lr=args.lr_embedder, weight_decay=args.weight_decay)
    opt_ext = torch.optim.Adam(extractor.parameters(), lr=args.lr_extractor, weight_decay=args.weight_decay)
    emb_params, ext_params = list(embedder.parameters()), list(extractor.parameters())
    lambda1, lambda2, lambda3 = args.lambdas

    checkpoint = args.run / "last.pt"
    config_path = args.run / "config.json"
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    start_epoch = 1
    if checkpoint.exists():
        # A resume may only extend the epoch target; any other change would silently mix two experiments.
        saved = json.loads(config_path.read_text())
        changed = {k: (saved.get(k), v) for k, v in config.items() if k != "epochs" and saved.get(k) != v}
        if changed:
            raise SystemExit(f"{args.run} was started with other settings (saved, given): {changed}")
        # CPU: RNG states must stay CPU ByteTensors; load_state_dict copies weights onto the model's device.
        state = torch.load(checkpoint, map_location="cpu")
        embedder.load_state_dict(state["embedder"])
        extractor.load_state_dict(state["extractor"])
        opt_emb.load_state_dict(state["opt_emb"])
        opt_ext.load_state_dict(state["opt_ext"])
        generator.set_state(state["generator"])
        torch.set_rng_state(state["torch_rng"])
        start_epoch = state["epoch"] + 1
    else:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        config_path.write_text(json.dumps({**config, "commit": commit, "device": str(run_device)}))

    log_path = args.run / "log.csv"
    new_log = not log_path.exists()
    with log_path.open("a", newline="") as log_file:
        log = csv.DictWriter(log_file, LOG_FIELDS)
        if new_log:
            log.writeheader()
        for epoch in range(start_epoch, args.epochs + 1):
            embedder.train()
            extractor.train()
            started = time.time()
            totals = dict.fromkeys(LOG_FIELDS[1:-1], 0.0)
            order = torch.randperm(len(images), generator=generator)
            for start in range(0, len(images), args.batch):
                index = order[start : start + args.batch]
                host = to_signal(images[index].to(run_device))
                # §4.2: a newly generated random WM for each mini-batch.
                mark = random_marks(len(index), host.shape[-1] // 16, generator).to(run_device)

                watermarked = embedder(host, mark)
                extracted = extractor(simulate(watermarked, host, args.attacks))
                l1 = (watermarked - host).pow(2).mean()  # Eq. 1
                l2 = (extracted - mark).abs().mean()  # Eq. 2
                l_emb = lambda1 * l1 + lambda2 * l2  # Eq. 3
                l_ext = lambda3 * l2  # Eq. 4

                # Each loss updates only its own network (Eqs. 3-4); L_emb still flows through the extractor.
                opt_emb.zero_grad()
                opt_ext.zero_grad()
                l_emb.backward(inputs=emb_params, retain_graph=True)
                l_ext.backward(inputs=ext_params)
                opt_emb.step()
                opt_ext.step()

                with torch.no_grad():
                    count = len(index)
                    totals["l1"] += l1.item() * count
                    totals["l2"] += l2.item() * count
                    totals["l_emb"] += l_emb.item() * count
                    totals["l_ext"] += l_ext.item() * count
                    totals["psnr"] += psnr(to_pixels(host), to_pixels(watermarked)).sum().item()
                    totals["ber"] += ber(mark, extracted).sum().item()

            row = {key: value / len(images) for key, value in totals.items()}
            # Checkpoint every epoch, before logging, so a resumed run never repeats a logged epoch.
            _save(checkpoint, embedder, extractor, opt_emb, opt_ext, generator, epoch)
            log.writerow({"epoch": epoch, **row, "seconds": round(time.time() - started, 2)})
            log_file.flush()


def _save(
    path: Path,
    embedder: nn.Module,
    extractor: nn.Module,
    opt_emb: torch.optim.Optimizer,
    opt_ext: torch.optim.Optimizer,
    generator: torch.Generator,
    epoch: int,
) -> None:
    """Atomic write: a crash during save must not destroy the previous checkpoint."""
    state = {
        "embedder": embedder.state_dict(),
        "extractor": extractor.state_dict(),
        "opt_emb": opt_emb.state_dict(),
        "opt_ext": opt_ext.state_dict(),
        "generator": generator.get_state(),
        "torch_rng": torch.get_rng_state(),
        "epoch": epoch,
    }
    partial = path.with_suffix(".part")
    torch.save(state, partial)
    partial.replace(path)


if __name__ == "__main__":
    main()

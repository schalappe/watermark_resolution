# Watermarking adaptive to the resolution of image

PyTorch + Kornia reproduction of **Convolutional Neural Network-Based Digital Image Watermarking Adaptive to the
Resolution of Image and Watermark**[^1] (Lee, Seo & Kim, 2020): a blind, invisible, and robust watermarking network
made of an embedder and an extractor, trained jointly through an attack simulation.

The goal is a faithful reproduction: architecture (§3.2), settings (Table 3), datasets (§4.1), and
measurements (§4.4). The plan and the measurement log live in [`docs/PLAN.md`](docs/PLAN.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/). Runs on Apple Silicon (MPS), CUDA, or CPU.

```bash
make sync
```

## Data

```bash
make data
```

Downloads the paper's datasets into `data/`:

- training: BOSSbase 1.01[^2], 10,000 grayscale 512×512 images (1.67 GB archive), down-sampled to 128×128;
- evaluation: the 49 standard 512×512 grayscale test images[^3].

## Training

```bash
make train ARGS="--run models/storage/paper"                         # Table 3 settings
make train ARGS="--run models/storage/dev --limit 128 --batch 64 --epochs 100"   # dev subset
```

Re-running the same command resumes from `<run>/last.pt`. Each epoch appends losses, PSNR, and BER to
`<run>/log.csv`.

## Evaluation

```bash
make evaluate ARGS="--checkpoint models/storage/dev/last.pt --limit 128"
```

## Results

Run `models/storage/paper`: Table 3 settings, full Table 2 attack mix, RTX 3090, stopped at epoch
2607 / 4000 once training loss, PSNR, and BER were stable (§4.2 "until the loss value is stable").
Full report: [`docs/report-paper.md`](docs/report-paper.md)
(`make evaluate ARGS="--checkpoint models/storage/paper/last.pt --limit 1000 --full"`).

| Measurement (evaluation set, 49 images, s = 1) | Measured | Paper | Acceptance |
|---|---|---|---|
| PSNR mean / min / max | 40.08 / 36.89 / 41.78 dB | 40.58 / 37.46 / 42.13 dB | ±1.5 dB: pass |
| BER, no attack | 1.18 % | 0.70 % | < 2 %: pass |
| BER, Table 4 attacks | see report | per row | trend holds except rotation: fail |
| PSNR at 64 / 128 / 256 / 512 px | 39.86 / 40.08 / 40.24 / 40.36 dB | 39.97 / 40.58 / 41.23 / 42.35 dB | rises with resolution: pass |
| Strength s = 0.5 → 2 | PSNR 41.91 → 37.85 dB, BER 3.22 → 0.48 % | both fall | monotonic: pass |

Selected Table 4 rows (BER %, measured / paper):

| Attack | Measured | Paper |
|---|---|---|
| Gaussian filter 9×9 | 8.26 | 18.46 |
| Median filter 5×5 | 8.93 | 10.55 |
| Gaussian noise σ = 0.08 | 17.41 | 11.99 |
| JPEG Q = 50 | 3.48 | 8.07 |
| JPEG Q = 10 | 28.51 | 31.47 |
| Rotation 15° / 45° / 75° | 51.24 / 50.16 / 50.29 | 2.07 / 5.04 / 1.79 |
| Crop 0.5 | 17.38 | 14.64 |
| Dropout 0.5 | 6.57 | 6.25 |

Deviations:

- **Rotation is not learned:** BER ≈ 50 % (chance) at every angle and resolution, against 1.8–5.0 % in
  the paper. Every other attack is within a few points of the paper, better on filtering and JPEG,
  worse on noise, salt and pepper, crop, and cropout. Cause not investigated.
- Training stopped at epoch 2607, not 4000.
- PSNR rises with resolution much less than in the paper (+0.5 dB from 64 to 512 px vs +2.4 dB).
- WM2 / WM3 adaptability columns of Table 4 are not reproduced (watermarks not published).
- Interpretation choices where the paper is silent: [`docs/PLAN.md` §2.3](docs/PLAN.md).

## Development

```bash
make test    # pytest
make check   # ruff format, ruff check, pyrefly
```

## Reference

Lee, Jae-Eun, Young-Ho Seo, and Dong-Wook Kim. 2020. "Convolutional Neural Network-Based Digital Image Watermarking
Adaptive to the Resolution of Image and Watermark" Applied Sciences 10, no. 19:6854. https://doi.org/10.3390/app10196854

[^1]: https://www.mdpi.com/2076-3417/10/19/6854

[^2]: https://dde.binghamton.edu/download/

[^3]: https://ccia.ugr.es/cvg/CG/base.htm

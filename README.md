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

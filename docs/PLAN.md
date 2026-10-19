# Reproduction plan: Lee, Seo & Kim (2020)

Paper: *Convolutional Neural Network-Based Digital Image Watermarking Adaptive to the Resolution of Image
and Watermark*, Appl. Sci. 10(19):6854 (`docs/applsci-10-06854.pdf`).

## Target

One straight path: a faithful PyTorch + Kornia reimplementation of the paper that trains with the
paper's settings (Table 3) on the paper's data (§4.1) and reports the paper's measurements (§4.4) side
by side with the published numbers. Anything that does not serve this target is removed (see
[Out of scope](#out-of-scope)).

## Decisions

| Decision | Choice | Evidence |
|---|---|---|
| Framework | PyTorch + Kornia, device `mps` → `cuda` → `cpu` | `tensorflow-metal` 1.2.0 (Jan 2025) is its last release, and the TF team states "we do not support the tensorflow-metal plugin" ([tensorflow#84167](https://github.com/tensorflow/tensorflow/issues/84167)). JAX has an experimental Metal GPU plugin ([Apple](https://developer.apple.com/metal/jax/)); `jax-metal` 0.1.1 (Oct 2024) is its last release, and its compatibility with this workload was not tested. Probe on M1 Pro, torch 2.14.1 + kornia 0.8.3 on MPS: the WM up-sampling block runs, and gradients flow through Gaussian/average/median filtering, sharpening, rotation, and differentiable JPEG (Q=50). Training also runs unchanged on CUDA, which Phase 3 needs. |
| Training data | BOSSbase 1.01, 10,000 grayscale 512×512 images, down-sampled to 128×128 | §4.1.1. Available at `https://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip` (1.67 GB, HTTP 200). |
| Evaluation data | The 49 standard 512×512 grayscale images, down-sampled to 128×128 | §4.1.1, ref. [17]. Available as `https://ccia.ugr.es/cvg/CG/images/base/{1..49}.gif` (HTTP 200). |
| Hyper-parameters | Paper Table 3, hard-coded as CLI defaults | Adam; embedder LR 1e-4; extractor LR 1e-5; batch 100; 4000 epochs; λ = 45 / 0.2 / 20; s = 1; weight decay 0.01 (§4.2). |
| Color | Grayscale only: the network input is Y | Both datasets are grayscale. §4.1.1 says BOSS was chosen because "it contains only gray images". |

## Development protocol (Phases 1–2)

Phases 1 and 2 never train on the full dataset. Measurements on a small fixed subset are sanity checks
against coding mistakes (crashes, no learning, a broken attack), not optimization: 128 images cannot
reach the paper's numbers. Results are judged in Phase 3.

- **Dev subset:** the first 128 BOSS images (sorted file names) for training. Evaluation uses the same
  128 images (can the networks learn?) and the 49 evaluation images (sanity check).
- **Fixed run:** same seed, batch 64 (two steps per epoch; dev-only deviation from batch 100), and the
  same epoch budget for every measurement. Start with 100 epochs, about 7 min on the M1 Pro
  [INFERENCE: extrapolated from the 2.96 s/step measured at batch 100]. If the first measurement shows
  no signal, raise the budget once, then keep it fixed so measurements stay comparable.
- **Who runs it:** runs longer than about 30 s go to a `verifier` agent with the exact command and the
  expected numbers.
- **Measurement log:** after each step, append one row to the table below.

| # | Step | Commit | L1 | L2 | PSNR train / eval (dB) | BER identity train / eval (%) | BER attacked train / eval (%) |
|---|---|---|---|---|---|---|---|
| M0 | Phase 1, identity only, 100 epochs (+2 resumed) | uncommitted | 0.0038 | 0.134 | 32.30 / 31.14 | 3.83 / 6.57 | – |
| M1 | + gaussian_filter, 100 epochs | uncommitted | 0.0038 | 0.317 | 31.89 / 30.78 | 11.21 / 16.04 | 24.21 / 27.25 (Table 4 mean) |
| M2 | + average_filter, 100 epochs | uncommitted | 0.0038 | 0.347 | 31.86 / 30.77 | 13.32 / 17.60 | 25.49 / 28.50 (Table 4 mean) |
| M3 | + median_filter, 100 epochs | uncommitted | 0.0038 | 0.344 | 31.86 / 30.77 | 13.16 / 17.76 | 25.43 / 28.49 (Table 4 mean) |
| M4 | + salt_pepper, 100 epochs | uncommitted | 0.0037 | 0.555 | 31.92 / 30.86 | 18.35 / 22.39 | 29.26 / 31.88 (Table 4 mean) |
| M5 | + gaussian_noise, 100 epochs | uncommitted | 0.0037 | 0.611 | 31.98 / 30.91 | 19.58 / 24.11 | 30.27 / 32.95 (Table 4 mean) |
| M6 | + sharpening, 100 epochs | uncommitted | 0.0034 | 0.811 | 32.36 / 31.24 | 27.14 / 31.79 | 36.30 / 39.39 (Table 4 mean) |
| M7 | + jpeg, 100 epochs | uncommitted | 0.0034 | 0.819 | 32.45 / 31.31 | 28.93 / 33.23 | 37.11 / 40.18 (Table 4 mean) |
| M8 | + rotation, 100 epochs | uncommitted | 0.0033 | 0.879 | 32.66 / 31.47 | 33.58 / 38.39 | 40.33 / 43.02 (Table 4 mean) |
| M9 | + crop, 100 epochs | uncommitted | 0.0032 | 0.892 | 32.70 / 31.50 | 34.72 / 38.62 | 41.16 / 43.77 (Table 4 mean) |
| M10 | + dropout, 100 epochs | uncommitted | 0.0033 | 0.885 | 32.67 / 31.47 | 32.79 / 36.48 | 39.95 / 42.37 (Table 4 mean) |

Expected direction:

- **M0, end of Phase 1, identity attack only:** BER identity well below 50 % on the dev subset, and PSNR
  rising across epochs. This is an indicator only: the extractor could learn a signal the embedder
  already leaks without any L2 gradient. The proof that the 2023 bug is gone is the L2-gradient test
  (1.6).
- **M1…M10, Phase 2, one row per added attack:** every run trains and evaluates without error, and the
  training BER falls below 50 %. Observed: harder mixes learn later (at epoch 100, M10's BER is 38.5 %
  and still falling, where M0 was at 3.9 %); with a fixed 100-epoch budget, BER rises as attacks are
  added. Expected, not a defect.
- **Final Phase 2 row, full Table 2 mix:** every attack's BER is reported. This row is the baseline that
  Phase 3 scales up.

## Phase 1: Rebuild (removes every crash and bug)

The rewrite replaces the TensorFlow code instead of patching it. Every finding from the review is
resolved either by deletion or by a requirement on the new code (table below).

### 1.1 Project and tooling

- [x] `pyproject.toml`: dependencies `torch`, `kornia`, `pillow`, `numpy`; dev group `pytest`, `ruff`,
      `pyrefly`. Remove `tensorflow`, `tensorflow-metal`, `imgaug`, `optuna`, `optuna-dashboard`,
      `scikit-learn`, `pydot`, `requests`, `loguru`, `rich`, `python-dotenv`, `black`, `isort`, `pylint`,
      `radon`, `jupyterlab`, `ipywidgets`.
- [x] Package `src/watermark/` (importable as `watermark`), replacing the top-level `src` package.
- [x] `Makefile`: targets `sync`, `data`, `train`, `evaluate`, `test`, `check`. Remove `.ONESHELL` and
      the `.env` round trip; paths are CLI arguments with defaults under `data/` and `models/`.
- [x] Code quality: `ruff format` (formatter), `ruff check` (linter), `pyrefly check` (type checker),
      all run by `make check`. Delete `pylintrc`; ruff replaces black, isort, and pylint.
- [x] Rewrite `README.md` for the new commands and data sources.

Target layout:

```text
src/watermark/
  data.py       download + decode + resize to 128×128, cached as one uint8 tensor per split
  model.py      Embedder (host pre-processing, WM pre-processing, embedding), Extractor
  attacks.py    training attack simulation (Table 2), evaluation attacks (Table 4)
  train.py      training loop, checkpoint/resume, CLI with Table 3 defaults
  evaluate.py   PSNR, BER, Table 4 / Table 5 / strength-sweep reports
tests/test_watermark.py
```

### 1.2 Data (`data.py`)

- [x] Download BOSSbase and the 49 evaluation images with fixed file names. Check the HTTP status,
      retry transient failures a bounded number of times, and fail the run if any image is missing.
- [x] Decode as grayscale, resize 512 → 128 with anti-aliasing, and store each split as one uint8
      tensor (`10000×1×128×128` ≈ 164 MB). Training keeps the whole set in memory, so no data loader
      and no cache. A `--limit N` option keeps the first N sorted images: the 128-image dev subset.
- [x] Watermarks: a fresh random 8×8 binary WM for every mini-batch (§4.1.2, §4.2), generated in the
      training loop. Evaluation WMs come from a fixed seed, so every run and checkpoint sees the same
      marks.

### 1.3 Model (`model.py`)

- [x] Host pre-processing: Conv 3×3, 64 filters, stride 1 (§3.2.1).
- [x] WM pre-processing: 3 × [ConvTranspose 3×3 stride 2 → BN → ReLU → AvgPool 2×2 stride 1], then
      [ConvTranspose → AvgPool], filters 512 / 256 / 128 / 1, output × s (§3.2.2). Each block maps
      n → 2n, so 8 → 128.
- [x] Embedder: concatenate (64 + 1) channels → 4 × [Conv → BN → ReLU, 64 filters] → Conv → tanh,
      1 filter, stride 1 (§3.2.3).
- [x] Extractor: 3 × [Conv stride 2 → BN → ReLU], filters 128 / 256 / 512 → Conv stride 2 → tanh,
      1 filter (§3.2.5).
- [x] Fully convolutional, no dense layer. The same weights must run at 64/4, 128/8, 256/16, and
      512/32 (§4.4.4).
- [x] `s` is a forward argument, not a constructor constant: Figure 7 sweeps it at inference.

### 1.4 Losses and metrics

- [x] L1 = MSE between host Y and embedder output, both in [−1, 1] (Eq. 1, §3.1).
- [x] L2 = MAE between original WM and extracted WM, both in {−1, +1} encoding (Eq. 2).
- [x] L_emb = λ1·L1 + λ2·L2 updates only the embedder; L_ext = λ3·L2 updates only the extractor
      (Eqs. 3–4).
- [x] PSNR (Eq. 5) on the final image: map [−1, 1] → [0, 255], round, clamp, then compute per image.
- [x] BER (Eq. 6): threshold the extracted WM at 0, compare bits, compute per image.
- [x] Every reported number is the mean over images, not over batches (§4.4.2).

### 1.5 Training loop (`train.py`)

- [x] One forward pass per step: embed → attack simulation → extract. Two Adam optimizers with
      separate parameter groups.
- [x] Seed torch, Python, and NumPy. Record the seed and git commit in the run directory.
- [x] Every epoch, save a checkpoint atomically (both models, both optimizers, epoch, RNG state);
      re-running the same `--run` resumes. A run lasts days (Phase 3), so a crash must not lose it.
- [x] Append a per-epoch CSV row: L1, L2, L_emb, L_ext, training PSNR, training BER.

### Review findings → resolution

| Review finding | Resolution |
|---|---|
| `keras.optimizers.legacy`, `reset_states`, `EarlyStopping` mode, lost tensor shapes | Deleted with the TF code. |
| No gradient from L2 to the embedder (`tf.numpy_function`) | Attacks are differentiable torch/Kornia ops. A test computes the gradient of L2 alone with respect to the embedder weights and asserts that it is finite and non-zero through every training attack. A total-loss gradient would not do: in the 2023 bug it was non-zero thanks to L1 alone. |
| BER compares continuous outputs with bits | Threshold at 0 (1.4). A test asserts BER = 0 for a perfect extraction and 100 for an inverted one. |
| Downloaded file names contain the URL signature | Hugging Face downloader deleted; fixed file names and completeness check (1.2). |
| Pixels outside [0, 255] wrap around on `uint8` cast | Training stays in [−1, 1] float; evaluation rounds and clamps (1.4). |
| `cache()` freezes WMs and shuffle order | No cache; fresh WM per mini-batch; reshuffle every epoch (1.2). |
| Broken and unused Arnold scrambling | Deleted (see Out of scope). |
| Image loss on RGB [0, 1] (¼ of the paper's scale) | L1 on Y in [−1, 1] (1.4). |
| Metrics averaged per batch | Per-image mean (1.4). |
| Unseeded trials, unsorted `glob` | Seeds recorded; sorted, fixed file lists (1.2, 1.5). |
| `tensorflow-metal` breaks `uv sync` off macOS, `optuna` in dev group, `imgaug` blocks NumPy 2 | All removed (1.1). |
| Silent `pylintrc` (`no-member` disabled) | Replaced by ruff (1.1). |
| No tests | `tests/test_watermark.py` (1.6). |

### 1.6 Checks

- [x] `tests/test_watermark.py`, kept small:
  - shapes at the four Table 5 resolutions;
  - gradient of L2 alone with respect to the embedder weights: finite and non-zero through each training
    attack;
  - BER boundaries (0 and 100);
- [x] Measurement M0 (Development protocol): identity attack only, dev subset. Phase 1 is done when the
      L2-gradient test passes and M0 shows the expected direction.

## Phase 2: Paper fidelity

### 2.1 Attack simulation (Table 2, §3.2.4)

- [x] Every mini-batch contains every attack at the Table 2 ratios. For a batch of size B, per-attack
      counts use the largest-remainder rule on `B × ratio`, and samples are assigned at random to attacks.
- [x] Each sample gets a random strength within its attack's Table 2 range.
- [x] Add attacks one at a time, in table order, with one measurement row each (Development protocol):
      `train --attacks identity,gaussian_filter,...` trains on a subset with renormalized ratios.
- [x] Test: per-batch attack allocation matches the Table 2 ratios.

| Attack | Ratio | Training strength | Implementation |
|---|---|---|---|
| Identity | 1/12 | – | pass-through |
| Gaussian filtering | 2/12 | kernel 3, 5, 7, 9 | `kornia.filters.gaussian_blur2d` |
| Average filtering | 2/12 | kernel 3, 5 | `kornia.filters.box_blur` |
| Median filtering | 1/12 | kernel 3, 5 | `kornia.filters.median_blur` |
| Salt & pepper | 0.5/12 | p = 0.1 | random mask set to 0 / 1 |
| Gaussian noise | 0.5/12 | σ = 0.1 | additive noise |
| Sharpening | 1/12 | 5-point, 9-point stencil | `kornia.filters.filter2d` |
| JPEG | 1/12 | Q = 50 | `kornia.enhance.jpeg_codec_differentiable` |
| Rotation | 1/12 | 0–90° | `kornia.geometry.transform.rotate` |
| Crop | 1/12 | keep 0.5–0.8 | mask outside the kept region |
| Dropout | 1/12 | keep 0.3–0.9 | keep watermarked pixels with prob. p, host pixels elsewhere |

### 2.2 Evaluation (`evaluate.py`)

- [x] §4.4.1: mean, min, and max PSNR on the 49 evaluation images at s = 1. Training-set PSNR is
      reported too.
- [x] Table 4: BER per attack and per strength, with Table 4's exact grid, including cropout (evaluation
      only). Evaluation JPEG uses real JPEG encoding (Pillow) on the quantized image, not the
      differentiable approximation.
- [-] Table 4, "WM adaptability" (WM2, WM3): not reproduced. The paper does not publish these two
      watermark images; the "Random (Average)" column, which the report covers, is the main claim.
- [x] Table 5 and Figure 6: PSNR and BER at 64/4, 128/8, 256/16, 512/32 with the 128/8-trained weights.
- [x] Figure 7: PSNR and BER for s ∈ [0.5, 2] at inference.
- [x] Output: one Markdown report per checkpoint, with the paper's value next to each measured value.

### 2.3 Interpretation decisions

The paper leaves these points open. Each choice is documented in code where it applies.

| Topic | Choice | Evidence |
|---|---|---|
| WM channel counts | 512 / 256 / 128 / 1 (text) | §3.2.2 text. Figure 2 labels 512 / 128 / 64 / 1; the text is the more explicit source. |
| Kernel size outside host pre-processing | 3×3 | Only §3.2.1 states 3×3; Figure 2 shows no kernel sizes. |
| WM encoding | bits {0, 1} → {−1, +1}; threshold 0 | Extractor ends with tanh (§3.2.5), the host is in [−1, 1] (§3.1). |
| Attack value domain | attacks run on [0, 1] images; σ and p are defined on [0, 1] | [INFERENCE] Not stated. Table 4's σ range (0.01–0.08) fits a [0, 1] scale. |
| Gaussian-filter σ for kernel k | OpenCV rule `0.3·((k−1)/2 − 1) + 0.8` | [INFERENCE] Not stated. |
| Crop p | kept area fraction; the rest is set to 0; image size unchanged | Table 4: BER falls as p rises (0.9 → 0.70 %, 0.1 → 39.6 %). The fixed-stride extractor needs a constant input size. |
| Cropout p | fraction of the area replaced by host pixels | Table 4: BER rises with p (0.1 → 2.2 %, 0.9 → 34.9 %). |
| Dropout p | fraction of watermarked pixels kept; the others come from the host | Table 4: BER falls as p rises (0.9 → 0.92 %, 0.1 → 34.1 %). |
| Weight decay 0.01 | `torch.optim.Adam(weight_decay=0.01)` (coupled L2) | §4.2 says "Adam optimizer" and "weight decay rate 0.01". AdamW is the alternative if training stalls. |
| Quantization during training | none (continuous [−1, 1]) | §4.3 quantizes only "as a final watermarked image used for the invisibility evaluation". |

## Phase 3: Run

### 3.1 Training-resource research (not on the Mac)

The Mac is excluded for the full run. Measured on the M1 Pro (MPS, batch 100, model + one Gaussian-blur
attack): 2.96 s/step ≈ 296 s/epoch ≈ **13.7 days for 4000 epochs**. The paper took about 4 days on an
RTX 2080 Ti (§4.2).

- [x] Compare the ways to train on a CUDA GPU. Candidates to check, none verified yet: hourly GPU
      rental (e.g. RunPod, Vast.ai, Lambda), free notebook quotas (e.g. Kaggle, Colab), and cloud spot
      instances.
- [x] Criteria for each option:
  - a GPU at least as fast as an RTX 2080 Ti, supported by current PyTorch wheels;
  - disk for the 1.67 GB BOSS archive, the cached tensors, and the checkpoints;
  - session limits and interruptions, which the checkpoint/resume of 1.5 must cover;
  - price per hour;
  - total = step time × 400,000 steps (4000 epochs × 100 steps).
- [x] Deliverable: [`docs/training-resources.md`](training-resources.md), a comparison table with sourced
      prices, estimated duration, and estimated total cost, plus a recommendation. The choice is yours
      (money).
- [x] On the chosen option, a short benchmark with the full Table 2 attack mix (run by a `verifier`)
      confirms the step time before the full run starts. Runpod RTX 4090 was chosen on 2026-10-18 but was
      out of stock; RTX 3090 Community ($0.22/h) used instead. Measured 2026-10-18: 0.212 s/step (mean of
      epochs 2–10, `--limit 1000`, batch 100), GPU utilisation 86–89 % → ≈ 23.6 h and ≈ $5.2 for 400k
      steps [INFERENCE]. Runbook: `docs/training-resources.md`.

### 3.2 Full run

- [x] Training length: 4000 epochs (Table 3), or stop once the loss is stable, since the paper trained
      "until the loss value is stable" (§4.2). The completed epoch count is recorded in the report.
      Stopped at **epoch 2607 / 4000** on 2026-10-19 07:28 UTC: from epoch 1124 on, training BER stayed
      within 7.14–7.39 % and training PSNR within 41.0–41.7 dB (hourly readings); last hour BER −0.02 pt.
- [x] Full training with Table 3 settings: run `models/storage/paper` (commit `8105a4a`, RTX 3090,
      torch 2.14.1+cu126), ≈ 21.6 s/epoch, ≈ 16 h.
- [x] Evaluation report (2.2) on the final checkpoint: [`docs/report-paper.md`](report-paper.md).
      Rotation BER ≈ 50 % (chance) at every angle: not learned; every other criterion passes.
- [x] `README.md` results section: the measured vs paper table, plus the deviations from 2.3.

## Acceptance criteria

| Measurement | Paper | Acceptance (proposal) |
|---|---|---|
| Mean PSNR, evaluation set, s = 1 | 40.58 dB | within ±1.5 dB |
| BER, no attack | 0.70 % | < 2 % |
| BER, Table 4 attacks | per row | reported for every row. Trend matches: BER rises with attack strength. |
| PSNR, Table 5 resolutions | 39.97 / 40.58 / 41.23 / 42.35 dB | PSNR rises with resolution |
| Strength sweep, Figure 7 | PSNR falls and BER falls as s rises | both monotonic |
| `make test` | – | green |

## Out of scope

Removed by this plan. Each item says when it could come back.

- **Optuna hyper-parameter search**, `models/best_params.json`, `models/search_space.json`: the paper
  gives every hyper-parameter (Table 3). Re-add only if the paper settings fail to converge.
- **Hugging Face datasets** and `src/data/build_dataset.py`: replaced by the paper's datasets.
- **Host-image augmentation** (flip, hue, saturation, brightness, contrast): not in the paper and
  meaningless on grayscale images.
- **Arnold scrambling** (§3.1, §4.1.2): scrambling a random i.i.d. WM with a fixed key does not change
  what the networks learn. Add it as pre/post-processing outside the networks when embedding a real logo.
- **RGB / YCbCr handling** (Figure 1): every experiment in the paper is grayscale. Add it when color hosts
  are needed.
- **Notebooks** (`notebooks/*.ipynb`): they import the deleted TF code, and their committed results are
  chance-level (BER ≈ 50 %). The Phase 2 Markdown report replaces them.

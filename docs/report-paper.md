## Evaluation of `models/storage/paper/last.pt`

- Training: run `models/storage/paper`, commit `8105a4a`, Table 3 settings, full Table 2 attack mix,
  RTX 3090. Completed **epoch 2607 / 4000**, stopped on 2026-10-19 07:28 UTC because the loss was stable
  (§4.2): from epoch 1124 on, training BER stayed within 7.14–7.39 % and training PSNR within
  41.0–41.7 dB.
- Command: `make evaluate ARGS="--checkpoint models/storage/paper/last.pt --limit 1000 --full"`; the
  train row covers the first 1000 training images.

| Split | Images | PSNR mean / min / max (dB) | BER no attack (%) | BER Table 4 mean (%) |
|---|---|---|---|---|
| train | 1000 | 41.70 / 37.07 / 46.47 | 0.42 | 13.33 |
| eval | 49 | 40.08 / 36.89 / 41.78 | 1.18 | 14.53 |
| paper eval | 49 | 40.58 / 37.46 / 42.13 | 0.70 | - |

### Table 4: BER (%) on the evaluation set, s = 1

| Attack | Strength | Measured | Paper |
|---|---|---|---|
| No attack | - | 1.18 | 0.70 |
| Gaussian filtering | 3x3 | 1.91 | 1.59 |
| Gaussian filtering | 5x5 | 3.48 | 7.53 |
| Gaussian filtering | 7x7 | 5.45 | 11.51 |
| Gaussian filtering | 9x9 | 8.26 | 18.46 |
| Average filtering | 3x3 | 4.11 | 4.27 |
| Average filtering | 5x5 | 7.91 | 5.23 |
| Median filtering | 3x3 | 3.25 | 8.42 |
| Median filtering | 5x5 | 8.93 | 10.55 |
| Salt and pepper | 0.01 | 1.37 | 0.86 |
| Salt and pepper | 0.03 | 1.75 | 1.18 |
| Salt and pepper | 0.05 | 2.10 | 1.40 |
| Salt and pepper | 0.07 | 3.51 | 1.82 |
| Salt and pepper | 0.09 | 4.40 | 2.58 |
| Gaussian noise | σ=0.01 | 1.40 | 0.83 |
| Gaussian noise | σ=0.03 | 3.67 | 1.98 |
| Gaussian noise | σ=0.05 | 9.41 | 6.09 |
| Gaussian noise | σ=0.08 | 17.41 | 11.99 |
| Sharpening | 5-point | 2.30 | 3.28 |
| Sharpening | 9-point | 2.39 | 3.92 |
| JPEG | 90 | 1.28 | 0.96 |
| JPEG | 70 | 2.23 | 4.24 |
| JPEG | 50 | 3.48 | 8.07 |
| JPEG | 30 | 8.77 | 14.89 |
| JPEG | 10 | 28.51 | 31.47 |
| Rotation | 15 | 51.24 | 2.07 |
| Rotation | 30 | 48.69 | 4.91 |
| Rotation | 45 | 50.16 | 5.04 |
| Rotation | 60 | 49.71 | 3.83 |
| Rotation | 75 | 50.29 | 1.79 |
| Crop | 0.9 | 1.66 | 0.70 |
| Crop | 0.7 | 8.20 | 2.14 |
| Crop | 0.5 | 17.38 | 14.64 |
| Crop | 0.3 | 28.48 | 24.97 |
| Crop | 0.1 | 41.20 | 39.64 |
| Cropout | 0.1 | 3.95 | 2.20 |
| Cropout | 0.3 | 11.67 | 9.06 |
| Cropout | 0.5 | 20.63 | 17.03 |
| Cropout | 0.7 | 29.69 | 24.04 |
| Cropout | 0.9 | 41.07 | 34.85 |
| Dropout | 0.9 | 1.47 | 0.92 |
| Dropout | 0.7 | 3.38 | 2.46 |
| Dropout | 0.5 | 6.57 | 6.25 |
| Dropout | 0.3 | 15.85 | 14.89 |
| Dropout | 0.1 | 34.09 | 34.12 |

### Table 5: PSNR (dB) by host resolution, s = 1

| Host | WM | Measured | Paper |
|---|---|---|---|
| 64x64 | 4x4 | 39.86 | 39.97 |
| 128x128 | 8x8 | 40.08 | 40.58 |
| 256x256 | 16x16 | 40.24 | 41.23 |
| 512x512 | 32x32 | 40.36 | 42.35 |

### Figure 6: BER (%) by host resolution

| Attack | Strength | 64x64 | 128x128 | 256x256 | 512x512 |
|---|---|---|---|---|---|
| No attack | - | 1.53 | 1.18 | 1.32 | 1.60 |
| Gaussian filtering | 3x3 | 3.06 | 1.91 | 1.88 | 2.36 |
| Gaussian filtering | 5x5 | 5.10 | 3.48 | 2.78 | 3.21 |
| Gaussian filtering | 7x7 | 7.14 | 5.45 | 4.19 | 4.55 |
| Gaussian filtering | 9x9 | 9.44 | 8.26 | 6.35 | 6.32 |
| Average filtering | 3x3 | 5.87 | 4.11 | 3.56 | 4.06 |
| Average filtering | 5x5 | 9.18 | 7.91 | 6.81 | 6.64 |
| Median filtering | 3x3 | 5.23 | 3.25 | 3.56 | 3.82 |
| Median filtering | 5x5 | 14.03 | 8.93 | 8.84 | 8.52 |
| Salt and pepper | 0.01 | 2.17 | 1.37 | 1.55 | 1.69 |
| Salt and pepper | 0.03 | 2.42 | 1.75 | 1.76 | 2.18 |
| Salt and pepper | 0.05 | 3.06 | 2.10 | 2.33 | 2.66 |
| Salt and pepper | 0.07 | 3.32 | 3.51 | 3.12 | 3.32 |
| Salt and pepper | 0.09 | 4.97 | 4.40 | 3.80 | 4.15 |
| Gaussian noise | σ=0.01 | 1.53 | 1.40 | 1.63 | 1.82 |
| Gaussian noise | σ=0.03 | 4.97 | 3.67 | 4.34 | 4.33 |
| Gaussian noise | σ=0.05 | 9.95 | 9.41 | 9.28 | 9.91 |
| Gaussian noise | σ=0.08 | 15.94 | 17.41 | 17.88 | 18.28 |
| Sharpening | 5-point | 2.17 | 2.30 | 2.01 | 2.18 |
| Sharpening | 9-point | 3.95 | 2.39 | 2.38 | 2.43 |
| JPEG | 90 | 1.66 | 1.28 | 1.55 | 1.80 |
| JPEG | 70 | 3.57 | 2.23 | 2.50 | 2.83 |
| JPEG | 50 | 4.85 | 3.48 | 3.96 | 4.08 |
| JPEG | 30 | 9.44 | 8.77 | 9.46 | 10.07 |
| JPEG | 10 | 29.34 | 28.51 | 30.29 | 32.41 |
| Rotation | 15 | 42.98 | 51.24 | 49.95 | 49.78 |
| Rotation | 30 | 50.26 | 48.69 | 49.85 | 49.88 |
| Rotation | 45 | 46.30 | 50.16 | 50.22 | 49.58 |
| Rotation | 60 | 51.91 | 49.71 | 49.80 | 50.26 |
| Rotation | 75 | 49.36 | 50.29 | 49.62 | 49.72 |
| Crop | 0.9 | 1.91 | 1.66 | 2.26 | 3.97 |
| Crop | 0.7 | 3.95 | 8.20 | 10.79 | 13.52 |
| Crop | 0.5 | 13.14 | 17.38 | 20.60 | 23.43 |
| Crop | 0.3 | 24.36 | 28.48 | 32.00 | 33.89 |
| Crop | 0.1 | 38.90 | 41.20 | 43.38 | 43.94 |
| Cropout | 0.1 | 3.32 | 3.95 | 5.09 | 5.84 |
| Cropout | 0.3 | 10.20 | 11.67 | 13.90 | 15.10 |
| Cropout | 0.5 | 17.35 | 20.63 | 23.49 | 24.46 |
| Cropout | 0.7 | 29.72 | 29.69 | 32.52 | 33.91 |
| Cropout | 0.9 | 45.03 | 41.07 | 42.94 | 43.60 |
| Dropout | 0.9 | 1.91 | 1.47 | 1.73 | 2.03 |
| Dropout | 0.7 | 4.59 | 3.38 | 3.15 | 3.67 |
| Dropout | 0.5 | 8.80 | 6.57 | 6.42 | 7.39 |
| Dropout | 0.3 | 18.37 | 15.85 | 14.82 | 16.51 |
| Dropout | 0.1 | 36.10 | 34.09 | 33.54 | 35.38 |

### Figure 7: PSNR (dB) by strength factor s

| s | PSNR |
|---|---|
| 0.5 | 41.91 |
| 1.0 | 40.08 |
| 1.5 | 38.85 |
| 2.0 | 37.85 |

### Figure 7: BER (%) by strength factor s

| Attack | Strength | s=0.5 | s=1.0 | s=1.5 | s=2.0 |
|---|---|---|---|---|---|
| No attack | - | 3.22 | 1.18 | 0.67 | 0.48 |
| Gaussian filtering | 3x3 | 5.17 | 1.91 | 1.18 | 0.89 |
| Gaussian filtering | 5x5 | 7.14 | 3.48 | 1.75 | 1.43 |
| Gaussian filtering | 7x7 | 10.14 | 5.45 | 3.92 | 2.81 |
| Gaussian filtering | 9x9 | 13.84 | 8.26 | 5.77 | 4.34 |
| Average filtering | 3x3 | 8.29 | 4.11 | 3.16 | 2.36 |
| Average filtering | 5x5 | 13.52 | 7.91 | 5.55 | 4.27 |
| Median filtering | 3x3 | 7.05 | 3.25 | 2.26 | 1.85 |
| Median filtering | 5x5 | 15.15 | 8.93 | 6.03 | 5.13 |
| Salt and pepper | 0.01 | 3.48 | 1.37 | 0.64 | 0.38 |
| Salt and pepper | 0.03 | 4.46 | 1.75 | 1.02 | 0.73 |
| Salt and pepper | 0.05 | 6.15 | 2.10 | 1.18 | 0.80 |
| Salt and pepper | 0.07 | 7.91 | 3.51 | 1.69 | 1.02 |
| Salt and pepper | 0.09 | 9.02 | 4.40 | 2.68 | 1.53 |
| Gaussian noise | σ=0.01 | 4.27 | 1.40 | 0.70 | 0.51 |
| Gaussian noise | σ=0.03 | 9.69 | 3.67 | 1.72 | 0.77 |
| Gaussian noise | σ=0.05 | 18.34 | 9.41 | 5.77 | 3.76 |
| Gaussian noise | σ=0.08 | 25.77 | 17.41 | 12.60 | 9.38 |
| Sharpening | 5-point | 5.20 | 2.30 | 1.53 | 0.89 |
| Sharpening | 9-point | 5.84 | 2.39 | 1.43 | 1.02 |
| JPEG | 90 | 4.11 | 1.28 | 0.83 | 0.54 |
| JPEG | 70 | 6.57 | 2.23 | 1.12 | 0.67 |
| JPEG | 50 | 9.34 | 3.48 | 2.14 | 1.15 |
| JPEG | 30 | 16.17 | 8.77 | 5.61 | 3.64 |
| JPEG | 10 | 34.22 | 28.51 | 25.16 | 21.62 |
| Rotation | 15 | 50.64 | 51.24 | 50.99 | 50.51 |
| Rotation | 30 | 49.20 | 48.69 | 49.01 | 49.11 |
| Rotation | 45 | 50.10 | 50.16 | 50.57 | 50.41 |
| Rotation | 60 | 49.55 | 49.71 | 49.52 | 49.46 |
| Rotation | 75 | 50.67 | 50.29 | 50.19 | 50.16 |
| Crop | 0.9 | 4.24 | 1.66 | 0.86 | 0.61 |
| Crop | 0.7 | 11.35 | 8.20 | 7.24 | 6.47 |
| Crop | 0.5 | 20.09 | 17.38 | 16.49 | 15.94 |
| Crop | 0.3 | 30.84 | 28.48 | 27.87 | 27.07 |
| Crop | 0.1 | 42.06 | 41.20 | 40.56 | 40.37 |
| Cropout | 0.1 | 6.38 | 3.95 | 3.28 | 2.71 |
| Cropout | 0.3 | 14.16 | 11.67 | 10.75 | 10.24 |
| Cropout | 0.5 | 23.12 | 20.63 | 19.83 | 19.23 |
| Cropout | 0.7 | 31.89 | 29.69 | 29.05 | 28.70 |
| Cropout | 0.9 | 42.38 | 41.07 | 40.31 | 39.73 |
| Dropout | 0.9 | 3.99 | 1.47 | 0.86 | 0.51 |
| Dropout | 0.7 | 7.17 | 3.38 | 1.79 | 1.08 |
| Dropout | 0.5 | 12.63 | 6.57 | 4.40 | 3.60 |
| Dropout | 0.3 | 21.91 | 15.85 | 12.79 | 10.27 |
| Dropout | 0.1 | 38.07 | 34.09 | 31.47 | 29.82 |

# Training resources for the full run (Phase 3.1)

All prices and limits were retrieved on 2026-10-18 from the linked pages. Prices are USD, before tax.
Lines marked [INFERENCE] are estimates, not measurements. A list of unverified claims is at the end.

## Workload

- 400,000 steps (4000 epochs × 100 steps), batch 100, 128×128 grayscale, two nets (≈1.6 M + ≈1.5 M
  params), FP32, single GPU, PyTorch 2.14 + Kornia 0.8.
- Disk: 1.67 GB BOSS zip downloaded at setup, ≈180 MB cached tensors, ≈37 MB checkpoint overwritten
  every epoch. Under 5 GB in total. RAM < 4 GB.
- Resumable: re-running the same `make train ARGS="--run …"` resumes from `<run>/last.pt`, so any
  interruption costs at most one epoch (100 steps) plus restart time.
- Reference points: paper ≈ 4 days on one RTX 2080 Ti (§4.2) → ≈ 0.86 s/step. M1 Pro (MPS), measured:
  ≈ 4.1 s/step with the full attack mix (≈ 19 days).

## Throughput estimate per GPU [INFERENCE]

Method: start from the paper's 96 h on an RTX 2080 Ti (13.4 TFLOPS FP32, 616 GB/s) and divide by a
speed-up ratio. The step is either compute-bound (convolutions, ≈ 2.5 TFLOP/step) or memory-bound
(element-wise ops, Kornia median filter, JPEG, per-layer activations of ≈ 10⁸ elements). Which one
dominates is unknown, so the range uses both ratios: speed-up ∈ [min, max] of
(FP32 TFLOPS ratio, memory-bandwidth ratio). Hours = 96 h / speed-up.

Caveats:
- The FP32 figure of Ampere/Ada/Blackwell consumer cards counts dual-issue FP32; real gains are
  usually lower, so the low end of their range is more likely.
- PyTorch enables TF32 for cuDNN convolutions by default on Ampere and newer, which could beat the FP32
  ratio, mostly on A100/H100.
- Kernel-launch overhead and small kernels (128×128 images, 64-channel layers) may leave big GPUs
  (A100, H100) under-used. This is not in the ranges and hits the fastest GPUs hardest, so their real
  time is likely above the range. Cheaper consumer GPUs are likely the cost optimum.
- Whether the paper's 4 days covered all 4000 epochs with the full attack mix is not stated.

GPU specs (FP32 TFLOPS / memory GB/s, from NVIDIA datasheets, not re-fetched): 2080 Ti 13.4/616,
RTX 3090 35.6/936, RTX 4090 82.6/1008, RTX 5090 104.8/1792, L4 30.3/300, L40S 91.6/864, A10 31.2/600,
A100 40GB 19.5/1555, A100 80GB SXM 19.5/2039, H100 SXM 67/3350, V100 15.7/900, P100 9.3/732, T4 8.1/320.

| GPU | Speed-up vs 2080 Ti | Hours for 400k steps | s/step |
|---|---|---|---|
| RTX 3090 | 1.5–2.7× | 36–63 | 0.33–0.57 |
| RTX 4090 | 1.6–6.2× | 16–59 | 0.14–0.53 |
| RTX 5090 | 2.9–7.8× | 12–33 | 0.11–0.30 |
| L4 | 0.5–2.3× | 42–197 | 0.38–1.77 |
| L40S | 1.4–6.8× | 14–68 | 0.13–0.62 |
| A10 | 1.0–2.3× | 41–99 | 0.37–0.89 |
| A100 40GB | 1.5–2.5× | 38–66 | 0.34–0.59 |
| A100 80GB SXM | 1.5–3.3× | 29–66 | 0.26–0.59 |
| H100 SXM | 5.0–5.4× (if saturated) | 18–19, likely more | 0.16–0.17 |
| V100 | 1.2–1.5× | 66–82 | 0.59–0.74 |
| P100 (Kaggle) | 0.7–1.2× | 81–138 | 0.73–1.24 |
| T4 (Kaggle, Colab) | 0.5–0.6× | 159–185 | 1.43–1.66 |

## Options

### Runpod (Pods)
Source: [pricing](https://www.runpod.io/pricing) (page "Updated October 8, 2026"),
[Pod pricing docs](https://docs.runpod.io/pods/pricing), [storage types](https://docs.runpod.io/pods/storage/types.md).
- GPUs, per hour, Community Cloud / Secure Cloud: RTX 3090 $0.22 / $0.50, RTX 4090 $0.34 / $0.89,
  RTX 5090 $0.69 / $1.19, L4 $0.44 / $0.59, L40S $0.79 / $1.09, A100 SXM 80GB $1.39 / $1.79,
  H100 SXM $2.69 / $3.99.
- Billing: per second, compute and storage; no ingress/egress fees. Deploying needs ≥ 1 h of credit.
- Interruption: on-demand Pods "cannot be displaced by other users". The docs list only on-demand and
  savings plans (no spot tier found). Pods stop when the balance reaches $0.
- Storage: container disk is erased on stop; volume disk (`/workspace`, $0.10/GB/month running,
  $0.20 stopped) survives stop/restart and is deleted with the Pod; network volume $0.07/GB/month,
  independent of the Pod.
- Access: SSH, web terminal, JupyterLab; tmux is documented for long jobs. Default spend limit $80/h.
- Free credits: none listed on the pricing page.

### Vast.ai (marketplace)
Source: [pricing docs](https://docs.vast.ai/guides/instances/pricing),
[instance types](https://docs.vast.ai/guides/instances/choosing/instance-types),
[public price feed](https://storage.googleapis.com/vast-public-gpu-pricing/gpu-pricing-public.json)
(feed timestamp 2026-10-18T14:30Z; prices change hourly).
- Current offers, $/h min / median (number of offers): RTX 3090 0.12 / 0.21 (207), RTX 4090
  0.23 / 0.49 (426), RTX 5090 0.21 / 0.68 (637), L4 0.20 / 0.32 (13), L40S 0.43 / 0.80 (60),
  A100 SXM4 0.40 / 0.80 (124), H100 SXM 1.73 / 2.40 (78). Host-set prices; storage and bandwidth are
  billed on top, rates vary per host.
- Billing: per second; prepaid credits; storage billed while the instance exists, even stopped.
- Interruption: on-demand/reserved are "never interrupted"; interruptible instances (bidding, often
  50 %+ cheaper) are paused when outbid and resume later, data preserved. On-demand offers carry a
  maximum rental duration; expired instances may be deleted 48 h after expiry.
- Reliability varies per host (reliability score on each offer).
- Free credits: none listed.

### Lambda (On-Demand Cloud)
Source: [pricing](https://lambda.ai/pricing), [billing docs](https://docs.lambda.ai/public-cloud/billing/).
- GPUs per hour: B200 $6.69, H100 SXM $3.99, A100 SXM 80GB $2.79, A100 SXM 40GB $1.99, V100 $0.79.
  No consumer GPUs, no L4/L40S/A10 listed.
- Billing: one-minute increments; persistent filesystems billed per GiB-month (price shown at
  creation); no ingress/egress fees.
- Interruption: on-demand, no spot tier listed. SSH access to a full VM.
- Free credits: none listed.

### Modal (serverless, per second)
Source: [pricing](https://modal.com/pricing), [timeouts](https://modal.com/docs/guide/timeouts),
[preemption](https://modal.com/docs/guide/preemption.md).
- GPUs per second (per hour): L4 $0.000222 ($0.80), A10 $0.000306 ($1.10), L40S $0.000542 ($1.95),
  A100 80GB $0.000694 ($2.50), H100 $0.001097 ($3.95). CPU ($0.0000131/core/s) and memory
  ($0.00000222/GiB/s) are billed on top. Region selection costs 1.15–1.75×; non-preemptible execution
  costs 3×.
- Free credits: Starter plan $30/month.
- Limits: one Function execution lasts at most 24 h, so a 2-day run needs chained calls (resume makes
  this work). Functions are preemptible by default and restarted on the same input.
- Storage: Volumes $0.09/GiB/month, 1 TiB/month included. No SSH workflow; the job is a Python
  function deployed with the Modal SDK, so the training entry point needs a small wrapper.

### Kaggle Notebooks (free)
Source: [efficient GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage),
[notebooks docs](https://www.kaggle.com/docs/notebooks).
- GPUs: P100 (docs); T4 ×2 also offered ([announcement](https://www.kaggle.com/product-feedback/361104)).
  Single-GPU training uses one T4. Both are slower than a 2080 Ti, the minimum speed required.
- Quota: 30 h/week GPU ("or sometimes higher"); 12 h per session (from the notebooks docs, via search
  snippet). 81–185 h needed → 3–7 weeks of quota and 7–16 manual restarts.
- Storage: up to 20 GB of output in `/kaggle/working` saved per version; resuming means attaching the
  previous version's output as input.
- PyTorch: 2.14 still ships CUDA 12.6 wheels with Pascal (P100) support; 2.15 drops them
  ([RELEASE.md](https://github.com/pytorch/pytorch/blob/main/RELEASE.md),
  [notice](https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432)).

### Google Colab (free and paid)
Source: [FAQ](https://research.google.com/colaboratory/faq.html),
[paid plans](https://colab.research.google.com/signup) (via search snippet; the page needs sign-in).
- Free: GPU type not guaranteed, limits not published, at most 12 h per runtime; SSH and
  non-interactive use may be terminated without warning.
- Paid: Pay As You Go $9.99 per 100 compute units; Pro+ $49.99/month; Pro $9.99/month for 100 units
  (third-party sources). Pro+ allows up to 24 h of continuous execution. Compute-unit burn per GPU is
  not published by Google, so $/h cannot be sourced.
- Not suitable for a multi-day unattended run.

### Hyperscaler: Google Cloud Spot (L4) and AWS
Source: [GCP Spot pricing](https://cloud.google.com/spot-vms/pricing),
[Spot VM docs](https://docs.cloud.google.com/compute/docs/instances/spot),
[VM pricing / billing model](https://cloud.google.com/products/compute/pricing),
[free trial](https://docs.cloud.google.com/free/docs/free-cloud-features).
- GCP `g2-standard-4` (1× L4, 4 vCPU, 16 GiB) Spot: $0.423/h in the page's default region; other
  regions on the same page show $0.49–0.52/h. Boot disk extra.
- Preemption at any time, 30 s best-effort shutdown (notice configurable up to 120 s); no maximum
  runtime. Billing per second after a 1-minute minimum.
- Free credits: $300 for 90 days, but a Free Trial account cannot add GPUs or request quota; after
  upgrading to a paid account, the remaining credit stays usable.
- AWS `g6.xlarge` (1× L4) on-demand: $0.8048/h in us-east-1
  ([AWS price list](https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20(N.%20Virginia)/Linux/index.json),
  published 2026-10-15). Current Spot price not verified: the public Spot feed was last updated in
  May 2024 and the [Spot pricing page](https://aws.amazon.com/ec2/spot/pricing/) renders it with JS.

## Comparison

Hours from the throughput table. Cost = $/h × hours, GPU compute only (storage under 5 GB adds
cents). [INFERENCE] for hours and cost.

| Option | GPU | $/h | Est. hours, 400k steps | Est. total cost | Limits / risks |
|---|---|---|---|---|---|
| Runpod Community | RTX 3090 | 0.22 | 36–63 | $8–14 | third-party host; volume lost if Pod deleted |
| Runpod Community | RTX 4090 | 0.34 | 16–59 | $5–20 | same |
| Runpod Community | RTX 5090 | 0.69 | 12–33 | $8–23 | same; needs a Blackwell-capable CUDA wheel (cu128+) |
| Runpod Secure | RTX 4090 | 0.89 | 16–59 | $14–52 | data-center host |
| Runpod Community | L40S | 0.79 | 14–68 | $11–54 | — |
| Runpod Community | H100 SXM | 2.69 | 18–19+ | $47–52+ | likely under-used |
| Vast.ai on-demand (median) | RTX 3090 | 0.21 | 36–63 | $8–13 | host reliability varies; max rental duration; storage/bandwidth extra |
| Vast.ai on-demand (median) | RTX 4090 | 0.49 | 16–59 | $8–29 | same |
| Vast.ai on-demand (median) | RTX 5090 | 0.68 | 12–33 | $8–22 | same |
| Vast.ai on-demand (median) | A100 SXM4 | 0.80 | 29–66 | $23–53 | same |
| Lambda | A100 40GB | 1.99 | 38–66 | $76–131 | no cheap GPUs |
| Lambda | H100 SXM | 3.99 | 18–19+ | $70–77+ | likely under-used |
| Modal | L40S | 1.95 (+CPU/RAM) | 14–68 | $27–134 (−$30 credit) | 24 h max per call; preemptible; SDK wrapper |
| Modal | A10 | 1.10 (+CPU/RAM) | 41–99 | $45–109 (−$30 credit) | same |
| GCP Spot | L4 | 0.42 | 42–197 | $18–83 (paid from $300 credit after upgrade) | preemption; GPU quota request; setup effort |
| AWS on-demand | L4 | 0.80 | 42–197 | $34–159 | Spot price not verified |
| Kaggle | P100 / T4 | 0 | 81–185 | $0 | 30 h/week, 12 h sessions → 3–7 weeks; slower than 2080 Ti |
| Colab free / paid | not guaranteed | n/a | n/a | n/a | 12–24 h runtimes, unpublished limits |

## Recommendation

- **Primary: Runpod Community Cloud, RTX 4090 ($0.34/h).** Fixed published price, per-second
  billing, SSH + tmux, not displaced by other users, `/workspace` volume survives Pod restarts. Estimated
  16–59 h and $5–20. The workload is small, so a large data-center GPU buys little and costs 4–10× more
  per hour.
- **Fallback: Vast.ai on-demand RTX 4090 or RTX 3090** (median $0.49 / $0.21 per hour) on a host with a
  high reliability score and a maximum rental duration above 4 days. Same cost range; more host variance.
- If the benchmark shows the 4090 far below its compute bound (memory- or overhead-bound), an RTX 3090
  at $0.22/h on the same platform gives nearly the same time for less money.
- Risk on Community/marketplace hosts: the host can fail. The checkpoint is overwritten every epoch on
  the host disk only; copy `<run>/last.pt` and `log.csv` off the machine periodically (e.g. `rsync`
  over SSH from the Mac every few hours).
- Not recommended: Kaggle/Colab (slower than a 2080 Ti, weeks of manual restarts), Lambda/H100
  (3–10× the cost for an estimated ≤ 3× speed-up), Modal (24 h call limit and SDK wrapper for no price
  gain).

## Decision and runbook

Chosen on 2026-10-18: **Runpod Community Cloud, RTX 4090**.

### 1. Pod (done by the account owner)

- Runpod account with prepaid credit (the full run is estimated at $5–20; the benchmark costs < $0.10).
- Deploy a Pod: Community Cloud, 1× RTX 3090 (the RTX 4090 was out of stock on 2026-10-18), container
  disk 20 GB, volume 20 GB mounted on `/workspace`, SSH enabled with the Mac's public key. The lockfile
  resolves `torch 2.14.1+cu130`, which needs a host driver with CUDA ≥ 13.0: pick a CUDA 13.0+ host. On
  a CUDA 12.x host, see the workaround in section 2.
- Share the SSH command shown by Runpod (`ssh root@<ip> -p <port>`).

### 2. Setup and ~10-minute benchmark

From the Mac (no git push needed; the code is copied as-is):

```bash
rsync -az --exclude .venv --exclude data --exclude models/storage -e "ssh -p <port>" \
  ./ root@<ip>:/workspace/watermark_resolution/
ssh -p <port> root@<ip>
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
cd /workspace/watermark_resolution && make sync && make data
make train ARGS="--run models/storage/bench --epochs 5"
```

- s/step = mean `seconds` of epochs 2–5 in `models/storage/bench/log.csv` / 100 (epoch 1 is warm-up).
- Projected hours = s/step × 400,000 / 3600; cost = hours × $/h. Check `nvidia-smi` during the run:
  low utilisation means the step is overhead-bound.
- CUDA 12.x host workaround (used for the 2026-10-18 benchmark): torch 2.14.1 has no cu128 wheel, so
  install `uv pip install --python .venv/bin/python --reinstall "torch==2.14.1" --index-url
  https://download.pytorch.org/whl/cu126` and run `.venv/bin/python -m watermark.train …` instead of
  `make train` (`uv run` would re-install the cu130 wheel).
- Measured 2026-10-18 on RTX 3090 Community: 0.212 s/step, GPU 86–89 % → ≈ 23.6 h, ≈ $5.2.

### 3. Full run

```bash
rm -rf models/storage/bench
tmux new -d -s paper 'make train ARGS="--run models/storage/paper"'
# CUDA 12.x host with the cu126 workaround of section 2:
tmux new -d -s paper '.venv/bin/python -m watermark.train --run models/storage/paper'
```

- Every few hours, from the Mac:
  `rsync -az -e "ssh -p <port>" root@<ip>:/workspace/watermark_resolution/models/storage/paper/ models/storage/paper/`.
- If the Pod dies, start a new Pod, copy the code and `models/storage/paper/` back, run `make data`, and
  re-run the same training command: it resumes from `last.pt`. On a CUDA 12.x host, `make data` re-syncs
  the cu130 wheel, so re-apply the cu126 workaround before training.

## Unverified claims

- All GPU throughput and duration figures are [INFERENCE] from datasheet ratios; GPU specs were quoted
  from NVIDIA datasheets without re-fetching them on 2026-10-18.
- Whether the paper's "≈ 4 days" covers 4000 epochs with the full attack mix.
- Kaggle 12 h session limit: read from a search snippet of the Kaggle notebooks docs, not from the
  fetched page body.
- Colab Pro price ($9.99/month for 100 units) comes from third-party pages; Google's pricing page needs
  sign-in. Colab compute-unit burn rates per GPU are not published.
- AWS g6/g5 current Spot price: not found on a primary source that renders without JS.
- GCP Spot $0.423/h: the region of the page's default view was not confirmed; other regions list
  $0.49–0.52/h. New-project GPU quota after upgrade (often 0, requiring a request) not checked.
- RTX 5090 (Blackwell) needs a CUDA 12.8+ PyTorch wheel; not checked against the torch 2.14 support
  matrix.
- Vast.ai prices are hourly market snapshots; storage and bandwidth rates per host were not sampled.

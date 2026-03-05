---
name: nnpbr-usage
description: Run NN-PBR dataset prep, training, inference, and true-BC6 export. Use this when you need commands, expected outputs, or quick troubleshooting.
argument-hint: "[task] (optional, e.g. train | full | infer | bc6)"
---

# NN-PBR Usage

This skill centralizes “how to run it” so it stays up to date as scripts evolve.

If you change any CLI flags or artifact formats, update:
- `.claude/skills/nnpbr-usage/SKILL.md`
- and run `/nnpbr-update-skills` (or follow its checklist).

## Setup (Python deps)

Typical venv setup (adjust to your environment):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch numpy pillow matplotlib
```

## 1) Prepare Dataset (FreePBR → `reference_8ch.pt`)

```bash
source .venv/bin/activate
python prepare_freepbr_material.py \
  --product-url https://freepbr.com/product/scratched-up-steel/ \
  --variant-keyword "-bl.zip" \
  --size 1024 \
  --out-root data/freepbr/materials
```

Expected outputs:
- `data/freepbr/materials/<material>/reference_8ch.pt`
- `data/freepbr/materials/<material>/dataset_report.json`
- `data/freepbr/materials/<material>/maps/*_preview.png`

## 2) Train Only (export artifacts)

```bash
source .venv/bin/activate
python infrerenfe_nural_mateirals.py \
  --mode train \
  --reference-pt data/freepbr/materials/<material>/reference_8ch.pt \
  --output-dir runs/train_long \
  --device auto \
  --phase1-iters 5000 \
  --phase2-iters 200000 \
  --phase3-iters 0 \
  --batch-size 4096 \
  --log-every 200
```

Expected outputs:
- `runs/train_long/export/metadata.json`
- `runs/train_long/export/decoder_fp16.bin`
- `runs/train_long/export/latents/latent_XX_mip_YY.pt`
- `runs/train_long/training_history.json`
- `runs/train_long/run_report.json`

## 3) Full Run (train + plots + mip0 inference)

Use small iters for a smoke run:
```bash
source .venv/bin/activate
python infrerenfe_nural_mateirals.py \
  --mode full \
  --reference-pt data/freepbr/materials/<material>/reference_8ch.pt \
  --output-dir runs/full_demo \
  --device auto \
  --phase1-iters 30 \
  --phase2-iters 60 \
  --phase3-iters 0 \
  --batch-size 1024 \
  --log-every 10 \
  --interactive-progress \
  --analysis-batch-size 131072
```

Expected outputs:
- `runs/full_demo/inference/pbr_preview.png`
- `runs/full_demo/analysis/gt_vs_neural.png`
- `runs/full_demo/analysis/gt_vs_neural_diff.png`
- `runs/full_demo/analysis/training_loss.png`
- `runs/full_demo/analysis/quality_metrics.json`

## 4) Infer Only (from exported artifacts)

```bash
source .venv/bin/activate
python infrerenfe_nural_mateirals.py \
  --mode infer \
  --export-dir runs/train_long/export \
  --output-dir runs/infer_only \
  --device auto
```

Notes:
- This path is mip0-only (loads `latent_XX_mip_00.pt` and upsamples), so it is a sanity check rather than a perfect runtime simulation.

## 5) Export True BC6 DDS (GPU-ready, pure Python)

No external tools required. Uses the built-in pure-Python BC6H Mode 11 encoder.

```bash
source .venv/bin/activate
python export_true_bc6_dds.py \
  --export-dir runs/train_long/export
```

With verification diff figures (default: enabled):
```bash
python export_true_bc6_dds.py \
  --export-dir runs/train_long/export \
  --out-dir runs/train_long/export/true_bc6_dds
```

Skip verification (faster, no diff PNGs):
```bash
python export_true_bc6_dds.py \
  --export-dir runs/train_long/export \
  --no-verify
```

Encode only specific latent or limit count:
```bash
python export_true_bc6_dds.py \
  --export-dir runs/train_long/export \
  --latent-index 0 \
  --max-latents 2
```

Force signed mode (BC6H_SF16):
```bash
python export_true_bc6_dds.py \
  --export-dir runs/train_long/export \
  --bc6-format SF16
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--export-dir` | required | Dir with `metadata.json` and latent `.pt` files |
| `--out-dir` | `<export-dir>/true_bc6_dds` | Output directory for DDS files |
| `--bc6-format` | from metadata | `UF16` or `SF16` |
| `--latent-index` | all | Encode only this latent index |
| `--max-latents` | 0 (all) | Cap on number of latents to encode |
| `--no-verify` | off | Skip BC6H decode + diff verification |

### Expected outputs

- `runs/train_long/export/true_bc6_dds/latent_XX.bc6.dds` — BC6H DDS with full mip chain
- `runs/train_long/export/true_bc6_dds/true_bc6_export_report.json` — export report with PSNR metrics
- `runs/train_long/export/true_bc6_dds/latent_XX_mip_YY.diff.png` — diff figures (unless `--no-verify`)

### Quality expectations

- Smooth PBR latent blocks (typical): PSNR > 35 dB
- Random/noisy content (worst case): PSNR ~12 dB (inherent BC6H Mode 11 with 8 interpolation steps)
- Flat regions: near-lossless (max error < 0.001)

### Encoder notes

- Implements BC6H Mode 11 (single-subset, 11-bit endpoints, 3-bit indices per texel)
- `[-1, 1]` tensors remapped to `[0, 1]` for UF16, used as-is for SF16
- Integer interpolation: `(ep0*(64-w) + ep1*w + 32) >> 6` per the BC6H spec
- No subprocess calls — pure Python + numpy + torch

## Troubleshooting (quick)

- If dataset prep fails to find maps, inspect `data/.../extracted/` and adjust keywords or `--variant-keyword`.
- If normals look inverted, check the DirectX/OpenGL Y-flip heuristic (filename contains `dx`/`directx`).
- If BC6 PSNR is unexpectedly low, check that latent tensors are locally smooth (rough/noisy latents compress poorly with 8 interpolation steps).


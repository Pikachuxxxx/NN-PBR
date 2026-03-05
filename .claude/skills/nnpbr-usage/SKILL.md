---
name: nnpbr-usage
description: Run NN-PBR dataset prep, training, inference, and true-BC6 export. Use this when you need commands, expected outputs, or quick troubleshooting.
argument-hint: "[task] (optional, e.g. train | full | infer)"
---

# NN-PBR Usage

This skill centralizes "how to run it" so it stays up to date as scripts evolve.

If you change any CLI flags or artifact formats, update:
- `.claude/skills/nnpbr-usage/SKILL.md`
- and run `/nnpbr-update-skills` (or follow its checklist).

## Setup (Python deps)

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

## 2) Train Only

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

Expected outputs (export layout v4):
```
runs/train_long/
  export/
    metadata.json          ← runtime (version: 4, lod_biases, decoder params)
    decoder_fp16.bin       ← runtime MLP weights
    latent_00.bc6.dds      ← runtime BC6H UF16 DDS (full mip chain, direct from block params)
    latent_01.bc6.dds      ← ...
    metadata/
      decoder_state.pt     ← debug (full PyTorch state dict)
      latent_00_mip_00.png ← visual preview from soft surrogate decode
      latent_00_mip_01.png
      ...
  training_history.json
  run_report.json
```

**Key: DDS files are written directly from trained block params (lossless packing, no float32 roundtrip).**

## 3) Full Run (train + plots + inference)

Quick smoke run:
```bash
source .venv/bin/activate
python infrerenfe_nural_mateirals.py \
  --mode full \
  --reference-pt data/freepbr/materials/<material>/reference_8ch.pt \
  --output-dir runs/full_demo \
  --device auto \
  --phase1-iters 20 \
  --phase2-iters 100 \
  --batch-size 1024 \
  --log-every 10 \
  --interactive-progress
```

Validate that exported latents produce the same output as the trained model:
```bash
python infrerenfe_nural_mateirals.py \
  --mode full \
  ... \
  --use-export-latents
```

Expected outputs:
- `runs/full_demo/inference/pbr_preview.png`
- `runs/full_demo/analysis/all_analysis.png`
- `runs/full_demo/analysis/training_loss.png`
- `runs/full_demo/analysis/quality_metrics.json`
- `runs/full_demo/run_report.json`

## 4) Infer Only (from exported artifacts)

```bash
source .venv/bin/activate
python infrerenfe_nural_mateirals.py \
  --mode infer \
  --export-dir runs/train_long/export \
  --output-dir runs/infer_only \
  --device auto
```

Loads latent mip0 from `metadata/latent_XX_mip_00.png` (reverse-mapped to [-1,1]) + `decoder_state.pt`.

## CLI flags reference (`infrerenfe_nural_mateirals.py`)

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `full` | `train` / `full` / `infer` |
| `--reference-pt` | required (train/full) | Path to `reference_8ch.pt` |
| `--output-dir` | required (train/full) | Run output root |
| `--export-dir` | `<output-dir>/export` | Export artifacts location |
| `--device` | `auto` | `auto` / `cpu` / `cuda` |
| `--phase1-iters` | 5000 | Warmup iterations |
| `--phase2-iters` | 200000 | BC-constrained iterations |
| `--phase3-iters` | 0 | Quantized finetune iterations |
| `--batch-size` | 4096 | UV/LOD sample batch size |
| `--latent-res` | `512,256,128,64` | Pyramid base resolutions |
| `--latent-mips` | `8,7,6,5` | Mip levels per pyramid |
| `--hidden-dim` | 16 | MLP hidden size |
| `--endpoint-bits` | 6 | BC6H endpoint quantization bits |
| `--index-bits` | 3 | BC6H index quantization bits |
| `--log-every` | 200 | Print loss every N iters |
| `--interactive-progress` | off | Enable tqdm progress bars |
| `--infer-size` | `auto` | Output resolution: `auto`, `1024`, `1024x1024` |
| `--analysis-batch-size` | 131072 | Random batch size for metrics |
| `--use-export-latents` | off | Full mode: infer from exported PNGs instead of in-memory model (validates export) |

## DDS Export (built-in, no separate step)

As of export layout v4, `latent_XX.bc6.dds` files are written automatically by
`export_trained_artifacts()` — **no separate script call needed**.

The encoder packs the trained block params (endpoints + indices) directly into
BC6H Mode 11 blocks: `endpoint_q (6-bit) → 11-bit → pack → 128-bit BC6H block`.
This is lossless with respect to the quantized training representation.

`export_true_bc6_dds.py` is now an informational stub:
```bash
python export_true_bc6_dds.py --export-dir runs/train_long/export
# prints DDS file list and sizes; does nothing for v4 exports
```

## Troubleshooting (quick)

- **NaN in phase2**: keep `--phase1-iters` ≤ 20 (larger warmup destabilizes BC init)
- **Wrong storage size**: check `latent_*.bc6.dds` exist in export root — `_collect_neural_storage_bytes` uses actual DDS sizes when present
- If dataset prep fails, inspect `data/.../extracted/` and adjust `--variant-keyword`
- If normals look inverted, check DX/GL Y-flip heuristic (filename contains `dx`/`directx`)

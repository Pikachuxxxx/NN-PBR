---
name: demo-testing
description: Own demos, smoke checks, true BC6 export validation, and runtime integration (Vulkan/shader wiring).
tools: Read, Glob, Grep, Bash, Edit, Write
skills:
  - nnpbr-usage
---

# Demo / Testing / Runtime Integration Subagent

## Mission
Own “everything else” needed to make the project demonstrable and verifiable:
- run training/inference experiments and sanity checks
- improve plots/metrics for comparisons
- validate true BC6 DDS export path
- drive runtime integration (shader contract + Vulkan stub)

Deep reference:
- `.claude/reference/NNPBR_GUIDE.md`

Usage commands:
- use `/nnpbr-usage`

## Core entry points
- `infrerenfe_nural_mateirals.py` (`--mode full` / `--mode infer`)
- `export_true_bc6_dds.py` (true BC6 DDS via external CLI)
- `shaders/neural_material_decode.hlsl` (runtime decode contract)

## Canonical smoke checks
- Dataset prep produces `reference_8ch.pt` + `maps/*_preview.png`.
- `--mode full` produces:
  - `analysis/gt_vs_neural.png`
  - `analysis/gt_vs_neural_diff.png`
  - `analysis/training_loss.png`
  - `analysis/quality_metrics.json`
- `--mode infer` produces `inference/pbr_preview.png`.
- True BC6 export produces `true_bc6_dds/true_bc6_export_report.json` and at least one `latent_XX.bc6.dds`.

## Vulkan integration (current TODO)
`examples/vulkan_neural_material/` is a stub. The integration goal is:
- load/bind `decoder_fp16.bin` as a raw buffer (ByteAddressBuffer equivalent)
- load/bind `latent_XX.bc6.dds` as sampled textures
- bind LOD biases (`metadata.json["lod_biases"]`) as constants/uniforms
- render fullscreen and decode via `shaders/neural_material_decode.hlsl`

If any CLI/options or artifact outputs change while building demos:
- update `.claude/skills/nnpbr-usage/SKILL.md` and run `/nnpbr-update-skills`.


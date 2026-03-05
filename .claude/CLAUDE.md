# NN-PBR — Claude Code Project Instructions

This file is intentionally **short** (project memory). Deep theory/architecture lives in:
- `.claude/reference/NNPBR_GUIDE.md`

## Project Summary

NN-PBR is a WIP implementation of *Real-Time Neural Materials using Block-Compressed Features*:
- arXiv (abs): https://arxiv.org/abs/2311.16121
- arXiv (pdf): https://arxiv.org/pdf/2311.16121.pdf

It “bakes” a single material into:
- a few latent texture pyramids (intended to be BC6-filterable at runtime), and
- a tiny MLP decoder (`decoder_fp16.bin`) evaluated in a shader to reconstruct 8 PBR channels.

## Where Things Are

Core Python:
- `prepare_freepbr_material.py` — FreePBR → `reference_8ch.pt`
- `infrerenfe_nural_mateirals.py` — `train|full|infer` runner + plots/metrics
- `neuralmaterials.py` — model + BC6-inspired surrogate + export formats
- `export_true_bc6_dds.py` — optional true BC6 DDS export via external encoder CLI

Runtime shader:
- `shaders/neural_material_decode.hlsl` — samples latent textures + runs FP16 MLP in-shader

Outputs:
- `data/` — datasets (FreePBR zips, extracted maps, `reference_8ch.pt`)
- `runs/` — training runs, export artifacts, plots, reports

## Skills (Usage Lives Here)

Usage commands and expected outputs are intentionally **not** embedded in this file.
Use the project skills:
- `/nnpbr-usage` → how to run dataset prep / training / inference / true BC6 export
- `/nnpbr-update-skills` → refresh skill/docs after CLI/options change

## Keeping Skills Up To Date (important)

Whenever you change any of these:
- add/remove CLI flags in any script
- change artifact formats (`metadata.json`, `decoder_fp16.bin`, latent file inventory)
- change output folders or report formats

Do this in the same PR:
1) Update `.claude/skills/nnpbr-usage/SKILL.md`
2) If the update is non-trivial, update `.claude/reference/NNPBR_GUIDE.md`
3) Keep this file small; add only high-level pointers + FAQ updates

## Subagents

Project subagents live in `.claude/agents/`:
- `paper-planner` — paper reading + roadmap/gap analysis
- `python-core` — Python training/export/inference implementation work
- `demo-testing` — demos, smoke checks, true BC6 export, runtime integration

## FAQ (from our discussion)

### What exactly is being trained?
Latent texture pyramids (per-material) + a tiny MLP decoder that reconstructs 8 material channels from sampled latents.

### What does “BC6-inspired surrogate” mean here?
Training uses a differentiable, BC6-like **block parameterization** (endpoints/indices/partition) with STE quantization, but it is not a bit-exact BC6H codec.

### Why are normals stored as X/Y only?
It matches common GPU practice (BC5 normals). `z` can be reconstructed as `sqrt(1 - x^2 - y^2)` with clamping.

### What’s the difference between `*.blocks128.bin` and `*.bc6.dds`?
`*.blocks128.bin` are compact, custom 128-bit-per-block records for accounting/debug (not guaranteed BC6-compliant). `*.bc6.dds` are true engine-ready BC6 textures produced by an external encoder CLI.

### Why train on random UV + continuous LOD?
To learn a function that behaves well across mip levels and filtering, not just at mip0.

### How do I run the pipeline?
Invoke `/nnpbr-usage`.

### I added a new CLI option—what now?
Invoke `/nnpbr-update-skills`, then update the usage skill + any docs that reference the changed behavior.


---
name: python-core
description: Implement/modify the Python training/export/inference pipeline while keeping the shader + artifact contracts stable.
tools: Read, Glob, Grep, Bash, Edit, Write
skills:
  - nnpbr-usage
---

# Python Core Subagent (Training / Export / Inference)

## Mission
Own changes to the Python pipeline and artifact formats:
- training loops and stability
- latent representation / BC6-surrogate blocks
- export formats (`metadata.json`, `decoder_fp16.bin`, latent `.pt` mips, packed blocks)
- inference/analysis tooling

This subagent must keep the runtime contract stable with:
- `shaders/neural_material_decode.hlsl`
- `runs/<run>/export/` artifact layout

Deep reference:
- `.claude/reference/NNPBR_GUIDE.md`

Usage commands:
- use `/nnpbr-usage` (don’t paste long command blocks into `.claude/CLAUDE.md`)

## Where to Work
- `neuralmaterials.py`
- `infrerenfe_nural_mateirals.py`
- `prepare_freepbr_material.py`
- `export_true_bc6_dds.py`

## Runtime Contract (do not break casually)
1) **Output channels and ranges**
   - 8 output channels trained in `[-1,1]`; shader remaps to `[0,1]`.
2) **Decoder blob**
   - `decoder_fp16.bin` layout: `fc1.weight`, `fc1.bias`, `fc2.weight`, `fc2.bias` (FP16, contiguous).
3) **LOD biases**
   - `metadata.json["lod_biases"]` must match model computation; shader uses these to bias sampling.

If you must change any contract:
- update shader + docs in the same PR
- update `.claude/skills/nnpbr-usage/SKILL.md` expected outputs if artifacts change

## “Done” checklist
- A short `--mode full` run works end-to-end and writes `runs/<run>/analysis/quality_metrics.json`.
- `--mode infer` from export produces sane maps.
- If any CLI flags changed, run `/nnpbr-update-skills` and update skills/docs.


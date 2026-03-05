---
name: nnpbr-update-skills
description: Maintenance checklist to keep NN-PBR skills/docs synced with evolving CLI options, scripts, and artifact formats.
disable-model-invocation: true
allowed-tools: Bash
---

# NN-PBR Skill Maintenance

Use this after you change any scripts, flags, exports, or runtime contracts.

## Goal
Keep these always correct:
- `.claude/skills/nnpbr-usage/SKILL.md` (commands + expected outputs)
- `.claude/CLAUDE.md` (short pointers + FAQ)
- `.claude/reference/NNPBR_GUIDE.md` (deep theory/architecture reference, no copy-pasted usage)

## Step-by-step checklist

### 1) Re-read current CLI help

Run these and compare against the usage skill:
```bash
source .venv/bin/activate || true
python infrerenfe_nural_mateirals.py --help
python prepare_freepbr_material.py --help
python export_true_bc6_dds.py --help
python neuralmaterials.py --help || true
```

If any flags/options changed:
- update `.claude/skills/nnpbr-usage/SKILL.md`
- update any examples to match new defaults/required args

### 2) Verify artifacts + paths still match docs

Run a short smoke pipeline (tiny iters) and confirm expected files exist:
```bash
python infrerenfe_nural_mateirals.py --mode full --reference-pt data/.../reference_8ch.pt --output-dir runs/_smoke --phase1-iters 1 --phase2-iters 1 --log-every 1
```

If export layouts or filenames changed:
- update `.claude/skills/nnpbr-usage/SKILL.md` expected outputs
- update `.claude/reference/NNPBR_GUIDE.md` export/runtime contract sections
- update `.claude/CLAUDE.md` FAQ entries that reference artifacts

### 3) Keep memory file short

Do not paste long theory or big command blocks into `.claude/CLAUDE.md`.
Instead:
- keep `.claude/CLAUDE.md` as pointers + FAQ
- keep deep content in `.claude/reference/NNPBR_GUIDE.md`

### 4) If runtime contract changed, update shader + docs together

If you changed any of:
- latent count / channels per latent
- MLP dims
- `decoder_fp16.bin` layout
- `metadata.json` fields used at runtime

Then update in the same PR:
- `shaders/neural_material_decode.hlsl`
- `.claude/reference/NNPBR_GUIDE.md`
- `.claude/skills/nnpbr-usage/SKILL.md` (if any commands/outputs changed)

## Done criteria
- `nnpbr-usage` commands run as written (or clearly note prerequisites).
- Docs do not contradict CLI `--help` output.
- `.claude/CLAUDE.md` stays readable and short.


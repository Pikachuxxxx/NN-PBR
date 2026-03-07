# BC6H Mode 10 Cleanup TODO

This file is the implementation checklist for bringing the training and export pipeline in line with:

- `/Users/phanisrikar/Downloads/bc6_neural_training_workflow.md`
- paper §4.2 (differentiable BC6 decode formulas)
- paper §5.2 (final export quantization)

## Canonical target

- Warm up neural features as unconstrained `float3` mip pyramids.
- Run a standard BC6 encoder once after warm-up to pick the best block partition, endpoints, indices, and mode for every 4×4 block.
- Keep `partitionID` fixed after initialization.
- Continue constrained training with block parameters only:
  - 4 endpoints per block
  - 16 interpolation weights / indices per block
  - fixed `partitionID`
- Export each trained feature layer as a mipmapped `DXGI_FORMAT_BC6H_UF16` texture.
- Quantize endpoints to 6 bits and indices to 3 bits during final export, as stated in §5.2.
- Store decoder weights directly as FP16.
- Pack every BC6 block as a real 16-byte Mode 10 packet that GPU hardware can decode directly.

## Current status

- Done: official Mode 10 partition table and fix-up anchors are shared by training init, software decode, and export packing.
- Done: warm-up now switches into a fixed-partition Mode 10 initialization pass that searches all 32 legal partitions and seeds quantized endpoints + indices per block.
- Done: constrained training uses the §4.2-style unsigned Mode 10 decode surrogate with fixed partitions, 6-bit endpoint space, and 3-bit index space.
- Done: export packs true 16-byte `DXGI_FORMAT_BC6H_UF16` Mode 10 blocks and validates pack/decode round-trips before writing DDS.
- Done: full-mode inference renders from the exported DDS latents instead of the in-memory training latents.
- Remaining: signed BC6H Mode 10 is still intentionally blocked until the training-side signed quantization path is made spec-correct.

## Required cleanup

### 1. Replace fake partition data with spec data

- Remove the procedural partition bank in `neuralmaterials.py`.
- Add the official BC6H two-subset partition table used by Mode 10.
- Add per-partition anchor / fix-up metadata from the real BC6 layout.
- Make partition lookup shared by training init, export packing, and validation decode.

### 2. Split the pipeline into three explicit stages

- Keep warm-up sampling and unconstrained mips separate from BC-constrained storage.
- Keep the one-time BC initialization encoder separate from the differentiable training decode.
- Keep final quantization / packing separate from training-time latent reconstruction.

### 3. Rewrite warm-up initialization to use a real BC6 search

- For each 4×4 block, search the chosen fixed BC6 mode and all legal partitions.
- Pick the best partition from the official partition set, not the procedural masks.
- Initialize block endpoints and per-pixel indices from that encode result.
- Freeze `partitionID` immediately after initialization.

### 4. Make the differentiable decode match the paper

- Implement the exact endpoint unquantization path from §4.2.
- Implement partitioned interpolation using the fixed partition mask chosen at initialization.
- Implement the FP16 reinterpretation surrogate from §4.2 without adding extra nonlinear remaps that do not exist in the paper/runtime path.
- Keep training parameters in a form that cleanly maps to final 6-bit / 3-bit quantization.

### 5. Make final export match §5.2 exactly

- Quantize trained endpoints to 6 bits per component.
- Quantize trained interpolation weights / indices to 3 bits.
- Export each latent mip level as BC6H Mode 10 blocks.
- Write DDS mip chains using those already-quantized block packets.
- Export decoder weights as raw FP16 exactly once per model.

### 6. Replace the current custom block packer with a spec-correct Mode 10 packer

- Remove legacy Mode 11 / Mode 12 assumptions from the active path.
- Stop writing custom 128-bit layouts under a BC6H DDS header.
- Encode the real Mode 10 endpoint fields, partition bits, fix-up anchor bits, and index layout.
- Guarantee that every block occupies exactly 16 bytes and round-trips through a spec decoder.

### 7. Add validation before trusting runtime output

- Add block-level pack/unpack tests for a few known partitions.
- Validate anchor / fix-up handling against the official partition table.
- Decode exported DDS blocks with a spec-matching software decoder and compare against the training-side quantized decode.
- Spot-check runtime GPU sampling against the software reference on mip0 and small mips.

### 8. Remove stale documentation and code paths

- Clean `README.md` so it only describes the canonical Mode 10 path.
- Remove or clearly mark legacy Mode 11 / Mode 12 helpers that are no longer part of the target implementation.
- Keep comments aligned with the paper terminology: warm-up, BC initialization, constrained decode, final quantization, DDS packing.

## File-by-file implementation plan

- `neuralmaterials.py`
  - replace procedural partitions
  - fix initialization search
  - fix differentiable decode
  - clean export path / comments
- `export_true_bc6_dds.py`
  - make it the validation/export tool for spec-correct Mode 10 DDS output
- `README.md`
  - remove stale Mode 11 / Mode 12 wording
  - point to this checklist until Mode 10 export is complete
- `shaders/neural_material_decode.hlsl`
  - keep runtime assumptions limited to “sample BC6 textures + run FP16 MLP”

## Acceptance criteria

- `partitionID` comes from the official BC6 partition table and stays fixed after initialization.
- Training-side constrained decode and export-side quantized decode agree up to expected quantization error.
- Exported DDS files contain real BC6H Mode 10 blocks, 16 bytes per block, full mip chain.
- GPU hardware sampling of the exported DDS matches the software reference closely enough for the decoder MLP to reconstruct the trained material.
- Decoder weights are exported as FP16 and loaded unchanged by runtime.

## End-to-end smoke run

- Run a 512-iteration full pipeline check with a real `reference_8ch.pt`.
- Keep the run split explicit:
  - phase 1 warm-up
  - one-time BC initialization
  - phase 2 BC-constrained training
  - export
  - BC6H DDS decode/infer check
- Verify after the run:
  - `metadata.json` exists
  - `decoder_fp16.bin` exists
  - `latent_XX.bc6.dds` files exist
  - DDS files decode through `decode_bc6h_dds_mip0`
  - infer/full mode can render from exported DDS latents
- If the 512-iteration run is unstable, lower batch size before changing the paper-aligned training/export math.

### Reference command

- `uv run --python 3.12 --with torch --with numpy --with pillow --with matplotlib --with tqdm python infrerenfe_nural_mateirals.py --mode full --reference-pt data/freepbr/materials/red-plaid-bl/reference_8ch.pt --output-dir runs/mode10_paper512_512_cpu --device cpu --phase1-iters 512 --phase2-iters 512 --phase3-iters 0 --batch-size 512 --analysis-batch-size 2048 --log-every 64 --infer-chunk 32768`
- `uv run --python 3.12 --with torch --with numpy --with pillow --with matplotlib --with tqdm python export_true_bc6_dds.py --export-dir runs/mode10_paper512_512_cpu/export --decode-check`
- `uv run --python 3.12 --with torch --with numpy --with pillow --with matplotlib --with tqdm python infrerenfe_nural_mateirals.py --mode full --reference-pt data/freepbr/materials/red-plaid-bl/reference_8ch.pt --output-dir runs/mode10_paper1024_1024_cpu --device cpu --phase1-iters 1024 --phase2-iters 1024 --phase3-iters 0 --batch-size 512 --analysis-batch-size 2048 --log-every 64 --infer-chunk 32768`

### Latest result

- Completed on `2026-03-07` with `phase1=128`, `phase2=384`, `phase3=0`.
- Export wrote four DDS latent pyramids plus `decoder_fp16.bin` and `metadata.json`.
- `export_true_bc6_dds.py --decode-check` reported all DDS headers/sizes valid and decodable as `BC6H_UF16`.

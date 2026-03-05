# NN-PBR (Neural Materials + Block-Compressed Features) — Repo Guide

This repo is a **WIP implementation** of:
- **Paper:** *Real-Time Neural Materials using Block-Compressed Features*
- **arXiv (abs):** https://arxiv.org/abs/2311.16121
- **arXiv (pdf):** https://arxiv.org/pdf/2311.16121.pdf

The goal is to represent a full set of **PBR material maps** using:
1) a small set of **block-compressed latent textures** (BC6/BC6-like), and
2) a tiny **MLP decoder** evaluated in a runtime shader to reconstruct material outputs.

In this repo, the trained decoder reconstructs **8 channels** (all in `[-1,1]` during training/export):
`[albedo_r, albedo_g, albedo_b, normal_x, normal_y, ao, roughness, metallic]`

---

## TL;DR (mental model)

You can think of this repo as learning a compact representation of a single material:

- The *ground truth* material is a function:
  - `f(u, v, lod) -> 8 PBR channels`
  - where `(u,v)` is UV and `lod` is the mip level (continuous).

- The model approximates `f` with:
  - a small set of **latent textures** `L_i(u, v, lod + bias_i)` (3 channels each, multiple resolutions)
  - a tiny **MLP decoder** `D( concat_i L_i(...) ) -> 8 channels`

- Training samples random `(u,v,lod)` points and minimizes an MSE reconstruction loss.
- Export produces:
  - latent mip pyramids (intended to live as BC6 textures at runtime),
  - plus FP16-packed MLP weights for a shader-side decoder.

This design targets **runtime bandwidth** and **hardware filtering** by keeping the runtime payload “texture-like”.

---

## What this repo is (and isn’t)

**It is:**
- A per-material optimization pipeline that “bakes” a material into:
  - a few latent textures + a small decoder network.
- A BC6-*inspired* training/export representation for those latents.
- A runtime decode contract (HLSL) that can be integrated into a renderer.

**It is not (yet):**
- A general model that learns from thousands of materials (the current pipeline fits one material at a time).
- A bit-exact implementation of the BC6H format encoder/decoder.
  - The repo’s `*.blocks128.bin` is a compact, BC6-like record, but not guaranteed BC6-compliant.
  - True BC6 DDS is produced by re-encoding float latents with an external CLI.

---

## Glossary (terms used throughout)

- **PBR**: Physically Based Rendering (material parameters used by a lighting model).
- **Albedo/BaseColor**: diffuse reflectance color (RGB).
- **Normal map**: per-texel surface orientation in tangent space; typically stored as RGB, but only `x,y` are needed because `z = sqrt(1 - x² - y²)`.
- **AO**: Ambient Occlusion (scalar).
- **Roughness**: microfacet roughness (scalar).
- **Metallic**: metalness (scalar).
- **ORM**: common packed texture of `(AO, Roughness, Metallic)` (RGB).
- **Mipmaps**: prefiltered downsampled versions of a texture (mip0 = full res).
- **LOD**: Level-of-detail, selecting/interpolating between mip levels.
- **BCn / Block compression**: GPU texture formats compressing 4×4 blocks into fixed-size records (fast to sample; hardware supports filtering).
- **BC6H**: BC format typically used for HDR/float-like data; 16 bytes per 4×4 block.
- **Endpoints / indices / partition**: typical block-compression vocabulary; store a few representative colors (endpoints), then indices per texel to interpolate.
- **STE**: Straight-Through Estimator; a way to treat quantization/rounding as identity in the backward pass.

---

## Theory Primer (why these pieces exist)

### Why neural materials + compressed latents?

Classic runtime material textures typically use multiple BC textures:
- Albedo: e.g. BC1/BC7
- Normal: e.g. BC5 (2 channels)
- ORM: e.g. BC1/BC7

That’s multiple texture fetches, plus multiple texture assets, plus streaming overhead.

This approach aims to reduce runtime payload to:
- a small set of latent textures (that still live in a **hardware-filterable** compressed format),
- and a tiny shader MLP (cheap to evaluate) that reconstructs full PBR outputs.

Intuition:
- The latent textures are “feature maps” that store material structure efficiently.
- The MLP learns how to map those features into standard PBR channels.

### Another perspective: “learned basis textures” + nonlinear mixing

If you squint, this is similar to classic texture compression ideas:
- You want to represent a complex signal (8-channel material) with fewer stored numbers.

Here:
- the latent textures act like **learned basis fields** over UV space,
- and the decoder is a small **nonlinear mixer** that maps those basis values into output channels.

Compared to a typical autoencoder:
- there is no explicit encoder network here; the “encoder” is effectively the optimization of the latent textures themselves.
- the decoder is intentionally tiny so it can run per-pixel in a shader.
- the latent representation is texture-like so you keep GPU sampling/filtering.

### Block compression 101 (BCn)

Most “BCn” formats compress textures in fixed-rate **4×4 blocks**:
- The texture is split into `(ceil(W/4) * ceil(H/4))` blocks per mip level.
- Each block is stored as a small fixed-size record (e.g., 8 bytes or 16 bytes).
- At runtime, the GPU:
  - fetches the compressed record for the relevant block(s),
  - decompresses in hardware,
  - and can still apply filtering across texels/mips.

Why this matters:
- You get strong compression and fast sampling without custom shader decompress code.
- Filtering and mips “just work” the same way as ordinary textures (critical for minification).

In this repo, the **latent textures** are intended to live in a BC format (BC6 in particular) so:
- they can be filtered as textures,
- and sampled cheaply in a shader.

### Why BC6H-like for latents?

Latents in this repo are 3-channel float-ish values bounded to `[-1,1]` during training (`tanh`).

BC6H is attractive in spirit because:
- it is a **16-byte-per-block** format suitable for HDR/float-like content,
- it can represent a wide range of values with fewer artifacts than low-dynamic-range BC formats,
- it is widely supported by graphics APIs and hardware.

This repo uses two related representations:
- **BC6-inspired surrogate parameters** during training/export (`BC6SurrogateBlockLevel`), because it’s differentiable and quantization-aware.
- **True BC6 DDS** optionally, by re-encoding the exported float latent mips with an external encoder CLI.

### Mipmaps, LOD, and filtering (the runtime reality)

At runtime, the GPU chooses which mip to sample based on screen-space derivatives:
- `ddx(uv)` and `ddy(uv)` measure how quickly UV changes across neighboring pixels.
- Large derivatives imply minification → higher mip levels (blurrier, less aliasing).

Key point:
- If your latent textures do not have correct mip chains (or don’t filter well),
  your reconstructed material will shimmer/alias even if mip0 looks good.

That’s why this repo explicitly trains over **continuous LOD**, not only mip0.

### LOD bias via derivative scaling (why `SampleGrad` works)

The shader applies per-latent **LOD bias** by scaling derivatives:
- If you multiply derivatives by `s`, the GPU effectively sees a different footprint size.
- `s = 2^(lodBias)` means “pretend the texture is scaled”.

So:
- `tex.SampleGrad(sampler, uv, ddx(uv)*s, ddy(uv)*s)` approximates sampling at `lod + lodBias`
  (without explicit `SampleLevel`).

This is why `metadata.json["lod_biases"]` is passed to the shader cbuffer and used in `SampleLatent(...)`.

### Why multiple latent resolutions?

Materials have multi-scale structure:
- fine scratches and micro-variation (high frequency)
- large gradients and macro patterns (low frequency)

Using multiple latent pyramids at different base resolutions:
- gives high-res latents capacity for fine detail,
- while low-res latents carry global context.

The `lod_biases` are used so each latent pyramid samples the “right” effective mip relative to the reference resolution.

### Why train on random UV + continuous LOD?

Training on random `(u,v,lod)` samples:
- makes the model learn a *continuous* function over the UV domain and LOD dimension,
- naturally covers all mip levels (important for runtime minification/magnification),
- avoids hard-coding one fixed output resolution.

Practically, this also avoids having to render full images every iteration.

### Why normalize to `[-1,1]`?

For learning:
- Keeping inputs/targets roughly centered and bounded stabilizes optimization.
- `tanh`-bounded latents are convenient to prevent runaway values.

For runtime:
- outputs are remapped back to `[0,1]` in the shader (albedo/ao/roughness/metallic).
- normals are reconstructed from `x,y` and normalized.

---

## Table of Contents

- [Repo Layout](#repo-layout-what-lives-where)
- [Architecture Diagram](#architecture-diagram-conceptual)
- [High-Level Pipeline](#high-level-pipeline-end-to-end)
- [Data: `reference_8ch.pt`](#data-reference_8chpt-format-training-target)
- [Model Architecture](#model-architecture-how-training-maps-to-runtime)
- [Training Phases](#training-phases-implementation-details)
- [BC6 Surrogate](#bc6-surrogate-representation-what-exactly-is-learned)
- [Code Walkthrough](#code-walkthrough-by-file)
- [Export Format](#export-format-runtime-contract)
- [Inference Paths](#inference-paths-python)
- [True BC6 DDS Export](#true-bc6-dds-export-gpu-ready-latents)
- [Gotchas](#gotchas--sharp-edges)
- [Runtime Shader Contract](#runtime-shader-contract-hlsl)

---

## Repo Layout (What lives where)

Top-level entrypoints:
- `infrerenfe_nural_mateirals.py`: unified runner for `train|full|infer` (the “orchestrator”).
- `neuralmaterials.py`: core model, BC6-surrogate latent representation, training loop, export formats.
- `prepare_freepbr_material.py`: downloads a FreePBR zip, picks maps, builds `reference_8ch.pt`.
- `export_true_bc6_dds.py`: converts exported latent `.pt` mips into **true BC6 DDS** via external encoder CLI.

Runtime shader:
- `shaders/neural_material_decode.hlsl`: samples BC latents + runs `decoder_fp16.bin` MLP to output PBR params.

Outputs / artifacts:
- `data/`: datasets (FreePBR zips, extracted maps, `reference_8ch.pt`).
- `runs/`: training runs, plots, export artifacts.

Example runtime (incomplete):
- `examples/vulkan_neural_material/`: stub for a Vulkan app + shaders (TODO in README).

---

## Architecture Diagram (Conceptual)

```mermaid
flowchart LR
  A["PBR maps (GT)\nAlbedo/Normal/AO/Rough/Metal"] -->|normalize to [-1,1]| B["reference_8ch.pt\n(base mip + meta)"]
  B --> C["build mip chain\n(load_reference_mips)"]
  C -->|random uv,lod| D["sample_mips_trilinear\n(grid_sample)"]
  D --> T["target material\n[B,8]"]

  U["random uv,lod"] --> L["latent feature sampling\n(N latents, 3ch each)"]
  L --> M["decoder MLP\n(12->16->8)"]
  M --> P["pred material\n[B,8]"]

  P -->|MSE| Loss["loss"]
  T -->|MSE| Loss

  subgraph Latents["Latent pyramids"]
    W["Phase1: WarmupLatentPyramid\n(unconstrained, tanh-bounded)"]
    S["Phase2+: BC6SurrogatePyramid\n(endpoints/indices/partition, STE quant)"]
  end

  L --- W
  L --- S

  S -->|export| E["export/\nlatents/*.pt, decoder_fp16.bin, metadata.json"]
  E -->|optional external CLI| F["true_bc6_dds/\nlatent_XX.bc6.dds"]
  E --> G["HLSL runtime\nsample BC6 latents + MLP decode"]
  F --> G
```

---

## High-Level Pipeline (End-to-end)

1) **Dataset preparation** (FreePBR → tensor)
   - Run `prepare_freepbr_material.py` to build a single material reference tensor:
     - `data/freepbr/materials/<material>/reference_8ch.pt`
     - This stores the *base mip* (mip0) as a tensor and includes metadata + original map paths.

2) **Training**
   - Run `infrerenfe_nural_mateirals.py --mode train|full` using the `reference_8ch.pt`.
   - Training is done in phases:
     - Phase 1: **warmup** with unconstrained latent pyramids (stable start).
     - Phase 2: **BC-constrained** (learns BC6-style surrogate block params + decoder).
     - Phase 3: optional **quantized finetune** (freeze quantized BC features, tune decoder).

3) **Export**
   - Produces a runtime package under `runs/<run>/export/`:
     - `decoder_fp16.bin`: flat FP16 weights for the 2-layer MLP.
     - `decoder_state.pt`: PyTorch state dict (debug/validation).
     - `metadata.json`: dims, LOD biases, file inventory, packing notes.
     - `latents/latent_XX_mip_YY.pt`: decoded float latent mips (canonical source for true BC6 export here).
     - `latents/*.blocks128.bin`: compact *custom* 128-bit records (BC6-inspired; not guaranteed BC6 bitstream).

4) **Optional: export true BC6 DDS**
   - Run `export_true_bc6_dds.py` against the export dir:
     - Writes `true_bc6_dds/latent_XX.bc6.dds` (GPU-ready textures).
     - Uses an **external** BC6 encoder CLI (e.g. Compressonator CLI).

5) **Runtime decode**
   - In shader (`shaders/neural_material_decode.hlsl`):
     - sample BC6 latent textures (hardware filtering)
     - run MLP from `decoder_fp16.bin`
     - reconstruct albedo / normalTS / ao / roughness / metallic

---

## Data: `reference_8ch.pt` format (training target)

Created by `prepare_freepbr_material.py`.

File: `data/freepbr/materials/<material>/reference_8ch.pt`

Structure (PyTorch `torch.save`):
```python
{
  "base": Tensor[C=8, H, W],  # normalized to [-1,1]
  "meta": {
    "product_url": "...",
    "download_name": "...zip",
    "variant_keyword": "...",
    "used_directx_normal_y_flip": bool,
    "map_paths": { "albedo": "...", "normal": "...", ... },
    "channels": ["albedo_r","albedo_g","albedo_b","normal_x","normal_y","ao","roughness","metallic"],
  }
}
```

Channel encoding details (from `prepare_freepbr_material.py`):
- **Albedo**: `[0,1]` → `[-1,1]` via `*2-1`.
- **Normal**:
  - reads RGB normal map, converts to `nx, ny` in `[-1,1]` using `*2-1`.
  - if the filename contains `dx`/`directx`, flips the sign of `ny` (DirectX vs OpenGL convention).
  - only `nx, ny` are stored; `nz` is reconstructed later via `sqrt(1 - nx² - ny²)` with clamping.
- **AO / Roughness / Metallic**: grayscale `[0,1]` → `[-1,1]` via `*2-1`.

Normal-map theory note:
- Tangent-space normals are unit vectors on the hemisphere.
- Storing only `(nx, ny)` is common (e.g., BC5 normal maps); `nz` can be reconstructed because `nx² + ny² + nz² = 1`.
- Any time `nx² + ny² > 1`, reconstruction clamps `nz` to 0, which can cause artifacts. If this is a problem:
  - consider adding a training penalty on `max(0, nx² + ny² - 1)`,
  - or renormalizing `(nx, ny)` before reconstructing `nz`.

ORM theory note:
- Many pipelines pack AO/Roughness/Metallic into one texture to reduce fetches.
- Asset packs sometimes use glossiness instead of roughness, or invert channels; adjust the dataset prep if needed.

Mip chain:
- The reference file stores only `base` (mip0).
- `neuralmaterials.load_reference_mips(...)` builds `ref-mips` mips via `build_mip_chain(...)`.

---

## Model Architecture (How training maps to runtime)

### Inputs and outputs

Training input is **procedural**:
- Random `uv ∈ [0,1]^2` and random continuous `lod ∈ [0, max_lod]` sampled per batch.

Training target:
- `sample_mips_trilinear(ref_mips, uv, lod)` produces the target material output at that UV/LOD.
  - Trilinear across mips is done manually by sampling two discrete mips and lerping.
  - Each mip sampling uses `grid_sample` (with padding differences for MPS).

Model output:
- `decoder(latent_features(uv,lod))` predicts 8 channels in the same `[-1,1]` space as the reference.

### Objective function (what is being optimized)

At a high level, training solves:

```
min_{latents, decoder}  E_{(u,v,lod) ~ U} [ ||  D( concat_i L_i(u,v,lod + b_i) )  -  T(u,v,lod)  ||^2 ]
```

Where:
- `T(u,v,lod)` is the reference material sample produced by `sample_mips_trilinear(ref_mips, uv, lod, ...)`.
- `L_i(...)` are latent pyramid samplers (warmup or BC-surrogate, depending on phase).
- `b_i` are per-latent LOD biases (`lod_biases`).
- `D(...)` is the 2-layer MLP decoder.

Important nuance:
- The current repo trains on *one material at a time*, so the “dataset” is effectively the continuous function defined by a single `reference_8ch.pt`.
- This is closer to “compressing a texture set” than training a general neural network across many assets.

### Tensor shapes / conventions (important when editing code)

Across Python:
- `uv`: `Tensor[B,2]` in `[0,1]`
- `lod`: `Tensor[B]` in `[0, max_lod]` (continuous)
- reference mip: `Tensor[C,H,W]` (C=8 for materials, C=3 for latents)
- sampled reference/latent: `Tensor[B,C]`
- model prediction: `Tensor[B,8]`

Across runtime (shader):
- 4 latent textures (default) each provide `float3` at a pixel.
- those 12 scalars are the MLP input vector.

### Reference mip chain + trilinear sampling (nitty gritty)

Reference mip construction:
- `load_reference_mips(...)` loads `base` from `reference_8ch.pt` and builds a mip chain with `build_mip_chain(...)`.
- `build_mip_chain(...)` uses `torch.nn.functional.interpolate(..., mode="bilinear")` repeatedly to downsample.

Reference sampling:
- `sample_mips_trilinear(mips, uv, lod, bilinear_mode=...)` performs:
  1) clamp `lod` to valid range
  2) compute `l0 = floor(lod)`, `l1 = l0 + 1`
  3) sample **both** mip levels at the same UV using `grid_sample`
  4) lerp between them by `a = lod - l0`

The sampler uses `align_corners=False` and:
- padding mode `border` on most backends (stable behavior at UV edges),
- a special-case for Apple MPS where border padding is unsupported:
  - clamps UV slightly inward and uses `zeros` padding.

Practical implication:
- Numerical results can differ slightly across devices/backends.
- If you’re comparing runs, keep device and PyTorch version consistent.

### Latent representation: multiple textures + LOD bias

The model uses `N_LATENT` latent pyramids (default 4):
- Each pyramid is a set of mip levels, each mip is a `3×H×W` latent texture (3 channels per latent “texture”).
- Latent resolutions and mip counts are configured via CLI flags:
  - `--latent-res 512,256,128,64`
  - `--latent-mips 8,7,6,5`

LOD bias:
- For each latent pyramid, a bias `b_i = log2(latent_res_i / reference_res)` is computed.
- At sampling time, model uses `lod + b_i` when sampling that latent pyramid.
- This mirrors the runtime shader trick where LOD bias is implemented by scaling derivatives.

Two equivalent viewpoints of “LOD bias”:
- **Training (explicit LOD):** sample latent pyramid at `li = lod + b_i`.
- **Runtime (implicit LOD):** the GPU derives LOD from derivatives; scaling derivatives by `2^(b_i)` shifts the effective LOD similarly.

In this repo:
- Python uses the “explicit LOD” view (because we manually control `lod` during training).
- HLSL uses the “derivative scaling” view (because the GPU chooses LOD during `SampleGrad`).

### Decoder MLP: tiny 2-layer network

Defined in `neuralmaterials.MaterialDecoderMLP`:
- `fc1: (N_LATENT*3) -> hidden_dim` (default `12 -> 16`)
- ReLU
- `fc2: hidden_dim -> out_channels` (default `16 -> 8`)

Runtime expects:
- The *exact* weight layout packed in `decoder_fp16.bin` and read by `shaders/neural_material_decode.hlsl`.

---

## Training Phases (Implementation details)

Core training is implemented in `neuralmaterials.train(...)` and wrapped by `infrerenfe_nural_mateirals.py`.

### Default hyperparameters (as of current code)

From `neuralmaterials.TrainConfig` (see source for exact defaults):
- Batch size: `4096` in the orchestrator (or `8192` in the standalone `neuralmaterials.py` CLI)
- Phase iters: `phase1=5000`, `phase2=200000`, `phase3=0`
- Learning rates:
  - phase1: feature LR `5e-2`, MLP LR `1e-3`
  - phase2: feature LR `1e-2`, MLP LR `1e-3`
  - phase3: MLP LR `5e-4`
- Schedulers:
  - phase1: exponential LR decay `gamma=0.9995`
  - phase2: exponential LR decay `gamma=0.9999`

Rationale (practical, not formal):
- Warmup features start with a higher LR to quickly shape latent textures.
- Decoder LR stays relatively modest to avoid chasing noisy early features.
- Phase2 feature LR drops because parameters become more constrained/quantized and gradients can get brittle.

### Phase 1: Warmup (unconstrained latent pyramids)

Latents:
- `WarmupLatentPyramid` stores each mip as an unconstrained learnable parameter.
- At decode time it applies `tanh` to bound values to `[-1,1]`.

Sampling:
- Collect features by sampling each pyramid at `lod + bias`, concatenate `[B, N_LATENT*3]`.

Optimization:
- Adam over warmup latents + decoder.
- Trains against reference samples using MSE.

Purpose:
- Prevents unstable optimization when directly learning quantized / block-structured parameters.

### Phase 2: BC-constrained (BC6-like surrogate block params)

This repo does **not** implement a bit-exact BC6H encoder.
Instead, it implements a **differentiable, BC6-inspired surrogate** that:
- stores per-block endpoints/indices/partition id
- uses STE rounding to simulate quantization
- decodes blocks back into a float latent mip (bounded via `tanh`)

Warm-start:
- `NeuralMaterialCompressionModel.initialize_bc_from_warmup()` initializes each BC pyramid from the warmup mip tensors.
- During init, each 4×4 block picks the best partition mask among 32 fixed patterns (`make_partition_bank(...)`).
- After init, partition ids are **fixed** (`fix_partition_ids()` disables partition gradient).

Optimization:
- Adam over BC endpoints+indices (partition fixed) + decoder.

Notes:
- After warm-start, `fix_partition_ids()` makes partition choice effectively discrete and stops it from drifting.
- If you want to experiment with learnable partitions (paper-dependent), this is one of the main knobs to revisit.

### Phase 3: Optional quantized finetune (export path)

If enabled (`phase3_iters > 0`):
- Quantizes BC features in-place.
- Freezes BC feature parameters.
- Finetunes decoder only.

What “quantize” means here:
- Endpoints and indices are snapped to discrete levels according to `endpoint_bits` / `index_bits`.
- Quantization uses “parameter space” transforms (sigmoid/logit / tanh/atanh) so the stored parameters remain valid.
- After quantization, BC feature params are frozen (`requires_grad=False`) and only the decoder is optimized.

---

## BC6 Surrogate Representation (What exactly is learned)

Implemented in `neuralmaterials.BC6SurrogateBlockLevel` and `neuralmaterials.BC6SurrogatePyramid`.

Per 4×4 block parameters:
- `endpoints`: `[4, 3]` (4 RGB endpoints; two line segments for two partitions)
- `indices`: `[16]` (one scalar per texel)
- `partition_id`: one of 32 fixed binary masks (`partition_bank`) splitting the 16 texels into two groups

Key implementation choices:
- **Partition bank**: `make_partition_bank(...)` procedurally generates 32 binary 4×4 masks.
- **Partition choice**: after warm-start, partition ids are fixed (not learned further).
- **Endpoint quantization**: `endpoint_bits` (default 6), STE rounding.
- **Index quantization**: `index_bits` (default 3), STE rounding.
- **Signed mode toggle**: `bc6_signed_mode` changes a constant used in unquantization (repo typically uses unsigned mode).

### How parameters are stored (why `sigmoid/logit` and `tanh/atanh` show up)

The BC surrogate wants values that live in constrained ranges:
- endpoints should map to either `[0,1]` (unsigned) or `[-1,1]` (signed-ish)
- indices should map to `[0,1]`
- partition should be one of 32 masks

To make optimization easier, the code stores **unconstrained** parameters and maps them:
- Unsigned endpoints:
  - stored as real numbers `E_raw`
  - decoded as `E_n = sigmoid(E_raw)` to force `[0,1]`
  - warm-start from a target `e` uses logit: `E_raw = log(e / (1-e))`
- Signed endpoints (optional):
  - stored as real numbers `E_raw`
  - decoded as `E_s = tanh(E_raw)` to force `[-1,1]`
  - warm-start from a target `e` uses atanh: `E_raw = atanh(e)`
- Indices:
  - stored as real numbers `I_raw`
  - decoded as `t = sigmoid(I_raw)` (interpreted as interpolation position)
  - warm-start from a target `t` uses `logit(t)`

Partition:
- stored as logits `P_raw[32]`
- soft choice would use `softmax(P_raw)` (not used after warm-start in current code)
- hard choice uses `argmax(P_raw)` → one-hot

This “store unconstrained → map to valid range” pattern is common in differentiable quantization pipelines.

### Partition bank (what are the 32 masks?)

`make_partition_bank(...)` builds `partition_bank: Tensor[32,16]` where:
- each row is a binary mask over the 16 texels in a 4×4 block,
- the mask splits the block into two groups (“partition 1” and “partition 2”),
- masks are unique up to inversion (a mask and its complement are treated as the same split).

Why a small fixed bank?
- It’s cheaper than learning arbitrary per-block partitions.
- It mirrors the “limited mode set” spirit of BC formats, where the block layout comes from a small menu.

### Warm-start block fitting (how init chooses endpoints/indices/partition)

`BC6SurrogateBlockLevel.init_from_unconstrained(mip_chw)` does a per-block search:

1) Extract all 4×4 blocks from `mip_chw`:
   - input: `Tensor[3, H, W]`
   - reshaped to: `blocks: Tensor[NB, 16, 3]` (NB = number of 4×4 blocks)

2) For each candidate partition mask `mask[16]`:
   - compute per-group min/max across the 16 texels:
     - group1 min/max (where mask==1)
     - group2 min/max (where mask==0)
   - treat these min/max as endpoints for two line segments:
     - `(g1_min, g1_max)` and `(g2_min, g2_max)`

3) Compute a “best t” per texel by projecting each texel color onto the segment direction:
   - `t = dot(x - min, dir) / dot(dir, dir)` clamped to `[0,1]`

4) Reconstruct the block under this partition:
   - `rec = mask * rec_from_seg1 + (1-mask) * rec_from_seg2`

5) Compute MSE error and keep the best partition per block.

Implementation details worth knowing:
- This is chunked (e.g., 4096 blocks at a time) to keep memory bounded.
- The code uses a large constant to “ignore” masked-out texels when computing min/max.
- After selecting the best partition, the code:
  - sets `partition_logits` to a one-hot (large value on chosen id),
  - stores `indices` as `logit(t)` so `sigmoid(indices)` recovers `t`,
  - stores endpoints in the raw parameterization expected by the decode path.

### STE quantization (why gradients still flow through rounding)

The key helper is:
```python
ste_round(x) = x + (round(x) - x).detach()
```

Meaning:
- forward pass uses the rounded value,
- backward pass behaves like identity (gradient of `x`).

This is a practical trick to make “quantized-at-runtime” parameters trainable.

Decode path (high-level):
1) Map endpoints params to `[0,1]` via sigmoid (or `tanh` remap for signed mode).
2) STE-round to quantized integers.
3) Unquantize endpoints using the paper-inspired formula (see code comments).
4) Quantize indices and scale to match expected interpolation behavior.
5) Interpolate per-texel endpoints based on partition mask.
6) Apply a nonlinear “reinterpret” surrogate (approximation of BC6 half-float behavior).
7) Bound final decoded latent values via `tanh`.

### Decode path (step-by-step, aligned to code)

The code path in `BC6SurrogateBlockLevel._decode_soft_blocks(...)` is annotated as following the paper’s Eq.7/Eq.8/Eq.9 structure.

At a per-mip level:
- `NB` = number of 4×4 blocks
- endpoints are shaped `[NB, 4, 3]`
- indices are shaped `[NB, 16]`
- output blocks are shaped `[NB, 16, 3]` then reshaped to `[3, H, W]`

Steps (conceptually):
1) **Endpoints → normalized domain**
   - unsigned mode: `E_n = sigmoid(E_raw)` in `[0,1]`
   - signed mode: `E_n = (tanh(E_raw)+1)/2` also in `[0,1]` (then treated as signed-ish later)

2) **Quantize endpoints**
   - `E_q = round(E_n * (2^b - 1))` using STE rounding

3) **Unquantize endpoints (Eq.7-like)**
   - `E_u = (a * 2^16 * E_q + 2^15) / 2^b`
   - `a` is a constant that differs between signed/unsigned paths in the implementation

4) **Quantize indices**
   - `x_n = sigmoid(I_raw)` in `[0,1]`
   - `x_q = round(x_n * (2^q - 1))` using STE rounding
   - `x_scaled = 2^q * (x_q / (2^q - 1))`

5) **Partitioned interpolation (Eq.8-like)**
   - each block chooses a fixed binary mask `mask[16]` (after warm-start)
   - interpolate along segment 1 where `mask==1`, segment 2 where `mask==0`

6) **Nonlinear reinterpret approximation (Eq.9-like)**
   - applies a piecewise function intended to approximate a BC6 “half reinterpret” style nonlinearity
   - then bounds the result with `tanh(...)` to keep latent values stable

The important practical takeaway:
- The surrogate is designed so that training “feels” the effect of quantization + block structure.
- It is not trying to be a perfect BC6 decoder; it is a differentiable proxy.

Output:
- a decoded latent mip as `Tensor[3, H, W]` where `H,W` are multiples of 4 (block aligned).

This decoded mip is then sampled with standard filtering and fed to the MLP decoder.

### Why this is “BC6-inspired” (not a codec)

BC6H in real hardware has:
- multiple encoding modes with different bit allocations,
- specific bitfield layouts,
- and a true bit-exact decode path.

This repo’s surrogate captures the *shape* of the problem:
- per-block endpoints + per-texel indices + partition selection,
- fixed quantization bit budgets (`endpoint_bits`, `index_bits`),
- and a nonlinear “reinterpret” approximation,

but it intentionally avoids:
- mode selection,
- full BC6 bitfield packing,
- and strict bit-exactness.

For actual engine-ready BC6 textures, use `export_true_bc6_dds.py` (external encoder CLI).

---

## Code Walkthrough (by file)

### `infrerenfe_nural_mateirals.py` (orchestrator)

Modes:
- `train`:
  - loads reference mips (`load_reference_mips`)
  - trains (`train`)
  - exports artifacts (`export_trained_artifacts`)
  - writes `runs/<run>/training_history.json` and `runs/<run>/run_report.json`
- `full`:
  - same as `train`, then:
  - renders full-res mip0 from the trained model (`_render_mip0_from_model`)
  - saves `inference/*.png` and `analysis/*.png`
  - computes quality metrics (`_quality_metrics`) and random-batch metrics (`_eval_random_batch_metrics`)
  - estimates storage savings against an analytical “BCn baseline” (`_bcn_bytes`, `_collect_neural_storage_bytes`)
  - writes `analysis/quality_metrics.json` with:
    - full-image `mse_all`, plus per-group MSE (albedo / normal_xy / orm)
    - a random UV/LOD batch MSE sample (optional, controlled by `--analysis-batch-size`)
  - writes `run_report.json` including:
    - baseline runtime bytes estimate (BC1 albedo + BC5 normal + BC1 ORM with full mips)
    - neural runtime bytes (sum of `*.blocks128.bin` + `decoder_fp16.bin`)
    - savings percentage vs the baseline
- `infer`:
  - loads exported latents mip0 (`latent_XX_mip_00.pt`)
  - upsamples to output size (`_infer_output_resolution`)
  - runs the exported MLP weights (`decoder_state.pt`) on CPU/GPU
  - saves `inference/*.png` + `run_report.json`

Storage/savings note:
- The “baseline” is an analytical GPU runtime payload estimate for a typical PBR setup:
  - BC1 albedo (8 bytes/block)
  - BC5 normal (16 bytes/block)
  - BC1 ORM (8 bytes/block)
  - summed across the full mip chain (`_bcn_bytes`)
- It is intentionally **not** based on PNG source file sizes (which aren’t representative of GPU memory).

Key helper behavior:
- `detect_device("auto")` prefers CUDA; forces CPU on Apple MPS due to `grid_sample` backward limitations.
- `_pbr_views` converts `pred[8,H,W]` into:
  - `albedo` RGB
  - `normal` RGB by reconstructing `nz`
  - `orm` RGB = `(ao, roughness, metallic)`

### `neuralmaterials.py` (core)

Major components:
- Sampling + mips:
  - `build_mip_chain(base_chw, levels)` builds reference mips
  - `sample_texture_chw(tex_chw, uv, mode)` uses `grid_sample`
  - `sample_mips_trilinear(mips, uv, lod, bilinear_mode)` manual trilinear over discrete mips
- Warmup latents:
  - `WarmupLatentPyramid`: unconstrained learnable mips, `tanh` bounded
- BC surrogate:
  - `make_partition_bank(...)`: generates 32 fixed partition masks
  - `BC6SurrogateBlockLevel`: per-mip per-block endpoints/indices/partition logits
    - `init_from_unconstrained(...)`: warm-start + best partition selection per block
    - `decode_mip(...)`: decodes blocks → `Tensor[3,H,W]`
    - `export_quantized_block_params()`: emits endpoints/indices/partition ids as quantized tensors
    - `quantize_inplace()`: hard-quantizes parameters for phase3/export-like behavior
  - `BC6SurrogatePyramid`: list of block levels (one per mip)
- Full model:
  - `NeuralMaterialCompressionModel`:
    - holds warmup pyramids + BC pyramids + decoder MLP
    - computes `lod_biases` and applies them when sampling latents
    - provides `forward_warmup(...)` and `forward_bc(...)`
- Export format:
  - `export_trained_artifacts(...)` writes decoder weights and decoded latent mips to disk
  - `pack_quantized_blocks_to_128b(...)` packs quantized params into custom 128-bit records

### `prepare_freepbr_material.py` (dataset builder)

Pipeline:
1) downloads FreePBR zip by parsing the product page HTML download form
2) extracts the zip
3) heuristically picks maps by filename keywords
4) resizes to a square resolution (`--size`)
5) assembles the 8-channel tensor in `[-1,1]`
6) writes:
   - `reference_8ch.pt`
   - `dataset_report.json`
   - `maps/*` (original chosen maps + quick previews)

Map picking heuristics (details):
- The script parses multiple downloadable variants from the product page and prefers one whose name contains `--variant-keyword` (default `-bl.zip`).
- It scans extracted files for images and rejects obvious preview/thumb assets (`preview`, `thumb`, `thumbnail` in the filename).
- For each required map type it tries multiple keyword sets:
  - albedo: `albedo`, `basecolor`, `diffuse`, `color`, ...
  - normal: prefers `normal+ogl` / `normal+gl`, otherwise any `normal`/`nrm`
  - AO: `ao`, `ambient occlusion`, `occlusion`
  - roughness: `roughness`, `rough`
  - metallic: `metallic`, `metalness`, `metal`
- If multiple candidates match, it chooses the **largest by pixel area** (ties broken by shorter filename).

Normal handedness:
- If the chosen normal filename contains `dx` or `directx`, the script flips the sign of `ny`.
- This is a pragmatic heuristic; some assets may still require manual correction.

### `export_true_bc6_dds.py` (true BC6 DDS via external encoder)

Steps:
1) reads `export/metadata.json` and enumerates `latent_files`
2) groups latent `.pt` tensors into mip chains (per latent index)
3) writes an intermediate mip-chained **RGBA16F** DDS (DX10 header)
4) calls an external BC encoder CLI to produce `BC6H` (or signed `BC6H_SF`) DDS
5) validates that mip count wasn’t dropped and writes a JSON report

### `shaders/neural_material_decode.hlsl` (runtime)

Key details:
- Uses `ByteAddressBuffer` for the FP16 blob; indices are “half indices”, not byte offsets.
- Uses `SampleGrad` and derivative scaling to apply per-latent LOD bias.
- Assumes:
  - `N_LATENT=4`, `LATENT_CHANNELS_PER_TEX=3`, `IN_DIM=12`
  - `HIDDEN_DIM=16`, `OUT_DIM=8`
- Output mapping:
  - `[-1,1]` → `[0,1]` for all PBR outputs
  - reconstructs `normal.z` and normalizes

---

## Export Format (Runtime contract)

Export is implemented in `neuralmaterials.export_trained_artifacts(...)`.

Export directory layout:
```
export/
  decoder_fp16.bin
  decoder_state.pt
  metadata.json
  latents/
    latent_00_mip_00.pt
    latent_00_mip_00.png
    latent_00_mip_00.bc6params.pt
    latent_00_mip_00.blocks128.bin
    ...
```

What each file is for:
- `decoder_state.pt`: PyTorch state dict for debugging/validation (not used by the shader).
- `decoder_fp16.bin`: the runtime-consumable weight blob for the shader MLP.
- `metadata.json`: the “glue” that tells runtime how many latents exist, how to bias LOD, and what dims to expect.
- `latents/latent_XX_mip_YY.pt`: decoded float latent mip tensors (canonical source for true BC6 export in this repo).
- `latents/latent_XX_mip_YY.png`: LDR visualization only (maps `[-1,1] -> [0,255]` for quick viewing).
- `latents/latent_XX_mip_YY.bc6params.pt`: quantized surrogate parameters (endpoints/indices/partition id) used to generate `.blocks128.bin`.
- `latents/latent_XX_mip_YY.blocks128.bin`: packed custom 128-bit records per block (BC6-like payload accounting).

### `decoder_fp16.bin` layout (critical for shader)

Flat FP16 blob concatenating:
1) `fc1.weight` (HIDDEN_DIM * IN_DIM)
2) `fc1.bias`   (HIDDEN_DIM)
3) `fc2.weight` (OUT_DIM * HIDDEN_DIM)
4) `fc2.bias`   (OUT_DIM)

The shader reads this via `ByteAddressBuffer` and converts FP16 → FP32 using `f16tof32`.

Important runtime detail (why the HLSL uses a “half index”):
- `ByteAddressBuffer.Load(byteOffset)` loads a **32-bit word** (4 bytes).
- The shader wants to index individual FP16 values, so it:
  - computes which 32-bit word contains the FP16,
  - selects low/high 16 bits based on `halfIndex & 1`.

In other words:
- `halfIndex` is an index in units of 16-bit halves, not bytes.
- This matches how `export_trained_artifacts(...)` writes a contiguous FP16 array.

### `metadata.json`

Includes:
- `latent_resolutions`, `latent_count`
- `lod_biases` (pass into runtime shader cbuffer)
- decoder dims and file names
- inventory of exported latent files/mips
- notes describing that `*.blocks128.bin` are **custom** BC-like records (not a true DDS bitstream)

### `*.blocks128.bin` (custom packed block records)

These are packed 128-bit records per block intended for:
- payload accounting
- debugging / eventual direct-block DDS writing experiments

They are **not guaranteed** to match BC6 DDS bit layout.

Packing details (from `neuralmaterials.pack_quantized_blocks_to_128b(...)`):
- Each block becomes exactly **128 bits** (16 bytes).
- Bit order is little-endian, least-significant-bit first within the record.
- Field order (default):
  1) `partition_id` (5 bits)
  2) `endpoints_q` (12 values × `endpoint_bits`)
  3) `indices_q` (16 values × `index_bits`)
  4) padding to 128 bits

This is intentionally simple and stable for debugging, but it is not a BC6 bitfield layout.

---

## Inference Paths (Python)

There are two inference flows implemented in `infrerenfe_nural_mateirals.py`:

1) **From the live model** (in `--mode full`)
   - Calls `_render_mip0_from_model(...)` to evaluate `model.forward_bc(...)` at every pixel of mip0.
   - Saves albedo/normal/orm PNGs and analysis plots.
   - Implementation detail:
     - builds a full `uv` grid of shape `[H*W,2]`
     - uses `lod=0` everywhere (mip0)
     - evaluates in chunks (`--infer-chunk`) to control memory

2) **From exported artifacts** (in `--mode infer`)
   - Loads `export/latents/latent_XX_mip_00.pt`
   - Upsamples each latent to a chosen output size
   - Runs the MLP from `decoder_state.pt`
   - Saves albedo/normal/orm PNGs

   What it does *not* do:
   - it does not sample across latent mip levels
   - it does not simulate trilinear LOD on the latent pyramids
   - it bypasses BC surrogate params entirely (it uses already-decoded float latents)

This second path is a sanity check that the exported artifacts are self-consistent.

---

## True BC6 DDS Export (GPU-ready latents)

`export_true_bc6_dds.py` converts exported latent float tensors into **true BC6 DDS textures** by:
1) building a mip-chained **RGBA16F** DDS from the latent `RGB` mip tensors
2) invoking an external encoder CLI to encode the DDS into BC6 (`BC6H` or `BC6H_SF`)

Outputs:
- `export/true_bc6_dds/latent_XX.bc6.dds`
- `export/true_bc6_dds/true_bc6_export_report.json`

Important:
- The BC6 DDS are encoded from the `.pt` tensors, **not** from the `.png` previews.
- This requires an encoder CLI (Compressonator CLI is the default expected).

### DDS nitty gritty (what the script actually writes)

The script writes an intermediate DDS that is:
- **DX10 DDS** (pixel format FourCC is `"DX10"`)
- `DXGI_FORMAT_R16G16B16A16_FLOAT` for the source (RGBA16F)
- contains a full mip chain payload (concatenated mip images)

Then the external CLI encodes that source DDS into:
- `DXGI_FORMAT_BC6H_UF16` (unsigned) or
- `DXGI_FORMAT_BC6H_SF16` (signed)

The script validates:
- output DDS exists and has non-zero size
- mip count is not lower than the source mip count (some encoders drop mips unless configured)

CLI robustness:
- It tries a few argument orderings because different encoder builds use different CLI conventions.
- Optional `--decode-smoke` decodes the first BC6 DDS back to a PNG to sanity check the encode path.

---

## Gotchas / Sharp Edges

- **Block alignment:** BC surrogate mips are forced to multiples of 4. If you change latent resolutions/mips, ensure every mip stays ≥4 and divisible by 4.
- **Mip dimensions can “snap”:** BC mips are floored to a multiple of 4. This means some downsample steps are not exact powers of two, which can slightly affect filtering/LOD behavior.
- **Normal validity:** only `nx, ny` are trained; `nz` is reconstructed later. If `nx^2 + ny^2 > 1`, `nz` clamps to 0 (can create artifacts).
- **Sampler mismatch:** reference sampling sometimes uses `bicubic` while latents use `bilinear`. This is a design choice for stability/quality but means “training target filtering” is not exactly the same as “runtime filtering”.
- **MPS backend:** `grid_sample` backward limitations are handled by forcing CPU in auto device selection.
- **“BC6-like” vs “BC6”:** `*.blocks128.bin` are compact *custom* records; “true BC6 DDS” requires external encoding.
- **Export infer is mip0-only:** `--mode infer` currently loads only `mip_00` latents and upsamples them; it does not reproduce trilinear LOD sampling across latent mips.

## Runtime Shader Contract (HLSL)

File: `shaders/neural_material_decode.hlsl`

Resources expected:
- `ByteAddressBuffer gDecoderWeightsFP16` (t0): `decoder_fp16.bin`
- `Texture2D<float3> gLatentTex0..3` (t1..t4): latent textures (BC6 / or any float3 texture in debug)
- `SamplerState gLinearWrapSampler` (s0)
- `cbuffer NeuralMaterialCB` (b0): LOD biases (`gLodBias0..3`) from `metadata.json`

Sampling:
- Uses `SampleGrad` and scales derivatives by `exp2(lodBias)` to apply a LOD bias without explicit lod sampling.

Decoder:
- Implements the same 2-layer MLP in-shader (unrolled loops).

Output mapping:
- `albedo`: `[-1,1] -> [0,1]` via `*0.5 + 0.5`
- `normalTS`: reconstruct `nz = sqrt(saturate(1 - nx^2 - ny^2))`, normalize
- `ao/roughness/metallic`: same `*0.5 + 0.5`

### Practical integration notes (nitty gritty)

- **Decoder blob size is tiny.** For default dims `IN_DIM=12`, `HIDDEN_DIM=16`, `OUT_DIM=8`:
  - FP16 values = `(16*12) + 16 + (8*16) + 8 = 344`
  - bytes = `344 * 2 = 688` bytes

- **Shader math cost is modest.**
  - `fc1`: `HIDDEN_DIM * IN_DIM` multiply-adds (192)
  - `fc2`: `OUT_DIM * HIDDEN_DIM` multiply-adds (128)
  - total ≈ 320 MACs/pixel + 4 texture fetches + some ALU (ReLU/sqrt/normalize)

- **Precision model:**
  - weights are stored as FP16 (compact), expanded to FP32 via `f16tof32`
  - accumulation is FP32 in the shader loops
  - outputs are clamped with `saturate` after remapping to `[0,1]`

- **Resource binding semantics:**
  - `ByteAddressBuffer` is a raw buffer view of the FP16 blob (no struct layout).
  - `Texture2D<float3>` assumes your latent textures are bound as 3-channel sampled textures.
    - In practice BC6 is stored as a compressed format; the API view still behaves like sampling `float3`.
  - `cbuffer` contains the `lod_biases` from `metadata.json`.

- **Debug strategy:**
  - Bind *uncompressed* float textures for latents first (to validate shader logic).
  - Then switch to true BC6 DDS latents (to validate the encode/decode path).

---

## Debugging + Verification (practical checklist)

When something looks “wrong”, these are the fastest sanity checks:

### Dataset / reference tensor
- Open `data/freepbr/materials/<material>/maps/*_preview.png`:
  - albedo preview should look like the expected basecolor.
  - normal preview should look like a mostly blue normal map (with correct handedness).
  - ORM preview should show plausible AO/roughness/metallic patterns.
- Confirm `reference_8ch.pt` channel ordering matches expectations:
  - `[albedo_rgb, normal_x, normal_y, ao, roughness, metallic]` normalized to `[-1,1]`.
- If normals look “inside out”, check the DirectX/OpenGL Y-flip detection:
  - the script flips `ny` when filename contains `dx` / `directx`.

### Training sanity
- Run a tiny `--mode full` (e.g. `phase1=10, phase2=10`) to verify the pipeline completes and exports artifacts.
- Check `runs/<run>/analysis/training_loss.png` decreases at least initially.
- Inspect `runs/<run>/analysis/gt_vs_neural_diff.png`:
  - albedo errors should cluster at edges/details first.
  - normal/ORM errors can remain higher depending on channel weighting (currently unweighted MSE).

### Export/infer sanity
- `--mode infer` should produce a reasonable `inference/pbr_preview.png`.
- If infer-from-export differs wildly from full-mode inference:
  - remember infer-from-export is mip0-only and upsamples latents; it’s not a perfect match for runtime LOD sampling.

### BC6 DDS sanity
- Run `export_true_bc6_dds.py --decode-smoke` and inspect the decoded PNG:
  - it should resemble the latent `.png` preview qualitatively (not identical due to tonemapping/clamping differences).
- If the encoder drops mips, the script will raise; adjust the encoder CLI flags/variant.

---

## Extensions + Research Directions (if you want to push it further)

These ideas are consistent with the repo’s design and are common next steps:

### Multi-material training

Current behavior:
- The repo optimizes both **latents** and the **decoder** for one material at a time.

If you want a more “paper-like” production setup:
- Train a **shared decoder** across many materials.
- Store only per-material latents as assets.

That requires:
- a dataset loader that samples across materials,
- a training loop that batches samples from multiple materials,
- and an export format where each material has its own latent package but references a shared decoder.

### Loss shaping (more perceptual / more physically relevant)

Right now the objective is plain MSE across 8 channels.

Common refinements:
- channel weighting (e.g., emphasize normals/roughness if visually dominant)
- normal-specific penalty to discourage invalid `nx,ny` magnitudes
- multi-scale losses (match errors at multiple reference mips explicitly)
- perceptual losses for albedo (less common for PBR pipelines, but possible)

### Latent/channel layout changes

The repo uses 4× `float3` latents → 12 inputs.

Alternatives:
- fewer latents with larger hidden dims
- more latents with smaller decoder
- packing latents differently (e.g., 2-channel latents and derive a third)

Any change must update both export and shader contract.

### Toward “direct BC6 blocks” export

Today:
- surrogate params → custom `*.blocks128.bin`
- true BC6 DDS via external encoder CLI

If you want direct engine-ready BC6 blocks without external tools:
- implement a real BC6H encoder (non-trivial)
- or implement enough of BC6 bitfield packing + mode selection to match a subset

This is a deeper project on its own; treat it as an optional stretch goal.

---

## If you change model dimensions (keep everything consistent)

If you change any of:
- latent count (`N_LATENT`)
- latent channels per texture (currently 3)
- MLP hidden dim
- output channel count

Then update in lockstep:
- Python:
  - `neuralmaterials.NeuralMaterialCompressionModel` (decoder in/out dims)
  - `export_trained_artifacts` (weight packing order and metadata)
  - `infrerenfe_nural_mateirals.py` inference-from-export path
- Shader:
  - `shaders/neural_material_decode.hlsl` macros (`N_LATENT`, `IN_DIM`, `HIDDEN_DIM`, `OUT_DIM`)
- Docs:
  - this guide (`.claude/reference/NNPBR_GUIDE.md`)

## Usage (moved to a Claude skill)

To keep this guide focused on architecture/theory, **all “how to run it” usage** lives in:
- `.claude/skills/nnpbr-usage/SKILL.md` (invoke as `/nnpbr-usage`)

When CLI flags or scripts evolve:
- run `/nnpbr-update-skills` and update the usage skill + any relevant docs.

---

## What’s intentionally incomplete / TODO

Current state:
- Training + export + Python inference is working for single-material datasets.
- The **Vulkan example** is a placeholder; it needs:
  - resource binding for `decoder_fp16.bin` + BC6 latent textures
  - descriptor set layout matching the HLSL shader
  - swapchain fullscreen pass wiring

See also README TODOs.

---

## Sub-Agents (recommended specialization)

This repo is easiest to iterate on with three specialized agents:
- `.claude/agents/paper-planner.md`: read paper → map to code → prioritize gaps.
- `.claude/agents/python-core.md`: implement/modify training/export/inference in Python.
- `.claude/agents/demo-testing.md`: run experiments, generate demos, validate runtime/shader integration.

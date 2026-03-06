# NN-PBR Full vs Infer Mode Discrepancy - Root Cause Analysis

## Executive Summary
The difference between `--mode full` and `--mode infer` outputs is caused by a **fundamental architectural mismatch** in how latent tensors are represented:
- **Soft Surrogate Training (Full Mode)**: Uses BC6H **unsigned [0, 1]** internally, but exports as if values were in **[-1, 1]**
- **BC6H DDS Files (Infer Mode)**: Stores true **unsigned [0, 1]** values, which are then converted to **[-1, 1]** on read

## Detailed Findings

### 1. Latent Tensor Ranges

**Actual trained latent values (from soft surrogate):**
```
Range: [0, 1] (unsigned mode)
Mean: ~0.5
Example mip values: min=0.0, max=1.0
```

**PNG preview export (from trained latents):**
- Assumes latents are in [-1, 1]: `uint8 = ((x + 1.0) * 0.5 * 255.0)`
- For actual [0, 1] values: uint8 ∈ [128, 255]
- Observed PNG uint8: [128, 255] ✓ (confirms latents are [0, 1])

**PNG preview import (debug script):**
- Reads as [0, 1] LDR: `float32 = uint8 / 255.0` ∈ [0.5, 1.0]
- Converts assuming [-1, 1]: `latent = float32 * 2 - 1` ∈ [0, 1]
- **WRONG:** Should be [-1, 1] but ends up in [0, 1]

**DDS export/import (true BC6H):**
- Stores unsigned [0, 1] natively (11-bit UF16)
- DDS decode returns [0, 1]: `value / 2047.0`
- Converts to [-1, 1]: `value * 2 - 1` ∈ [-1, 1] ✓
- Observed DDS values: [-0.4, 0.4] ✓

### 2. The Mismatch

| Operation | Full Mode | Infer Mode | Decoder Expects |
|-----------|-----------|-----------|-----------------|
| Latent source | Soft surrogate (trained) | BC6H DDS (exported) | [-1, 1] |
| Latent range | [0, 1] | After decode: [-1, 1] | [-1, 1] |
| Actual input to decoder | [0, 1] (unchanged!) | [0, 1] rescaled as [-1, 1] | [-1, 1] |
| **Result** | **Biased positive** | **Balanced around 0** | **Incorrect** |

### 3. Why Outputs Differ

**Full Mode (using trained latents [0, 1]):**
- Decoder receives biased positive values (mean ≈ 0 in the [-1, 1] interpretation)
- Network trained on these specific distributions
- Produces biased predictions

**Infer Mode (using DDS latents [-1, 1] after conversion):**
- Decoder receives balanced values around 0
- Different distribution than training
- Produces different predictions (MSE ≈ 0.07 for albedo)

### 4. Discrepancy Magnitude

**Albedo channel:**
- Full vs Infer MSE: 6.92e-02
- Per-channel error: B=1.12e-01, G=9.46e-02, R=1.39e-03

**ORM channel:**
- Full vs Infer MSE: 1.09e-01 (severe)

**Normal channel:**
- Full vs Infer MSE: 1.39e-03 (acceptable)

## Root Cause Summary

The soft BC6 surrogate for **unsigned mode** naturally operates in [0, 1] space. However:
1. PNG save assumes inputs are in [-1, 1] and converts accordingly
2. PNG load assumes inputs are in [-1, 1] and converts back
3. But actual trained latents are in [0, 1], so both conversions create offsets
4. DDS uses true BC6H unsigned encoding (preserves [0, 1] space correctly)
5. DDS decode applies the [-1, 1] conversion, changing the distribution
6. Decoder was trained with [0, 1] soft surrogate values but receives [-1, 1] values from DDS

## Fix Options

### Option A: Change BC6H Mode to Signed (SF16)
- Use `bc6_signed_mode=True` in training and export
- Latents would naturally be in [-1, 1]
- PNG conversions would work correctly
- DDS would store signed values
- **Requires retraining**

### Option B: Fix PNG Save/Load to Match Actual Space
- Save trained latents as-is (they're in [0, 1])
- Load PNG without conversion (read directly)
- No conversion in DDS pipeline
- **Requires code changes in export/infer**

### Option C: Fix Decoder Input Normalization
- Add normalization layer before decoder
- Rescale DDS values [0, 1] to match training distribution
- **Requires retraining with normalization**

### Option D: Use Export-Latents During Training
- Run full mode with `--use-export-latents` flag
- Ensures training uses same DDS-decoded latents as inference
- **Requires code changes, validates current approach**

## Recommendation
**Option A (signed BC6)** is the cleanest architecturally, but requires retraining.
**Option B (fix PNG pipeline)** is quickest, requires code changes to clarify the space (0, 1 for unsigned).
**Option D (validation mode)** should be tested first to confirm this theory.

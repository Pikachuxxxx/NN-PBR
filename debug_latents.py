#!/usr/bin/env python3
"""
Compare latent tensors from:
1. Full mode: in-memory model.bc_pyramids
2. Infer mode: DDS decode_bc6h_dds_mip0
"""
import json
import torch
import numpy as np
from pathlib import Path
from neuralmaterials import decode_bc6h_dds_mip0, load_reference_mips, TrainConfig, NeuralMaterialCompressionModel

export_dir = Path("runs/debug_full_256/export")

# Load metadata to understand structure
with open(export_dir / "metadata.json") as f:
    meta = json.load(f)

print("=" * 80)
print("LATENT TENSOR ANALYSIS")
print("=" * 80)

# Load DDS-decoded latents (what infer mode uses)
print("\n[1] Loading latents from DDS files (infer mode):")
dds_latents = []
signed_mode = bool(meta.get("bc6_signed_mode", False))

for i in range(4):
    dds_path = export_dir / f"latent_{i:02d}.bc6.dds"
    if dds_path.exists():
        t = decode_bc6h_dds_mip0(dds_path, signed_mode=signed_mode)
        dds_latents.append(t)
        print(f"  latent_{i}: shape={t.shape}, min={t.min():.4f}, max={t.max():.4f}, "
              f"mean={t.mean():.4f}, std={t.std():.4f}")

# Now compare with PNG previews (if available)
print("\n[2] Checking PNG previews (decode_mips from BC params):")
for i in range(4):
    png_path = export_dir / "metadata" / f"latent_{i:02d}_mip_00.png"
    if png_path.exists():
        from PIL import Image
        img = np.array(Image.open(png_path), dtype=np.float32) / 255.0
        # PNGs are saved as LDR [0, 1], need to map back to [-1, 1]
        ldr_latent = img.transpose(2, 0, 1)  # [H, W, 3] -> [3, H, W]
        print(f"  latent_{i} PNG (raw LDR [0,1]): min={ldr_latent.min():.4f}, max={ldr_latent.max():.4f}, "
              f"mean={ldr_latent.mean():.4f}")
        latent = ldr_latent * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        print(f"  latent_{i} PNG (converted [-1,1]): shape={latent.shape}, min={latent.min():.4f}, max={latent.max():.4f}, "
              f"mean={latent.mean():.4f}, std={latent.std():.4f}")

        # Compare with DDS
        if i < len(dds_latents):
            dds = dds_latents[i].numpy()
            diff = np.abs(dds - latent)
            print(f"    vs DDS: MSE={np.mean(diff**2):.2e}, MAE={np.mean(diff):.2e}, max={np.max(diff):.2e}")

print("\n" + "=" * 80)
print("DECODER ANALYSIS")
print("=" * 80)

# Load decoder state
decoder_state_path = export_dir / "metadata" / "decoder_state.pt"
if decoder_state_path.exists():
    state = torch.load(decoder_state_path, map_location="cpu")
    print("\nDecoder state dict keys:", list(state.keys()))
    for key, val in state.items():
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}, "
              f"min={val.min():.4f}, max={val.max():.4f}")

# Load FP16 blob
fp16_path = export_dir / "decoder_fp16.bin"
if fp16_path.exists():
    fp16_bytes = fp16_path.read_bytes()
    fp16_data = np.frombuffer(fp16_bytes, dtype=np.float16)
    print(f"\nDecoder FP16 blob: {len(fp16_data)} elements, "
          f"min={fp16_data.min():.4f}, max={fp16_data.max():.4f}")

    # Try to reconstruct from FP16
    fc1_weight_els = 16 * 12  # hidden_dim * in_dim
    fc1_bias_els = 16
    fc2_weight_els = 8 * 16  # out_dim * hidden_dim
    fc2_bias_els = 8

    fc1_w = fp16_data[0:fc1_weight_els].astype(np.float32).reshape(16, 12)
    fc1_b = fp16_data[fc1_weight_els:fc1_weight_els+fc1_bias_els].astype(np.float32)
    fc2_w_start = fc1_weight_els + fc1_bias_els
    fc2_w = fp16_data[fc2_w_start:fc2_w_start+fc2_weight_els].astype(np.float32).reshape(8, 16)
    fc2_b = fp16_data[fc2_w_start+fc2_weight_els:].astype(np.float32)

    print(f"\n  fc1.weight (from FP16): shape={fc1_w.shape}, min={fc1_w.min():.4f}, max={fc1_w.max():.4f}")
    print(f"  fc1.bias (from FP16):   shape={fc1_b.shape}, min={fc1_b.min():.4f}, max={fc1_b.max():.4f}")
    print(f"  fc2.weight (from FP16): shape={fc2_w.shape}, min={fc2_w.min():.4f}, max={fc2_w.max():.4f}")
    print(f"  fc2.bias (from FP16):   shape={fc2_b.shape}, min={fc2_b.min():.4f}, max={fc2_b.max():.4f}")

    # Compare with state dict
    if decoder_state_path.exists():
        state = torch.load(decoder_state_path, map_location="cpu")
        for key, expected in [("fc1.weight", fc1_w), ("fc1.bias", fc1_b),
                              ("fc2.weight", fc2_w), ("fc2.bias", fc2_b)]:
            actual = state[key].numpy().astype(np.float32)
            if key.endswith(".weight"):
                actual = actual.reshape(expected.shape)
            diff = np.abs(actual - expected)
            print(f"\n  {key} FP16 vs state: MSE={np.mean(diff**2):.2e}, MAE={np.mean(diff):.2e}")

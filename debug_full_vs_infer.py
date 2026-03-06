#!/usr/bin/env python3
"""
Compare full mode vs infer mode outputs to find discrepancies.
"""
import os
import numpy as np
from PIL import Image
import json

full_dir = "runs/debug_full_256/inference"
infer_dir = "runs/debug_infer_256"
output_dir = "runs/debug_comparison"
os.makedirs(output_dir, exist_ok=True)

channels = ["albedo", "normal", "orm"]
results = {}

print("=" * 80)
print("FULL MODE vs INFER MODE COMPARISON")
print("=" * 80)

for channel in channels:
    full_path = f"{full_dir}/{channel}.png"
    infer_path = f"{infer_dir}/{channel}.png"

    if not os.path.exists(full_path) or not os.path.exists(infer_path):
        print(f"\n⚠️  Missing {channel}: full={os.path.exists(full_path)}, infer={os.path.exists(infer_path)}")
        continue

    # Load images
    full_img = np.array(Image.open(full_path), dtype=np.float32) / 255.0
    infer_img = np.array(Image.open(infer_path), dtype=np.float32) / 255.0

    # Compute differences
    abs_diff = np.abs(full_img - infer_img)
    mse = np.mean(abs_diff ** 2)
    mae = np.mean(abs_diff)
    max_diff = np.max(abs_diff)

    # Per-channel stats if multi-channel
    stats_by_ch = {}
    if len(full_img.shape) == 3:
        for c in range(full_img.shape[2]):
            ch_diff = abs_diff[:, :, c]
            stats_by_ch[f"channel_{c}"] = {
                "mse": float(np.mean(ch_diff ** 2)),
                "mae": float(np.mean(ch_diff)),
                "max": float(np.max(ch_diff))
            }

    results[channel] = {
        "shape_full": full_img.shape,
        "shape_infer": infer_img.shape,
        "mse": float(mse),
        "mae": float(mae),
        "max_diff": float(max_diff),
        "per_channel": stats_by_ch,
        "match": "✅" if mse < 1e-6 else "❌"
    }

    print(f"\n{channel.upper()}")
    print(f"  Shape: {full_img.shape} vs {infer_img.shape}")
    print(f"  MSE:      {mse:.2e}")
    print(f"  MAE:      {mae:.2e}")
    print(f"  Max diff: {max_diff:.2e}")
    print(f"  Match:    {results[channel]['match']}")

    if len(stats_by_ch) > 0:
        print(f"  Per-channel stats:")
        for ch_name, ch_stats in stats_by_ch.items():
            print(f"    {ch_name}: MSE={ch_stats['mse']:.2e}, MAE={ch_stats['mae']:.2e}, max={ch_stats['max']:.2e}")

    # Generate visual diff (if there's a difference)
    if mse > 1e-6:
        diff_viz = (abs_diff * 10.0).clip(0, 1)  # Scale for visibility
        diff_img = Image.fromarray((diff_viz * 255).astype(np.uint8))
        diff_path = f"{output_dir}/{channel}_diff_viz.png"
        diff_img.save(diff_path)
        print(f"  Diff visualization saved: {diff_path}")

# Check metadata and decoder
print("\n" + "=" * 80)
print("ARTIFACT ANALYSIS")
print("=" * 80)

metadata_path = "runs/debug_full_256/export/metadata.json"
with open(metadata_path) as f:
    metadata = json.load(f)

print(f"\nMetadata version: {metadata.get('version')}")
print(f"Decoder params: hidden_dim={metadata.get('hidden_dim')}, hidden_act={metadata.get('hidden_act')}")
print(f"BC6H params: endpoint_bits={metadata.get('endpoint_bits')}, index_bits={metadata.get('index_bits')}")
print(f"Latent resolutions: {metadata.get('latent_resolutions')}")
print(f"Latent mip counts: {metadata.get('latent_mip_counts')}")

# Check decoder weights
decoder_path = "runs/debug_full_256/export/decoder_fp16.bin"
decoder_size = os.path.getsize(decoder_path)
print(f"\nDecoder weights: {decoder_size} bytes")

# Check latent DDS files
print("\nLatent DDS files:")
for i in range(4):
    dds_path = f"runs/debug_full_256/export/latent_{i:02d}.bc6.dds"
    if os.path.exists(dds_path):
        size = os.path.getsize(dds_path)
        print(f"  latent_{i:02d}.bc6.dds: {size} bytes")

# Save comparison report
report_path = f"{output_dir}/comparison_report.json"
with open(report_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nComparison report saved: {report_path}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
all_match = all(r['match'] == "✅" for r in results.values())
if all_match:
    print("✅ All channels match perfectly between full and infer modes!")
else:
    print("❌ Discrepancies found between full and infer modes")
    for ch, r in results.items():
        if r['match'] == "❌":
            print(f"   - {ch}: MSE={r['mse']:.2e}")

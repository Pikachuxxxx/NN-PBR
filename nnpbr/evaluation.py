"""Deterministic, export-based validation at integer and fractional reference LODs."""

import argparse, json, math
from pathlib import Path
import torch
from .artifacts import ExportedMaterial
from .sampling import load_reference_mips, sample_mips_trilinear


@torch.no_grad()
def evaluate(export_dir, reference, device="cpu", query_side=64):
    runtime = ExportedMaterial(export_dir, device)
    mips = load_reference_mips(Path(reference), 32, 8, torch.device(device))
    if mips[0].shape[-1] != runtime.meta["reference_resolution"]:
        raise ValueError("Reference/export resolution mismatch")
    axis = (torch.arange(query_side, device=device) + 0.5) / query_side
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    uv = torch.stack([xx, yy], -1).reshape(-1, 2)
    results = []
    for level in [i * 0.5 for i in range(2 * len(mips) - 1)]:
        lod = torch.full((len(uv),), level, device=device)
        target = sample_mips_trilinear(mips, uv, lod, "bicubic")
        pred = runtime(uv, lod)
        d = pred - target
        scaled = d.clone()
        scaled[:, 3:] *= 0.5
        groups = {
            name: float(d[:, a:b].square().mean())
            for name, a, b in [("albedo", 0, 3), ("normal_xy", 3, 5), ("orm", 5, 8)]
        }
        results.append(
            {
                "reference_lod": level,
                "mse": float(d.square().mean()),
                "normalized_psnr_db": -10
                * math.log10(max(float(scaled.square().mean()), 1e-30)),
                "p99_normalized_abs_error": float(
                    torch.quantile(scaled.abs().flatten(), 0.99)
                ),
                "channel_mse": groups,
            }
        )
    return {
        "query_grid": [query_side, query_side],
        "source": "DDS + FP16 blob",
        "levels": results,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export-dir", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    a.output.write_text(
        json.dumps(evaluate(a.export_dir, a.reference, a.device), indent=2)
    )

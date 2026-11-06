"""Rebuild channel-correct data from existing source maps; no downloads."""

import argparse
import hashlib
from pathlib import Path

import torch
from .freepbr import _build_reference_tensor
from nnpbr.sampling import build_mip_chain

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--maps", required=True, type=Path)
p.add_argument("--output", required=True, type=Path)
p.add_argument("--size", type=int, default=128)
p.add_argument("--levels", type=int, default=8)
a = p.parse_args()
paths = [
    a.maps.resolve() / (s + ".png")
    for s in ("albedo", "normal", "ao", "roughness", "metallic")
]
base, files, dx = _build_reference_tensor(*paths, a.size)
a.output.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "base": base,
        "mips": build_mip_chain(base, a.levels),
        "meta": {
            "channel_schema": "albedo_srgb01_normal_xy_signed_orm_signed_v1",
            "map_paths": files,
            "normal_y_flipped": dx,
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
            },
        },
    },
    a.output,
)
print(
    f"Wrote {a.output}: {tuple(base.shape)}, albedo range {base[:3].min():.4f}..{base[:3].max():.4f}"
)

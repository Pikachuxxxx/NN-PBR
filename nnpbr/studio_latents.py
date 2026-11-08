"""Render actual BC6H mip values to temporary inspection thumbnails."""

import json
import sys
from pathlib import Path

import torch
from PIL import Image

from .codec import decode_bc6h_dds


def main():
    output = Path(sys.argv[1])
    torch.set_num_threads(2)
    layers = []
    for i, path in enumerate(sys.argv[2:]):
        levels = []
        for j, texture in enumerate(decode_bc6h_dds(Path(path), False)):
            lo, hi = float(texture.min()), float(texture.max())
            rgb = (
                ((texture - lo) / max(hi - lo, 1e-12) * 255)
                .round()
                .byte()
                .permute(1, 2, 0)
                .numpy()
            )
            image = Image.fromarray(rgb)
            levels.append({"size": image.width, "min": lo, "max": hi})
            image.thumbnail((512, 512))
            image.save(output / f"{i}-{j}.png")
        layers.append(levels)
    (output / "levels.json").write_text(json.dumps(layers))


if __name__ == "__main__":
    main()

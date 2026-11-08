"""Generate display-sized comparison maps from an NN-PBR export."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

from .artifacts import ExportedMaterial
from .cli import _pbr_views, _render_mip0_from_export
from .freepbr import _build_reference_tensor
from .sampling import load_reference_mips


def thumbnail(values, file):
    pix = (np.clip(values, 0, 1) * 255).round().astype("uint8")
    image = Image.fromarray(pix)
    image.thumbnail((384, 384), Image.Resampling.LANCZOS)
    image.save(file)


def channels(tensor):
    albedo, normal, orm = _pbr_views(tensor)
    return [albedo, normal, *(orm[:, :, i] for i in range(3))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--files", nargs=5, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    material = ExportedMaterial(args.export)
    for i, chain in enumerate(material.mips):
        rgb = chain[0].permute(1, 2, 0).numpy()
        thumbnail(
            (rgb - rgb.min()) / max(float(np.ptp(rgb)), 1e-12),
            args.output / f"latent-{i}.png",
        )
    del material
    decoded = _render_mip0_from_export(args.export, torch.device("cpu"))
    run = args.export.parent
    reference = run / "reference.pt"
    report = run / "run_report.json"
    if not reference.exists() and report.exists():
        ref = json.loads(report.read_text()).get("reference_pt")
        if ref:
            reference = Path(ref)
    label = None
    truth = None
    if reference.is_file():
        truth = load_reference_mips(reference, 1, 8, torch.device("cpu"))[0]
        label = "Training reference"
    elif args.files:
        truth, _, _ = _build_reference_tensor(*args.files, decoded.shape[-1])
        label = "Selected source maps"
    if truth is not None and truth.shape != decoded.shape:
        raise ValueError("Reference and export dimensions differ")
    gt = channels(truth) if truth is not None else []
    stats = []
    for i, (view, components) in enumerate(
        zip(
            channels(decoded),
            [slice(0, 3), slice(3, 5), slice(5, 6), slice(6, 7), slice(7, 8)],
        )
    ):
        thumbnail(view, args.output / f"decoded-{i}.png")
        if truth is None:
            continue
        thumbnail(gt[i], args.output / f"source-{i}.png")
        error = np.abs(gt[i] - view)
        if error.ndim == 3:
            error = error.mean(axis=2)
        peak = float(error.max())
        thumbnail(
            colormaps["inferno"](error / max(peak, 1e-12))[:, :, :3],
            args.output / f"error-{i}.png",
        )
        stats.append(
            {
                "mse": float((truth[components] - decoded[components]).square().mean()),
                "peak": peak,
            }
        )
    (args.output / "summary.json").write_text(
        json.dumps({"source": label, "channels": stats})
    )


if __name__ == "__main__":
    main()

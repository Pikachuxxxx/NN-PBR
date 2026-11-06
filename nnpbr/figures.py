"""Regenerate README figures from a finished bake without retraining."""

import argparse
import json
from pathlib import Path

import torch

from .cli import _render_mip0_from_export, _save_full_plots
from .plot import save_loss
from .sampling import load_reference_mips


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, help="Defaults to the run's analysis directory")
    parser.add_argument("--reference", type=Path, help="Original reference if the run was moved")
    args = parser.parse_args()
    run = args.run
    report = json.loads((run / "run_report.json").read_text())
    if report["mode"] != "full":
        parser.error("A completed full-mode run is required for the storage comparison")
    history = json.loads((run / "training_history.json").read_text())
    output = args.out or run / "analysis"
    reference = args.reference or Path(report["reference_pt"])
    if not reference.is_file():
        parser.error("Supply the original reference tensor with --reference")
    device = torch.device("cpu")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    base = load_reference_mips(reference, 1, 8, device)[0]
    export = run / "export"
    decoded = _render_mip0_from_export(export, device)
    if base.shape != decoded.shape:
        parser.error("Reference and export dimensions differ")
    storage = report["storage_bytes"]
    images = _save_full_plots(base, decoded, history, output, storage["baseline_pbr_bcn"],
                              storage["neural_runtime_total"], export)
    images["loss_timeline"] = str(save_loss(run, history, output))
    for path in images.values():
        print(path)


if __name__ == "__main__":
    main()

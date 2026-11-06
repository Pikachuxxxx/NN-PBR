"""Bake a bundled material and inspect the runtime files from the same run."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def python(*args, cpu=False):
    env = os.environ.copy()
    if cpu:
        env.setdefault("OMP_NUM_THREADS", "2")
    subprocess.run([sys.executable, *args], cwd=ROOT, env=env, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("smoke", "Bake a small CPU test"), ("train", "Bake a full 2K material on GPU")):
        bake = commands.add_parser(name, help=help_text)
        bake.add_argument("material", nargs="?", default="red-plaid-bl",
                          choices=sorted(p.name for p in (ROOT / "data/freepbr/materials").iterdir() if p.is_dir()))
        bake.add_argument("--out", type=Path, help="New run directory")
    check = commands.add_parser("check", help="Score exported DDS and FP16 weights against the reference")
    check.add_argument("run", type=Path)
    view = commands.add_parser("view", help="Open a baked material in Vulkan")
    view.add_argument("run", type=Path)
    view.add_argument("--hlsl", action="store_true", help="Compile and display the embedded-weight HLSL shader")
    view.add_argument("--viewer", type=Path, help="Path to a custom viewer build")
    args = parser.parse_args(argv)
    if args.command == "view":
        export = args.run.resolve() / "export"
        if not (export / "metadata.json").is_file():
            parser.error(f"Expected a finished export in {export}")
        binary = "vk_neural_demo.exe" if os.name == "nt" else "vk_neural_demo"
        candidates = [args.viewer.resolve()] if args.viewer else [ROOT / "build" / sub / binary for sub in ("Release", "")]
        exe = next((p for p in candidates if p.is_file()), None)
        if exe is None:
            parser.error("Build the Vulkan viewer first or pass --viewer PATH")
        flags = ["--final-view"]
        if args.hlsl:
            python("-m", "nnpbr.hlsl", export, "--spirv")
            flags.append("--embedded-hlsl")
        subprocess.run([exe, "--export-dir", export, *flags], cwd=ROOT, check=True)
        return

    if args.command == "check":
        run = args.run.resolve()
    else:
        small = args.command == "smoke"
        run = (args.out or ROOT / "runs" / args.material / args.command).resolve()
        if run.exists():
            parser.error(f"{run} already exists. Choose a new directory with --out.")
        if not small:
            import torch
            if not torch.cuda.is_available():
                parser.error("GPU training requires a GPU-enabled PyTorch installation.")
        run.mkdir(parents=True)
        levels = "7" if small else "9"
        python("-m", "nnpbr.prepare", "--maps", ROOT / "data/freepbr/materials" / args.material / "maps",
               "--output", run / "reference.pt", "--size", "64" if small else "2048", "--levels", levels, cpu=True)
        options = (["--latent-res", "32,16,8,4", "--latent-mips", "4,3,2,1",
                    "--phase1-iters", "150", "--phase2-iters", "400", "--phase3-iters", "100",
                    "--batch-size", "512", "--microbatch-size", "256"] if small else
                   ["--microbatch-size", "262144", "--log-every", "100"])
        python("-m", "nnpbr", "--mode", "full", "--reference-pt", run / "reference.pt",
               "--output-dir", run, "--device", "cpu" if small else "cuda:0",
               "--ref-mips", levels, *options, cpu=small)
    if not (run / "reference.pt").is_file() or not (run / "export/metadata.json").is_file():
        parser.error(f"Expected reference.pt and export/metadata.json in {run}")
    python("-m", "nnpbr.evaluation", "--export-dir", run / "export", "--reference", run / "reference.pt",
           "--output", run / "mip_metrics.json", cpu=True)
    print(f"Per-mip scores: {run / 'mip_metrics.json'}")


if __name__ == "__main__":
    main()

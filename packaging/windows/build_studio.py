"""Build a portable Windows directory from the active NN-PBR Python environment."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP_CACHE = shutil.ignore_patterns("__pycache__", "*.pyc")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("LOCALAPPDATA", ROOT / "build"))
        / "NN-PBR Studio/App",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    build = args.build.resolve()
    if sys.platform != "win32" or Path(sys.prefix) == Path(sys.base_prefix):
        parser.error(
            "Build with the Windows virtual environment containing PyTorch and PySide6."
        )
    if output.exists():
        parser.error("Choose a fresh output folder.")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch, PySide6.QtWidgets, numpy, PIL, matplotlib",
        ],
        check=True,
    )
    subprocess.run(["cmake", "-S", ROOT, "-B", build], check=True)
    subprocess.run(
        [
            "cmake",
            "--build",
            build,
            "--config",
            "Release",
            "--target",
            "nnpbr_studio_launcher",
            "vk_neural_demo",
        ],
        check=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        app = Path(tmp) / "application"
        runtime = app / "runtime"
        base = Path(sys.base_prefix)
        runtime.mkdir(parents=True)
        for file in base.iterdir():
            if file.is_file() and (
                file.suffix.lower() in (".exe", ".dll")
                or file.name.startswith("LICENSE")
            ):
                shutil.copy2(file, runtime)
        for name in ("DLLs", "Lib", "include", "libs", "tcl"):
            if (base / name).is_dir():
                shutil.copytree(
                    base / name,
                    runtime / name,
                    ignore=shutil.ignore_patterns(
                        "site-packages", "__pycache__", "*.pyc"
                    ),
                )
        packages = Path(sysconfig.get_path("purelib"))
        # ROCm's generated kernel filenames can exceed legacy Windows path limits.
        shutil.copytree(
            packages,
            Path("\\\\?\\" + str(runtime / "Lib/site-packages")),
            ignore=SKIP_CACHE,
        )
        shutil.copytree(ROOT / "nnpbr", app / "nnpbr", ignore=SKIP_CACHE)
        model = Path("models/hammered-copper-512")
        shutil.copytree(ROOT / model / "export", app / model / "export")
        for file in ("training_history.json", "quality_metrics.json"):
            shutil.copy2(ROOT / model / file, app / model / file)
        shutil.copy2(build / "Release/NN-PBR Studio.exe", app)
        (app / "build/Release").mkdir(parents=True)
        shutil.copy2(build / "Release/vk_neural_demo.exe", app / "build/Release")
        (app / "shaders").mkdir()
        for file in (build / "shaders").glob("*.spv"):
            shutil.copy2(file, app / "shaders")
        info = {
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "working_tree_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=ROOT, text=True
                ).strip()
            ),
        }
        (app / "build-info.json").write_text(json.dumps(info), encoding="utf-8")
        app.rename(output)
    print(output / "NN-PBR Studio.exe")


if __name__ == "__main__":
    main()

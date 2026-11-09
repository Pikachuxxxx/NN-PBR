"""Install a per-user Linux desktop entry for Studio in this checkout and environment."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def desktop_arg(value):
    # Exec is parsed by the desktop-entry spec, not by a shell.
    value = str(value).replace("%", "%%")
    for ch in ("\\", '"', "$", "`"):
        value = value.replace(ch, "\\" + ch)
    return '"' + value.replace("\\", "\\\\") + '"'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("Install this launcher from a native Linux Python environment.")
    if any(c in str(ROOT) + sys.executable + str(args.build.resolve()) for c in "\n\r"):
        parser.error("Installation paths must fit on one line.")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import PySide6.QtWidgets, torch, numpy, matplotlib, PIL",
        ],
        check=True,
    )
    build = args.build.resolve()
    subprocess.run(
        ["cmake", "-S", ROOT, "-B", build, "-DCMAKE_BUILD_TYPE=Release"], check=True
    )
    subprocess.run(
        ["cmake", "--build", build, "--target", "vk_neural_demo", "--parallel"],
        check=True,
    )
    data_home = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
    entry = data_home / "applications/nn-pbr-studio.desktop"
    entry.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "nnpbr.studio",
        "--viewer",
        build / "vk_neural_demo",
        "--export-dir",
        ROOT / "models/hammered-copper-512/export",
    ]
    text = "\n".join(
        (
            "[Desktop Entry]",
            "Type=Application",
            "Name=NN-PBR Studio",
            "Terminal=false",
            "Categories=Graphics;3DGraphics;",
            "Path=" + str(ROOT).replace("\\", "\\\\"),
            "Exec=" + " ".join(map(desktop_arg, command)),
            "",
        )
    )
    entry.write_text(text, encoding="utf-8")
    print(entry)


if __name__ == "__main__":
    main()

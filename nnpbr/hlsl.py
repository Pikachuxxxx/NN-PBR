"""Generate an engine include and optionally the Vulkan demo fragment shader."""

import argparse
import math
from pathlib import Path
import struct
import subprocess
from tempfile import TemporaryDirectory


TEMPLATE = Path(__file__).resolve().parents[1] / "shaders/neural_material_decode.hlsl"


def write_hlsl(directory):
    values = struct.unpack("<344e", (directory / "decoder_fp16.bin").read_bytes())
    if not all(map(math.isfinite, values)):
        raise ValueError("Decoder contains a nonfinite value")
    rows = [", ".join(repr(v) for v in values[i:i + 8]) for i in range(0, 344, 8)]
    source = TEMPLATE.read_text().replace("/* NNPBR_WEIGHTS */", ",\n    ".join(rows))
    target = directory / "material.hlsl"
    target.write_text(source, encoding="utf-8")
    return target


def compile_viewer_shader(directory):
    source = write_hlsl(directory)
    target = directory / "material.frag.spv"
    with TemporaryDirectory(dir=directory) as tmp:
        binary = Path(tmp) / target.name
        subprocess.run(["glslc", "-x", "hlsl", "-fshader-stage=frag", "-DNNPBR_VIEWER",
                        "-fauto-combined-image-sampler", "-O", str(source), "-o", str(binary)],
                       check=True)
        binary.replace(target)
    return target


if __name__ == "__main__":
    from .artifacts import ExportedMaterial

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("--spirv", action="store_true", help="Compile the viewer shader with glslc")
    args = parser.parse_args()
    ExportedMaterial(args.export)
    print(compile_viewer_shader(args.export) if args.spirv else write_hlsl(args.export))

"""Validate an existing exact Mode 10 export; training already writes DDS files."""

import argparse
from pathlib import Path
from nnpbr.artifacts import ExportedMaterial

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True, type=Path)
    args = parser.parse_args()
    result = ExportedMaterial(args.export_dir)
    print(
        f"Validated four DDS pyramids, {sum(map(len, result.mips))} mips and FP16 decoder"
    )

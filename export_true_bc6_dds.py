#!/usr/bin/env python3
"""
BC6H DDS export — integrated into export_trained_artifacts() since layout v4.

As of export layout v4, latent_XX.bc6.dds files are written directly during
training export (neuralmaterials.export_trained_artifacts) with no float32
roundtrip.  This script is now an informational stub.

For new exports (v4+), check the export root for DDS files:
  <run>/export/latent_00.bc6.dds
  <run>/export/latent_01.bc6.dds
  ...

If you have a legacy v3 export (with latent_XX_mip_YY.pt files), retrain and
re-export with the current code to get native v4 DDS output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export-dir", type=Path, required=True)
    args = ap.parse_args()

    export_dir = args.export_dir.resolve()
    meta_path = export_dir / "metadata.json"
    if not meta_path.exists():
        raise RuntimeError(f"metadata.json not found: {meta_path}")
    meta = json.loads(meta_path.read_text())

    version = int(meta.get("version", 1))
    dds_files = sorted(export_dir.glob("latent_*.bc6.dds"))

    if version >= 4:
        print(f"[bc6h] Export v{version} — DDS already written by export_trained_artifacts().")
    else:
        print(f"[bc6h] Legacy export v{version} detected. Re-export with current code to get v4 DDS.")

    if dds_files:
        total = sum(f.stat().st_size for f in dds_files)
        for f in dds_files:
            print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
        print(f"  Total: {total // 1024} KB")
    else:
        print("  No DDS files found in export root.")


if __name__ == "__main__":
    main()

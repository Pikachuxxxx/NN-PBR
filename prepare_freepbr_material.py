#!/usr/bin/env python3
"""
Download one FreePBR material, extract maps, and build an 8-channel reference tensor:
  [albedo_rgb, normal_x, normal_y, ao, roughness, metallic]
- Albedo: stays in [0, 1] (physical reflectance)
- Normal XY: normalized to [-1, 1] (signed)
- AO/Roughness/Metallic: normalized to [-1, 1] for training
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}


def _fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _parse_download_forms(page_html: str, fallback_url: str) -> List[Dict[str, str]]:
    forms: List[Dict[str, str]] = []
    form_blocks = re.findall(
        r"(<form[^>]*class=\"[^\"]*somdn-download-form[^\"]*\"[^>]*>.*?</form>)",
        page_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for block in form_blocks:
        block = html.unescape(block)
        action_m = re.search(r"<form[^>]*action=\"([^\"]+)\"", block, flags=re.IGNORECASE)
        action = action_m.group(1).strip() if action_m else fallback_url

        fields: Dict[str, str] = {}
        for name, value in re.findall(
            r"<input[^>]*name=\"([^\"]+)\"[^>]*value=\"([^\"]*)\"[^>]*>",
            block,
            flags=re.IGNORECASE,
        ):
            fields[name] = value

        name_m = re.search(
            r"<a[^>]*class=\"[^\"]*somdn-download-link[^\"]*\"[^>]*>([^<]+)</a>",
            block,
            flags=re.IGNORECASE,
        )
        if name_m:
            fields["download_name"] = name_m.group(1).strip()
        fields["action_url"] = action
        if {"somdn_download_key", "somdn_product", "somdn_productfile", "action"}.issubset(fields.keys()):
            forms.append(fields)
    return forms


def _download_zip(form: Dict[str, str], out_zip: Path):
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "somdn_download_key": form["somdn_download_key"],
        "action": form["action"],
        "somdn_product": form["somdn_product"],
        "somdn_productfile": form["somdn_productfile"],
    }
    body = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(
        form["action_url"],
        data=body,
        method="POST",
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": form["action_url"],
        },
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read()
    out_zip.write_bytes(data)


def _find_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]


def _image_area(path: Path) -> int:
    try:
        with Image.open(path) as im:
            w, h = im.size
            return int(w * h)
    except Exception:
        return -1


def _pick_map(candidates: List[Path], keys: List[str], reject: Optional[List[str]] = None) -> Optional[Path]:
    reject = reject or []
    hit: List[Path] = []
    for p in candidates:
        stem = p.stem.lower()
        if any(r in stem for r in reject):
            continue
        if all(k in stem for k in keys):
            hit.append(p)
    if not hit:
        return None
    hit.sort(key=lambda x: (_image_area(x), len(x.name)), reverse=True)
    return hit[0]


def _pick_first_any(candidates: List[Path], key_options: List[List[str]], reject: Optional[List[str]] = None) -> Optional[Path]:
    for keys in key_options:
        m = _pick_map(candidates, keys=keys, reject=reject)
        if m is not None:
            return m
    return None


def _load_rgb(path: Path, size: Optional[int]) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size is not None and size > 0:
            im = im.resize((size, size), Image.Resampling.LANCZOS)
        t = torch.from_numpy(np.array(im, dtype="float32") / 255.0)
    return t.permute(2, 0, 1).contiguous()  # [3,H,W]


def _load_gray(path: Path, size: Optional[int]) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("L")
        if size is not None and size > 0:
            im = im.resize((size, size), Image.Resampling.LANCZOS)
        t = torch.from_numpy(np.array(im, dtype="float32") / 255.0)
    return t.unsqueeze(0).contiguous()  # [1,H,W]


def _save_rgb01(t: torch.Tensor, out_path: Path):
    arr = (t.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype("uint8")
    Image.fromarray(arr, mode="RGB").save(out_path)


def _normal_xy_to_rgb(nx: torch.Tensor, ny: torch.Tensor) -> torch.Tensor:
    nz = torch.sqrt(torch.clamp(1.0 - nx * nx - ny * ny, min=0.0))
    return torch.stack([(nx + 1.0) * 0.5, (ny + 1.0) * 0.5, nz], dim=0)


def _build_reference_tensor(
    albedo_path: Path,
    normal_path: Path,
    ao_path: Path,
    roughness_path: Path,
    metallic_path: Path,
    size: Optional[int],
) -> Tuple[torch.Tensor, Dict[str, str], bool]:
    albedo = _load_rgb(albedo_path, size=size)
    normal = _load_rgb(normal_path, size=size)
    ao = _load_gray(ao_path, size=size)
    rough = _load_gray(roughness_path, size=size)
    metal = _load_gray(metallic_path, size=size)

    normal_name = normal_path.stem.lower()
    is_dx = ("dx" in normal_name) or ("directx" in normal_name)
    nx = normal[0] * 2.0 - 1.0
    ny = normal[1] * 2.0 - 1.0
    if is_dx:
        ny = -ny

    base = torch.cat(
        [
            albedo,  # Keep albedo in [0, 1] (physical reflectance)
            nx.unsqueeze(0),
            ny.unsqueeze(0),
            ao * 2.0 - 1.0,  # Normalize AO to [-1, 1]
            rough * 2.0 - 1.0,  # Normalize roughness to [-1, 1]
            metal * 2.0 - 1.0,  # Normalize metallic to [-1, 1]
        ],
        dim=0,
    )
    paths = {
        "albedo": str(albedo_path),
        "normal": str(normal_path),
        "ao": str(ao_path),
        "roughness": str(roughness_path),
        "metallic": str(metallic_path),
    }
    return base, paths, is_dx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product-url", required=True, help="FreePBR product page URL")
    ap.add_argument("--variant-keyword", default="-bl.zip", help="Preferred zip name keyword (e.g. -bl.zip, -ue.zip, -unity.zip)")
    ap.add_argument("--size", type=int, default=1024, help="Square size for all maps; <=0 keeps source size")
    ap.add_argument("--material-name", default=None, help="Optional material output folder name")
    ap.add_argument("--out-root", type=Path, default=Path("data/freepbr/materials"))
    args = ap.parse_args()

    page_html = _fetch_text(args.product_url)
    forms = _parse_download_forms(page_html, fallback_url=args.product_url)
    if not forms:
        raise RuntimeError("No downloadable forms found on product page.")

    chosen = None
    kw = args.variant_keyword.lower()
    for f in forms:
        if kw in f.get("download_name", "").lower():
            chosen = f
            break
    if chosen is None:
        chosen = forms[0]

    zip_name = chosen.get("download_name", "material.zip")
    material_name = args.material_name or Path(zip_name).stem
    mat_dir = args.out_root / material_name
    raw_dir = mat_dir / "raw"
    extracted_dir = mat_dir / "extracted"
    maps_dir = mat_dir / "maps"
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / zip_name
    print(f"[download] {zip_name}")
    _download_zip(chosen, zip_path)
    if zip_path.stat().st_size < 1024:
        raise RuntimeError(f"Downloaded file too small: {zip_path}")

    print(f"[extract] {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extracted_dir)

    candidates = _find_images(extracted_dir)
    if not candidates:
        raise RuntimeError("No image maps found in extracted zip.")

    reject_preview = ["preview", "thumb", "thumbnail"]
    albedo = _pick_first_any(candidates, [["albedo"], ["basecolor"], ["base_color"], ["diffuse"], ["color"]], reject=reject_preview)
    normal = _pick_first_any(
        candidates,
        [["normal", "ogl"], ["normal", "gl"], ["normal"], ["nrm"]],
        reject=reject_preview,
    )
    ao = _pick_first_any(candidates, [["ao"], ["ambient", "occlusion"], ["occlusion"]], reject=reject_preview)
    rough = _pick_first_any(candidates, [["roughness"], ["rough"]], reject=reject_preview)
    metal = _pick_first_any(candidates, [["metallic"], ["metalness"], ["metal"]], reject=reject_preview)

    missing = [k for k, v in [("albedo", albedo), ("normal", normal), ("ao", ao), ("roughness", rough), ("metallic", metal)] if v is None]
    if missing:
        raise RuntimeError(f"Missing required maps: {missing}")

    base, paths, used_dx = _build_reference_tensor(
        albedo_path=albedo,
        normal_path=normal,
        ao_path=ao,
        roughness_path=rough,
        metallic_path=metal,
        size=(args.size if args.size > 0 else None),
    )

    ref_pt = mat_dir / "reference_8ch.pt"
    torch.save(
        {
            "base": base,
            "meta": {
                "product_url": args.product_url,
                "download_name": zip_name,
                "variant_keyword": args.variant_keyword,
                "used_directx_normal_y_flip": used_dx,
                "map_paths": paths,
                "channels": ["albedo_r", "albedo_g", "albedo_b", "normal_x", "normal_y", "ao", "roughness", "metallic"],
            },
        },
        ref_pt,
    )

    # Save normalized map previews for quick inspection.
    albedo_preview = (base[0:3] + 1.0) * 0.5
    nx = base[3]
    ny = base[4]
    normal_preview = _normal_xy_to_rgb(nx, ny)
    orm_preview = torch.stack([(base[5] + 1.0) * 0.5, (base[6] + 1.0) * 0.5, (base[7] + 1.0) * 0.5], dim=0)
    _save_rgb01(albedo_preview, maps_dir / "albedo_preview.png")
    _save_rgb01(normal_preview, maps_dir / "normal_preview.png")
    _save_rgb01(orm_preview, maps_dir / "orm_preview.png")

    # Keep original selected maps for provenance.
    for key, src in paths.items():
        src_path = Path(src)
        dst = maps_dir / f"{key}{src_path.suffix.lower()}"
        shutil.copy2(src_path, dst)

    report = {
        "material_name": material_name,
        "reference_pt": str(ref_pt),
        "download_zip": str(zip_path),
        "maps": paths,
        "size_chw": list(base.shape),
        "variant_used": zip_name,
    }
    (mat_dir / "dataset_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

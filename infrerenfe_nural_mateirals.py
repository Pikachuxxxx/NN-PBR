#!/usr/bin/env python3
"""
Unified runner for neural material compression.

Modes:
- train: train + export only (for long runs on another PC)
- full: train + export + compare plots
- infer: load exported latents + decoder and produce PBR outputs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from neuralmaterials import (
    NeuralMaterialCompressionModel,
    TrainConfig,
    export_trained_artifacts,
    load_reference_mips,
    make_partition_bank,
    random_uv_lod,
    sample_mips_trilinear,
    train,
)


def detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    # grid_sample backward is currently unsupported on MPS in this pipeline.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "cpu"
    return "cpu"


def parse_list_int(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _bcn_bytes(w: int, h: int, block_bytes: int, include_mips: bool) -> int:
    total = 0
    cw, ch = int(w), int(h)
    while True:
        bw = (cw + 3) // 4
        bh = (ch + 3) // 4
        total += bw * bh * block_bytes
        if not include_mips or (cw == 1 and ch == 1):
            break
        cw = max(1, cw // 2)
        ch = max(1, ch // 2)
    return total


def _collect_neural_storage_bytes(export_dir: Path) -> Dict[str, int]:
    lat_dir = export_dir / "latents"
    block_files = sorted(lat_dir.glob("*.blocks128.bin"))
    block_bytes = sum(p.stat().st_size for p in block_files)
    decoder_bytes = (export_dir / "decoder_fp16.bin").stat().st_size
    total = block_bytes + decoder_bytes
    return {
        "blocks128_bytes": int(block_bytes),
        "decoder_fp16_bytes": int(decoder_bytes),
        "runtime_total_bytes": int(total),
    }


def _collect_source_map_bytes(reference_pt: Path) -> int:
    try:
        obj = torch.load(reference_pt, map_location="cpu")
    except Exception:
        return 0
    meta = obj.get("meta", {})
    map_paths = meta.get("map_paths", {})
    total = 0
    for p in map_paths.values():
        path = Path(p)
        if path.exists():
            total += int(path.stat().st_size)
    return total


def _save_rgb01(rgb: np.ndarray, out_path: Path):
    x = np.clip(rgb, 0.0, 1.0)
    img = (x * 255.0).round().astype(np.uint8)
    Image.fromarray(img, mode="RGB").save(out_path)


def _to_rgb_chw(t: torch.Tensor) -> np.ndarray:
    return ((t[:3].clamp(-1.0, 1.0) + 1.0) * 0.5).permute(1, 2, 0).cpu().numpy()


def _normal_xy_to_rgb(t: torch.Tensor) -> np.ndarray:
    nx = t[3].clamp(-1.0, 1.0)
    ny = t[4].clamp(-1.0, 1.0)
    nz = torch.sqrt(torch.clamp(1.0 - nx * nx - ny * ny, min=0.0))
    rgb = torch.stack([(nx + 1.0) * 0.5, (ny + 1.0) * 0.5, nz], dim=0)
    return rgb.permute(1, 2, 0).cpu().numpy()


def _pbr_views(pred: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    albedo = _to_rgb_chw(pred)
    normal = _normal_xy_to_rgb(pred)
    orm = torch.stack(
        [(pred[5] + 1.0) * 0.5, (pred[6] + 1.0) * 0.5, (pred[7] + 1.0) * 0.5],
        dim=0,
    ).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
    return albedo, normal, orm


def _quality_metrics(ref_base: torch.Tensor, pred_base: torch.Tensor) -> Dict[str, float]:
    ref = ref_base.float()
    pred = pred_base.float()
    d = pred - ref
    return {
        "mse_all": float(torch.mean(d * d).item()),
        "mse_albedo": float(torch.mean((d[0:3]) ** 2).item()),
        "mse_normal_xy": float(torch.mean((d[3:5]) ** 2).item()),
        "mse_orm": float(torch.mean((d[5:8]) ** 2).item()),
        "mae_all": float(torch.mean(torch.abs(d)).item()),
    }


@torch.no_grad()
def _eval_random_batch_metrics(
    model: NeuralMaterialCompressionModel,
    ref_mips: List[torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    if batch_size <= 0:
        return {}
    max_lod = float(len(ref_mips) - 1)
    uv, lod = random_uv_lod(batch_size, max_lod=max_lod, device=device)
    target = sample_mips_trilinear(ref_mips, uv, lod, bilinear_mode="bicubic")
    pred = model.forward_bc(uv, lod)
    d = pred - target
    return {
        "batch_size": int(batch_size),
        "mse_all": float(torch.mean(d * d).item()),
        "mse_albedo": float(torch.mean((d[:, 0:3]) ** 2).item()),
        "mse_normal_xy": float(torch.mean((d[:, 3:5]) ** 2).item()),
        "mse_orm": float(torch.mean((d[:, 5:8]) ** 2).item()),
    }


@torch.no_grad()
def _render_mip0_from_model(
    model: NeuralMaterialCompressionModel,
    h: int,
    w: int,
    out_channels: int,
    device: torch.device,
    chunk: int = 65536,
) -> torch.Tensor:
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    uv = torch.stack(
        [(xs.to(torch.float32) + 0.5) / float(w), (ys.to(torch.float32) + 0.5) / float(h)],
        dim=-1,
    ).view(-1, 2)
    lod = torch.zeros((uv.shape[0],), dtype=torch.float32)

    out_parts = []
    for i in range(0, uv.shape[0], chunk):
        u = uv[i : i + chunk].to(device)
        l = lod[i : i + chunk].to(device)
        p = model.forward_bc(u, l)
        out_parts.append(p.detach().cpu())
    return torch.cat(out_parts, dim=0).view(h, w, out_channels).permute(2, 0, 1).contiguous()


def _infer_output_resolution(meta: Dict, latents: List[torch.Tensor], infer_size: str) -> Tuple[int, int]:
    if infer_size and infer_size.lower() != "auto":
        if "x" in infer_size.lower():
            a, b = infer_size.lower().split("x")
            return int(a), int(b)
        s = int(infer_size)
        return s, s

    latent_resolutions = meta.get("latent_resolutions", [])
    lod_biases = meta.get("lod_biases", [])
    if latent_resolutions and lod_biases and len(latent_resolutions) == len(lod_biases):
        refs = []
        for r, b in zip(latent_resolutions, lod_biases):
            try:
                refs.append(int(round(float(r) / (2.0 ** float(b)))))
            except Exception:
                pass
        if refs:
            s = max(refs)
            return s, s

    # Fallback: largest latent resolution.
    h = max(int(t.shape[1]) for t in latents)
    w = max(int(t.shape[2]) for t in latents)
    return h, w


@torch.no_grad()
def _render_mip0_from_export(
    export_dir: Path,
    device: torch.device,
    chunk: int = 65536,
    infer_size: str = "auto",
) -> torch.Tensor:
    meta = json.loads((export_dir / "metadata.json").read_text())
    lat_dir = export_dir / "latents"

    latent_count = int(meta["latent_count"])
    latents = []
    for i in range(latent_count):
        p = lat_dir / f"latent_{i:02d}_mip_00.pt"
        if not p.exists():
            raise RuntimeError(f"Missing mip0 latent tensor: {p}")
        latents.append(torch.load(p, map_location="cpu").float())
    out_h, out_w = _infer_output_resolution(meta, latents, infer_size=infer_size)
    upsampled = [
        F.interpolate(t.unsqueeze(0), size=(out_h, out_w), mode="bilinear", align_corners=False).squeeze(0)
        for t in latents
    ]

    x = torch.cat(upsampled, dim=0).permute(1, 2, 0).contiguous().view(-1, latent_count * 3)
    state = torch.load(export_dir / "decoder_state.pt", map_location="cpu")

    fc1_w = state["fc1.weight"].to(device)
    fc1_b = state["fc1.bias"].to(device)
    fc2_w = state["fc2.weight"].to(device)
    fc2_b = state["fc2.bias"].to(device)
    out_dim = int(fc2_b.shape[0])

    y_parts = []
    for i in range(0, x.shape[0], chunk):
        xi = x[i : i + chunk].to(device)
        hi = torch.relu(xi @ fc1_w.t() + fc1_b)
        yi = hi @ fc2_w.t() + fc2_b
        y_parts.append(yi.detach().cpu())
    y = torch.cat(y_parts, dim=0).view(out_h, out_w, out_dim).permute(2, 0, 1).contiguous()
    return y


def _save_inference_maps(pred: torch.Tensor, out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    albedo, normal, orm = _pbr_views(pred)

    albedo_p = out_dir / "albedo.png"
    normal_p = out_dir / "normal.png"
    orm_p = out_dir / "orm.png"
    _save_rgb01(albedo, albedo_p)
    _save_rgb01(normal, normal_p)
    _save_rgb01(orm, orm_p)

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(albedo)
    ax.set_title("Albedo")
    ax.axis("off")
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(normal)
    ax.set_title("Normal")
    ax.axis("off")
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(orm)
    ax.set_title("ORM")
    ax.axis("off")
    fig.tight_layout()
    preview_p = out_dir / "pbr_preview.png"
    fig.savefig(preview_p, dpi=150)
    plt.close(fig)

    return {
        "albedo": str(albedo_p),
        "normal": str(normal_p),
        "orm": str(orm_p),
        "preview": str(preview_p),
    }


def _save_full_plots(
    ref_base: torch.Tensor,
    pred_base: torch.Tensor,
    loss_history: List[dict],
    compare_dir: Path,
    baseline_bytes: int,
    neural_bytes: int,
) -> Dict[str, str]:
    compare_dir.mkdir(parents=True, exist_ok=True)

    ref_albedo = _to_rgb_chw(ref_base)
    pred_albedo = _to_rgb_chw(pred_base)
    ref_normal = _normal_xy_to_rgb(ref_base)
    pred_normal = _normal_xy_to_rgb(pred_base)
    ref_orm = torch.stack(
        [(ref_base[5] + 1.0) * 0.5, (ref_base[6] + 1.0) * 0.5, (ref_base[7] + 1.0) * 0.5],
        dim=0,
    ).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
    pred_orm = torch.stack(
        [(pred_base[5] + 1.0) * 0.5, (pred_base[6] + 1.0) * 0.5, (pred_base[7] + 1.0) * 0.5],
        dim=0,
    ).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()

    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(ref_albedo)
    ax.set_title("GT Albedo")
    ax.axis("off")
    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(ref_normal)
    ax.set_title("GT Normal")
    ax.axis("off")
    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(ref_orm)
    ax.set_title("GT ORM")
    ax.axis("off")
    ax = fig.add_subplot(2, 3, 4)
    ax.imshow(pred_albedo)
    ax.set_title("Neural Albedo")
    ax.axis("off")
    ax = fig.add_subplot(2, 3, 5)
    ax.imshow(pred_normal)
    ax.set_title("Neural Normal")
    ax.axis("off")
    ax = fig.add_subplot(2, 3, 6)
    ax.imshow(pred_orm)
    ax.set_title("Neural ORM")
    ax.axis("off")
    fig.tight_layout()
    gt_vs_neural = compare_dir / "gt_vs_neural.png"
    fig.savefig(gt_vs_neural, dpi=150)
    plt.close(fig)

    diff_albedo = np.abs(ref_albedo - pred_albedo).mean(axis=2)
    diff_normal = np.abs(ref_normal - pred_normal).mean(axis=2)
    diff_orm = np.abs(ref_orm - pred_orm).mean(axis=2)

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(diff_albedo, cmap="inferno")
    ax.set_title("Abs Diff Albedo")
    ax.axis("off")
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(diff_normal, cmap="inferno")
    ax.set_title("Abs Diff Normal")
    ax.axis("off")
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(diff_orm, cmap="inferno")
    ax.set_title("Abs Diff ORM")
    ax.axis("off")
    fig.tight_layout()
    diff_maps = compare_dir / "gt_vs_neural_diff.png"
    fig.savefig(diff_maps, dpi=150)
    plt.close(fig)

    p1_x = [h["iter"] for h in loss_history if h["phase"] == 1]
    p1_y = [h["mse"] for h in loss_history if h["phase"] == 1]
    p2_x = [h["iter"] for h in loss_history if h["phase"] == 2]
    p2_y = [h["mse"] for h in loss_history if h["phase"] == 2]
    p3_x = [h["iter"] for h in loss_history if h["phase"] == 3]
    p3_y = [h["mse"] for h in loss_history if h["phase"] == 3]

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    if p1_x:
        ax.plot(p1_x, p1_y, label="phase1")
    if p2_x:
        ax.plot(p2_x, p2_y, label="phase2")
    if p3_x:
        ax.plot(p3_x, p3_y, label="phase3")
    ax.set_xlabel("iteration")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    training_loss = compare_dir / "training_loss.png"
    fig.savefig(training_loss, dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    names = ["PBR BCn", "Neural"]
    vals_mb = [baseline_bytes / (1024.0 * 1024.0), neural_bytes / (1024.0 * 1024.0)]
    bars = ax.bar(names, vals_mb, color=["#c96b2c", "#3c77a8"])
    ax.set_ylabel("MB")
    ax.set_title("Runtime Storage")
    ax.grid(True, axis="y", alpha=0.2)
    for b, v in zip(bars, vals_mb):
        ax.text(b.get_x() + b.get_width() * 0.5, b.get_height(), f"{v:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    cost_savings = compare_dir / "cost_savings.png"
    fig.savefig(cost_savings, dpi=150)
    plt.close(fig)

    return {
        "gt_vs_neural": str(gt_vs_neural),
        "gt_vs_neural_diff": str(diff_maps),
        "training_loss": str(training_loss),
        "cost_savings": str(cost_savings),
    }


def _run_true_bc6_export(export_dir: Path, bc6_cli: str | None, bc6_format: str | None):
    cmd = [
        sys.executable,
        str((Path(__file__).parent / "export_true_bc6_dds.py").resolve()),
        "--export-dir",
        str(export_dir),
        "--decode-smoke",
    ]
    if bc6_cli:
        cmd.extend(["--bc6-cli", bc6_cli])
    if bc6_format:
        cmd.extend(["--bc6-format", bc6_format])
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "full", "infer"], default="full")
    ap.add_argument("--reference-pt", type=Path, default=None, help="Required for train/full")
    ap.add_argument("--output-dir", type=Path, default=None, help="train/full output root; infer output dir for maps")
    ap.add_argument("--export-dir", type=Path, default=None, help="Use existing export dir in infer mode")

    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-channels", type=int, default=8)
    ap.add_argument("--ref-mips", type=int, default=9)
    ap.add_argument("--latent-res", type=str, default="512,256,128,64")
    ap.add_argument("--latent-mips", type=str, default="8,7,6,5")
    ap.add_argument("--hidden-dim", type=int, default=16)
    ap.add_argument("--endpoint-bits", type=int, default=6)
    ap.add_argument("--index-bits", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--phase1-iters", type=int, default=5000)
    ap.add_argument("--phase2-iters", type=int, default=200000)
    ap.add_argument("--phase3-iters", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--interactive-progress", action="store_true", help="Enable tqdm-like live progress bars during training.")
    ap.add_argument("--infer-chunk", type=int, default=65536)
    ap.add_argument("--infer-size", type=str, default="auto", help="'auto', '1024', or '1024x1024'")
    ap.add_argument("--analysis-batch-size", type=int, default=131072, help="Extra random UV/LOD batch size for analysis metrics in full mode.")

    ap.add_argument("--export-true-bc6", action="store_true")
    ap.add_argument("--bc6-cli", type=str, default=None)
    ap.add_argument("--bc6-format", type=str, default=None)
    args = ap.parse_args()

    mode = args.mode
    device_str = detect_device(args.device)
    device = torch.device(device_str)

    if mode in ("train", "full") and args.reference_pt is None:
        raise ValueError("--reference-pt is required for train/full mode.")
    if mode in ("train", "full") and args.output_dir is None:
        raise ValueError("--output-dir is required for train/full mode.")
    if mode == "infer" and args.export_dir is None:
        raise ValueError("--export-dir is required for infer mode.")

    if mode in ("train", "full"):
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        export_dir = args.export_dir or (output_dir / "export")

        latent_res = parse_list_int(args.latent_res)
        latent_mips = parse_list_int(args.latent_mips)
        if len(latent_res) != len(latent_mips):
            raise ValueError("--latent-res and --latent-mips length mismatch")

        ref_mips = load_reference_mips(args.reference_pt, args.ref_mips, args.out_channels, device=device)
        ref_base = ref_mips[0].detach().cpu()
        h, w = int(ref_base.shape[1]), int(ref_base.shape[2])

        model = NeuralMaterialCompressionModel(
            latent_resolutions=latent_res,
            latent_mips=latent_mips,
            out_channels=args.out_channels,
            hidden_dim=args.hidden_dim,
            partition_bank=make_partition_bank(device=device),
            endpoint_bits=args.endpoint_bits,
            index_bits=args.index_bits,
            ref_base_res=h,
            bc6_signed_mode=False,
        ).to(device)

        cfg = TrainConfig(
            device=device_str,
            batch_size=args.batch_size,
            phase1_iters=args.phase1_iters,
            phase2_iters=args.phase2_iters,
            phase3_iters=args.phase3_iters,
            log_every=args.log_every,
            interactive_progress=args.interactive_progress,
        )
        print(f"[train] mode={mode} device={device_str} resolution={w}x{h}")
        history = train(model, ref_mips, cfg)
        (output_dir / "training_history.json").write_text(json.dumps(history, indent=2))

        print("[export] writing neural artifacts")
        export_trained_artifacts(model=model, out_dir=export_dir)

        if mode == "train":
            report = {
                "mode": "train",
                "reference_pt": str(args.reference_pt),
                "export_dir": str(export_dir),
                "device": device_str,
                "training_iters": {
                    "phase1": args.phase1_iters,
                    "phase2": args.phase2_iters,
                    "phase3": args.phase3_iters,
                },
            }
            (output_dir / "run_report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))
        else:
            print("[infer] rendering full-resolution mip0 from trained model")
            pred_base = _render_mip0_from_model(model, h=h, w=w, out_channels=args.out_channels, device=device, chunk=args.infer_chunk)
            infer_out = output_dir / "inference"
            infer_paths = _save_inference_maps(pred_base, infer_out)

            baseline_bytes = (
                _bcn_bytes(w, h, block_bytes=8, include_mips=True)
                + _bcn_bytes(w, h, block_bytes=16, include_mips=True)
                + _bcn_bytes(w, h, block_bytes=8, include_mips=True)
            )
            neural_storage = _collect_neural_storage_bytes(export_dir)
            neural_bytes = neural_storage["runtime_total_bytes"]
            savings_pct = (1.0 - float(neural_bytes) / float(baseline_bytes)) * 100.0

            analysis_paths = _save_full_plots(
                ref_base=ref_base,
                pred_base=pred_base,
                loss_history=history,
                compare_dir=output_dir / "analysis",
                baseline_bytes=baseline_bytes,
                neural_bytes=neural_bytes,
            )
            quality_metrics = _quality_metrics(ref_base=ref_base, pred_base=pred_base)
            batch_metrics = _eval_random_batch_metrics(
                model=model,
                ref_mips=ref_mips,
                batch_size=args.analysis_batch_size,
                device=device,
            )
            (output_dir / "analysis" / "quality_metrics.json").write_text(
                json.dumps(
                    {
                        "full_image": quality_metrics,
                        "random_batch": batch_metrics,
                    },
                    indent=2,
                )
            )

            report = {
                "mode": "full",
                "reference_pt": str(args.reference_pt),
                "export_dir": str(export_dir),
                "device": device_str,
                "training_iters": {
                    "phase1": args.phase1_iters,
                    "phase2": args.phase2_iters,
                    "phase3": args.phase3_iters,
                },
                "storage_bytes": {
                    "baseline_pbr_bcn": int(baseline_bytes),
                    "baseline_source_map_files_total_bytes": int(_collect_source_map_bytes(args.reference_pt)),
                    "neural_runtime_total": int(neural_bytes),
                    **neural_storage,
                },
                "savings_method": (
                    "Baseline BCn bytes are analytical runtime memory estimates "
                    "(BC1 albedo + BC5 normal + BC1 ORM with full mip chain), "
                    "not PNG source-file sizes."
                ),
                "savings_percent_vs_bcn": float(savings_pct),
                "quality_metrics": quality_metrics,
                "random_batch_metrics": batch_metrics,
                "inference_outputs": infer_paths,
                "analysis_outputs": analysis_paths,
            }
            (output_dir / "run_report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2))

    else:
        export_dir = args.export_dir
        infer_out = args.output_dir or (export_dir / "inference")
        print(f"[infer] loading exported latents+decoder from {export_dir}")
        pred = _render_mip0_from_export(
            export_dir=export_dir,
            device=device,
            chunk=args.infer_chunk,
            infer_size=args.infer_size,
        )
        infer_paths = _save_inference_maps(pred, infer_out)
        storage = _collect_neural_storage_bytes(export_dir)
        report = {
            "mode": "infer",
            "export_dir": str(export_dir),
            "device": device_str,
            "decoded_shape_chw": list(pred.shape),
            "storage_bytes": storage,
            "inference_outputs": infer_paths,
        }
        (infer_out / "run_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))

    if args.export_true_bc6:
        print("[export] true BC6 DDS")
        target_export_dir = args.export_dir if mode == "infer" else (args.export_dir or (args.output_dir / "export"))
        _run_true_bc6_export(target_export_dir, bc6_cli=args.bc6_cli, bc6_format=args.bc6_format)

if __name__ == "__main__":
    main()

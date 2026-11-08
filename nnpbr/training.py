"""Three explicit training stages, effective batches, and resumable checkpoints."""

from dataclasses import dataclass, asdict
from pathlib import Path
import time
import hashlib
import torch
import torch.nn.functional as F
from .sampling import random_uv_lod, sample_mips_trilinear


@dataclass
class TrainConfig:
    device: str = "cpu"
    batch_size: int = 262144  # samples per optimizer update, not per microbatch
    microbatch_size: int = 8192
    phase1_iters: int = 5000
    phase2_iters: int = 200000
    phase3_iters: int = 1000
    lr_feat_phase1: float = 0.05
    lr_mlp_phase1: float = 0.001
    gamma_phase1: float = 0.9995
    lr_feat_phase2: float = 0.01
    lr_mlp_phase2: float = 0.001
    gamma_phase2: float = 0.9999
    lr_mlp_phase3: float = 0.0005
    seed: int = 7
    shared_scale: bool = True
    compile_model: bool = False
    fused_adam: bool = False
    log_every: int = 100
    checkpoint_every: int = 1000
    checkpoint_path: str | None = None
    resume: str | None = None
    interactive_progress: bool = False


def _restore_model(model, state):
    # Frozen mip caches acquire their shape only at the quantization transition.
    for key, value in state.items():
        if key.endswith(".frozen_mip"):
            path = key.rsplit(".", 1)[0]
            model.get_submodule(path).frozen_mip = torch.empty_like(
                value, device=next(model.parameters()).device
            )
    model.load_state_dict(state)


def train(model, ref_mips, cfg: TrainConfig):
    if (
        min(cfg.batch_size, cfg.microbatch_size, cfg.log_every, cfg.checkpoint_every)
        < 1
    ):
        raise ValueError("Batch, logging, and checkpoint intervals must be positive")
    if min(cfg.phase1_iters, cfg.phase2_iters, cfg.phase3_iters) < 0:
        raise ValueError("Stage lengths cannot be negative")
    device = torch.device(cfg.device)
    compile_backend = "inductor"
    if cfg.compile_model and device.type == "cuda":
        from torch.utils._triton import has_triton

        if not has_triton():
            compile_backend = "aot_eager"
            print("[compile] Triton unavailable; using PyTorch AOT Autograd with eager GPU kernels.", flush=True)
        else:
            import triton

            print(f"[compile] Inductor with Triton {triton.__version__} on {torch.cuda.get_device_name(device)}", flush=True)
    max_lod = float(len(ref_mips) - 1)
    digest = hashlib.sha256()
    for mip in ref_mips:
        digest.update(mip.detach().cpu().contiguous().numpy().tobytes())
    reference_digest = digest.hexdigest()
    architecture = {
        "resolutions": model.latent_resolutions,
        "mips": [len(p.mips) for p in model.bc_pyramids],
        "reference_resolution": model.ref_base_res,
    }
    torch.manual_seed(cfg.seed + 1)
    history = []
    resume = None
    global_step = 0
    start = time.perf_counter()
    if cfg.resume:
        resume = torch.load(cfg.resume, map_location=device, weights_only=False)
        if (
            resume.get("reference_digest") != reference_digest
            or resume.get("architecture") != architecture
        ):
            raise ValueError(
                "Resume requires the same reference tensors and model architecture"
            )
        saved = resume["config"]
        current = asdict(cfg)
        mutable = {
            "resume",
            "checkpoint_path",
            "checkpoint_every",
            "log_every",
            "interactive_progress",
            "device",
        }
        if any(saved[k] != current[k] for k in current if k not in mutable):
            raise ValueError(
                "Resume configuration differs from checkpoint; use its original training configuration"
            )
        _restore_model(model, resume["model"])
        history = resume["history"]
        global_step = resume["global_step"]
        torch.set_rng_state(resume["rng"].cpu())
        if device.type == "cuda" and resume.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([r.cpu() for r in resume["cuda_rng"]])
    stages = [(1, cfg.phase1_iters), (2, cfg.phase2_iters), (3, cfg.phase3_iters)]
    for phase, iterations in stages:
        if resume and phase < resume["phase"]:
            continue
        same_phase = resume is not None and phase == resume["phase"]
        if phase == 2 and not same_phase:
            print(
                "[transition] Fit exact Mode 10 palettes and freeze partition IDs",
                flush=True,
            )
            model.initialize_bc_from_warmup()
            history.append({"phase": "bc_init", "global_step": global_step})
        if phase == 3:
            model.quantize_and_freeze_bc_features()
            model.decoder.quantized_forward = True
            history.append({"phase": "quantized", "global_step": global_step})
        if phase == 1:
            groups = [
                {"params": list(model.warmup_parameters()), "lr": cfg.lr_feat_phase1},
                {"params": list(model.decoder_parameters()), "lr": cfg.lr_mlp_phase1},
            ]
            gamma = cfg.gamma_phase1
            forward = model.forward_warmup
        elif phase == 2:
            groups = [
                {
                    "params": list(model.bc_feature_parameters()),
                    "lr": cfg.lr_feat_phase2,
                },
                {"params": list(model.decoder_parameters()), "lr": cfg.lr_mlp_phase2},
            ]
            gamma = cfg.gamma_phase2
            forward = model.forward_bc
        else:
            groups = [
                {"params": list(model.decoder_parameters()), "lr": cfg.lr_mlp_phase3}
            ]
            gamma = 1.0
            forward = model.forward_bc
        opt = torch.optim.Adam(groups, fused=(cfg.fused_adam and device.type == "cuda"))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
        if same_phase:
            opt.load_state_dict(resume["optimizer"])
            scheduler.load_state_dict(resume["scheduler"])
        begin = resume["completed"] if same_phase else 0
        if cfg.compile_model:
            forward = torch.compile(forward, backend=compile_backend, fullgraph=False)
        phase_start = time.perf_counter()
        for step in range(begin, iterations):
            opt.zero_grad(set_to_none=True)
            # One common continuous reference scale per effective batch (paper profile).
            uv, lod = random_uv_lod(cfg.batch_size, max_lod, device, cfg.shared_scale)
            total = torch.zeros((), device=device)
            for first in range(0, cfg.batch_size, cfg.microbatch_size):
                u = uv[first : first + cfg.microbatch_size]
                l = lod[first : first + cfg.microbatch_size]
                with torch.no_grad():
                    target = sample_mips_trilinear(ref_mips, u, l, "bicubic")
                prediction = forward(u, l)
                loss = F.mse_loss(prediction, target)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Nonfinite loss in phase {phase}, step {step}"
                    )
                fraction = len(u) / cfg.batch_size
                (loss * fraction).backward()
                total += loss.detach() * fraction
            torch.nn.utils.clip_grad_norm_(
                [p for group in groups for p in group["params"]],
                float("inf"),
                error_if_nonfinite=True,
            )
            opt.step()
            scheduler.step()
            global_step += 1
            if step % cfg.log_every == 0 or step + 1 == iterations:
                entry = {
                    "phase": phase,
                    "iter": step + 1,
                    "global_step": global_step,
                    "samples": global_step * cfg.batch_size,
                    "mse": float(total),
                    "lr": opt.param_groups[0]["lr"],
                    "elapsed_seconds": time.perf_counter() - start,
                }
                history.append(entry)
                message = f"[phase {phase} {step + 1}/{iterations}] mse={entry['mse']:.6g}"
                if cfg.interactive_progress:
                    done = step + 1
                    filled = 24 * done // iterations
                    rate = (done - begin) / max(time.perf_counter() - phase_start, 1e-9)
                    eta = round((iterations - done) / rate)
                    bar = "#" * filled + "-" * (24 - filled)
                    message = f"\r[{bar}] {done/iterations:6.1%} {message}  {rate:.1f} updates/s  ETA {eta//60}m {eta%60:02}s"
                    print(message.ljust(110), end="\n" if done == iterations else "", flush=True)
                else:
                    print(message, flush=True)
            if cfg.checkpoint_path and (
                (step + 1) % cfg.checkpoint_every == 0 or step + 1 == iterations
            ):
                checkpoint = {
                    "version": 1,
                    "phase": phase,
                    "completed": step + 1,
                    "global_step": global_step,
                    "reference_digest": reference_digest,
                    "architecture": architecture,
                    "config": asdict(cfg),
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "history": history,
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()
                    if device.type == "cuda"
                    else None,
                }
                path = Path(cfg.checkpoint_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                torch.save(checkpoint, temporary)
                temporary.replace(path)
        del opt, scheduler, groups, forward
        resume = None
    # Even a zero-step phase3 ends with the exact runtime representation.
    model.quantize_and_freeze_bc_features()
    model.decoder.quantized_forward = True
    return history

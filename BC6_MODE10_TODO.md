# BC6 implementation status

The old checklist was stale. See [README](README.md) and [illustrated tutorial](docs/tutorial.html).

Completed: official Mode 10 bit packing; fixed subset ownership; exact final features; FP16-aware final tuning; runtime-file inference; all-mip DDS validation; clamp and LOD parity; shared shader decoder; corrected dataset ranges and normal previews; module split; effective microbatches; deterministic resumable checkpoints; regression suite; small matched experiments.

Before a full run: validate the intended GPU PyTorch backend, measure realistic batch throughput/VRAM, rebuild 2K references from source maps, review the selected feature resolution and evaluation criteria, and obtain the requested user confirmation.

Remaining scope: full-resolution quality reproduction and performance optimization; full numerical shader/filtering/MLP readback test; full lit PBR renderer; displacement channels; signed trainable BC6; other BC6 modes; periodic border training. These are not silently supported by the current runtime.

# Export comparison

**Coverage: partial.**
Weighted PSNR pools squared errors by texel count before converting to dB. Higher is better. Filtered PSNR gives each of 89 viewing distances equal weight. FLIP averages four shaded views. Lower is better.

## ukulele-01

Group 1 has identical source hashes, channels, evaluation UVs and lights.

| Run | Updates | Payload B | Weighted dB | Mip-0 dB | Filtered dB | FLIP |
|---|---:|---:|---:|---:|---:|---:|
| Neural ASTC · 0002 | 18225 | 349,848 | 28.101 | 27.810 | 29.312 | 0.0985 |
| Neural ASTC · 0005 | 20000 | 349,848 | 28.106 | 27.814 | 29.317 | 0.0980 |
| Local Neural BC6H · 0002 | 20000 | 464,880 | 27.336 | 27.047 | 30.361 | 0.0914 |
| Uncompressed 2K reference · 0001 | — | 33,554,432 | exact | exact | exact | 0.0000 |

Highest measured weighted quality, including ties within 0.05 dB: Neural ASTC · 0002, Neural ASTC · 0005. Lowest shaded error: Local Neural BC6H · 0002. The table gives each run’s training budget. Imported exports may have unknown training cost.

[Mip and rate plots](group-1/figures/quality.png) · [Viewing windows](group-1/windows.csv) · [Runtime protocol and timings](group-1/runtime.csv). Timing uses resident, expanded tensors. Native phone timing is [PENDING].
[Reconstruction and error crops 1](group-1/figures/maps-1.png)
[Training loss](group-1/figures/training.png) shows fits with a recorded minibatch trace.

## Missing or incompatible evidence

- ukulele-01 lacks ntc; fit or import these methods on the same source maps.

[Editable comparison and workflow](comparison.drawio)

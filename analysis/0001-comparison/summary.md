# Export comparison

**Coverage: partial.**
Weighted PSNR pools squared errors by texel count before converting to dB. Higher is better. Filtered PSNR gives each of 89 viewing distances equal weight. FLIP averages four shaded views. Lower is better.

## ukulele-01

Group 1 has identical source hashes, channels, evaluation UVs and lights.

| Run | Updates | Payload B | Mip-0 dB | Weighted dB | Filtered dB | FLIP |
|---|---:|---:|---:|---:|---:|---:|
| neuralastc · 0001 | unknown | 349,848 | 27.809 | 28.101 | 29.311 | 0.0987 |
| ubisoft · 0001 | unknown | 464,880 | 27.047 | 27.336 | 30.361 | 0.0914 |
| ref · 0001 | — | 33,554,432 | exact | exact | exact | 0.0000 |

Highest measured weighted quality, including ties within 0.05 dB: neuralastc · 0001. Lowest shaded error: ubisoft · 0001. The table gives each run’s training budget. Imported exports may have unknown training cost.

[Mip and rate plots](group-1/figures/quality.png) · [Viewing windows](group-1/windows.csv) · [Runtime protocol and timings](group-1/runtime.csv). Timing uses resident, expanded tensors. Native phone timing is [PENDING].

## Missing or incompatible evidence

- ukulele-01 lacks ntc; fit or import these methods on the same source maps.

[Editable comparison and workflow](comparison.drawio)

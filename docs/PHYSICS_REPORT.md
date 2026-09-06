# Phase 0: real-meter acceptance report

Status: NOT YET MEASURED. No external meter connection or separate helper endpoint
has been supplied. Host GPU power telemetry is not independent meter evidence.

Initial environment inspection: Windows, Python 3.13.9, NVIDIA GeForce RTX 5060
Laptop GPU, 8151 MiB, driver 577.13. The laptop battery and charging state must be
recorded when wall-power experiments begin. No CUDA toolkit or Python numerical
dependencies were present initially. Serial device enumeration was denied by the
environment; this is not evidence that no meter exists.

| Required measurement | Result |
| --- | --- |
| True device sample cadence | Unmeasured |
| Smoothing, lag, washout | Unmeasured |
| Tensor versus idle power separation | Unmeasured |
| Memory versus tensor separation | Unmeasured |
| Mixed workload response | Unmeasured |
| Local full-pipeline time under the physical test configuration | Unmeasured |
| Separate unmetered helper full-pipeline time | Unmeasured |
| Repeatability on fresh held-out schedules | Unmeasured |

Backend development and computational tests may proceed. The physical acceptance
gate remains closed, and no polished dashboard is in scope. Synthetic test data
must remain labelled as simulation; it cannot fill this table with real results.

## Computational measurements completed separately

CPU/CUDA digest agreement passed for IDLE, TENSOR, MEMORY and MIXED on the actual
RTX 5060 Laptop GPU. Three local and three forwarded epochs using actual HTTP on
this same laptop passed all small-workload computational targets. A second process
on the same host is not an unmetered helper and cannot establish the requested
physical disagreement. See validation/loopback-cuda.json for auditor timings.

Fresh trusted local profiling of the larger example workload recorded TENSOR total
worker times of 0.497–0.570 s, MEMORY 0.845–0.926 s, and MIXED 1.067–1.459 s. TENSOR
execution itself was only 0.0074–0.0079 s: CPU input generation, copies, aggregation
and hashing dominate. These are tuning observations without network or meter data,
not proof of sustained GPU utilization. The ten-second example segments are not
yet a calibrated physical experiment.

The reporting implementation distinguishes exact device timestamp evidence from
collector-observed fresh-update cadence. Repeated polling alone never establishes
native sample rate. Smoothing/lag reports describe the observed workload-plus-meter
response and explicitly retain that limitation.

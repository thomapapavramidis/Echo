# Implementation and remaining work

Updated September 6, 2026.

The repository is prepared to receive two real RunPod endpoint configurations. No
RunPod deployment or hardware-dependent calibration has occurred. `NVML_DEMO` host
telemetry is proof-of-concept diagnostics and is permanently excluded from the
independent physical-evidence detector. See [RUNPOD_POC.md](RUNPOD_POC.md).

Two matching H100 Pods and their proxy URLs are now available, but live connectivity
and profile artifacts have not yet been verified from this repository. The bank-free
synchronized NVML profiler and first-run commands are documented in H100_FIRST_RUN.md.

The judge-facing pivot is implemented separately as `HYBRID_DEMO`: real fresh
H100 challenges and routing, separate simulated US/Iceland facility-meter traces,
private helper-response ground truth, capture, explicit replay, a projection
dashboard, and no production receipt or physical-verification endpoint. Raw live
NVML is confined to the collapsed diagnostic panel and excluded from detection.

Reported H100 profiles place the current verified GPU work at approximately 12–15
milliseconds with warm active power approximately 1–3 W above warm idle. Those
results motivated the dual digital-twin stage view; they are not calibration data
or independent physical evidence.

## Implemented

- Empty repository scaffolded as an installable Python package with locked
  dependencies, optional CUDA/serial support, CLI, and operator documentation.
- Encrypted trusted-reference challenge bank, fresh per-operation inputs, aggregate
  hashes over every output, atomic durable retirement before network delivery,
  concurrency protection, replay rejection, and crash recovery without reissuance.
- Executable IDLE, TENSOR, dependent MEMORY, and interleaved MIXED workloads on CPU
  and CUDA. Exact integer/PRNG/serialization/hash protocol documented and tested.
- Full pipeline timing, auditor-controlled completion timing, response-byte/receive
  measurements, and explicit limits on one-way network timing inference.
- Facility and helper worker services with real HTTP forwarding, per-role
  credentials, persistent replay ledgers, bounded cancellation, quiescence, and
  operational GPU temperature protection.
- Independent meter HTTP/serial normalized interfaces, authenticated auditor
  ingestion, persistent sequence checks, freshness/clock/quality enforcement,
  power-limit and sensor-loss stop behavior.
- Balanced random epoch schedules, just-in-time issuance, measured/provisional
  washout distinction, progress events, new-window findings, explicit transitions.
- Step/cadence characterization, training and held-out validation, fine-mode
  qualification and ACTIVE fallback, separate shape/gain, empirical ranges, and
  abstention on unsupported or unreliable observations.
- Separate correctness/performance/physical findings; PENDING, DEADLINE_FAILED,
  INCORRECT, interruption and execution errors preserved distinctly.
- Offline evaluation joins private routing labels after detection; counts include
  failures and insufficient outcomes with denominators and equivalent-target checks.
- Raw evidence, signed Ed25519 receipts, trusted-issuer/evidence verification.
- Unit/integration tests and a real TCP/CUDA forwarding smoke script. No polished
  dashboard, fake production sensor, attestation integration, or GPU attribution.
- Authenticated direct-NVML telemetry with a bounded `nvidia-smi` fallback, separate
  declared-site/helper streams, attack-ground-truth labeling, an unpolished dashboard,
  RunPod startup scripts, and deployment-readiness checks.

## Validated here

- Final test suite after the hybrid-demo correction: **72 passed**, including CUDA tests, on September 6, 2026.
  Ruff lint and Python compilation checks pass. All services launched for validation were
  stopped afterward; no background demo service is intentionally left running.

- CPU and RTX 5060 Laptop GPU agree for all four workload modes, including multiple
  independent seeds and memory sequences crossing cancellation chunk boundaries.
- Scalar arithmetic tests validate CPU GEMM and dependent-memory recurrence.
- Skipped memory work, corrupted first batch output, old results, duplicate seeds,
  stale/replayed meters, unknown specs, mixed provenance, interrupted epochs,
  overlapping calibration/evaluation sets, and altered receipts are exercised.
- Three local and three forwarded real-HTTP GPU epochs pass equivalent small
  computational targets. Both services run on this same host/GPU, so this is
  computational integration only. See validation/loopback-cuda.json.
- A second final HTTP/CUDA run passed three fresh epochs on each route after the
  cancellation/deadline changes; see validation/loopback-cuda-final.json.
- Trusted local profiling recorded complete pipeline stage costs for larger example
  workloads. Input generation, copies and hashing dominate these starting sizes;
  the configuration is not a GPU-saturation or physical detection result.

## Physical gates still open

| Gate | Status / required next input |
| --- | --- |
| Identify and connect actual external meter | Meter model/API/serial connection not supplied |
| Model-specific meter adapter | Normalize actual meter fields and genuine update marker |
| Inspect branch and independent collector path | Requires physical setup confirmation |
| Native cadence, smoothing and lag | Requires real independent samples |
| Battery/power state and stable operating conditions | Must be recorded at the measured laptop/setup |
| Sustained useful tensor/idle response | Tune from actual full-pipeline and meter traces |
| Separate helper endpoint and location | Not supplied; loopback is not a physical substitute |
| Both routes pass the chosen real computational target | Requires equivalent real-machine runs |
| Memory/mixed timing and electrical distinction | Timing profiler exists; physical qualification pending |
| Held-out repeated local/forwarded physical separation | Must pass final evaluation with independent samples |
| Polished dashboard | Explicitly withheld until preceding acceptance gate passes |

No value from synthetic unit tests or loopback measurements fills a missing
electrical measurement. The project is a tested backend implementation with an
outstanding hardware experiment, not a completed physical verification demo.

## Implementation limitations to keep visible

- V1 uses CPU SHAKE256 input generation and CPU SHA-256 over full locally copied
  results. Compact network output does not eliminate local transfer/hash cost.
  Profile before increasing batch counts. Device-side generation/hashing may be
  worth implementing if the actual meter experiment establishes that need.
- The physical detector uses settled segment means, not a full dynamic convolution
  model. Bursts that end before settling, long UPS buffering, or short segments can
  make calibration invalid. Preserve that failure rather than manufacture a match.
- The memory challenge defeats the unchanged-buffer shortcut but has no formal
  memory-hardness, residency, or minimum-access guarantee.
- A local dummy load may reproduce the expected response while remote compute
  supplies valid answers. Fine workload modes are experimental and do not remove
  this fundamental absence of device-to-result binding.
- Known-local commissioning and correct sensor placement remain assumptions.
  Backup rollback/cloning of an issuance bank is forbidden operationally; the local
  database is not a hardware monotonic counter.
- CUDA cancellation waits for an already executing GEMM/bounded memory chunk. It
  does not implement hard GPU preemption. Long kernel behavior must be bounded by
  tested configurations.
- Cross-OS/cross-GPU digest interoperability has a portable specification and test
  vectors, but only this CPU/GPU pair has been executed here.
- Exact device counts, reference H100 equivalents, GPU-class fingerprints, device
  serial identity, hardware enforcement and export-control claims are not emitted.

## Suggested next session

Supply the two matching RunPod proxy URLs and credentials described in
RUNPOD_POC.md. Probe both hosts, profile the four configurable workload modes, tune
the equivalent targets, generate a new private bank, and run fresh held-out local
and forwarded schedules. External-meter commissioning remains a later production
phase; the NVML proof of concept cannot substitute for it.

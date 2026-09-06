# Compute Echo

Compute Echo challenges declared compute capacity and checks for its physical echo
at the claimed site. This repository implements the hackathon backend: private
single-use challenges, deterministic CPU/CUDA workers, real forwarding, independent
meter ingestion, epoch scheduling, calibration, routing-blind findings, and signed
evidence receipts.

**Current status:** CPU/CUDA correctness and real HTTP loopback forwarding have been
tested on the available RTX 5060 Laptop GPU. The independent external meter and
separate helper machine have not been provided. Real electrical acceptance is
therefore **NOT MEASURED**. There is no polished dashboard and no fabricated power
generator. See [the physics report](docs/PHYSICS_REPORT.md) and
[implementation status](docs/STATUS.md).

The two-Pod RunPod preparation, unpolished dashboard, required environment values,
and startup commands are documented in [the RunPod POC runbook](docs/RUNPOD_POC.md).
The first bank-free H100 deployment/profile sequence is in
[the H100 first-run guide](docs/H100_FIRST_RUN.md).
The polished transparent hackathon flow, capture/replay controls, and exact stage
script are in [the hybrid demo runbook](docs/HYBRID_DEMO_RUNBOOK.md).

## Install and verify

Python 3.11+ is supported; the checked environment uses Python 3.13.9. `uv.lock`
records the exact dependencies used. Commands below are PowerShell from this folder.

```powershell
uv sync --python 3.13 --extra dev --cache-dir .uv-cache
# Optional, on NVIDIA workers / a trusted CUDA reference machine:
uv sync --python 3.13 --extra dev --extra cuda --cache-dir .uv-cache

$env:CUPY_CACHE_DIR = Join-Path (Get-Location) '.cupy'
.venv\Scripts\echo-audit.exe doctor --cuda
New-Item -ItemType Directory -Path runs -Force | Out-Null
.venv\Scripts\python.exe -m pytest -q --basetemp runs/pytest-local
.venv\Scripts\ruff.exe check src tests scripts
```

Pytest's `--basetemp` is disposable and cleared by pytest: use only a dedicated
test folder under `runs`, never an evidence/private folder. A CPU installation
skips CUDA-specific tests; a CUDA installation verifies all four GPU modes against
the independent CPU implementation. Request `--cuda` explicitly; no silent fallback
labels CPU work as GPU work.

For an actual HTTP smoke test that starts and stops two local processes:

```powershell
.venv\Scripts\python.exe scripts/loopback_smoke.py --backend cuda --output runs/loopback-report.json
```

This test uses the **same host/GPU for both processes** and no meter. It validates
protocol and forwarding integration, not location or electrical detection.

## Backend components

| Module | Responsibility |
| --- | --- |
| `workloads.py` | IDLE, INT8/INT32 TENSOR, dependent MEMORY, interleaved MIXED; full pipeline timings |
| `bank.py` | Offline trusted reference computation; encrypted private payloads; atomic permanent retirement |
| `worker.py`, `transport.py` | Authenticated worker/helper service, real HTTP forwarding, cancellation and quiescence |
| `meter.py`, `physics.py` | Independent HTTP/serial adapter contracts, durable sample replay checks, cadence/step analysis |
| `epochs.py` | Balanced random schedules, just-in-time seeds, washout, progress, fresh epoch evidence |
| `calibration.py`, `detector.py` | Training/validation separation, empirical envelopes, ACTIVE fallback, abstention |
| `evaluation.py` | Join routing ground truth only after inference; counts, denominators, timing comparisons |
| `receipt.py` | Ed25519 receipts, trusted issuer verification, hashes of saved evidence |
| `service.py`, `cli.py` | Auditor API, explicit transition control, command-line experimental workflow |
| `telemetry.py`, `demo.py`, `dashboard.py` | Role-separated NVML_DEMO host diagnostics and the unpolished POC display; excluded from production meter evidence |
| `deployment.py`, `deploy/runpod/` | RunPod environment validation, live endpoint probe, and Pod startup scripts |
| `profiling.py` | Bank-free exploratory CUDA profiles synchronized with raw, non-independent NVML host telemetry |
| `hybrid.py`, `hybrid_dashboard.py` | Separate receipt-free HYBRID_DEMO with live compute/routing, digital-twin inference, capture and explicit replay |

## The protocol

The trusted reference machine computes unique workloads offline and privately stores
their seeds/specifications/expected aggregate digests. Issuance permanently retires
a seed **before** sending it. A worker derives fresh inputs for every operation,
computes and hashes all outputs, and returns a 32-byte digest plus diagnostic
metadata. Local and forwarded epochs use equivalent specifications with different
seeds. The auditor checks its private expected digest and its own monotonic clock.

The memory workload performs state-dependent reads and writes on per-lane mutable
buffers. Both the mutated buffers and terminal states enter the digest. It does
not claim provable memory hardness or GPU memory residency. Details are in
[the memory design](docs/MEMORY_CHALLENGE.md) and [wire specification](docs/PROTOCOL.md).

## Run the real experiment

Follow [the runbook](docs/RUNBOOK.md). Required equipment is an independently
controlled external meter/collector, one measured worker, one separately powered
helper, and an auditor outside the measured branch. The repository provides a
normalized JSON HTTP/serial adapter interface; mapping a particular commercial
meter still requires that model's actual refresh counter/timestamp and API.

The sequence is:

1. Configure and independently inspect the meter connection and clocks.
2. Tune complete fresh workloads on the actual machines; precompute the private bank.
3. Capture long provisional physics epochs and measure cadence/step response.
4. Fit on at least three trusted local training epochs; qualify on three new local
   validation epochs. Characterization, training, and validation are disjoint.
5. Freeze the model and evaluate at least three new local and three new forwarded
   epochs under equivalent targets. Report all failures and insufficient outcomes.
6. Permit a polished dashboard only after both routes pass the same computational
   target and the independent power difference repeats.

The default example workload sizes are **not tuned or physically validated**.
Full-pipeline duration can be dominated by CPU input generation and hashing even
when GEMM runs correctly on tensor hardware. The workload profiler reports that
overhead explicitly. Increasing a window length alone does not create sustained
GPU activity.

## Documentation

- [Updated architecture](docs/ARCHITECTURE.md)
- [Deterministic protocol and trust boundaries](docs/PROTOCOL.md)
- [Operator runbook, commands, and API](docs/RUNBOOK.md)
- [Measurement results and missing equipment](docs/PHYSICS_REPORT.md)
- [Implemented work, acceptance gates, remaining work](docs/STATUS.md)
- [Recorded computational validation](docs/validation/README.md)

Private banks, keys, raw run folders, and environment secrets are ignored by Git.
An audit receipt authenticates the issuer and preserved evidence; it does not prove
device identity, exact GPU count, ownership, exclusivity, or resistance to an
adaptive local dummy load. The backend always reports
`gpu_class_fingerprint_supported: false`.

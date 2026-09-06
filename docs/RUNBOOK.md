# Operator runbook

All commands assume the repository root and PowerShell. Replace equipment URLs and
installation information with the real setup. The commands do not start a dashboard.

## 1. Equipment and installation

Use the external meter only for the measured facility worker. Auditor, collector,
helper, and display must be outside that electrical branch. A separate process on
the same GPU is not an unmetered helper. Record meter model/serial, worker/helper
hardware, driver/library versions, power limits, battery charge/assistance state,
background jobs, and clock synchronization before calibration. Prefer stable mains
power and document laptop battery behavior. Use existing rated metering equipment.

The independent adapter must emit this normalized JSON object:

```json
{
  "meter_id": "bench-meter-1",
  "sequence": 123,
  "acquired_at_ns": 1788739200000000000,
  "watts": 132.5,
  "device_sequence": 456,
  "device_time_ns": 1788739200000000000,
  "quality": "GOOD"
}
```

The numbers above illustrate the schema, not measurements. Supply current UTC
timestamps. `sequence` is the collector's persistent monotonic sequence; the device
fields come from actual fresh device updates, not the polling loop. Configure
`source_freshness` as `device_sequence` or `device_timestamp`. If the hardware exposes
neither, select `unknown`; the physical result will remain insufficient until its
refresh behavior can be established independently. Do not assign a new hardware
counter just because an HTTP read happened.

An HTTP meter adapter returns one such object. A serial adapter emits one JSON object
per line. Raw vendor protocols need a small model-specific adapter; no meter model
was supplied for this implementation. Keep device identity and sequence durable
across collector restarts. A meter reset requires a new commissioned identity.

Copy `config/meter.example.json` into private configuration and fill in the actual
installation and limits. Sample age/skew allowances must match measured clock and
transport behavior. Acquisition times and auditor epoch times share the documented
UTC clock domain; deadline timing uses the auditor's monotonic clock.

## 2. Credentials and process placement

Generate long independent secrets on the auditor using your normal secret storage.
Do not place secrets in command arguments, Git, result files, or shared worker source
trees. Set environment variables in each process's shell as follows:

| Process | Environment variables |
| --- | --- |
| Unmetered helper | ECHO_WORKER_TOKEN=helper credential; ECHO_CONTROL_TOKEN=helper control credential |
| Measured facility worker | ECHO_WORKER_TOKEN=facility credential; ECHO_CONTROL_TOKEN=facility control credential; ECHO_HELPER_TOKEN=helper credential |
| Auditor | ECHO_WORKER_TOKEN=facility credential; ECHO_CONTROL_TOKEN=facility control credential; ECHO_ADMIN_TOKEN=auditor API credential; ECHO_METER_TOKEN=collector credential |
| Independent collector | ECHO_METER_TOKEN=collector credential only |

All credentials must be at least 24 characters. Worker/control credentials must
differ, as must admin/meter credentials. For real separated deployments use distinct
credentials for every role. Use HTTPS/tunnels beyond a controlled lab network.
Worker credentials cannot change routing or submit meter samples. The worker and
helper must never be given the auditor private directory or reference bank.

## 3. Start helper and worker

On the separate helper, after installing CUDA dependencies:

```powershell
.venv\Scripts\echo-audit.exe worker --backend cuda --ledger private/helper-ledger.sqlite3 --host 0.0.0.0 --port 8102
```

On the measured facility machine:

```powershell
.venv\Scripts\echo-audit.exe worker --backend cuda --ledger private/worker-ledger.sqlite3 --helper-url http://HELPER_HOST:8102 --host 0.0.0.0 --port 8101
```

The API warms up all four modes before reporting readiness. Warmup uses public test
inputs, is not counted as audit work, and occurs before washout. Default CUDA safety
checks stop at 85 C or when temperature diagnostics become unavailable. Meter-side
power limits are configured separately on the auditor. `--backend cpu` is available
for reference/testing and is always labelled CPU.

## 4. Tune and precompute on trusted equipment

Profile the entire worker pipeline first:

```powershell
.venv\Scripts\echo-audit.exe workload-profile --backend cuda --specs config/workloads.physics.example.json --repeats 3 --output runs/workload-profile.json
```

The example configuration is uncalibrated. On the available laptop the initial
tensor batch finished in about half a second inside a ten-second segment, with
input generation/hashing dominating. This is a tuning signal, not a ready demo
configuration. Increase fresh operation counts/change dimensions and measure again;
leave deadline headroom on both worker and helper. Do not repeat identical matrices
or insert unverified dummy load to extend work. If CPU/transfer overhead prevents a
useful physical signal, a faster deterministic device input/hash implementation is
an experimental optimization; do not claim the existing pipeline is GPU-saturated.

For initial TENSOR/IDLE commissioning, create a spec file containing only those two
entries from the example. All four modes remain available for later qualification.

Generate a bank using the exact final specifications on trusted equipment:

```powershell
.venv\Scripts\echo-audit.exe bank-generate --private private --specs private/workloads.json --count-per-mode 96 --backend cuda
```

Each four-mode epoch with repeats=4 consumes 12 IDLE and four challenges of each
active mode. A two-mode epoch consumes four of each. Bank counts include every
issued/aborted challenge; provision extra challenges for rehearsals. The command
prints only inventory counts and specification hashes, never seeds/expected answers.

Never restore a used bank from a snapshot. To recover lost state, archive the bank
and provision new seeds. Sequential reference generation is permitted; never
precompute a supposedly secret bank on the untrusted facility worker.

## 5. Capture physics epochs

Two supported options follow. Use only one auditor owner for a given bank.

### Standalone CLI with direct independent HTTP meter adapter

```powershell
.venv\Scripts\echo-audit.exe physics-run --private private --runs runs/physics --meter-config private/meter.json --meter-source-url http://METER_ADAPTER/sample --worker-url http://WORKER_HOST:8101 --specs private/workloads.json --route local --repeats 4 --provisional-washout 5
```

The provisional washout above is a declared starting experiment parameter, not a
measured result. The command starts an auditor-owned collector, runs a fresh epoch,
preserves data, and closes its connections. Missing/stale readings stop the epoch.
Use `--route forwarded` for a separate fresh experiment under identical specs.

### Persistent auditor API with independent collector

```powershell
.venv\Scripts\echo-audit.exe auditor --private private --runs runs/physics --meter-config private/meter.json --worker-url http://WORKER_HOST:8101 --site demo-bench --site-assurance 'Auditor inspected the metered outlet and worker connection'
```

In a separate auditor-controlled collector shell:

```powershell
.venv\Scripts\echo-audit.exe collect-http --source-url http://METER_ADAPTER/sample --auditor-url http://127.0.0.1:8100 --interval 0.5
# Or install the serial extra and use the independent serial adapter:
.venv\Scripts\echo-audit.exe collect-serial --port COM4 --baudrate 115200 --auditor-url http://127.0.0.1:8100
```

Polling twice per second is not an assertion of two fresh measurements per second.
Duplicate hardware updates are rejected. The device adapter must persist its own
collector sequence and expose genuine hardware freshness.

Submit a physics run to the auditor (never put expected digests here):

```powershell
$headers = @{Authorization = "Bearer $env:ECHO_ADMIN_TOKEN"}
$body = @{
    specs = @(Get-Content private/workloads.json -Raw | ConvertFrom-Json)
    repeats = 4
    provisional_washout_s = 5
    timing_only = $false
    route = 'local'
} | ConvertTo-Json -Depth 10
Invoke-RestMethod -Uri http://127.0.0.1:8100/epochs/start -Method Post -Headers $headers -ContentType application/json -Body $body
Invoke-RestMethod -Uri http://127.0.0.1:8100/health -Headers $headers
```

Change the body route and POST to `/epochs/transition` to close/drain the current
epoch, change routing, wash out, and begin a fresh schedule. `/epochs/stop` interrupts
and preserves the current epoch. `/health` exposes `completed_segments` and
`total_segments`. Insufficient evidence has no fabricated progress-derived verdict.

`--timing-only` (CLI) / `timing_only=true` (API) explicitly permits computational
integration without a meter. Such a run cannot qualify independent physical
acceptance. Use `config/meter.timing-only.json` and label the experiment accordingly.

## 6. Characterize, calibrate, and evaluate

Characterization consumes previously captured known-local epochs with repeated
TENSOR/IDLE steps:

```powershell
.venv\Scripts\echo-audit.exe characterize runs/physics/CHARACTERIZATION_EPOCH --output runs/meter-profile.json
```

If `profile` is null, adjust meter access, workload duration, or segment duration and
run new experiments. Inspect true versus observed cadence, lag, rise time, step
amplitudes, and missingness. The measured washout must leave enough settled samples
per segment. Do not change thresholds until a desired result appears.

Collect three new trusted-local training epochs and three new held-out local
validation epochs with the same workload configuration. For these pre-model runs,
explicitly use the characterized washout as the provisional washout argument.

```powershell
.venv\Scripts\echo-audit.exe calibrate --training runs/physics/TRAIN1 runs/physics/TRAIN2 runs/physics/TRAIN3 --validation runs/physics/VAL1 runs/physics/VAL2 runs/physics/VAL3 --profile runs/meter-profile.json --hardware-configuration 'Actual GPU, power cap, battery state, driver, package lock, workload specs and background jobs' --output runs/calibration-v1.json
```

Only a model reporting `validated=true` can support physical consistency. It may
collapse all supported active modes to ACTIVE. A changed spec or previously unseen
mode requires new calibration. The raw computational measurements still retain
all four modes.

Restart the auditor with `--calibration runs/calibration-v1.json`. Omit provisional
washout when starting normal epochs: the model supplies its measured washout.
Run at least three new local and three new forwarded epochs; both routes must meet
the same target. Evaluate using the separately stored labels:

```powershell
.venv\Scripts\echo-audit.exe evaluate runs/physics/LOCAL1 runs/physics/LOCAL2 runs/physics/LOCAL3 runs/physics/FWD1 runs/physics/FWD2 runs/physics/FWD3 --labels private/evaluation-labels.jsonl --calibration runs/calibration-v1.json --output runs/evaluation.json
```

The report includes denominators, physical statuses, computational passes, false
alarms, and per-mode pipeline timing. `minimum_acceptance_passed` is the gate for
the polished dashboard. A generic dummy-load experiment is labelled separately and
must report a bypass if it succeeds. Adaptive waveform imitation is not defeated
by this architecture.

## 7. Evidence and receipts

Each run directory contains issued-challenge/progress/result JSONL events, complete
meter samples in evidence.json, and a routing-blind finding.json. The private label
file is not part of detector inputs. The bank, meter journal, and signing key remain
private. Incomplete epochs are retained for troubleshooting and evaluation.

After an epoch closes, POST `/epochs/{id}/receipt` with the admin credential. GET
the same path to retrieve the immutable receipt. Verify with a separately trusted
issuer ID, not merely the key embedded inside the receipt:

```powershell
.venv\Scripts\echo-audit.exe verify-receipt runs/physics/EPOCH/receipt.json --trusted-key-id TRUSTED_KEY_SHA256 --evidence-directory runs/physics/EPOCH
```

## API summary

| Method/path | Credential | Purpose |
| --- | --- | --- |
| GET /health | admin | Epoch progress and independent meter health |
| POST /meter/samples | meter | Submit one fresh independent sample |
| POST /epochs/start | admin | Start a fresh epoch when idle |
| POST /epochs/transition | admin | Close/drain, route, washout, new epoch |
| POST /epochs/stop | admin | Stop and preserve current evidence |
| GET /epochs/{id} | admin | Evidence and finding after completion |
| GET /epochs/{id}/events | admin | Server-sent observations/progress stream |
| POST /epochs/{id}/receipt | admin | Sign a closed epoch |
| GET /epochs/{id}/receipt | admin | Retrieve signed receipt |
| POST /worker/task | worker | Execute/forward one issued challenge |
| POST /worker/cancel/{id} | worker | Request cancellation and await bounded work |
| POST /worker/quiesce | worker | Drain local and helper work |
| POST /control/route | worker control | Change facility routing while quiescent |

Workers reject concurrent tasks rather than silently creating workload contention.
If quiescence fails, investigate the worker before starting another epoch. API
OpenAPI documentation is available from each service's `/docs` for schema inspection;
this is a backend development interface, not the postponed judge dashboard.

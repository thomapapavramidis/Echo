# Hybrid hackathon demonstration runbook

## Permanent evidence boundary

The judge-facing mode is `HYBRID_DEMO`. Every screen says **LIVE COMPUTE +
SIMULATED FACILITY METERS**. The two prominent graphs say **DECLARED FACILITY
METER DIGITAL TWIN — SIMULATED** and **REMOTE FACILITY METER DIGITAL TWIN —
SIMULATED DEMO GROUND TRUTH**. Replay adds **REPLAY OF PREVIOUSLY CAPTURED
DEMONSTRATION**.

In the local scene, the simulated US trace follows the challenge barcode and the
simulated Iceland trace remains idle. In the forwarded scene, the US trace remains
idle and the simulated Iceland trace follows the barcode. The remote simulation is
private demonstration ground truth; it is not independent evidence available to a
normal auditor.

Live NVML from both workers appears only inside the collapsed **RAW HOST
TELEMETRY** technical panel under **LIVE NVML HOST TELEMETRY — DIAGNOSTIC ONLY**.
It uses fixed 0–700 W and 0–100% axes and may look nearly flat. It is not amplified,
treated as a facility meter, or supplied to the detector.

The app exposes no meter-ingestion, production physical-verification, calibration,
or receipt endpoint. The route-blind digital-twin detector receives only the
challenge schedule and simulated US facility trace. Its function accepts no route
argument. The controller retains the confirmed route and simulated Iceland trace
separately as demonstration ground truth.

## Publish from the development checkout

The RunPod proof-of-concept is published at `ba60595`; the hybrid-demo changes are
the uncommitted update. From the repository root:

```bash
git status --short
git add README.md docs/STATUS.md src/compute_echo/cli.py tests/test_service_epochs.py
git add config/hybrid.env.example config/workloads.hybrid-demo.json
git add docs/HYBRID_DEMO_RUNBOOK.md scripts/hybrid_health.ps1
git add scripts/reset_hybrid_demo.ps1 scripts/start_hybrid_demo.ps1
git add src/compute_echo/hybrid.py src/compute_echo/hybrid_dashboard.py tests/test_hybrid.py
git commit -m "Add transparent hybrid H100 demonstration"
git push -u origin main
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

Do not add `.env.hybrid`, `.env.runpod`, `private/`, or `runs/`; they are ignored.

## Pull the update onto the helper

In the helper Pod terminal, stop the previous service if its PID file exists:

```bash
cd /workspace/Echo
if [ -s /workspace/compute-echo-state/helper-service.pid ]; then
  kill "$(cat /workspace/compute-echo-state/helper-service.pid)" 2>/dev/null || true
  wait "$(cat /workspace/compute-echo-state/helper-service.pid)" 2>/dev/null || true
fi
git fetch origin main
git checkout main
git pull --ff-only origin main
bash deploy/runpod/bootstrap.sh
git rev-parse HEAD
```

## Pull the update onto the facility

In the facility Pod terminal:

```bash
cd /workspace/Echo
if [ -s /workspace/compute-echo-state/facility-service.pid ]; then
  kill "$(cat /workspace/compute-echo-state/facility-service.pid)" 2>/dev/null || true
  wait "$(cat /workspace/compute-echo-state/facility-service.pid)" 2>/dev/null || true
fi
git fetch origin main
git checkout main
git pull --ff-only origin main
bash deploy/runpod/bootstrap.sh
git rev-parse HEAD
```

The two printed commit IDs must match the laptop's published commit.

## Credentials

Generate four distinct URL-safe secrets on the laptop:

```bash
python3 - <<'PY'
import secrets
for name in (
    "SITE_WORKER_TOKEN",
    "SITE_CONTROL_TOKEN",
    "HELPER_WORKER_TOKEN",
    "HELPER_CONTROL_TOKEN",
):
    print(f"{name}={secrets.token_urlsafe(32)}")
PY
```

Helper `/workspace/Echo/.env.runpod`:

```dotenv
HELPER_WORKER_TOKEN=REPLACE_WITH_HELPER_WORKER_TOKEN
HELPER_CONTROL_TOKEN=REPLACE_WITH_HELPER_CONTROL_TOKEN
```

Facility `/workspace/Echo/.env.runpod`:

```dotenv
SITE_WORKER_TOKEN=REPLACE_WITH_SITE_WORKER_TOKEN
SITE_CONTROL_TOKEN=REPLACE_WITH_SITE_CONTROL_TOKEN
HELPER_WORKER_TOKEN=REPLACE_WITH_HELPER_WORKER_TOKEN
HELPER_API_URL=https://wgm4pcdvdsnvlp-8102.proxy.runpod.net
```

Apply restrictive permissions on each Pod:

```bash
chmod 600 /workspace/Echo/.env.runpod
```

Laptop setup in PowerShell:

```powershell
Copy-Item config\hybrid.env.example .env.hybrid
notepad .env.hybrid
```

Fill the same `SITE_WORKER_TOKEN`, `SITE_CONTROL_TOKEN`, and
`HELPER_WORKER_TOKEN`. The supplied configurable endpoint values are:

```dotenv
SITE_API_URL=https://bakoybwz37y4yf-8101.proxy.runpod.net
HELPER_API_URL=https://wgm4pcdvdsnvlp-8102.proxy.runpod.net
```

Worker and control tokens remain in the laptop backend process and never enter
browser JavaScript.

## Start the services

Start the helper first:

```bash
cd /workspace/Echo
nohup bash deploy/runpod/start_worker.sh helper \
  > /workspace/compute-echo-state/helper-service.log 2>&1 &
echo $! > /workspace/compute-echo-state/helper-service.pid
sleep 4
tail -n 30 /workspace/compute-echo-state/helper-service.log
```

From the facility, confirm the helper and start the facility service:

```bash
cd /workspace/Echo
set -a
source .env.runpod
set +a
curl --fail --silent --show-error \
  --header "Authorization: Bearer ${HELPER_WORKER_TOKEN}" \
  "${HELPER_API_URL}/health" | python -m json.tool

nohup bash deploy/runpod/start_worker.sh site \
  > /workspace/compute-echo-state/facility-service.log 2>&1 &
echo $! > /workspace/compute-echo-state/facility-service.pid
sleep 4
tail -n 30 /workspace/compute-echo-state/facility-service.log
```

Start the dashboard on the Windows laptop with one command:

```powershell
.\scripts\start_hybrid_demo.ps1 -EnvFile .env.hybrid
```

Open `http://127.0.0.1:8100/`. The server binds only to laptop loopback.

## Health and reset

Health check:

```powershell
.\scripts\hybrid_health.ps1
```

Deterministic reset to pattern `echo-demo-001`, `LIVE HYBRID`, and local route:

```powershell
.\scripts\reset_hybrid_demo.ps1
```

Reset drains any current worker task before confirming the local route. It does not
erase a capture or reset worker replay ledgers.

## Five-minute setup checklist

1. Confirm both Pod services use the same published Git commit.
2. Confirm only `8101/http` is exposed on the facility and `8102/http` on the helper.
3. Run the PowerShell health check and open the dashboard.
4. Wait for both connection indicators to turn green and confirm both models say H100.
5. Press `RESET DEMO`; verify the fixed pattern and local route.
6. Select `CAPTURE`, press `AUTO DEMO`, and let both scenes finish once.
7. Select `REPLAY` and verify the permanent replay banner before relying on backup mode.
8. Return to `LIVE HYBRID`, press `RESET DEMO`, and use full-screen browser mode.

Never switch to replay without deliberately choosing `REPLAY` in the visible mode
control.

## Exact 45-second presentation

- **0–5 seconds:** Point to the permanent hybrid disclosure, the US Northeast H100,
  Iceland H100, and both green live connection indicators. Say: “The computation and
  network route are live. Both facility meters are clearly labeled digital twins.”
- **5 seconds:** Press `AUTO DEMO` from `LIVE HYBRID` or `CAPTURE`.
- **5–21 seconds:** Scene A confirms the actual local route, issues four fresh
  challenges, verifies returned digests, reports laptop-measured round-trip times,
  and animates the simulated US facility echo while the simulated Iceland facility
  remains idle. The result becomes `ECHO MATCHED` and `CONSISTENT WITH EXECUTION AT
  DECLARED SITE`.
- **21–23 seconds:** Auto Demo drains the worker, changes the authenticated route to
  the Iceland helper, and starts a different fresh challenge sequence with the same
  workload multiset and deadlines.
- **23–39 seconds:** Scene B verifies the live forwarded answers and reports actual
  helper responses returned through the facility. The simulated US facility remains
  idle while the simulated Iceland facility follows the challenge barcode. The
  route label and remote trace remain outside the detector.
- **39–45 seconds:** Hold on `VERIFIED`, `COMPLETED BEFORE DEMO DEADLINE`, `ECHO
  MISSING`, and **CORRECT ANSWERS — WRONG PHYSICAL SITE**. Say: “This simulates what
  independent facility meters would observe; the physical layer is simulated, not
  measured.”

For a manual presentation, press `RUN LOCALLY`, `START CHALLENGE`, wait for Scene A,
then press `FORWARD TO REMOTE HELPER` and `START CHALLENGE` for Scene B.

## Capture and replay

`CAPTURE` performs live computation and writes successful scene data to
`runs/hybrid-demo/capture.json`, including actual computational outcomes, both
displayed digital-twin traces, raw diagnostic host telemetry and event timing. It is
not a signed receipt.

`REPLAY` makes no worker route or compute calls. It replays only a deliberately
selected saved capture and permanently shows the replay disclosure. A live outage
produces `CONNECTION_ERROR`; it never activates replay automatically.

## Remaining manual steps

- Publish the hybrid-demo commit and pull the same revision onto both Pods.
- Confirm the four role-specific secrets and current Pod service processes.
- Run one full `CAPTURE` rehearsal while the venue connection is stable.
- Visually inspect projection scaling and browser full-screen mode.
- Keep `config/workloads.hybrid-demo.json` configurable. Do not generate the final
  challenge bank, tune physical workloads, or choose calibration thresholds in this
  iteration.

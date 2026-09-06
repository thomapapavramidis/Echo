# Two-Pod RunPod proof of concept

## Evidence boundary

`NVML_DEMO` is labeled **HOST GPU TELEMETRY — PROOF OF CONCEPT** in every sample and dashboard view. It is reported by each GPU host, is not auditor-controlled, never becomes a `MeterSample`, and is ineligible for the production independent-meter trust gate. Helper telemetry is attack ground truth for the demonstration and is marked **ATTACK GROUND TRUTH — NOT NORMALLY AVAILABLE TO AUDITOR**.

No RunPod deployment, GPU calibration, external-meter characterization, or held-out physical validation has been performed in this repository. The included workload sizes are starting values only.

Hardware is now allocated as two matching, physically separated H100 80GB HBM3
Pods. Follow [H100_FIRST_RUN.md](H100_FIRST_RUN.md) to deploy, expose only ports
8101/8102, collect the first bank-free profiles, and retrieve their reports.

## Values needed from the two Pods

Create two matching Secure Cloud GPU Pods from the same official RunPod PyTorch image. Expose the declared-site Pod's TCP port `8101` as HTTP and the helper Pod's TCP port `8102` as HTTP. Supply:

- the declared-site proxy URL, `https://SITE_POD_ID-8101.proxy.runpod.net`;
- the helper proxy URL, `https://HELPER_POD_ID-8102.proxy.runpod.net`;
- confirmation that both `/health` responses report the CUDA backend;
- telemetry from two distinct GPU UUIDs with the same GPU model;
- six distinct random credentials of at least 24 characters: site worker/control, helper worker/control, laptop admin, and unused demo meter-ingestion credentials.

Generate URL-safe values on the laptop and assign one different output to each
credential field:

```powershell
1..6 | ForEach-Object { .venv\Scripts\python -c "import secrets; print(secrets.token_urlsafe(32))" }
```

The meter credential remains separate so host telemetry cannot enter the meter-ingestion endpoint. Nothing should post NVML samples to `/meter/samples`.

## Pod setup

Keep the checkout and state under persistent `/workspace` storage. On each Pod:

```bash
cd /workspace
git clone YOUR_REPOSITORY_URL Echo
cd Echo
bash deploy/runpod/bootstrap.sh
cp config/runpod.env.example .env.runpod
chmod 600 .env.runpod
```

Fill `.env.runpod` on both Pods. The helper needs `HELPER_WORKER_TOKEN` and `HELPER_CONTROL_TOKEN`. The site needs `SITE_WORKER_TOKEN`, `SITE_CONTROL_TOKEN`, `HELPER_WORKER_TOKEN`, and the real `HELPER_API_URL` so forwarding reaches the helper. Start the helper first:

```bash
cd /workspace/Echo
bash deploy/runpod/start_worker.sh helper
```

Then start the declared-site worker:

```bash
cd /workspace/Echo
bash deploy/runpod/start_worker.sh site
```

Both startup paths bind to `0.0.0.0`; authentication remains mandatory on health, task, control, and telemetry APIs. `nvidia-ml-py` is installed for direct NVML access. The collector falls back to a bounded `nvidia-smi` query if direct bindings fail.

RunPod's HTTP proxy currently has a 100-second request limit. The deployment
configuration therefore requires `DEADLINE_S` below 95 seconds; the supplied
starting value is 9 seconds.

## Laptop setup and readiness

Copy the example to the ignored environment file and fill the same URLs and credentials:

```powershell
Copy-Item config/runpod.env.example .env.runpod
.venv\Scripts\python -m compute_echo.cli deployment-readiness --env-file .env.runpod
.venv\Scripts\python -m compute_echo.cli deployment-readiness --env-file .env.runpod --probe
```

The static command lists every missing endpoint, secret, timing value, or workload file. `--probe` authenticates to both endpoints, validates the role-specific telemetry schemas, and checks for matching models and distinct UUIDs. It does not claim calibration or deployment readiness.

After the two endpoints pass the probe, profile both machines, tune the workload file, and only then generate a private single-use challenge bank. Until that happens, the dashboard service can start but epoch execution will correctly fail for lack of compatible unused challenges.

The eventual one-command laptop startup is:

```powershell
.\scripts\start_demo.ps1 -EnvFile .env.runpod
```

Open `http://127.0.0.1:8100/dashboard` and enter `ECHO_ADMIN_TOKEN`. Route buttons call the declared-site worker's authenticated `/control/route` endpoint. They do not modify telemetry arrays or detector output.

## Measurements deferred until Pods arrive

The live validation report must record:

1. observed poll arrival cadence and missed polls for each NVML stream, while stating that NVML does not provide an independent meter update counter;
2. host-telemetry response lag and smoothing behavior;
3. tensor versus idle power/utilization separation;
4. memory versus tensor separation and mixed-mode response;
5. auditor round-trip and complete worker-pipeline time for local and forwarded routes;
6. repeatability on unused, held-out randomized schedules.

Mode-level host telemetry remains descriptive. If held-out data does not reliably distinguish `TENSOR`, `MEMORY`, and `MIXED`, the later display should group them as `ACTIVE`. No GPU-class fingerprint is supported in this phase.

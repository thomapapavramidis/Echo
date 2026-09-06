# First H100 RunPod run

The supplied hardware inventory is two physically separated NVIDIA H100 80GB HBM3
GPUs with driver 570.195.03, host CUDA 12.8, PyTorch 2.4.1+cu124, MIG disabled,
and a 700 W limit. The stated idle readings (approximately 72 W facility and 76 W
helper) are operator observations, not repository-produced measurements.

This run gathers exploratory host profiles. It does not freeze a workload, create a
challenge bank, choose a washout, set calibration thresholds, or satisfy the
independent-meter trust gate.

## 1. Expose one HTTP port on each Pod

Run these commands from a trusted Linux terminal with a RunPod API key. Updating a
running Pod resets its container, so do this before installing into `/workspace`.

```bash
export RUNPOD_API_KEY='REPLACE_WITH_RUNPOD_API_KEY'
export FACILITY_POD_ID='REPLACE_WITH_FACILITY_POD_ID'
export HELPER_POD_ID='REPLACE_WITH_HELPER_POD_ID'

curl --fail-with-body --silent --show-error \
  --request PATCH \
  --url "https://rest.runpod.io/v1/pods/${FACILITY_POD_ID}" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
  --header 'Content-Type: application/json' \
  --data '{"ports":["8101/http"]}'

curl --fail-with-body --silent --show-error \
  --request PATCH \
  --url "https://rest.runpod.io/v1/pods/${HELPER_POD_ID}" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
  --header 'Content-Type: application/json' \
  --data '{"ports":["8102/http"]}'
```

After both Pods return to `RUNNING`, verify the API lists only the intended
application port:

```bash
curl --fail --silent \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
  "https://rest.runpod.io/v1/pods/${FACILITY_POD_ID}" | jq '.ports'

curl --fail --silent \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}" \
  "https://rest.runpod.io/v1/pods/${HELPER_POD_ID}" | jq '.ports'
```

Expected arrays are `["8101/http"]` and `["8102/http"]`.

## 2. Install on each Pod

Use each Pod's RunPod web terminal. The repository URL must be readable from the
Pod. Run the following on both Pods:

```bash
export REPO_URL='REPLACE_WITH_REPOSITORY_CLONE_URL'
export REPO_REF='REPLACE_WITH_BRANCH_OR_COMMIT'
cd /workspace
git clone --branch "$REPO_REF" --single-branch "$REPO_URL" Echo
cd /workspace/Echo
bash deploy/runpod/bootstrap.sh
mkdir -p /workspace/compute-echo-state
```

If `/workspace/Echo` already contains the intended checkout, omit `git clone` and
run `git status --short` plus `git rev-parse HEAD` before bootstrapping.

## 3. Create role-specific environment files

Generate four distinct URL-safe service secrets in a trusted terminal and transfer
them through a password manager or RunPod secret facility:

```bash
python3 - <<'PY'
import secrets
for name in ('SITE_WORKER_TOKEN', 'SITE_CONTROL_TOKEN',
             'HELPER_WORKER_TOKEN', 'HELPER_CONTROL_TOKEN'):
    print(f'{name}={secrets.token_urlsafe(32)}')
PY
```

On the helper Pod:

```bash
cd /workspace/Echo
umask 077
cat > .env.runpod <<'EOF'
HELPER_WORKER_TOKEN=REPLACE_WITH_HELPER_WORKER_TOKEN
HELPER_CONTROL_TOKEN=REPLACE_WITH_HELPER_CONTROL_TOKEN
EOF
```

On the facility Pod, substitute the actual helper Pod ID and the same helper worker
token used above:

```bash
cd /workspace/Echo
umask 077
cat > .env.runpod <<'EOF'
SITE_WORKER_TOKEN=REPLACE_WITH_SITE_WORKER_TOKEN
SITE_CONTROL_TOKEN=REPLACE_WITH_SITE_CONTROL_TOKEN
HELPER_WORKER_TOKEN=REPLACE_WITH_HELPER_WORKER_TOKEN
HELPER_API_URL=https://REPLACE_WITH_HELPER_POD_ID-8102.proxy.runpod.net
EOF
```

## 4. Run the first safe, bank-free profiles

Do this before starting either worker service so no second process competes for the
GPU. The initial specification file is exploratory and remains editable.

On the facility Pod:

```bash
cd /workspace/Echo
echo-audit doctor --cuda \
  --output /workspace/compute-echo-state/facility-doctor.json
echo-audit host-profile \
  --specs config/workloads.h100.initial-profile.json \
  --role declared_site \
  --repeats 3 \
  --poll-interval 0.1 \
  --baseline 2 \
  --cooldown 3 \
  --max-run 30 \
  --max-temperature 82 \
  --output /workspace/compute-echo-state/facility-initial-profile.json
```

On the helper Pod:

```bash
cd /workspace/Echo
echo-audit doctor --cuda \
  --output /workspace/compute-echo-state/helper-doctor.json
echo-audit host-profile \
  --specs config/workloads.h100.initial-profile.json \
  --role remote_helper \
  --repeats 3 \
  --poll-interval 0.1 \
  --baseline 2 \
  --cooldown 3 \
  --max-run 30 \
  --max-temperature 82 \
  --output /workspace/compute-echo-state/helper-initial-profile.json
```

Each report contains worker pipeline timings, raw role-labeled NVML snapshots,
observed polling cadence, availability failures, per-run power/utilization summaries,
and explicit `false` values for bank use, frozen specifications, calibration, and
production trust eligibility. The 82 C stop and 30-second cancellation bound are
operational protections, not calibration choices.

## 5. Start and connect the services

Start the helper first:

```bash
cd /workspace/Echo
nohup bash deploy/runpod/start_worker.sh helper \
  > /workspace/compute-echo-state/helper-service.log 2>&1 &
echo $! > /workspace/compute-echo-state/helper-service.pid
```

From the facility Pod, verify the helper before starting the facility service:

```bash
cd /workspace/Echo
set -a
source .env.runpod
set +a
curl --fail --silent --show-error \
  --header "Authorization: Bearer ${HELPER_WORKER_TOKEN}" \
  "${HELPER_API_URL}/health"
```

Then start the facility worker:

```bash
cd /workspace/Echo
nohup bash deploy/runpod/start_worker.sh site \
  > /workspace/compute-echo-state/facility-service.log 2>&1 &
echo $! > /workspace/compute-echo-state/facility-service.pid
```

The facility's authenticated routing endpoint switches real execution between its
local CUDA engine and `HELPER_API_URL`; it cannot modify either telemetry stream.

## 6. Verify and retrieve the reports

From the trusted Linux laptop terminal:

```bash
export SITE_API_URL="https://${FACILITY_POD_ID}-8101.proxy.runpod.net"
export HELPER_API_URL="https://${HELPER_POD_ID}-8102.proxy.runpod.net"
export SITE_WORKER_TOKEN='REPLACE_WITH_SITE_WORKER_TOKEN'
export SITE_CONTROL_TOKEN='REPLACE_WITH_SITE_CONTROL_TOKEN'
export HELPER_WORKER_TOKEN='REPLACE_WITH_HELPER_WORKER_TOKEN'

curl --fail --silent --show-error \
  --header "Authorization: Bearer ${SITE_WORKER_TOKEN}" \
  "${SITE_API_URL}/health" | jq
curl --fail --silent --show-error \
  --header "Authorization: Bearer ${HELPER_WORKER_TOKEN}" \
  "${HELPER_API_URL}/health" | jq

curl --fail --silent --show-error \
  --header "Authorization: Bearer ${SITE_WORKER_TOKEN}" \
  "${SITE_API_URL}/telemetry/nvml-demo" | jq
curl --fail --silent --show-error \
  --header "Authorization: Bearer ${HELPER_WORKER_TOKEN}" \
  "${HELPER_API_URL}/telemetry/nvml-demo" | jq

# Confirm the facility accepted its real helper configuration, then restore local.
curl --fail --silent --show-error --request POST \
  --header "Authorization: Bearer ${SITE_CONTROL_TOKEN}" \
  --header 'Content-Type: application/json' \
  --data '{"route":"forwarded"}' \
  "${SITE_API_URL}/control/route" | jq
curl --fail --silent --show-error --request POST \
  --header "Authorization: Bearer ${SITE_CONTROL_TOKEN}" \
  --header 'Content-Type: application/json' \
  --data '{"route":"local"}' \
  "${SITE_API_URL}/control/route" | jq

curl --fail --silent --show-error \
  --header "Authorization: Bearer ${SITE_WORKER_TOKEN}" \
  "${SITE_API_URL}/diagnostics/host-profile" \
  --output facility-initial-profile.json
curl --fail --silent --show-error \
  --header "Authorization: Bearer ${HELPER_WORKER_TOKEN}" \
  "${HELPER_API_URL}/diagnostics/host-profile" \
  --output helper-initial-profile.json
```

Do not run `bank-generate` yet. Return both profile JSON files before changing
`config/workloads.h100.initial-profile.json` or selecting any detector threshold.

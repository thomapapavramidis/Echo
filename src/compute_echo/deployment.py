"""RunPod demo configuration and readiness checks. No calibration occurs here."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx

from .models import Mode, WorkloadSpec
from .telemetry import NVML_DEMO_LABEL, HostGPUTelemetry, TelemetryRole

PLACEHOLDERS = {"", "CHANGE_ME", "POD_ID", "https://POD_ID-INTERNAL_PORT.proxy.runpod.net"}


def load_env_file(path: Path, *, override: bool = False) -> dict[str, str]:
    """Load a small KEY=VALUE file without executing it as shell code."""
    loaded = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise ValueError(f"{path}:{number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise ValueError(f"{path}:{number}: invalid environment key")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value in PLACEHOLDERS or "POD_ID" in value:
        raise ValueError(f"{name} is not configured")
    return value


def load_specs(path: Path, segment_s: float, deadline_s: float) -> list[WorkloadSpec]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for row in rows:
        result.append(
            WorkloadSpec.model_validate({**row, "segment_s": segment_s, "deadline_s": deadline_s})
        )
    if len(result) != 4 or {spec.mode for spec in result} != set(Mode):
        raise ValueError("workload file must contain exactly IDLE, TENSOR, MEMORY, and MIXED")
    return result


def runtime_config() -> dict:
    readiness = static_readiness()
    if readiness["still_required"]:
        raise ValueError("configuration incomplete: " + "; ".join(readiness["still_required"]))
    segment_s = float(os.environ.get("SEGMENT_DURATION_S", "10"))
    deadline_s = float(os.environ.get("DEADLINE_S", "9"))
    polling_s = float(os.environ.get("SENSOR_POLL_INTERVAL_S", "0.5"))
    repeats = int(os.environ.get("EPOCH_REPEATS", "4"))
    washout_s = float(os.environ.get("PROVISIONAL_WASHOUT_S", "3"))
    if not (0.1 <= polling_s <= 60):
        raise ValueError("SENSOR_POLL_INTERVAL_S must be 0.1..60")
    if deadline_s >= 95:
        raise ValueError("DEADLINE_S must be below 95 seconds for the RunPod HTTP proxy")
    if not (1 <= repeats <= 100):
        raise ValueError("EPOCH_REPEATS must be 1..100")
    if not (0 <= washout_s <= 120):
        raise ValueError("PROVISIONAL_WASHOUT_S must be 0..120")
    spec_path = Path(os.environ.get("WORKLOAD_SPECS", "config/workloads.runpod.example.json"))
    return {
        "site_url": _required("SITE_API_URL"),
        "helper_url": _required("HELPER_API_URL"),
        "site_worker_token": _required("SITE_WORKER_TOKEN"),
        "site_control_token": _required("SITE_CONTROL_TOKEN"),
        "helper_worker_token": _required("HELPER_WORKER_TOKEN"),
        "admin_token": _required("ECHO_ADMIN_TOKEN"),
        "meter_token": _required("ECHO_METER_TOKEN"),
        "private": Path(os.environ.get("ECHO_PRIVATE_DIR", "private/runpod-demo")),
        "runs": Path(os.environ.get("ECHO_RUNS_DIR", "runs/runpod-demo")),
        "specs": load_specs(spec_path, segment_s, deadline_s),
        "spec_path": spec_path,
        "polling_s": polling_s,
        "repeats": repeats,
        "washout_s": washout_s,
        "host": os.environ.get("AUDITOR_HOST", "127.0.0.1"),
        "port": int(os.environ.get("AUDITOR_PORT", "8100")),
    }


def _endpoint_issue(name: str, value: str, port: int) -> str | None:
    if not value or value in PLACEHOLDERS or "POD_ID" in value:
        return f"{name}: actual RunPod proxy URL for internal port {port}"
    pattern = rf"^https://[A-Za-z0-9-]+-{port}\.proxy\.runpod\.net/?$"
    if not re.fullmatch(pattern, value):
        return f"{name}: expected https://POD_ID-{port}.proxy.runpod.net"
    return None


def static_readiness(env: dict[str, str] | None = None) -> dict:
    values = dict(os.environ if env is None else env)
    required = []
    for name, port in (("SITE_API_URL", 8101), ("HELPER_API_URL", 8102)):
        issue = _endpoint_issue(name, values.get(name, ""), port)
        if issue:
            required.append(issue)
    if values.get("SITE_API_URL") and values.get("SITE_API_URL") == values.get("HELPER_API_URL"):
        required.append("SITE_API_URL and HELPER_API_URL must identify different Pods")
    token_names = [
        "SITE_WORKER_TOKEN",
        "SITE_CONTROL_TOKEN",
        "HELPER_WORKER_TOKEN",
        "HELPER_CONTROL_TOKEN",
        "ECHO_ADMIN_TOKEN",
        "ECHO_METER_TOKEN",
    ]
    tokens = []
    for name in token_names:
        value = values.get(name, "")
        if value in PLACEHOLDERS or len(value) < 24:
            required.append(f"{name}: distinct secret with at least 24 characters")
        elif value in tokens:
            required.append(f"{name}: secret must be distinct")
        tokens.append(value)
    spec_path = Path(values.get("WORKLOAD_SPECS", "config/workloads.runpod.example.json"))
    if not spec_path.exists():
        required.append(f"WORKLOAD_SPECS: existing path (currently {spec_path})")
    for name in ("SEGMENT_DURATION_S", "DEADLINE_S", "SENSOR_POLL_INTERVAL_S"):
        try:
            if float(values.get(name, "")) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            required.append(f"{name}: positive numeric value")
    try:
        segment_s = float(values.get("SEGMENT_DURATION_S", ""))
        deadline_s = float(values.get("DEADLINE_S", ""))
        if deadline_s > segment_s:
            required.append("DEADLINE_S must fit within SEGMENT_DURATION_S")
        if deadline_s >= 95:
            required.append("DEADLINE_S must be below 95 seconds for the RunPod HTTP proxy")
    except (TypeError, ValueError):
        pass
    return {
        "phase": "CONFIGURATION_ONLY_NO_CLOUD_VALIDATION",
        "ready_to_probe_pods": not required,
        "ready_for_demonstration": False,
        "still_required": required,
        "after_pods_are_connected": [
            "confirm both health endpoints report CUDA",
            "confirm two distinct GPU UUIDs and matching GPU models",
            "measure real NVML sample cadence, lag, and missing-sample behavior",
            "profile full-pipeline local and forwarded completion times",
            "tune workload sizes, then generate the private single-use challenge bank",
            "run fresh held-out schedules and report repeatability",
        ],
        "telemetry_provenance": NVML_DEMO_LABEL,
        "production_trust_gate_eligible": False,
    }


async def probe_readiness(config: dict) -> dict:
    checks = {}
    identities = []
    for name, role, url, token in (
        ("declared_site", TelemetryRole.DECLARED_SITE, config["site_url"], config["site_worker_token"]),
        ("remote_helper", TelemetryRole.REMOTE_HELPER, config["helper_url"], config["helper_worker_token"]),
    ):
        try:
            async with httpx.AsyncClient(
                base_url=url.rstrip("/"), headers={"Authorization": "Bearer " + token}, timeout=15
            ) as client:
                health_response = await client.get("/health")
                health_response.raise_for_status()
                health = health_response.json()
                response = await client.get("/telemetry/nvml-demo")
                response.raise_for_status()
                sample = HostGPUTelemetry.model_validate(response.json())
                if sample.role != role:
                    raise ValueError(f"reported role {sample.role}, expected {role}")
                if health.get("backend") != "cuda":
                    raise ValueError(f"reported backend {health.get('backend')}, expected cuda")
                if not sample.available or not sample.fresh:
                    raise ValueError("NVML_DEMO sample is unavailable or stale")
                identities.append((sample.gpu_uuid, sample.gpu_model))
                checks[name] = {"reachable": True, "health": health, "telemetry": sample.model_dump(mode="json")}
        except Exception as error:
            checks[name] = {"reachable": False, "error": f"{type(error).__name__}: {error}"}
    matching = len(identities) == 2 and identities[0][1] == identities[1][1]
    distinct = len(identities) == 2 and identities[0][0] != identities[1][0]
    endpoint_ready = all(x.get("reachable") for x in checks.values())
    return {
        "phase": "LIVE_ENDPOINT_PROBE",
        "checks": checks,
        "matching_gpu_models": matching,
        "distinct_gpu_uuids": distinct,
        "ready_for_workload_tuning": endpoint_ready and matching and distinct,
        "ready_for_demonstration": False,
        "reason": "workload tuning, private bank generation, and held-out validation remain required",
        "production_trust_gate_eligible": False,
    }

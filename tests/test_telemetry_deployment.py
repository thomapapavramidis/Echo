import inspect
import subprocess
import sys
import time
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from compute_echo.demo import describe_host_response
from compute_echo.deployment import static_readiness
from compute_echo.models import MeterSample
from compute_echo.telemetry import (
    NVML_DEMO_LABEL,
    HostGPUTelemetry,
    NVMLDemoSensor,
    TelemetryRole,
    parse_nvidia_smi_csv,
)
from compute_echo.worker import WorkerRuntime, create_worker_app

TOKEN = "test-worker-token-never-use-in-production"
CONTROL = "test-control-token-never-use-in-production"


def test_nvidia_smi_parser_accepts_one_complete_no_units_row():
    parsed = parse_nvidia_smi_csv("GPU-abc, NVIDIA A100-SXM4-80GB, 123.45, 97, 64\n")
    assert parsed == {
        "gpu_uuid": "GPU-abc",
        "gpu_model": "NVIDIA A100-SXM4-80GB",
        "board_or_module_power_draw_w": 123.45,
        "gpu_compute_utilization_pct": 97.0,
        "gpu_memory_utilization_pct": 64.0,
    }


@pytest.mark.parametrize(
    "row", ["GPU-a, model, N/A, 1, 2", "GPU-a, model, 3, 4", "GPU-a,model,3,4,5\nGPU-b,model,3,4,5"]
)
def test_nvidia_smi_parser_rejects_missing_or_ambiguous_data(row):
    with pytest.raises(ValueError):
        parse_nvidia_smi_csv(row)


def test_direct_nvml_is_preferred_and_provenance_cannot_be_a_meter(monkeypatch):
    calls = []
    fake = SimpleNamespace(
        nvmlInit=lambda: calls.append("init"),
        nvmlShutdown=lambda: calls.append("shutdown"),
        nvmlDeviceGetHandleByIndex=lambda index: ("handle", index),
        nvmlDeviceGetUUID=lambda handle: b"GPU-direct",
        nvmlDeviceGetName=lambda handle: b"matching-model",
        nvmlDeviceGetPowerUsage=lambda handle: 212500,
        nvmlDeviceGetUtilizationRates=lambda handle: SimpleNamespace(gpu=88, memory=44),
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    sensor = NVMLDemoSensor(TelemetryRole.DECLARED_SITE)
    sample = sensor.sample()
    sensor.close()
    assert sample.collector_backend == "pynvml"
    assert sample.board_or_module_power_draw_w == 212.5
    assert sample.provenance_label == NVML_DEMO_LABEL
    assert sample.independent_physical_evidence is False
    assert sample.production_trust_gate_eligible is False
    assert calls == ["init", "shutdown"]
    with pytest.raises(ValidationError):
        MeterSample.model_validate(sample.model_dump())


def test_nvidia_smi_fallback_and_unavailable_are_explicit(monkeypatch):
    sensor = NVMLDemoSensor(TelemetryRole.REMOTE_HELPER)
    sensor._nvml = None
    sensor._nvml_error = "test direct failure"
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="GPU-helper, matching-model, 90, 55, 25\n", stderr=""
        ),
    )
    sample = sensor.sample()
    assert sample.collector_backend == "nvidia-smi"
    assert sample.helper_attack_ground_truth
    assert not sample.normal_auditor_visibility
    assert "NOT NORMALLY AVAILABLE" in sample.visibility_label

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    missing = sensor.sample()
    assert not missing.available and not missing.fresh
    assert missing.collector_backend == "unavailable"
    assert missing.board_or_module_power_draw_w is None


async def test_worker_nvml_endpoint_is_authenticated_and_route_control_is_real(tmp_path):
    class Sensor:
        def sample(self):
            return HostGPUTelemetry(
                role=TelemetryRole.DECLARED_SITE,
                sampled_at_ns=time.time_ns(),
                sequence=0,
                gpu_index=0,
                gpu_uuid="GPU-site",
                gpu_model="matching-model",
                board_or_module_power_draw_w=100,
                gpu_compute_utilization_pct=0,
                gpu_memory_utilization_pct=0,
                available=True,
                fresh=True,
                freshness_basis="successful_direct_nvml_query_at_timestamp",
                collector_backend="pynvml",
                helper_attack_ground_truth=False,
                normal_auditor_visibility=True,
                visibility_label="DECLARED-SITE HOST DIAGNOSTIC",
            )

        def close(self):
            pass

    class Helper:
        async def quiesce(self):
            pass

        async def close(self):
            pass

    profile_path = tmp_path / "profile.json"
    profile_path.write_text('{"production_trust_gate_eligible":false}')
    runtime = WorkerRuntime(
        "cpu",
        tmp_path / "worker.sqlite3",
        Helper(),
        telemetry=Sensor(),
        profile_path=profile_path,
    )
    app = create_worker_app(runtime, TOKEN, CONTROL)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://worker") as client:
        assert (await client.get("/telemetry/nvml-demo")).status_code == 401
        response = await client.get(
            "/telemetry/nvml-demo", headers={"Authorization": "Bearer " + TOKEN}
        )
        assert response.status_code == 200
        telemetry_before = response.json()
        changed = await client.post(
            "/control/route",
            json={"route": "forwarded"},
            headers={"Authorization": "Bearer " + CONTROL},
        )
        assert changed.json() == {"route": "forwarded"}
        assert runtime.route == "forwarded"
        telemetry_after = (
            await client.get("/telemetry/nvml-demo", headers={"Authorization": "Bearer " + TOKEN})
        ).json()
        assert telemetry_after["board_or_module_power_draw_w"] == telemetry_before[
            "board_or_module_power_draw_w"
        ]
        profile = await client.get(
            "/diagnostics/host-profile", headers={"Authorization": "Bearer " + TOKEN}
        )
        assert profile.status_code == 200
        assert profile.json()["production_trust_gate_eligible"] is False
    await runtime.close()


def test_host_description_is_route_blind_and_never_passes_production_gate():
    assert "route" not in inspect.signature(describe_host_response).parameters
    result = describe_host_response([], None)
    assert result["production_physical_response"] == "INSUFFICIENT"
    assert result["production_trust_gate_eligible"] is False


def test_readiness_names_every_missing_pod_value(tmp_path):
    report = static_readiness(
        {
            "WORKLOAD_SPECS": str(tmp_path / "missing.json"),
            "SEGMENT_DURATION_S": "10",
            "DEADLINE_S": "9",
            "SENSOR_POLL_INTERVAL_S": "0.5",
        }
    )
    missing = "\n".join(report["still_required"])
    for name in (
        "SITE_API_URL",
        "HELPER_API_URL",
        "SITE_WORKER_TOKEN",
        "SITE_CONTROL_TOKEN",
        "HELPER_WORKER_TOKEN",
        "HELPER_CONTROL_TOKEN",
        "ECHO_ADMIN_TOKEN",
        "ECHO_METER_TOKEN",
        "WORKLOAD_SPECS",
    ):
        assert name in missing
    assert report["ready_for_demonstration"] is False

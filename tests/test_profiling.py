import time

from compute_echo.models import Mode, PipelineTimes, WorkloadSpec
from compute_echo.profiling import run_host_profile
from compute_echo.telemetry import HostGPUTelemetry, TelemetryRole


class FakeSensor:
    def __init__(self):
        self.sequence = 0

    def sample(self):
        sample = HostGPUTelemetry(
            role=TelemetryRole.DECLARED_SITE,
            sampled_at_ns=time.time_ns(),
            sequence=self.sequence,
            gpu_index=0,
            gpu_uuid="GPU-profile",
            gpu_model="H100 fixture",
            board_or_module_power_draw_w=80 + self.sequence,
            gpu_compute_utilization_pct=10,
            gpu_memory_utilization_pct=5,
            available=True,
            fresh=True,
            freshness_basis="successful_direct_nvml_query_at_timestamp",
            collector_backend="pynvml",
            helper_attack_ground_truth=False,
            normal_auditor_visibility=True,
            visibility_label="DECLARED-SITE HOST DIAGNOSTIC",
        )
        self.sequence += 1
        return sample


class FakeEngine:
    identity = "cuda:H100 fixture"

    def execute(self, seed, spec, cancel=None):
        time.sleep(0.025)
        return "ab" * 32, PipelineTimes(execution_s=0.01, total_worker_s=0.025)


def test_host_profile_is_bank_free_non_calibrating_and_records_all_modes(tmp_path):
    specifications = [
        WorkloadSpec(
            mode=mode,
            m=16,
            n=16,
            k=16,
            lanes=1,
            words_per_lane=16,
            memory_steps=1,
            segment_s=0.03,
            deadline_s=0.02,
        )
        for mode in Mode
    ]
    output = tmp_path / "profile.json"
    report = run_host_profile(
        specifications,
        TelemetryRole.DECLARED_SITE,
        output,
        repeats=1,
        poll_interval_s=0.02,
        baseline_s=0,
        cooldown_s=0,
        max_run_s=1,
        sensor=FakeSensor(),
        engine=FakeEngine(),
    )
    assert output.exists()
    assert {record["mode"] for record in report["records"]} == {mode.value for mode in Mode}
    assert report["challenge_bank_used"] is False
    assert report["workload_specifications_frozen"] is False
    assert report["calibration_thresholds_selected"] is False
    assert report["independent_physical_evidence"] is False
    assert report["production_trust_gate_eligible"] is False
    assert report["raw_host_telemetry"]
    assert all("seed" not in record for record in report["records"])

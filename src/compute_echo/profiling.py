"""Bank-free workload profiling with explicitly non-independent host telemetry."""

from __future__ import annotations

import hashlib
import secrets
import statistics
import threading
import time
from pathlib import Path

from .models import Mode, WorkloadSpec
from .safety import GPUSafetyGuard
from .storage import write_json
from .telemetry import NVML_DEMO_LABEL, NVMLDemoSensor, TelemetryRole
from .workloads import CancelledWork, WorkloadEngine


def _summary(samples, start_ns: int, end_ns: int) -> dict:
    selected = [
        sample
        for sample in samples
        if start_ns <= sample.sampled_at_ns < end_ns and sample.available
    ]
    power = [sample.board_or_module_power_draw_w for sample in selected]
    compute = [sample.gpu_compute_utilization_pct for sample in selected]
    memory = [sample.gpu_memory_utilization_pct for sample in selected]
    return {
        "samples": len(selected),
        "power_w_median": statistics.median(power) if power else None,
        "power_w_min": min(power) if power else None,
        "power_w_max": max(power) if power else None,
        "compute_utilization_pct_median": statistics.median(compute) if compute else None,
        "compute_utilization_pct_max": max(compute) if compute else None,
        "memory_utilization_pct_median": statistics.median(memory) if memory else None,
        "memory_utilization_pct_max": max(memory) if memory else None,
    }


def run_host_profile(
    specifications: list[WorkloadSpec],
    role: TelemetryRole,
    output: Path,
    *,
    repeats: int = 3,
    poll_interval_s: float = 0.1,
    baseline_s: float = 2,
    cooldown_s: float = 3,
    max_run_s: float = 30,
    max_temperature_c: float = 82,
    sensor: NVMLDemoSensor | None = None,
    engine: WorkloadEngine | None = None,
) -> dict:
    """Run exploratory local profiles without generating or consuming bank entries."""
    if len(specifications) != 4 or {spec.mode for spec in specifications} != set(Mode):
        raise ValueError("profile requires exactly one specification for every workload mode")
    if not 1 <= repeats <= 20:
        raise ValueError("profile repeats must be 1..20")
    if not 0.02 <= poll_interval_s <= 5:
        raise ValueError("profile poll interval must be 0.02..5 seconds")
    if not 0 <= baseline_s <= 30 or not 0 <= cooldown_s <= 60:
        raise ValueError("invalid baseline/cooldown duration")
    if not 1 <= max_run_s <= 120:
        raise ValueError("max run time must be 1..120 seconds")

    owns_sensor = sensor is None
    sensor = sensor or NVMLDemoSensor(role)
    engine = engine or WorkloadEngine("cuda", GPUSafetyGuard(max_temperature_c))
    samples = []
    stop_sampling = threading.Event()

    def collect():
        while not stop_sampling.is_set():
            samples.append(sensor.sample())
            stop_sampling.wait(poll_interval_s)

    sampler = threading.Thread(target=collect, name="echo-profile-nvml", daemon=True)
    sampler.start()
    started_at_ns = time.time_ns()
    records = []
    fatal_error = None
    try:
        time.sleep(baseline_s)
        for replicate in range(repeats):
            for spec in specifications:
                seed = secrets.token_bytes(32)
                run_start_ns = time.time_ns()
                run_start_mono = time.monotonic()
                digest = None
                timings = None
                error = None
                timer = None
                try:
                    if spec.mode == Mode.IDLE:
                        # The protocol's IDLE digest is immediate; profiling supplies a
                        # real observation window without pretending it is GPU work.
                        time.sleep(spec.segment_s)
                        digest, timings = engine.execute(seed, spec)
                    else:
                        cancel = threading.Event()
                        timer = threading.Timer(max_run_s, cancel.set)
                        timer.start()
                        digest, timings = engine.execute(seed, spec, cancel)
                except CancelledWork:
                    error = "PROFILE_MAX_RUN_EXCEEDED"
                except Exception as failure:
                    error = f"{type(failure).__name__}: {failure}"[:500]
                    fatal_error = error
                finally:
                    if timer:
                        timer.cancel()
                run_end_ns = time.time_ns()
                records.append(
                    {
                        "replicate": replicate,
                        "mode": spec.mode.value,
                        "spec": spec.model_dump(mode="json"),
                        "spec_key": spec.key,
                        "seed_sha256": hashlib.sha256(seed).hexdigest(),
                        "digest": digest,
                        "started_at_ns": run_start_ns,
                        "ended_at_ns": run_end_ns,
                        "wall_s": time.monotonic() - run_start_mono,
                        "worker_pipeline": timings.model_dump(mode="json") if timings else None,
                        "local_pipeline_meets_configured_deadline": bool(
                            timings and timings.total_worker_s < spec.deadline_s
                        ),
                        "error": error,
                    }
                )
                time.sleep(cooldown_s)
                if fatal_error:
                    break
            if fatal_error:
                break
    finally:
        stop_sampling.set()
        sampler.join(timeout=max(2, poll_interval_s * 2))
        if owns_sensor:
            sensor.close()
    ended_at_ns = time.time_ns()
    for record in records:
        record["host_telemetry_summary"] = _summary(
            samples, record["started_at_ns"], record["ended_at_ns"]
        )
    available = [sample for sample in samples if sample.available]
    intervals = [
        (right.sampled_at_ns - left.sampled_at_ns) / 1e9
        for left, right in zip(available, available[1:], strict=False)
    ]
    report = {
        "kind": "exploratory-host-gpu-profile-v1",
        "role": role.value,
        "backend": engine.identity,
        "started_at_ns": started_at_ns,
        "ended_at_ns": ended_at_ns,
        "telemetry_provenance": NVML_DEMO_LABEL,
        "independent_physical_evidence": False,
        "production_trust_gate_eligible": False,
        "cloud_route_measured": False,
        "challenge_bank_used": False,
        "workload_specifications_frozen": False,
        "calibration_thresholds_selected": False,
        "requested_poll_interval_s": poll_interval_s,
        "observed_host_snapshot_interval_s": statistics.median(intervals) if intervals else None,
        "true_independent_meter_cadence_s": None,
        "telemetry_available_samples": len(available),
        "telemetry_unavailable_samples": len(samples) - len(available),
        "fatal_error": fatal_error,
        "records": records,
        "raw_host_telemetry": [sample.model_dump(mode="json") for sample in samples],
    }
    write_json(output, report)
    return report

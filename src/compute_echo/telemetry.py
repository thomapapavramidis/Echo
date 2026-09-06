"""Host GPU telemetry for the proof-of-concept dashboard.

NVML_DEMO is operator-provided host telemetry. It is never converted to MeterSample
and cannot enter the production physical-evidence detector or calibration pipeline.
"""

from __future__ import annotations

import asyncio
import csv
import os
import statistics
import subprocess
import time
from collections import deque
from enum import StrEnum
from pathlib import Path
from typing import Literal

import httpx
from pydantic import Field, model_validator

from .models import StrictModel
from .storage import EventLog

NVML_DEMO_LABEL = "HOST GPU TELEMETRY \u2014 PROOF OF CONCEPT"


class TelemetryRole(StrEnum):
    DECLARED_SITE = "declared_site"
    REMOTE_HELPER = "remote_helper"


class HostGPUTelemetry(StrictModel):
    schema_version: Literal["nvml-demo-v1"] = "nvml-demo-v1"
    source: Literal["NVML_DEMO"] = "NVML_DEMO"
    provenance_label: Literal["HOST GPU TELEMETRY \u2014 PROOF OF CONCEPT"] = NVML_DEMO_LABEL
    role: TelemetryRole
    sampled_at_ns: int = Field(gt=0)
    sequence: int = Field(ge=0)
    gpu_index: int = Field(ge=0)
    gpu_uuid: str | None = None
    gpu_model: str | None = None
    board_or_module_power_draw_w: float | None = Field(default=None, ge=0, le=100_000)
    gpu_compute_utilization_pct: float | None = Field(default=None, ge=0, le=100)
    gpu_memory_utilization_pct: float | None = Field(default=None, ge=0, le=100)
    available: bool
    fresh: bool
    freshness_basis: Literal[
        "successful_direct_nvml_query_at_timestamp",
        "successful_nvidia_smi_snapshot_at_timestamp",
        "unavailable",
    ]
    collector_backend: Literal["pynvml", "nvidia-smi", "unavailable"]
    error: str | None = Field(default=None, max_length=500)
    independent_physical_evidence: Literal[False] = False
    production_trust_gate_eligible: Literal[False] = False
    helper_attack_ground_truth: bool
    normal_auditor_visibility: bool
    visibility_label: Literal[
        "DECLARED-SITE HOST DIAGNOSTIC",
        "ATTACK GROUND TRUTH \u2014 NOT NORMALLY AVAILABLE TO AUDITOR",
    ]

    @model_validator(mode="after")
    def enforce_role_labels(self):
        helper = self.role == TelemetryRole.REMOTE_HELPER
        if self.helper_attack_ground_truth != helper or self.normal_auditor_visibility == helper:
            raise ValueError("telemetry visibility fields do not match role")
        expected = (
            "ATTACK GROUND TRUTH \u2014 NOT NORMALLY AVAILABLE TO AUDITOR"
            if helper
            else "DECLARED-SITE HOST DIAGNOSTIC"
        )
        if self.visibility_label != expected:
            raise ValueError("telemetry visibility label does not match role")
        metrics = (
            self.gpu_uuid,
            self.gpu_model,
            self.board_or_module_power_draw_w,
            self.gpu_compute_utilization_pct,
            self.gpu_memory_utilization_pct,
        )
        if self.available and (not self.fresh or any(value is None for value in metrics)):
            raise ValueError("available sample must be fresh and complete")
        if not self.available and self.fresh:
            raise ValueError("unavailable sample cannot be fresh")
        return self


def _decode(value):
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def _number(value: str) -> float:
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"n/a", "[n/a]", "not supported", "[not supported]"}:
        raise ValueError(f"NVML metric unavailable: {cleaned or 'empty'}")
    return float(cleaned)


def parse_nvidia_smi_csv(output: str) -> dict:
    """Parse one no-units CSV row without assuming whitespace or locale formatting."""
    rows = list(csv.reader(line for line in output.splitlines() if line.strip()))
    if len(rows) != 1 or len(rows[0]) != 5:
        raise ValueError("expected one GPU row with five nvidia-smi fields")
    uuid, model, power, compute, memory = (value.strip() for value in rows[0])
    if not uuid or not model:
        raise ValueError("GPU identity unavailable")
    return {
        "gpu_uuid": uuid,
        "gpu_model": model,
        "board_or_module_power_draw_w": _number(power),
        "gpu_compute_utilization_pct": _number(compute),
        "gpu_memory_utilization_pct": _number(memory),
    }


class NVMLDemoSensor:
    """Prefer direct pynvml; fall back to one bounded nvidia-smi snapshot."""

    def __init__(self, role: TelemetryRole, gpu_index: int = 0, allow_smi_fallback: bool = True):
        self.role = role
        self.gpu_index = gpu_index
        self.allow_smi_fallback = allow_smi_fallback
        self.sequence = 0
        self._nvml = None
        self._handle = None
        self._nvml_error = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self._nvml = pynvml
        except Exception as error:  # optional dependency, driver, or permissions
            self._nvml_error = f"{type(error).__name__}: {error}"

    def _role_fields(self):
        helper = self.role == TelemetryRole.REMOTE_HELPER
        return {
            "helper_attack_ground_truth": helper,
            "normal_auditor_visibility": not helper,
            "visibility_label": (
                "ATTACK GROUND TRUTH \u2014 NOT NORMALLY AVAILABLE TO AUDITOR"
                if helper
                else "DECLARED-SITE HOST DIAGNOSTIC"
            ),
        }

    def _direct(self):
        utilization = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
        return {
            "gpu_uuid": _decode(self._nvml.nvmlDeviceGetUUID(self._handle)),
            "gpu_model": _decode(self._nvml.nvmlDeviceGetName(self._handle)),
            # NVML describes this as device power usage. Depending on hardware it is
            # board or module scope, so the API avoids claiming a narrower boundary.
            "board_or_module_power_draw_w": self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0,
            "gpu_compute_utilization_pct": float(utilization.gpu),
            "gpu_memory_utilization_pct": float(utilization.memory),
        }

    def _smi(self):
        command = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=uuid,name,power.draw,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ]
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if process.returncode:
            raise RuntimeError(process.stderr.strip() or f"nvidia-smi exited {process.returncode}")
        return parse_nvidia_smi_csv(process.stdout)

    def sample(self) -> HostGPUTelemetry:
        sampled_at_ns = time.time_ns()
        sequence = self.sequence
        self.sequence += 1
        errors = []
        if self._nvml is not None:
            try:
                return HostGPUTelemetry(
                    role=self.role,
                    sampled_at_ns=sampled_at_ns,
                    sequence=sequence,
                    gpu_index=self.gpu_index,
                    available=True,
                    fresh=True,
                    freshness_basis="successful_direct_nvml_query_at_timestamp",
                    collector_backend="pynvml",
                    **self._direct(),
                    **self._role_fields(),
                )
            except Exception as error:
                errors.append(f"pynvml {type(error).__name__}: {error}")
        elif self._nvml_error:
            errors.append("pynvml unavailable: " + self._nvml_error)
        if self.allow_smi_fallback:
            try:
                return HostGPUTelemetry(
                    role=self.role,
                    sampled_at_ns=sampled_at_ns,
                    sequence=sequence,
                    gpu_index=self.gpu_index,
                    available=True,
                    fresh=True,
                    freshness_basis="successful_nvidia_smi_snapshot_at_timestamp",
                    collector_backend="nvidia-smi",
                    **self._smi(),
                    **self._role_fields(),
                )
            except Exception as error:
                errors.append(f"nvidia-smi {type(error).__name__}: {error}")
        return HostGPUTelemetry(
            role=self.role,
            sampled_at_ns=sampled_at_ns,
            sequence=sequence,
            gpu_index=self.gpu_index,
            available=False,
            fresh=False,
            freshness_basis="unavailable",
            collector_backend="unavailable",
            error="; ".join(errors)[:500] or "no telemetry collector available",
            **self._role_fields(),
        )

    def close(self):
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml = None


class TelemetryObservation(StrictModel):
    received_at_ns: int = Field(gt=0)
    endpoint_reachable: bool
    sample: HostGPUTelemetry | None = None
    error: str | None = Field(default=None, max_length=500)


class RemoteTelemetryStream:
    def __init__(
        self,
        role: TelemetryRole,
        url: str,
        token: str,
        interval_s: float,
        log_path: Path | None = None,
        max_samples: int = 1200,
    ):
        if interval_s < 0.1 or interval_s > 60:
            raise ValueError("telemetry polling interval must be 0.1..60 seconds")
        self.role, self.interval_s = role, interval_s
        self.samples = deque(maxlen=max_samples)
        self.client = httpx.AsyncClient(
            base_url=url.rstrip("/"),
            timeout=max(2.0, min(10.0, interval_s * 2)),
            headers={"Authorization": "Bearer " + token},
            trust_env=False,
        )
        self.log = EventLog(log_path) if log_path else None
        self.task = None

    async def poll_once(self):
        try:
            response = await self.client.get("/telemetry/nvml-demo")
            response.raise_for_status()
        except Exception as error:
            observation = TelemetryObservation(
                received_at_ns=time.time_ns(),
                endpoint_reachable=False,
                error=f"{type(error).__name__}: {error}"[:500],
            )
        else:
            try:
                sample = HostGPUTelemetry.model_validate(response.json())
                if sample.role != self.role:
                    raise ValueError(f"expected {self.role}, received {sample.role}")
                observation = TelemetryObservation(
                    received_at_ns=time.time_ns(), endpoint_reachable=True, sample=sample
                )
            except Exception as error:
                observation = TelemetryObservation(
                    received_at_ns=time.time_ns(),
                    endpoint_reachable=True,
                    error=f"invalid telemetry: {type(error).__name__}: {error}"[:500],
                )
        self.samples.append(observation)
        if self.log:
            self.log.append(observation)
        return observation

    async def run(self):
        while True:
            started = time.monotonic()
            await self.poll_once()
            await asyncio.sleep(max(0.0, self.interval_s - (time.monotonic() - started)))

    def start(self):
        if self.task is None:
            self.task = asyncio.create_task(self.run())

    async def close(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self.client.aclose()

    def snapshot(self):
        return [sample.model_dump(mode="json") for sample in self.samples]


class DemoTelemetryHub:
    def __init__(self, site: RemoteTelemetryStream, helper: RemoteTelemetryStream):
        if site.role != TelemetryRole.DECLARED_SITE or helper.role != TelemetryRole.REMOTE_HELPER:
            raise ValueError("telemetry hub requires declared-site and remote-helper streams")
        self.site, self.helper = site, helper

    def start(self):
        self.site.start()
        self.helper.start()

    async def close(self):
        await asyncio.gather(self.site.close(), self.helper.close())

    def snapshot(self):
        def summary(stream):
            rows = list(stream.samples)
            received = [row.received_at_ns for row in rows]
            successful = [row.sample for row in rows if row.sample and row.sample.available]
            intervals = [(b - a) / 1e9 for a, b in zip(received, received[1:], strict=False)]
            sample_intervals = [
                (b.sampled_at_ns - a.sampled_at_ns) / 1e9
                for a, b in zip(successful, successful[1:], strict=False)
            ]
            return {
                "polls": len(rows),
                "available_samples": len(successful),
                "unavailable_or_failed_polls": len(rows) - len(successful),
                "median_observed_poll_arrival_s": statistics.median(intervals) if intervals else None,
                "median_host_snapshot_interval_s": statistics.median(sample_intervals)
                if sample_intervals
                else None,
                "independent_meter_update_cadence": "NOT_AVAILABLE_FROM_NVML_DEMO",
            }

        return {
            "provenance_label": NVML_DEMO_LABEL,
            "production_trust_gate_eligible": False,
            "declared_site": self.site.snapshot(),
            "remote_helper": self.helper.snapshot(),
            "stream_summaries": {"declared_site": summary(self.site), "remote_helper": summary(self.helper)},
            "helper_disclosure": "ATTACK GROUND TRUTH \u2014 NOT NORMALLY AVAILABLE TO AUDITOR",
        }

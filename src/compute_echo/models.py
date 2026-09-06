"""Strict wire contracts. No routing ground truth belongs in evidence models."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def canonical(value: object) -> bytes:
    """Protocol JSON: UTF-8, sorted keys, compact separators, finite numbers only."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Mode(StrEnum):
    IDLE = "IDLE"
    TENSOR = "TENSOR"
    MEMORY = "MEMORY"
    MIXED = "MIXED"


class Outcome(StrEnum):
    PENDING = "PENDING"
    PASS = "PASS"
    DEADLINE_FAILED = "DEADLINE_FAILED"
    INCORRECT = "INCORRECT"
    INTERRUPTED = "INTERRUPTED"
    ERROR = "ERROR"


class WorkloadSpec(StrictModel):
    protocol: Literal["echo-int-v1"] = "echo-int-v1"
    mode: Mode
    operations: int = Field(default=1, ge=1, le=4096)
    m: int = Field(default=128, ge=16, le=8192, multiple_of=16)
    n: int = Field(default=128, ge=16, le=8192, multiple_of=16)
    k: int = Field(default=256, ge=16, le=32768, multiple_of=16)
    lanes: int = Field(default=64, ge=1, le=16384)
    words_per_lane: int = Field(default=1024, ge=16, le=65536)
    memory_steps: int = Field(default=128, ge=1, le=65536)
    segment_s: float = Field(default=5.0, ge=0.01, le=120)
    deadline_s: float = Field(default=4.0, ge=0.005, le=120)

    @model_validator(mode="after")
    def bounded(self):
        if self.words_per_lane & (self.words_per_lane - 1):
            raise ValueError("words_per_lane must be a power of two")
        if self.lanes * self.words_per_lane > 64 * 1024 * 1024:
            raise ValueError("memory buffer exceeds 256 MiB")
        if (self.m * self.k + self.k * self.n + 4 * self.m * self.n) > 512 * 1024 * 1024:
            raise ValueError("tensor working set exceeds 512 MiB")
        if self.deadline_s > self.segment_s:
            raise ValueError("deadline must fit inside segment")
        # Inputs are [-8,7]; every partial sum has absolute value <= 64*k.
        if 64 * self.k >= 2**31:
            raise ValueError("INT32 overflow bound exceeded")
        return self

    @property
    def key(self) -> str:
        return hashlib.sha256(canonical(self)).hexdigest()


class Challenge(StrictModel):
    challenge_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    epoch_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    seed: str = Field(pattern=r"^[a-f0-9]{64}$")
    spec: WorkloadSpec


class PipelineTimes(StrictModel):
    safety_check_s: float = Field(default=0, ge=0)
    input_generation_s: float = Field(default=0, ge=0)
    host_to_device_s: float = Field(default=0, ge=0)
    execution_s: float = Field(default=0, ge=0)
    device_to_host_s: float = Field(default=0, ge=0)
    aggregation_s: float = Field(default=0, ge=0)
    hashing_s: float = Field(default=0, ge=0)
    total_worker_s: float = Field(default=0, ge=0)


class WorkerResult(StrictModel):
    challenge_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    epoch_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    backend: str = Field(max_length=160)
    operations_completed: int = Field(ge=0)
    timings: PipelineTimes
    timing_provenance: Literal["worker_reported_diagnostic"] = "worker_reported_diagnostic"


class MeterSample(StrictModel):
    meter_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    sequence: int = Field(ge=0)
    acquired_at_ns: int = Field(gt=0)
    watts: float = Field(ge=0, le=1_000_000)
    device_sequence: int | None = Field(default=None, ge=0)
    device_time_ns: int | None = Field(default=None, gt=0)
    quality: Literal["GOOD", "MISSING", "INVALID"] = "GOOD"


class Segment(StrictModel):
    challenge_id: str
    mode: Mode
    spec_key: str
    start_ns: int
    end_ns: int
    outcome: Outcome
    digest_match: bool | None = None
    deadline_met: bool = False
    nominal_integer_operations: int = Field(default=0, ge=0)
    dependent_memory_updates: int = Field(default=0, ge=0)
    elapsed_s: float | None = None
    timings: PipelineTimes | None = None


class EpochEvidence(StrictModel):
    epoch_id: str
    started_at_ns: int
    ended_at_ns: int
    meter_id: str
    provenance: Literal["external", "simulation", "host_telemetry"]
    measurement_intent: Literal["physical_audit", "timing_only"] = "physical_audit"
    source_freshness_verified: bool
    segments: list[Segment]
    samples: list[MeterSample]
    complete: bool
    stop_reason: str | None = None

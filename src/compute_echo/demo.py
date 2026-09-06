"""Laptop-side state for the explicitly non-independent NVML proof of concept."""

from __future__ import annotations

from dataclasses import dataclass

from .models import EpochEvidence, Mode, Outcome, WorkloadSpec
from .telemetry import NVML_DEMO_LABEL, DemoTelemetryHub


def describe_host_response(observations: list[dict], epoch: EpochEvidence | None) -> dict:
    """Route-blind descriptive statistics. Never returns a production physical verdict."""
    available = [o["sample"] for o in observations if o.get("sample") and o["sample"]["available"]]
    result = {
        "result": "NO_HOST_TELEMETRY" if not available else "UNCALIBRATED_HOST_TELEMETRY",
        "production_physical_response": "INSUFFICIENT",
        "production_trust_gate_eligible": False,
        "provenance_label": NVML_DEMO_LABEL,
        "claim": "descriptive host diagnostic only; no independent physical corroboration",
        "mode_means": {},
    }
    if not epoch or not available:
        return result
    for mode in Mode:
        values = []
        for segment in epoch.segments:
            if segment.mode != mode:
                continue
            values.extend(
                sample["board_or_module_power_draw_w"]
                for sample in available
                if segment.start_ns <= sample["sampled_at_ns"] < segment.end_ns
            )
        if values:
            result["mode_means"][mode.value] = sum(values) / len(values)
    if {"IDLE", "TENSOR"} <= result["mode_means"].keys():
        result["tensor_minus_idle_w"] = (
            result["mode_means"]["TENSOR"] - result["mode_means"]["IDLE"]
        )
    return result


@dataclass
class DemoRuntime:
    telemetry: DemoTelemetryHub
    specs: list[WorkloadSpec]
    repeats: int
    washout_s: float
    current_route: str = "unknown"
    last_evidence: EpochEvidence | None = None

    def state(self, runner) -> dict:
        snapshot = self.telemetry.snapshot()
        finished = runner.status.get("state") in {"COMPLETE", "INTERRUPTED"}
        same_epoch = self.last_evidence and runner.status.get("epoch_id") == self.last_evidence.epoch_id
        evidence = self.last_evidence if finished and same_epoch else None
        outcomes = [] if evidence is None else [segment.outcome.value for segment in evidence.segments]
        return {
            "phase": "RUNPOD_CONFIGURATION_READY_NO_DEPLOYMENT_CLAIM",
            "route": self.current_route,
            "runner": runner.status,
            "expected_challenge_sequence": runner.expected_sequence,
            "telemetry": snapshot,
            "computational_correctness": (
                "INCORRECT"
                if Outcome.INCORRECT.value in outcomes
                else ("PASS" if outcomes and all(x == Outcome.PASS.value for x in outcomes) else "PENDING")
            ),
            "deadline_status": (
                "DEADLINE_FAILED"
                if Outcome.DEADLINE_FAILED.value in outcomes
                else ("MEETS_TARGET" if outcomes and evidence and all(s.deadline_met for s in evidence.segments) else "PENDING")
            ),
            "computational_segments": []
            if evidence is None
            else [
                {
                    "mode": segment.mode.value,
                    "outcome": segment.outcome.value,
                    "digest_match": segment.digest_match,
                    "deadline_met": segment.deadline_met,
                    "auditor_elapsed_s": segment.elapsed_s,
                    "worker_pipeline_s": segment.timings.model_dump(mode="json")
                    if segment.timings
                    else None,
                }
                for segment in evidence.segments
            ],
            "local_physical_response": describe_host_response(snapshot["declared_site"], evidence),
        }

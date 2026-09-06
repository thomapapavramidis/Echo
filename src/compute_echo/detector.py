"""Routing-blind physical inference over independent settled segment means."""

from __future__ import annotations

import numpy as np
from pydantic import Field

from .models import EpochEvidence, Outcome, StrictModel
from .physics import MeterProfile, segment_observations


class Calibration(StrictModel):
    calibration_id: str
    method: str = "settled-segment-linear-response-v1"
    profile: MeterProfile
    training_ids: list[str]
    validation_ids: list[str]
    spec_keys: list[str]
    category_map: dict[str, str]
    response_watts: dict[str, float]
    gain_min: float
    gain_max: float
    correlation_min: float
    residual_limit_watts: float = Field(gt=0)
    within_segment_noise_limit_watts: float = Field(default=1_000_000, gt=0)
    validated: bool
    validation_summary: dict
    experimental_distinctions: dict
    hardware_configuration: str
    gpu_class_fingerprint_supported: bool = False


def fit_response(expected, observed, times):
    expected, observed, times = (np.asarray(x, dtype=float) for x in (expected, observed, times))
    if len(expected) < 4 or len(expected) != len(observed):
        raise ValueError("insufficient or mismatched observations")
    nuisance = np.column_stack([np.ones(len(times)), times - np.mean(times)])
    s = expected - nuisance @ np.linalg.lstsq(nuisance, expected, rcond=None)[0]
    r = observed - nuisance @ np.linalg.lstsq(nuisance, observed, rcond=None)[0]
    energy = float(s @ s)
    if energy < 1e-8:
        raise ValueError("unidentifiable challenge response after detrending")
    gain = float(r @ s / energy)
    norm = float(np.linalg.norm(r) * np.linalg.norm(s))
    correlation = float(np.clip(r @ s / norm, -1, 1)) if norm > 1e-8 else None
    residual = r - gain * s
    return {
        "correlation": correlation,
        "gain": gain,
        "residual_rmse_watts": float(np.sqrt(np.mean(residual**2))),
        "statistic_unit": "one_mean_per_settled_segment",
        "local_fraction": None,
    }


def physical_metrics(epoch: EpochEvidence, calibration: Calibration):
    rows = segment_observations(epoch, calibration.profile)
    for row in rows:
        if row["mode"] not in calibration.category_map or row["spec_key"] not in calibration.spec_keys:
            raise ValueError("workload specification absent from calibration")
    expected = [calibration.response_watts[calibration.category_map[r["mode"]]] for r in rows]
    metrics = fit_response(expected, [r["watts"] for r in rows], [r["time_s"] for r in rows])
    metrics["sample_coverage"] = min(r["coverage"] for r in rows)
    metrics["max_within_segment_std_watts"] = max(r["within_segment_std_watts"] for r in rows)
    metrics["segments"] = len(rows)
    return metrics


def classify_metrics(metrics: dict, model: Calibration):
    # Noise invalidates evidence before a low fitted gain becomes an inconsistency.
    if (
        metrics["residual_rmse_watts"] > model.residual_limit_watts
        or metrics.get("max_within_segment_std_watts", 0) > model.within_segment_noise_limit_watts
    ):
        return "INSUFFICIENT", "background/noise outside calibrated residual range"
    gain = metrics["gain"]
    if gain < model.gain_min or gain > model.gain_max:
        return "INCONSISTENT", "response amplitude outside calibrated range"
    rho = metrics["correlation"]
    if rho is None or rho < model.correlation_min:
        return "INCONSISTENT", "response shape outside calibrated range"
    return "CONSISTENT", "response within the validated empirical envelope"


def detect(epoch: EpochEvidence, calibration: Calibration | None) -> dict:
    outcomes = [s.outcome for s in epoch.segments]
    incorrect = any(s.digest_match is False or s.outcome == Outcome.INCORRECT for s in epoch.segments)
    correct = bool(outcomes) and all(s.digest_match is True for s in epoch.segments)
    computation = "INCORRECT" if incorrect else ("PASS" if correct else "PENDING")
    performance = (
        "MEETS_TARGET" if outcomes and all(s.deadline_met for s in epoch.segments) else "BELOW_TARGET"
    )
    known_deadline_failure = any(s.outcome == Outcome.DEADLINE_FAILED for s in epoch.segments)
    if not epoch.complete and not known_deadline_failure:
        performance = "INCOMPLETE"
    result = {
        "computational_correctness": computation,
        "performance": performance,
        "computational_outcomes": [o.value for o in outcomes],
        "physical_response": "INSUFFICIENT",
        "physical_reason": "no validated calibration",
        "metrics": None,
        "verdict": "INCONCLUSIVE",
        "provenance": epoch.provenance,
        "gpu_class_fingerprint_supported": False,
    }
    duration = max((epoch.ended_at_ns - epoch.started_at_ns) / 1e9, 1e-9)
    accepted = [s for s in epoch.segments if s.digest_match and s.deadline_met]
    result["demonstrated_work"] = {
        "on_time_valid_tasks": len(accepted),
        "audit_window_s": duration,
        "nominal_integer_operations": sum(s.nominal_integer_operations for s in accepted),
        "nominal_integer_operations_per_audit_second": sum(s.nominal_integer_operations for s in accepted)
        / duration,
        "dependent_memory_updates": sum(s.dependent_memory_updates for s in accepted),
        "gpu_count": None,
        "memory_traffic_lower_bound": None,
        "units_note": "nominal operations for verified prescribed outputs; includes idle time; not minimum work proof",
    }
    reason = None
    if not epoch.complete:
        reason = "epoch incomplete or interrupted"
    elif epoch.measurement_intent != "physical_audit":
        reason = "timing-only runs cannot establish physical consistency"
    elif epoch.provenance != "external":
        reason = "independent external measurements required"
    elif not epoch.source_freshness_verified:
        reason = "true device measurement freshness is unverified"
    elif calibration is None or not calibration.validated:
        reason = "no validated calibration"
    elif epoch.meter_id != calibration.profile.meter_id:
        reason = "meter differs from calibration"
    elif epoch.epoch_id in (
        calibration.training_ids + calibration.validation_ids + calibration.profile.source_epoch_ids
    ):
        reason = "calibration/characterization run cannot serve as independent evaluation"
    elif calibration.profile.provenance != "external" or not calibration.profile.source_freshness_verified:
        reason = "calibration lacks independent fresh physical evidence"
    else:
        try:
            metrics = physical_metrics(epoch, calibration)
            status, reason = classify_metrics(metrics, calibration)
            result.update(metrics=metrics, physical_response=status)
        except ValueError as error:
            reason = str(error)
    result["physical_reason"] = reason
    if incorrect:
        result["verdict"] = "INVALID_COMPUTATION"
    elif performance == "BELOW_TARGET":
        result["verdict"] = "DEMONSTRATED_CAPACITY_BELOW_CLAIM"
    elif not epoch.complete:
        result["verdict"] = "INCONCLUSIVE"
    elif computation == "PASS" and result["physical_response"] == "CONSISTENT":
        result["verdict"] = "CONSISTENT_UNDER_TESTED_ASSUMPTIONS"
    elif computation == "PASS" and result["physical_response"] == "INCONSISTENT":
        result["verdict"] = "LOCAL_EXECUTION_NOT_CORROBORATED"
    if calibration is not None:
        result["physical_categories"] = calibration.category_map
        result["experimental_distinctions"] = calibration.experimental_distinctions
    return result

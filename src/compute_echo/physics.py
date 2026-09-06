"""Measurement reports and observed step-response characterization.

These estimates describe this complete workload+meter setup. Observed step lag
also includes workload startup; it cannot isolate instrument filtering by itself.
"""

from __future__ import annotations

import itertools
import statistics

import numpy as np
from pydantic import Field

from .models import EpochEvidence, Mode, StrictModel


class MeterProfile(StrictModel):
    meter_id: str
    cadence_s: float = Field(gt=0)
    cadence_max_s: float = Field(gt=0)
    lag_s: float = Field(ge=0)
    rise_10_90_s: float = Field(ge=0)
    washout_s: float = Field(ge=0)
    transitions: int = Field(ge=3)
    source_epoch_ids: list[str]
    provenance: str
    source_freshness_verified: bool
    cadence_basis: str = "distinct-device-updates-observed-at-collector"
    method: str = "empirical-load-step-10-90-v1"


def characterize(epochs: list[EpochEvidence]) -> dict:
    intervals, native_intervals, lags, rises, settling, amplitudes = [], [], [], [], [], []
    timestamp_intervals = []
    rising_count = falling_count = 0
    meters = {e.meter_id for e in epochs}
    if len(meters) != 1:
        raise ValueError("characterization requires one meter")
    if any(e.measurement_intent != "physical_audit" for e in epochs):
        raise ValueError("timing-only runs cannot characterize the physical meter")
    if any(
        not e.complete or any(not s.digest_match or not s.deadline_met for s in e.segments) for e in epochs
    ):
        raise ValueError("characterization requires complete computationally passing local runs")
    if len({e.provenance for e in epochs}) != 1:
        raise ValueError("mixed provenance in characterization")
    for epoch in epochs:
        samples = [s for s in epoch.samples if s.quality == "GOOD"]
        # Only genuine fresh device updates count; ingestion enforces uniqueness.
        times = np.array([s.acquired_at_ns for s in samples], dtype=np.int64)
        if epoch.source_freshness_verified:
            intervals.extend((np.diff(times) / 1e9).tolist())
            if samples and all(s.device_time_ns is not None for s in samples):
                timestamp_intervals.extend((np.diff([s.device_time_ns for s in samples]) / 1e9).tolist())
                for left, right in itertools.pairwise(samples):
                    if left.device_sequence is not None and right.device_sequence is not None:
                        updates = right.device_sequence - left.device_sequence
                        if updates <= 0 or right.device_time_ns <= left.device_time_ns:
                            raise ValueError("invalid hardware update markers in characterization")
                        native_intervals.append((right.device_time_ns - left.device_time_ns) / 1e9 / updates)
        for previous, current in itertools.pairwise(epoch.segments):
            rising = previous.mode == Mode.IDLE and current.mode == Mode.TENSOR
            falling = previous.mode == Mode.TENSOR and current.mode == Mode.IDLE
            if not (rising or falling):
                continue
            duration = (current.end_ns - current.start_ns) / 1e9
            prior_duration = (previous.end_ns - previous.start_ns) / 1e9
            before = [
                s.watts
                for s in samples
                if previous.end_ns - int(prior_duration * 0.3e9) <= s.acquired_at_ns < previous.end_ns
            ]
            trace = [s for s in samples if current.start_ns <= s.acquired_at_ns < current.end_ns]
            tail = [s.watts for s in trace if s.acquired_at_ns >= current.end_ns - int(duration * 0.3e9)]
            if len(before) < 2 or len(tail) < 2 or len(trace) < 5:
                continue
            baseline, plateau = statistics.median(before), statistics.median(tail)
            amplitude = plateau - baseline
            noise = max(float(np.std(before)), float(np.std(tail)), 0.1)
            if (rising and amplitude <= 5 * noise) or (falling and amplitude >= -5 * noise):
                continue
            crossing = []
            for fraction in (0.1, 0.9):
                value = next(
                    (
                        (s.acquired_at_ns - current.start_ns) / 1e9
                        for s in trace
                        if (
                            s.watts >= baseline + fraction * amplitude
                            if rising
                            else s.watts <= baseline + fraction * amplitude
                        )
                    ),
                    None,
                )
                crossing.append(value)
            if None in crossing or crossing[1] < crossing[0]:
                continue
            lags.append(crossing[0])
            rises.append(crossing[1] - crossing[0])
            settling.append(crossing[1])
            amplitudes.append(amplitude)
            rising_count += int(rising)
            falling_count += int(falling)
    # Distinct counters establish freshness. Without hardware timestamps, collector
    # arrival cadence is an observed upper bound, not an exact device sample period.
    result = {
        "true_cadence_s": statistics.median(native_intervals) if native_intervals else None,
        "observed_fresh_update_cadence_s": statistics.median(intervals) if intervals else None,
        "observed_device_timestamp_cadence_s": statistics.median(timestamp_intervals)
        if timestamp_intervals
        else None,
        "cadence_max_s": max(intervals) if intervals else None,
        "observed_lag_s": statistics.median(lags) if lags else None,
        "observed_rise_10_90_s": statistics.median(rises) if rises else None,
        "tensor_idle_step_watts": amplitudes,
        "rising_step_watts": [a for a in amplitudes if a > 0],
        "falling_step_watts": [a for a in amplitudes if a < 0],
        "usable_rising_transitions": rising_count,
        "usable_falling_transitions": falling_count,
        "profile": None,
        "status": "INSUFFICIENT",
        "caveat": "Lag includes workload behavior. Native cadence requires genuine device timestamps and "
        "update counters; dividing elapsed device time by counter increments accounts for skipped polls. "
        "Otherwise only observed sample/arrival intervals are known. Rise-time field includes falling recovery.",
    }
    if rising_count >= 3 and falling_count >= 3 and intervals and min(intervals) > 0:
        cadence = statistics.median(intervals)
        profile = MeterProfile(
            meter_id=next(iter(meters)),
            cadence_s=cadence,
            cadence_max_s=max(intervals),
            lag_s=statistics.median(lags),
            rise_10_90_s=statistics.median(rises),
            washout_s=max(settling) + 2 * cadence,
            transitions=len(lags),
            source_epoch_ids=[e.epoch_id for e in epochs],
            provenance=epochs[0].provenance,
            cadence_basis="device-timestamps-and-collector-arrivals"
            if native_intervals
            else "distinct-device-updates-observed-at-collector",
            source_freshness_verified=all(e.source_freshness_verified for e in epochs),
        )
        result.update(profile=profile.model_dump(), status="CHARACTERIZED")
    return result


def segment_observations(epoch: EpochEvidence, profile: MeterProfile):
    """One mean per usable settled segment. Never count smoothed samples as IID."""
    if len(epoch.segments) < 8:
        raise ValueError("at least eight independent challenge segments required")
    for left, right in itertools.pairwise(epoch.samples):
        if right.sequence <= left.sequence or right.acquired_at_ns <= left.acquired_at_ns:
            raise ValueError("replayed or unordered meter evidence")
        if left.device_sequence is not None and right.device_sequence is not None:
            if right.device_sequence <= left.device_sequence:
                raise ValueError("reused device measurement in evidence")
        if left.device_time_ns is not None and right.device_time_ns is not None:
            if right.device_time_ns <= left.device_time_ns:
                raise ValueError("reused device timestamp in evidence")
    if any(s.meter_id != epoch.meter_id for s in epoch.samples):
        raise ValueError("foreign meter sample in epoch")
    if any(s.start_ns >= s.end_ns for s in epoch.segments):
        raise ValueError("invalid segment bounds")
    if any(b.start_ns < a.end_ns for a, b in itertools.pairwise(epoch.segments)):
        raise ValueError("overlapping experimental segments")
    observations = []
    for segment in epoch.segments:
        start = segment.start_ns + int(profile.washout_s * 1e9)
        stop = segment.end_ns
        if stop <= start:
            raise ValueError("segment too short for measured washout")
        samples = [s for s in epoch.samples if start <= s.acquired_at_ns < stop and s.quality == "GOOD"]
        expected = (stop - start) / 1e9 / profile.cadence_s
        coverage = min(1.0, len(samples) / max(expected, 1))
        gaps = np.diff([start, *[s.acquired_at_ns for s in samples], stop]) / 1e9
        if len(samples) < 2 or coverage < 0.8 or max(gaps) > 2.5 * profile.cadence_s:
            raise ValueError("missing, stale, or insufficient settled meter samples")
        observations.append(
            {
                "mode": segment.mode.value,
                "spec_key": segment.spec_key,
                "time_s": ((start + stop) / 2 - epoch.started_at_ns) / 1e9,
                "watts": float(np.mean([s.watts for s in samples])),
                "within_segment_std_watts": float(np.std([s.watts for s in samples])),
                "coverage": coverage,
                "elapsed_s": segment.elapsed_s,
            }
        )
    return observations


def pipeline_report(epochs: list[EpochEvidence]) -> dict:
    report = {}
    for mode in Mode:
        segments = [s for e in epochs for s in e.segments if s.mode == mode and s.elapsed_s is not None]
        if not segments:
            continue
        elapsed = [s.elapsed_s for s in segments]
        report[mode.value] = {
            "count": len(segments),
            "auditor_end_to_end_s": {
                "min": min(elapsed),
                "median": statistics.median(elapsed),
                "max": max(elapsed),
            },
            "outcomes": {
                str(o): sum(s.outcome == o for s in segments) for o in {s.outcome for s in segments}
            },
            "worker_reported_pipeline": [s.timings.model_dump() for s in segments if s.timings],
        }
    return report

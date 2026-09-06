"""Trusted commissioning, frozen candidate models, separate qualification runs.

Validation qualifies prefit candidates; it never retunes their thresholds. Final
evaluation must use a third set of fresh schedules. Ranges are empirical, not CIs.
"""

from __future__ import annotations

import itertools
import statistics
import uuid

import numpy as np

from .detector import Calibration, classify_metrics, fit_response, physical_metrics
from .models import EpochEvidence, Mode
from .physics import MeterProfile, segment_observations


def calibrate(
    training: list[EpochEvidence],
    validation: list[EpochEvidence],
    profile: MeterProfile,
    hardware_configuration: str,
) -> Calibration:
    train_ids, val_ids = [e.epoch_id for e in training], [e.epoch_id for e in validation]
    all_ids = train_ids + val_ids
    if len(training) < 3 or len(validation) < 3:
        raise ValueError("at least three independent training and three validation epochs required")
    if len(set(all_ids)) != len(all_ids) or set(all_ids) & set(profile.source_epoch_ids):
        raise ValueError("characterization, training, and held-out validation epochs must be disjoint")
    for epoch in training + validation:
        if epoch.measurement_intent != "physical_audit":
            raise ValueError("timing-only runs cannot calibrate physical response")
        if epoch.meter_id != profile.meter_id or epoch.provenance != profile.provenance:
            raise ValueError("calibration meter/provenance mismatch")
        if not epoch.complete or any(not s.digest_match or not s.deadline_met for s in epoch.segments):
            raise ValueError("only complete computationally passing trusted-local epochs can calibrate")
        if any(s.elapsed_s is None for s in epoch.segments):
            raise ValueError("calibration requires complete auditor timing measurements")
    train_rows = [segment_observations(e, profile) for e in training]
    val_rows = [segment_observations(e, profile) for e in validation]
    modes = set(r["mode"] for r in train_rows[0])
    if not {Mode.IDLE.value, Mode.TENSOR.value} <= modes:
        raise ValueError("minimum calibration is tensor versus idle")
    for rows in train_rows + val_rows:
        if set(r["mode"] for r in rows) != modes or len(rows) < 8:
            raise ValueError("each calibration epoch needs >=8 segments and the same set of modes")
    keys = {r["spec_key"] for rows in train_rows for r in rows}
    if any(r["spec_key"] not in keys for rows in val_rows for r in rows):
        raise ValueError("held-out validation must use equivalent workload specifications")

    # Fit baseline, drift, and independent mode amplitudes within each training run.
    active = sorted(modes - {"IDLE"})
    amplitudes = {m: [] for m in active}
    for rows in train_rows:
        t = np.array([r["time_s"] for r in rows])
        design = np.column_stack(
            [np.ones(len(rows)), t - t.mean(), *[[float(r["mode"] == m) for r in rows] for m in active]]
        )
        if np.linalg.matrix_rank(design) < design.shape[1] or np.linalg.cond(design) > 1e6:
            raise ValueError("workload schedule does not identify distinct mode responses")
        coefficients = np.linalg.lstsq(design, [r["watts"] for r in rows], rcond=None)[0]
        for m, amplitude in zip(active, coefficients[2:]):
            amplitudes[m].append(float(amplitude))
    centers = {m: statistics.median(v) for m, v in amplitudes.items()}
    if min(centers.values()) <= 1.0:
        raise ValueError("active/idle response too weak to calibrate all requested modes")

    pairs = {}
    distinct = len(active) > 1
    for a, b in itertools.combinations(active, 2):
        margin = max(1.0, 0.1 * max(abs(centers[a]), abs(centers[b])))
        separated = max(amplitudes[a]) + margin < min(amplitudes[b]) or max(amplitudes[b]) + margin < min(
            amplitudes[a]
        )
        heldout_correct = True
        for rows in val_rows:
            idle = statistics.mean(r["watts"] for r in rows if r["mode"] == "IDLE")
            for mode in (a, b):
                delta = statistics.mean(r["watts"] for r in rows if r["mode"] == mode) - idle
                chosen = min((a, b), key=lambda m: abs(delta - centers[m]))
                heldout_correct &= chosen == mode
        pairs[f"{a}_vs_{b}"] = {
            "training_ranges_separated": separated,
            "held_out_nearest_center_correct_all_runs": heldout_correct,
        }
        distinct &= separated and heldout_correct

    experimental = {
        "per_mode_training_delta_watts": amplitudes,
        "pairwise_qualification": pairs,
        "finer_modes_experimental": not distinct,
        "interval_type": "empirical_run_ranges",
        "timing_ranges_s": {
            m: {
                "training": [r["elapsed_s"] for rows in train_rows for r in rows if r["mode"] == m],
                "held_out": [r["elapsed_s"] for rows in val_rows for r in rows if r["mode"] == m],
            }
            for m in sorted(modes)
        },
    }
    timing_pairs = {}
    for a, b in itertools.combinations(active, 2):
        intervals = {}
        for phase, datasets in (("training", train_rows), ("held_out", val_rows)):
            medians = {
                m: [
                    statistics.median(
                        r["elapsed_s"] for r in rows if r["mode"] == m and r["elapsed_s"] is not None
                    )
                    for rows in datasets
                ]
                for m in (a, b)
            }
            intervals[phase] = {m: [min(values), max(values)] for m, values in medians.items()}
        separated_train = (
            intervals["training"][a][1] < intervals["training"][b][0]
            or intervals["training"][b][1] < intervals["training"][a][0]
        )
        same_direction = (
            statistics.mean(intervals["training"][a]) < statistics.mean(intervals["training"][b])
        ) == (statistics.mean(intervals["held_out"][a]) < statistics.mean(intervals["held_out"][b]))
        separated_val = (
            intervals["held_out"][a][1] < intervals["held_out"][b][0]
            or intervals["held_out"][b][1] < intervals["held_out"][a][0]
        )
        timing_pairs[f"{a}_vs_{b}"] = {
            "run_median_ranges_s": intervals,
            "held_out_timing_separation_supported": separated_train and separated_val and same_direction,
        }
    experimental["timing_pairwise_qualification"] = timing_pairs

    def candidate(fine: bool):
        mapping = {m: (m if fine or m == "IDLE" else "ACTIVE") for m in modes}
        response = {"IDLE": 0.0}
        if fine:
            response.update(centers)
        else:
            response["ACTIVE"] = statistics.median(centers.values())
        stats = [
            fit_response(
                [response[mapping[r["mode"]]] for r in rows],
                [r["watts"] for r in rows],
                [r["time_s"] for r in rows],
            )
            for rows in train_rows
        ]
        minimum = min(s["gain"] for s in stats) - 0.2
        if minimum <= 0.15:
            raise ValueError("local calibration cannot exclude an absent physical response")
        return Calibration(
            calibration_id=uuid.uuid4().hex,
            profile=profile,
            training_ids=train_ids,
            validation_ids=val_ids,
            spec_keys=sorted(keys),
            category_map=mapping,
            response_watts=response,
            gain_min=minimum,
            gain_max=max(s["gain"] for s in stats) + 0.2,
            correlation_min=max(0.0, min(s["correlation"] or 0.0 for s in stats) - 0.15),
            residual_limit_watts=max(2.0, max(s["residual_rmse_watts"] for s in stats) * 1.5),
            within_segment_noise_limit_watts=max(
                1.0, 1.5 * max(r["within_segment_std_watts"] for rows in train_rows for r in rows)
            ),
            validated=False,
            validation_summary={},
            experimental_distinctions=experimental,
            hardware_configuration=hardware_configuration,
        )

    # Both candidates use training only. Validation can select a supported candidate,
    # never adjust an amplitude, lag, or threshold to make a validation run pass.
    options = [candidate(True), candidate(False)] if distinct else [candidate(False)]
    selected, summary = options[-1], []
    for model in options:
        summary = [
            {"epoch_id": epoch.epoch_id, "metrics": physical_metrics(epoch, model)} for epoch in validation
        ]
        passed = all(classify_metrics(x["metrics"], model)[0] == "CONSISTENT" for x in summary)
        selected = model
        if passed:
            break
    qualified = all(classify_metrics(x["metrics"], selected)[0] == "CONSISTENT" for x in summary)
    return selected.model_copy(
        update={
            "validated": qualified,
            "validation_summary": {"passed": qualified, "runs": summary},
            "experimental_distinctions": {
                **experimental,
                "finer_modes_experimental": "ACTIVE" in selected.category_map.values(),
            },
        }
    )

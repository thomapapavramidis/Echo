"""Ground truth is joined only here, after routing-blind findings are computed."""

from collections import Counter, defaultdict

from .detector import Calibration, detect
from .models import EpochEvidence
from .physics import pipeline_report


def evaluate(epochs: list[EpochEvidence], labels: dict[str, str], calibration: Calibration):
    if len({e.epoch_id for e in epochs}) != len(epochs):
        raise ValueError("duplicate evaluation epochs")
    reserved = set(
        calibration.training_ids + calibration.validation_ids + calibration.profile.source_epoch_ids
    )
    if reserved & {e.epoch_id for e in epochs}:
        raise ValueError("evaluation must be disjoint from characterization and calibration")
    grouped = defaultdict(list)
    for epoch in epochs:
        if epoch.epoch_id not in labels:
            raise ValueError("evaluation label missing")
        grouped[labels[epoch.epoch_id]].append(epoch)
    output = {}
    for label, runs in grouped.items():
        findings = [detect(e, calibration) for e in runs]
        counts = Counter(f["physical_response"] for f in findings)
        output[label] = {
            "denominator": len(runs),
            "physical_counts": dict(counts),
            "verdict_counts": dict(Counter(f["verdict"] for f in findings)),
            "computation_and_performance_pass_count": sum(
                f["computational_correctness"] == "PASS" and f["performance"] == "MEETS_TARGET"
                for f in findings
            ),
            "pipeline": pipeline_report(runs),
            "runs": [{"epoch_id": e.epoch_id, "finding": f} for e, f in zip(runs, findings)],
        }
    local = output.get("local", {})
    forwarded = output.get("forwarded", {})
    enough = local.get("denominator", 0) >= 3 and forwarded.get("denominator", 0) >= 3
    equivalent = (
        bool(epochs)
        and len(
            {
                tuple(sorted(s.spec_key for s in e.segments))
                for e in epochs
                if labels[e.epoch_id] in {"local", "forwarded"}
            }
        )
        == 1
    )
    passed = (
        enough
        and equivalent
        and all(
            g.get("computation_and_performance_pass_count", 0) == g.get("denominator", -1)
            for g in (local, forwarded)
        )
    )
    passed &= local.get("physical_counts", {}).get("CONSISTENT", 0) == local.get(
        "denominator", -1
    ) and forwarded.get("physical_counts", {}).get("INCONSISTENT", 0) == forwarded.get("denominator", -1)
    return {
        "groups": output,
        "minimum_acceptance_passed": bool(passed),
        "equivalent_targets": equivalent,
        "required_independent_runs_per_route": 3,
        "false_alarm_count": local.get("physical_counts", {}).get("INCONSISTENT", 0),
        "false_alarm_denominator": local.get("denominator", 0),
        "outsourcing_detection_count": forwarded.get("physical_counts", {}).get("INCONSISTENT", 0),
        "outsourcing_denominator": forwarded.get("denominator", 0),
        "claim": "tested configurations only; empirical counts, no population guarantee",
    }

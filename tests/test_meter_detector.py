import numpy as np
import pytest

from compute_echo.calibration import calibrate
from compute_echo.detector import detect, fit_response
from compute_echo.evaluation import evaluate
from compute_echo.meter import MeterConfig, MeterStore
from compute_echo.models import MeterSample, Outcome
from compute_echo.physics import characterize


def model(make_epoch, profile, amplitudes=None):
    return calibrate(
        [make_epoch(i, amplitudes) for i in range(3)],
        [make_epoch(i, amplitudes) for i in range(10, 13)],
        profile,
        "synthetic test fixture",
    )


def test_correlation_is_not_local_fraction():
    s = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=float)
    result = fit_response(s, 0.5 * s, np.arange(len(s)))
    assert result["correlation"] == pytest.approx(1)
    assert result["gain"] == pytest.approx(0.5)
    assert result["local_fraction"] is None


def test_real_ingestion_freshness_and_restart(tmp_path):
    cfg = MeterConfig(
        meter_id="fixture", installation="independent fixture adapter", source_freshness="device_sequence"
    )
    store = MeterStore(tmp_path / "meter.db", cfg)
    now = 1_700_000_000_000_000_000
    sample = MeterSample(meter_id="fixture", sequence=10, device_sequence=5, acquired_at_ns=now, watts=100)
    store.ingest(sample, now)
    for changed in (
        {},
        {"sequence": 11, "acquired_at_ns": now + 1},
        {"sequence": 11, "device_sequence": 6, "acquired_at_ns": now - 10_000_000_000},
        {"meter_id": "forged", "sequence": 11},
    ):
        with pytest.raises(ValueError):
            store.ingest(sample.model_copy(update=changed), now)
    reopened = MeterStore(tmp_path / "meter.db", cfg)
    with pytest.raises(ValueError):
        reopened.ingest(sample, now)
    assert store.health(now) == "OK"
    assert store.health(now + 10_000_000_000) == "SENSOR_LOST"
    with pytest.raises(ValueError):
        MeterSample(meter_id="fixture", sequence=12, acquired_at_ns=now, watts=float("nan"))


def test_fine_modes_require_heldout_support(make_epoch, profile):
    calibrated = model(make_epoch, profile)
    assert calibrated.validated
    assert calibrated.category_map["MEMORY"] == "MEMORY"
    good = detect(make_epoch(20), calibrated)
    outsourced = detect(make_epoch(21, factor=0), calibrated)
    assert good["verdict"] == "CONSISTENT_UNDER_TESTED_ASSUMPTIONS"
    assert outsourced["verdict"] == "LOCAL_EXECUTION_NOT_CORROBORATED"
    assert good["gpu_class_fingerprint_supported"] is False


def test_indistinguishable_modes_collapse_active(make_epoch, profile):
    amps = {"TENSOR": 100, "MEMORY": 101, "MIXED": 99}
    calibrated = model(make_epoch, profile, amps)
    assert calibrated.validated
    assert calibrated.category_map == {
        "IDLE": "IDLE",
        "TENSOR": "ACTIVE",
        "MEMORY": "ACTIVE",
        "MIXED": "ACTIVE",
    }
    assert detect(make_epoch(25, amps), calibrated)["physical_response"] == "CONSISTENT"
    assert calibrated.experimental_distinctions["finer_modes_experimental"]


def test_missing_noisy_simulated_and_unknown_evidence_abstains(make_epoch, profile):
    calibrated = model(make_epoch, profile)
    epoch = make_epoch(20)
    variants = [
        epoch.model_copy(update={"samples": []}),
        epoch.model_copy(update={"samples": epoch.samples[::5]}),
        epoch.model_copy(update={"provenance": "simulation"}),
        epoch.model_copy(update={"measurement_intent": "timing_only"}),
        epoch.model_copy(update={"source_freshness_verified": False}),
        epoch.model_copy(update={"meter_id": "different"}),
        epoch.model_copy(update={"complete": False}),
        make_epoch(22, noise=100),
    ]
    for variant in variants:
        result = detect(variant, calibrated)
        assert result["physical_response"] == "INSUFFICIENT", result
        assert result["verdict"] == "INCONCLUSIVE", result


def test_calibration_leakage_and_unsupported_specs_rejected(make_epoch, profile):
    training = [make_epoch(i) for i in range(3)]
    with pytest.raises(ValueError):
        calibrate(training, training, profile, "test")
    calibrated = model(make_epoch, profile)
    epoch = make_epoch(20)
    assert (
        detect(epoch.model_copy(update={"epoch_id": calibrated.training_ids[0]}), calibrated)[
            "physical_response"
        ]
        == "INSUFFICIENT"
    )
    bad = epoch.segments[0].model_copy(update={"spec_key": "unsupported"})
    assert (
        detect(epoch.model_copy(update={"segments": [bad, *epoch.segments[1:]]}), calibrated)[
            "physical_response"
        ]
        == "INSUFFICIENT"
    )


def test_outcomes_do_not_hide_incorrect_computation(make_epoch, profile):
    calibrated = model(make_epoch, profile)
    epoch = make_epoch(20)
    wrong = epoch.segments[0].model_copy(update={"outcome": Outcome.INCORRECT, "digest_match": False})
    result = detect(
        epoch.model_copy(update={"segments": [wrong, *epoch.segments[1:]], "samples": []}), calibrated
    )
    assert result["computational_correctness"] == "INCORRECT"
    assert result["physical_response"] == "INSUFFICIENT"
    assert result["verdict"] == "INVALID_COMPUTATION"


def test_detector_contract_rejects_ground_truth(make_epoch):
    from compute_echo.models import EpochEvidence

    with pytest.raises(ValueError):
        EpochEvidence.model_validate({**make_epoch().model_dump(), "route": "forwarded"})


def test_evaluation_denominators_and_equivalent_targets(make_epoch, profile):
    calibrated = model(make_epoch, profile)
    epochs = [make_epoch(i) for i in range(20, 23)] + [make_epoch(i, factor=0) for i in range(23, 26)]
    labels = {e.epoch_id: "local" if i < 3 else "forwarded" for i, e in enumerate(epochs)}
    report = evaluate(epochs, labels, calibrated)
    assert report["minimum_acceptance_passed"]
    assert report["false_alarm_count"] == 0
    assert report["outsourcing_detection_count"] == report["outsourcing_denominator"] == 3


def test_characterization_never_fabricates_unknown_cadence(make_epoch):
    epoch = make_epoch().model_copy(update={"source_freshness_verified": False})
    assert characterize([epoch])["true_cadence_s"] is None
    assert characterize([epoch])["profile"] is None


def test_characterization_observes_multiple_steps(make_epoch):
    report = characterize([make_epoch(5)])
    assert report["usable_rising_transitions"] >= 3
    assert report["profile"] is not None
    assert report["observed_fresh_update_cadence_s"] == pytest.approx(0.1)
    assert report["true_cadence_s"] is None


def test_device_timestamp_cadence_is_reported_separately(make_epoch):
    epoch = make_epoch(5)
    epoch = epoch.model_copy(
        update={"samples": [s.model_copy(update={"device_time_ns": s.acquired_at_ns}) for s in epoch.samples]}
    )
    report = characterize([epoch])
    assert report["true_cadence_s"] == pytest.approx(0.1)


def test_native_cadence_accounts_for_skipped_hardware_updates(make_epoch):
    epoch = make_epoch(5)
    samples = [s.model_copy(update={"device_time_ns": s.acquired_at_ns}) for s in epoch.samples[::2]]
    report = characterize([epoch.model_copy(update={"samples": samples})])
    assert report["true_cadence_s"] == pytest.approx(0.1)
    assert report["observed_fresh_update_cadence_s"] == pytest.approx(0.2)


def test_validation_failure_does_not_retrain(make_epoch, profile):
    calibrated = calibrate(
        [make_epoch(i) for i in range(3)], [make_epoch(i, factor=0.2) for i in range(10, 13)], profile, "test"
    )
    assert not calibrated.validated
    assert calibrated.gain_min > 0.5


def test_replayed_samples_and_reused_device_markers_abstain(make_epoch, profile):
    calibrated = model(make_epoch, profile)
    epoch = make_epoch(20)
    duplicated = [epoch.samples[0], epoch.samples[0], *epoch.samples[1:]]
    assert (
        detect(epoch.model_copy(update={"samples": duplicated}), calibrated)["physical_response"]
        == "INSUFFICIENT"
    )
    forged_counter = [s.model_copy(update={"device_sequence": 1}) for s in epoch.samples]
    assert (
        detect(epoch.model_copy(update={"samples": forged_counter}), calibrated)["physical_response"]
        == "INSUFFICIENT"
    )


def test_missing_timing_and_timing_only_cannot_calibrate(make_epoch, profile):
    training = [make_epoch(i) for i in range(3)]
    validation = [make_epoch(i) for i in range(10, 13)]
    training[0] = training[0].model_copy(update={"measurement_intent": "timing_only"})
    with pytest.raises(ValueError):
        calibrate(training, validation, profile, "test")


def test_known_deadline_failure_remains_below_target(make_epoch, profile):
    epoch = make_epoch(20)
    failed = epoch.segments[0].model_copy(
        update={"outcome": Outcome.DEADLINE_FAILED, "digest_match": None, "deadline_met": False}
    )
    result = detect(epoch.model_copy(update={"segments": [failed], "complete": False}), None)
    assert result["performance"] == "BELOW_TARGET"
    assert result["computational_correctness"] == "PENDING"
    assert result["computational_outcomes"] == ["DEADLINE_FAILED"]
    assert result["verdict"] == "DEMONSTRATED_CAPACITY_BELOW_CLAIM"

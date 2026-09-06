"""Synthetic fixtures only: these are not laboratory observations."""

import random
import uuid

import pytest

from compute_echo.models import EpochEvidence, MeterSample, Mode, Outcome, Segment, WorkloadSpec
from compute_echo.physics import MeterProfile


@pytest.fixture
def small_spec():
    return WorkloadSpec(
        mode=Mode.TENSOR,
        m=32,
        n=48,
        k=64,
        operations=2,
        lanes=8,
        words_per_lane=64,
        memory_steps=32,
        segment_s=1,
        deadline_s=0.9,
    )


@pytest.fixture
def profile():
    return MeterProfile(
        meter_id="fixture",
        cadence_s=0.1,
        cadence_max_s=0.1,
        lag_s=0,
        rise_10_90_s=0,
        washout_s=0.1,
        transitions=3,
        source_epoch_ids=[uuid.uuid4().hex],
        provenance="external",
        source_freshness_verified=True,
    )


@pytest.fixture
def make_epoch():
    def make(seed=0, amplitudes=None, factor=1, provenance="external", noise=0.15):
        rng = random.Random(seed)
        amplitudes = amplitudes or {"TENSOR": 100, "MEMORY": 60, "MIXED": 150}
        active = [Mode.TENSOR, Mode.MEMORY, Mode.MIXED] * 4
        rng.shuffle(active)
        modes = []
        for mode in active:
            pair = [Mode.IDLE, mode]
            rng.shuffle(pair)
            modes.extend(pair)
        started = 1_700_000_000_000_000_000 + seed * 100_000_000_000
        segments, samples = [], []
        for index, mode in enumerate(modes):
            start = started + index * 1_000_000_000
            spec = WorkloadSpec(mode=mode, segment_s=1, deadline_s=0.9)
            segments.append(
                Segment(
                    challenge_id=uuid.uuid4().hex,
                    mode=mode,
                    spec_key=spec.key,
                    start_ns=start,
                    end_ns=start + 1_000_000_000,
                    outcome=Outcome.PASS,
                    digest_match=True,
                    deadline_met=True,
                    elapsed_s=0.05,
                )
            )
            for j in range(10):
                sequence = len(samples)
                samples.append(
                    MeterSample(
                        meter_id="fixture",
                        sequence=sequence,
                        device_sequence=sequence,
                        acquired_at_ns=start + j * 100_000_000,
                        watts=200 + factor * amplitudes.get(mode.value, 0) + rng.uniform(-noise, noise),
                    )
                )
        return EpochEvidence(
            epoch_id=uuid.uuid4().hex,
            started_at_ns=started,
            ended_at_ns=started + len(modes) * 1_000_000_000,
            meter_id="fixture",
            provenance=provenance,
            source_freshness_verified=True,
            segments=segments,
            samples=samples,
            complete=True,
        )

    return make

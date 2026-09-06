import hashlib
import struct
import threading

import numpy as np
import pytest

from compute_echo.models import Mode, WorkloadSpec, canonical
from compute_echo.workloads import (
    DOMAIN,
    CancelledWork,
    WorkloadEngine,
    memory_cpu,
    memory_inputs,
    stream,
    tensor_inputs,
)


def test_scalar_tensor_oracle(small_spec):
    seed = bytes(range(32))
    a, b = tensor_inputs(seed, small_spec, 0)
    expected = [
        [sum(int(a[i, k]) * int(b[k, j]) for k in range(small_spec.k)) for j in range(small_spec.n)]
        for i in range(small_spec.m)
    ]
    np.testing.assert_array_equal(a.astype(np.int32) @ b.astype(np.int32), expected)
    assert a.min() >= -8 and a.max() <= 7


def test_prng_protocol_is_library_independent(small_spec):
    seed = bytes(range(32))
    prefix = DOMAIN + bytes.fromhex(small_spec.key) + seed + struct.pack("<I", 3)
    expected = hashlib.shake_256(prefix + struct.pack("<I", 1) + b"A").digest(100)
    assert stream(seed, small_spec, 3, b"A", 100) == expected
    assert stream(seed, small_spec, 4, b"A", 100) != expected
    assert stream(seed, small_spec, 3, b"B", 100) != expected


def test_dependent_memory_scalar_oracle(small_spec):
    buf, state = memory_inputs(bytes(32), small_spec, 0)
    expected = buf.copy()
    expected_states = state.copy()
    mask = 0xFFFFFFFF

    def rot(x, r):
        return ((x << r) | (x >> (32 - r))) & mask

    for lane in range(len(state)):
        s = int(state[lane])
        for step in range(small_spec.memory_steps):
            index = s & (small_spec.words_per_lane - 1)
            old = int(expected[lane, index])
            s = (rot(s ^ old, 13) + 0x9E3779B9 + step) & mask
            expected[lane, index] = s ^ rot(old, 7)
        expected_states[lane] = s
    memory_cpu(buf, state, small_spec.memory_steps)
    np.testing.assert_array_equal(buf, expected)
    np.testing.assert_array_equal(state, expected_states)


def test_skipping_memory_work_changes_digest(small_spec, monkeypatch):
    import compute_echo.workloads as workloads

    spec = small_spec.model_copy(update={"mode": Mode.MEMORY})
    expected = WorkloadEngine().execute(bytes(32), spec)[0]
    monkeypatch.setattr(workloads, "memory_cpu", lambda b, s, steps, offset: (b, s))
    assert WorkloadEngine().execute(bytes(32), spec)[0] != expected


def test_every_operation_and_output_is_hashed(small_spec, monkeypatch):
    import compute_echo.workloads as workloads

    expected = WorkloadEngine().execute(bytes(32), small_spec)[0]
    original = workloads.tensor_inputs

    def corrupted(seed, spec, operation):
        a, b = original(seed, spec, operation)
        if operation == 0:  # Corrupt the first output, not only the last batch result.
            a[0, 0] ^= 1
        return a, b

    monkeypatch.setattr(workloads, "tensor_inputs", corrupted)
    assert WorkloadEngine().execute(bytes(32), small_spec)[0] != expected


@pytest.mark.parametrize("mode", list(Mode))
def test_repeatability_freshness_and_timing(mode, small_spec):
    spec = small_spec.model_copy(update={"mode": mode})
    engine = WorkloadEngine()
    digest, times = engine.execute(bytes(32), spec)
    assert digest == engine.execute(bytes(32), spec)[0]
    assert digest != engine.execute(bytes([1]) * 32, spec)[0]
    assert len(digest) == 64
    assert times.total_worker_s >= times.execution_s
    assert all(v >= 0 for v in times.model_dump().values())


def test_cancellation_and_bounds(small_spec):
    event = threading.Event()
    event.set()
    with pytest.raises(CancelledWork):
        WorkloadEngine().execute(bytes(32), small_spec, event)
    for update in ({"words_per_lane": 33}, {"k": 2**25}, {"deadline_s": 20}, {"mode": "UNKNOWN"}):
        with pytest.raises(ValueError):
            WorkloadSpec.model_validate({**small_spec.model_dump(), **update})


@pytest.mark.cuda
@pytest.mark.parametrize("mode", list(Mode))
def test_cuda_matches_independent_cpu(mode, small_spec):
    pytest.importorskip("cupy")
    try:
        gpu = WorkloadEngine("cuda")
    except Exception as error:
        pytest.skip(f"CUDA unavailable: {error}")
    spec = small_spec.model_copy(update={"mode": mode, "memory_steps": 513})
    cpu = WorkloadEngine()
    for seed in [bytes(32), bytes(range(32)), hashlib.sha256(b"second-device-vector").digest()]:
        assert gpu.execute(seed, spec)[0] == cpu.execute(seed, spec)[0]


def test_canonical_rejects_nonfinite():
    with pytest.raises(ValueError):
        canonical({"power": float("nan")})


def test_published_protocol_vectors():
    import json
    from pathlib import Path

    values = json.loads((Path(__file__).parents[1] / "docs/validation/protocol-vectors.json").read_bytes())
    engine = WorkloadEngine()
    for vector in values["vectors"]:
        actual, _ = engine.execute(bytes.fromhex(vector["seed"]), WorkloadSpec.model_validate(vector["spec"]))
        assert actual == vector["digest"]

"""Deterministic protocol implementation shared by trusted reference and workers.

The digest is a known-answer check, not a succinct proof of computational effort.
The CPU implementation is also the independent oracle for CUDA tests.
"""

from __future__ import annotations

import hashlib
import struct
import threading
import time
from collections import defaultdict

import numpy as np

from .models import Mode, PipelineTimes, WorkloadSpec, canonical

DOMAIN = b"ComputeEcho/int-v1\x00"


def stream(seed: bytes, spec: WorkloadSpec, operation: int, role: bytes, size: int) -> bytes:
    """SHAKE256 is the PRNG; no NumPy/CUDA generator implementation is involved."""
    if len(seed) != 32:
        raise ValueError("seed must be exactly 32 bytes")
    prefix = DOMAIN + bytes.fromhex(spec.key) + seed + struct.pack("<I", operation)
    return hashlib.shake_256(prefix + struct.pack("<I", len(role)) + role).digest(size)


def tensor_inputs(seed: bytes, spec: WorkloadSpec, operation: int):
    def matrix(role, rows, cols):
        raw = np.frombuffer(stream(seed, spec, operation, role, rows * cols), dtype=np.uint8)
        return ((raw & 15).astype(np.int8) - 8).reshape(rows, cols)

    return matrix(b"A", spec.m, spec.k), matrix(b"B", spec.k, spec.n)


def memory_inputs(seed: bytes, spec: WorkloadSpec, operation: int):
    buf = (
        np.frombuffer(
            stream(seed, spec, operation, b"memory", 4 * spec.lanes * spec.words_per_lane), dtype="<u4"
        )
        .astype(np.uint32)
        .reshape(spec.lanes, spec.words_per_lane)
    )
    state = np.frombuffer(stream(seed, spec, operation, b"state", 4 * spec.lanes), dtype="<u4").astype(
        np.uint32
    )
    return buf, state


def memory_cpu(buf, state, steps: int, offset: int = 0):
    lanes = np.arange(len(state))
    mask = np.uint32(buf.shape[1] - 1)
    for j in range(offset, offset + steps):
        indices = state & mask
        old = buf[lanes, indices].copy()
        x = state ^ old
        state[:] = ((x << np.uint32(13)) | (x >> np.uint32(19))) + np.uint32(0x9E3779B9)
        state[:] += np.uint32(j)
        buf[lanes, indices] = state ^ ((old << np.uint32(7)) | (old >> np.uint32(25)))
    return buf, state


MEMORY_KERNEL = r"""
extern "C" __global__ void dependent_memory(
    unsigned int* buf, unsigned int* states, int lanes, int words, int steps, int offset) {
  int lane = blockDim.x * blockIdx.x + threadIdx.x;
  if (lane >= lanes) return;
  unsigned int s = states[lane];
  unsigned int* local = buf + ((unsigned long long)lane * words);
  for (int j=offset; j<offset+steps; ++j) {
    unsigned int idx = s & (words-1), old = local[idx], x = s ^ old;
    s = ((x << 13) | (x >> 19)) + 0x9e3779b9u + (unsigned int)j;
    local[idx] = s ^ ((old << 7) | (old >> 25));
  }
  states[lane] = s;
}
"""


class CancelledWork(RuntimeError):
    pass


class WorkloadEngine:
    def __init__(self, backend: str = "cpu", safety_check=None):
        if backend not in {"cpu", "cuda"}:
            raise ValueError("backend must be explicitly cpu or cuda")
        self.backend = backend
        self.safety_check = safety_check
        self.cp = None
        if backend == "cuda":
            import cupy as cp

            self.cp = cp
            cp.cuda.runtime.getDeviceCount()  # Fail explicitly; never silently fall back to CPU.
            self.memory_kernel = cp.RawKernel(MEMORY_KERNEL, "dependent_memory")

    @property
    def identity(self):
        if self.cp is None:
            return "cpu:numpy-int32"
        props = self.cp.cuda.runtime.getDeviceProperties(self.cp.cuda.Device().id)
        name = props["name"]
        return "cuda:" + (name.decode() if isinstance(name, bytes) else str(name))

    def execute(self, seed: bytes, spec: WorkloadSpec, cancel: threading.Event | None = None):
        start = time.perf_counter()
        timings = defaultdict(float)
        cp = self.cp

        def check():
            if cancel is not None and cancel.is_set():
                raise CancelledWork("work cancelled between bounded operations")
            if self.safety_check is not None:
                before = time.perf_counter()
                try:
                    self.safety_check()
                finally:
                    timings["safety_check_s"] += time.perf_counter() - before

        def measured(name, fn):
            before = time.perf_counter()
            result = fn()
            if cp is not None and name in {"host_to_device_s", "execution_s", "device_to_host_s"}:
                cp.cuda.get_current_stream().synchronize()
            timings[name] += time.perf_counter() - before
            return result

        digest = hashlib.sha256()
        digest.update(DOMAIN + b"result\x00" + seed + canonical(spec))

        def aggregate(operation, tag, arrays):
            for index, arr in enumerate(arrays):
                payload = measured(
                    "aggregation_s",
                    lambda: arr.astype("<i4" if tag == b"tensor" else "<u4", copy=False).tobytes(order="C"),
                )
                header = struct.pack("<IIIQ", operation, len(tag), index, len(payload)) + tag
                measured("hashing_s", lambda: (digest.update(header), digest.update(payload)))

        for op in range(spec.operations if spec.mode != Mode.IDLE else 0):
            check()
            if spec.mode in {Mode.TENSOR, Mode.MIXED}:
                a, b = measured("input_generation_s", lambda: tensor_inputs(seed, spec, op))
                if cp is None:
                    result = measured("execution_s", lambda: a.astype(np.int32) @ b.astype(np.int32))
                else:
                    da, db = measured("host_to_device_s", lambda: (cp.asarray(a), cp.asarray(b)))
                    dc = cp.empty((spec.m, spec.n), dtype=cp.int32)
                    from cupy_backends.cuda.libs import cublas

                    alpha, beta = np.array(1, dtype=np.int32), np.array(0, dtype=np.int32)

                    def gemm():
                        # Row-major C=A@B is column-major C.T=B.T@A.T. No transpose copies.
                        cublas.gemmEx(
                            cp.cuda.device.get_cublas_handle(),
                            cublas.CUBLAS_OP_N,
                            cublas.CUBLAS_OP_N,
                            spec.n,
                            spec.m,
                            spec.k,
                            alpha.ctypes.data,
                            db.data.ptr,
                            3,
                            spec.n,
                            da.data.ptr,
                            3,
                            spec.k,
                            beta.ctypes.data,
                            dc.data.ptr,
                            10,
                            spec.n,
                            cublas.CUBLAS_COMPUTE_32I,
                            cublas.CUBLAS_GEMM_DEFAULT,
                        )

                    measured("execution_s", gemm)
                    result = measured("device_to_host_s", lambda: cp.asnumpy(dc))
                aggregate(op, b"tensor", [result])
            check()
            if spec.mode in {Mode.MEMORY, Mode.MIXED}:
                buf, state = measured("input_generation_s", lambda: memory_inputs(seed, spec, op))
                if cp is not None:
                    buf, state = measured("host_to_device_s", lambda: (cp.asarray(buf), cp.asarray(state)))
                for offset in range(0, spec.memory_steps, 256):
                    check()
                    steps = min(256, spec.memory_steps - offset)
                    if cp is None:
                        measured("execution_s", lambda: memory_cpu(buf, state, steps, offset))
                    else:
                        measured(
                            "execution_s",
                            lambda: self.memory_kernel(
                                ((spec.lanes + 127) // 128,),
                                (128,),
                                (buf, state, spec.lanes, spec.words_per_lane, steps, offset),
                            ),
                        )
                if cp is not None:
                    buf, state = measured("device_to_host_s", lambda: (cp.asnumpy(buf), cp.asnumpy(state)))
                aggregate(op, b"memory", [buf, state])
        check()
        value = measured("hashing_s", digest.hexdigest)
        timings["total_worker_s"] = time.perf_counter() - start
        return value, PipelineTimes(**timings)

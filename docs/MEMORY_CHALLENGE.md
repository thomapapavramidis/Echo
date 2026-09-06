# Dependent memory challenge proposal and v1 specification

The simplest implementation uses independent lanes, each owning a mutable uint32
buffer. It is vectorized across lanes on the CPU and uses one CUDA thread per lane
on the GPU, avoiding data races and device-specific atomic ordering.

For each fresh batch operation, domain-separated SHAKE256 produces a new buffer
and a new state per lane. At step `j`, for each lane:

```
index = state & (words_per_lane - 1)
old = buffer[index]
state = (rotate_left_32(state XOR old, 13) + 0x9e3779b9 + j) modulo 2**32
buffer[index] = state XOR rotate_left_32(old, 7)
```

Subsequent addresses depend on prior read values and mutations. The ordered final
buffer and every final lane state are included in the aggregate digest. Skipping
steps or hashing the unchanged initialization fails the reference comparison.

This is an experimental dependent read/write workload, not a formally proven
memory-hard function. It does not prove physical VRAM residence, require a
particular implementation, or establish a lower bound on memory traffic. An
optimized solver could simulate accesses differently. Buffer size and lane count
must be tuned relative to caches; small test configurations are functional tests,
not bandwidth or GPU-class evidence. MIXED interleaves one GEMM and one complete
dependent-memory operation for each operation index, and hashes both outputs.

Cancellation is checked between operations and bounded memory chunks. No runtime
duration loop repeats identical inputs or changes the deterministic operation
count. Scheduler segment duration is a fixed measurement window; it is distinct
from measured execution duration.

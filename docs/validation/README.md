# Recorded validation artifacts

These files contain actual computational measurements from this workspace; none
contains independently measured electricity.

- `hardware-doctor.json`: detected RTX 5060 Laptop GPU and small CPU/CUDA digest
  comparisons. Initial CUDA stage timings include context/kernel startup effects.
- `loopback-cuda.json`: three local and three forwarded epochs using real HTTP and
  two services on the **same laptop/GPU**. Both routes passed every computational
  deadline, and every physical finding remained insufficient. Full worker stage
  measurements are diagnostic; auditor completion timings are measured locally.
- `loopback-cuda-final.json`: a second successful three-epoch-per-route run after
  worker deadline/disconnect hardening. It has the same same-host/no-meter limits.
- `protocol-vectors.json`: public fixed seed/spec/digest conformance vectors for all
  four modes. They are tested and must never be used as operational challenges.
- `workload-profile-cuda.json`: three fresh repetitions per mode after a discarded
  warmup using config/workloads.physics.example.json. This is trusted local tuning,
  not an independently verified audit or electrical characterization.

The loopback small-workload active-mode median completion times ranged roughly
19–72 ms locally and 22–23 ms through the forwarding service in this run. Local
temperature-query and scheduling overhead can dominate these tiny workloads; this
is not a comparison of remote-site latency or hardware capacity.

For the larger starting tensor configuration, total worker time was 0.497–0.570 s
while the synchronized execution stage was 0.0074–0.0079 s. Memory was 0.845–0.926 s
and mixed was 1.067–1.459 s total worker time. These observations expose substantial
host/transfer/hash overhead and demonstrate why physical parameters must be tuned
from measured pipeline behavior rather than GEMM throughput alone.

Raw temporary integration runs remain under the ignored `runs/` directory. Their
paths are recorded in the JSON report. They are not reference banks for future
audits. Re-run scripts/loopback_smoke.py to create entirely new seeds and epochs.

The test suite's synthetic power fixtures exist only under tests/. They exercise
statistical behavior and trust-policy branches; their results are not reported as
laboratory detection statistics.

Final verification: 55 tests passed (including CUDA), with Ruff lint and formatting
checks passing. The added cases cover native cadence with skipped updates,
independent meter loss during work, worker-local deadlines, auditor disconnect,
cancelled epoch tasks, timing-only quarantine, and modified raw receipt evidence.

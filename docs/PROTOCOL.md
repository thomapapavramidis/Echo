# Deterministic compact-result protocol: echo-int-v1

This is a versioned known-answer protocol. A trusted offline machine computes the
same function and stores the expected digest privately. SHA-256 equality validates
the expected output digest under the hash assumptions; it is not a proof of work,
proof of knowledge, Freivalds test, or device/location attestation.

## Canonical specification

Every specification is validated by WorkloadSpec. The defaults are expanded before
serialization. JSON uses sorted keys, compact `,`/`:` separators, ASCII escaping,
UTF-8 bytes, and finite numbers. Pydantic normalizes declared float fields to floats
before encoding. `spec_key = SHA256(canonical_spec_bytes)`. This is a defined local
encoding, not a claim of general RFC 8785 compliance. Protocol/spec changes require
a new bank and new calibration.

Seed: 32 bytes from the operating system's cryptographic random generator. Wire
encoding: exactly 64 lowercase hexadecimal characters. Challenge and epoch IDs are
32 lowercase hexadecimal characters. No challenge contains its expected digest.

## PRNG

Let `D = b"ComputeEcho/int-v1\x00"`. For each operation index `i` and ASCII role `R`:

```
prefix = D || raw_32_byte_spec_key || seed || uint32_le(i)
bytes = SHAKE256(prefix || uint32_le(len(R)) || R).digest(required_length)
```

The SHAKE stream is specified independently of NumPy/CuPy random generators. Each
matrix/operation has a separate role/counter. It is generated locally from the
compact seed; v1 generates bytes on the CPU even on CUDA workers, and measures that
cost. No seed, specification, or generator is reused across issued tasks.

## Tensor operation

Roles `A` and `B` produce `m*k` and `k*n` bytes. Map each byte to signed INT8 using
`(byte & 15) - 8`, giving [-8,7]. Reshape A as (m,k) and B as (k,n), row-major.
Compute C=A@B using signed INT32 accumulation, alpha=1, beta=0. The maximum absolute
partial sum is bounded by `64*k`; k<=32768 makes it <=2,097,152, safely inside INT32.

CPU widens inputs to INT32 before NumPy multiplication. CUDA uses cuBLAS gemmEx
with signed INT8 inputs, INT32 output, and CUBLAS_COMPUTE_32I. It reinterprets the
row-major multiplication as column-major C.T=B.T@A.T without copying a transpose.
Dimension alignment and allocation limits are checked before execution.

Output C is serialized row-major, signed 32-bit two's-complement, little-endian.
It is never hashed as a float, a platform-native endianness buffer, or INT8 output.

## Dependent memory operation

Roles `memory` and `state` yield `lanes*words_per_lane*4` and `lanes*4` bytes,
interpreted as little-endian uint32 and then manipulated as unsigned 32-bit values.
words_per_lane must be a power of two. Each lane owns a separate contiguous buffer.
The exact read/modify/write recurrence is specified in MEMORY_CHALLENGE.md.

Every addition wraps modulo 2**32; shifts are logical; rotations use 32-bit words.
Steps run from 0 through memory_steps-1. Chunking every 256 steps preserves the
global step index and state. Lane ordering is deterministic and no writes race.

Outputs: the entire final buffer (lane-major, each lane's words in increasing
index order), then all terminal lane states; both unsigned uint32 little-endian.
Skipping the dependent sequence changes the expected digest. The protocol does
not establish a minimum memory footprint, access latency, or physical VRAM use.

## Aggregate hash

Initialize SHA-256 and update it with:

```
D || b"result\x00" || seed || canonical_spec_bytes
```

For operation indices 0..operations-1:

- TENSOR includes its C under tag `tensor` and array index 0.
- MEMORY includes its final buffer under tag `memory`, array index 0, then states,
  array index 1.
- MIXED includes the tensor output followed by both memory outputs for that index.
- IDLE includes no arrays; the challenge-bound initial hash is its response.

For every included array append:

```
uint32_le(operation_index)
|| uint32_le(tag_byte_length)
|| uint32_le(array_index)
|| uint64_le(payload_byte_length)
|| tag_bytes
|| serialized_array_bytes
```

The final SHA-256 digest is returned as 64 lowercase hexadecimal characters. Every
operation and output is included, not only the last result. The wire response also
contains challenge/epoch IDs, operation count, backend description, and pipeline
diagnostics, so total HTTP payload size exceeds 32 bytes. Expected digests and bank
keys never appear in request bodies, public configuration, or pre-issue events.

## Deadline, issuance, and completion

Durable retirement precedes seed delivery. Only the current challenge is revealed.
The auditor checks task/epoch identity, operation count, and constant-time digest
comparison. Worker-reported timing is diagnostic. Auditor monotonic elapsed time
determines deadline compliance; comparison/storage overhead is separate from the
completion interval. Deadline, segment duration, operation count, and difficulty
remain fixed for equivalent local/forwarded experiments.

An early IDLE answer does not shorten an idle segment. An early ACTIVE answer leaves
the rest of that measurement window idle; the implementation never pads it with
unverified dummy computation. Tune operation count and segment duration from actual
pipeline and meter behavior. A late response, interrupted connection, failed
challenge, or process crash never permits reissuance.

Requests/results reject unknown fields. A response for an old challenge cannot be
made valid by changing its ID: the new seed and full specification bind the digest.
Different reference/CUDA implementations are tested against scalar CPU checks and
cross-backend known answers before building an operational bank.

Public conformance vectors are in validation/protocol-vectors.json. Their seeds
and answers are deliberately public and are never suitable for audit issuance.

## Primary implementation references

- [NVIDIA cuBLAS gemmEx](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex)
  documents supported integer input/compute/output types and alignment requirements.
- [CuPy installation](https://docs.cupy.dev/en/stable/install.html) describes optional
  CUDA component wheels used by the cuda extra.
- [CuPy RawKernel](https://docs.cupy.dev/en/stable/reference/generated/cupy.RawKernel.html)
  is used for the explicitly defined dependent-memory CUDA implementation.
- [Python sqlite3](https://docs.python.org/3/library/sqlite3.html) is the local
  transaction interface. Permanent issuance additionally depends on the code's
  pre-delivery commit ordering and operational prohibition on snapshot rollback.

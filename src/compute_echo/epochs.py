"""Epoch scheduler. All seeds are issued just in time, and only once.

This module sees no ground-truth routing mode. The control plane may change routing
only after stop/quiescence and before starting another measured washout.
"""

from __future__ import annotations

import asyncio
import secrets
import time
import uuid
from pathlib import Path

from .bank import ChallengeBank
from .detector import Calibration, detect
from .meter import MeterStore
from .models import EpochEvidence, Mode, Outcome, Segment, WorkloadSpec
from .storage import EventLog, write_json
from .transport import WorkerTransport


def balanced_schedule(specs: list[WorkloadSpec], repeats: int):
    if repeats < 1 or repeats > 100:
        raise ValueError("repeats must be 1..100")
    by_mode = {s.mode: s for s in specs}
    if len(by_mode) != len(specs) or Mode.IDLE not in by_mode or Mode.TENSOR not in by_mode:
        raise ValueError("unique mode specifications including IDLE and TENSOR required")
    if len({s.segment_s for s in specs}) != 1 or len({s.deadline_s for s in specs}) != 1:
        raise ValueError("all modes must use equivalent segment durations and deadlines")
    active = [s for s in specs if s.mode != Mode.IDLE] * repeats
    rng = secrets.SystemRandom()
    rng.shuffle(active)
    schedule = []
    for spec in active:
        pair = [by_mode[Mode.IDLE], spec]
        rng.shuffle(pair)
        schedule.extend(pair)
    return schedule


class EpochRunner:
    def __init__(
        self,
        bank: ChallengeBank,
        worker: WorkerTransport,
        meter: MeterStore,
        runs: Path,
        calibration: Calibration | None = None,
    ):
        self.bank, self.worker, self.meter = bank, worker, meter
        self.runs, self.calibration = runs, calibration
        self.stop_event = asyncio.Event()
        self.current_id = None
        self.status = {"state": "READY"}
        self.expected_sequence: list[str] = []
        self.lock = asyncio.Lock()

    async def pause(self, seconds: float, require_meter: bool):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.stop_event.is_set():
                return "OPERATOR_STOP"
            if require_meter and self.meter.health() != "OK":
                return self.meter.health()
            await asyncio.sleep(min(0.1, max(0, deadline - time.monotonic())))
        return None

    async def run(
        self,
        specs: list[WorkloadSpec],
        repeats: int = 4,
        *,
        washout_s: float | None = None,
        timing_only: bool = False,
    ):
        if self.lock.locked():
            raise RuntimeError("an epoch is already running")
        async with self.lock:
            self.stop_event.clear()
            schedule = balanced_schedule(specs, repeats)
            self.expected_sequence = [spec.mode.value for spec in schedule]
            for spec in specs:
                required = sum(s.key == spec.key for s in schedule)
                if self.bank.available(spec) < required:
                    raise ValueError(f"bank needs {required} unused {spec.mode} challenges")
            if self.calibration is not None:
                washout = self.calibration.profile.washout_s
                if washout_s is not None and abs(washout_s - washout) > 1e-6:
                    raise ValueError("washout must equal measured calibration value")
            elif washout_s is not None and 0 <= washout_s <= 120:
                washout = washout_s  # Explicit provisional physics/timing experiment.
            else:
                raise ValueError("physics runs require an explicit provisional washout")
            await self.worker.quiesce()
            epoch_id = uuid.uuid4().hex
            self.current_id = epoch_id
            directory = self.runs / epoch_id
            events = EventLog(directory / "events.jsonl")

            def status(state, **extra):
                self.status = {
                    "state": state,
                    "epoch_id": epoch_id,
                    "expected_challenge_sequence": self.expected_sequence,
                    **extra,
                }
                events.append({"time_ns": time.time_ns(), **self.status})

            status("WASHOUT", washout_s=washout, measured=self.calibration is not None)
            reason = await self.pause(washout, not timing_only)
            started = time.time_ns()
            start_mono = time.monotonic()
            segments = []
            request = None
            try:
                for index, spec in enumerate(schedule):
                    if reason or self.stop_event.is_set():
                        reason = reason or "OPERATOR_STOP"
                        break
                    if not timing_only and self.meter.health() != "OK":
                        reason = self.meter.health()
                        break
                    segment_start = time.time_ns()
                    segment_mono = time.monotonic()
                    challenge = self.bank.issue(spec, epoch_id)
                    # Only issued seed is logged. Expected digest is never public.
                    events.append(
                        {
                            "type": "challenge_issued",
                            "time_ns": segment_start,
                            "challenge": challenge.model_dump(mode="json"),
                        }
                    )
                    status(
                        "COLLECTING",
                        completed_segments=index,
                        total_segments=len(schedule),
                        current_challenge_index=index,
                        current_mode=spec.mode.value,
                    )
                    request = asyncio.create_task(self.worker.execute(challenge))
                    response, failure = None, None
                    try:
                        while not request.done():
                            if self.stop_event.is_set():
                                failure, reason = Outcome.INTERRUPTED, "OPERATOR_STOP"
                                break
                            if not timing_only and self.meter.health() != "OK":
                                failure, reason = Outcome.INTERRUPTED, self.meter.health()
                                break
                            remaining = spec.deadline_s - (time.monotonic() - segment_mono)
                            if remaining <= 0:
                                failure = Outcome.DEADLINE_FAILED
                                break
                            await asyncio.wait({request}, timeout=min(0.05, remaining))
                        if request.done():
                            response = request.result()
                    except Exception as error:
                        failure = Outcome.ERROR
                        events.append({"type": "transport_error", "error_type": type(error).__name__})
                    elapsed = time.monotonic() - segment_mono
                    if response is None:
                        # A cancelled HTTP await does not stop GPU execution. Explicitly drain it.
                        try:
                            await self.worker.cancel(challenge.challenge_id)
                        except Exception:
                            reason = "WORKER_NOT_QUIESCENT"
                        request.cancel()
                        await asyncio.gather(request, return_exceptions=True)
                    completion = self.bank.complete(challenge, response, elapsed, failure)
                    events.append(
                        {
                            "type": "challenge_completed",
                            "challenge_id": challenge.challenge_id,
                            **completion,
                            "response": response.model_dump(mode="json") if response else None,
                            "transport": getattr(self.worker, "last_transport", None),
                        }
                    )
                    reason = reason or await self.pause(
                        max(0, spec.segment_s - (time.monotonic() - segment_mono)), not timing_only
                    )
                    segment_end = time.time_ns()
                    # UTC clock jumps invalidate physical alignment; deadlines use monotonic time.
                    if abs((segment_end - segment_start) / 1e9 - (time.monotonic() - segment_mono)) > 0.25:
                        reason = "AUDITOR_CLOCK_JUMP"
                    segments.append(
                        Segment(
                            challenge_id=challenge.challenge_id,
                            mode=spec.mode,
                            spec_key=spec.key,
                            start_ns=segment_start,
                            end_ns=segment_end,
                            outcome=completion["outcome"],
                            digest_match=completion["digest_match"],
                            deadline_met=completion["deadline_met"],
                            elapsed_s=elapsed,
                            nominal_integer_operations=2 * spec.m * spec.n * spec.k * spec.operations
                            if spec.mode in {Mode.TENSOR, Mode.MIXED}
                            else 0,
                            dependent_memory_updates=spec.lanes * spec.memory_steps * spec.operations
                            if spec.mode in {Mode.MEMORY, Mode.MIXED}
                            else 0,
                            timings=response.timings if response else None,
                        )
                    )
                    if completion["outcome"] in {Outcome.INCORRECT, Outcome.ERROR}:
                        reason = reason or "COMPUTATIONAL_FAILURE"
                    if completion["outcome"] == Outcome.DEADLINE_FAILED:
                        reason = reason or "DEADLINE_FAILED"
                    status("COLLECTING", completed_segments=len(segments), total_segments=len(schedule))
            except asyncio.CancelledError:
                reason = "TASK_CANCELLED"
                self.stop_event.set()
            except Exception as error:
                reason = "EPOCH_ERROR:" + type(error).__name__
            finally:
                try:
                    await self.worker.quiesce()
                except Exception:
                    reason = "WORKER_NOT_QUIESCENT"
                if request is not None:
                    request.cancel()
                    await asyncio.gather(request, return_exceptions=True)
                self.bank.interrupt_epoch(epoch_id)
            ended = time.time_ns()
            if abs((ended - started) / 1e9 - (time.monotonic() - start_mono)) > 0.25:
                reason = "AUDITOR_CLOCK_JUMP"
            evidence = EpochEvidence(
                epoch_id=epoch_id,
                started_at_ns=started,
                ended_at_ns=ended,
                meter_id=self.meter.config.meter_id,
                provenance=self.meter.config.provenance,
                measurement_intent="timing_only" if timing_only else "physical_audit",
                source_freshness_verified=self.meter.config.source_freshness != "unknown",
                segments=segments,
                samples=self.meter.samples(started, ended),
                complete=reason is None and len(segments) == len(schedule),
                stop_reason=reason,
            )
            write_json(directory / "evidence.json", evidence)
            finding = detect(evidence, self.calibration)
            write_json(directory / "finding.json", finding)
            status(
                "COMPLETE" if evidence.complete else "INTERRUPTED",
                completed_segments=len(segments),
                total_segments=len(schedule),
                finding=finding,
            )
            self.current_id = None
            return evidence, finding

    def stop(self):
        self.stop_event.set()

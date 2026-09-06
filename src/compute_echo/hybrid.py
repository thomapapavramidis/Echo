"""Transparent hybrid demo: live computation plus an illustrative digital twin."""

from __future__ import annotations

import asyncio
import hashlib
import math
import random
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from enum import StrEnum
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import Field

from .hybrid_dashboard import HYBRID_DASHBOARD_HTML
from .models import Challenge, Mode, StrictModel, WorkloadSpec
from .storage import write_json
from .telemetry import DemoTelemetryHub
from .transport import HTTPWorker
from .workloads import WorkloadEngine

HYBRID_DISCLOSURE = "LIVE COMPUTE + SIMULATED FACILITY METERS"
TWIN_LABEL = "DECLARED FACILITY METER DIGITAL TWIN — SIMULATED"
REMOTE_LABEL = "REMOTE FACILITY METER DIGITAL TWIN — SIMULATED DEMO GROUND TRUTH"
RAW_NVML_LABEL = "LIVE NVML HOST TELEMETRY — DIAGNOSTIC ONLY"
REPLAY_LABEL = "REPLAY OF PREVIOUSLY CAPTURED DEMONSTRATION"
DECISIVE_HEADLINE = "CORRECT ANSWERS — WRONG PHYSICAL SITE"


class HybridRunMode(StrEnum):
    LIVE_HYBRID = "LIVE HYBRID"
    CAPTURE = "CAPTURE"
    REPLAY = "REPLAY"


class PatternRequest(StrictModel):
    seed: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$")


TWIN_TARGETS = {
    Mode.IDLE: (120.0, 2.0, 3.0),
    Mode.TENSOR: (535.0, 94.0, 26.0),
    Mode.MEMORY: (350.0, 34.0, 91.0),
    Mode.MIXED: (585.0, 86.0, 82.0),
}


def randomized_schedule(specs: list[WorkloadSpec], pattern_seed: str, scene_number: int):
    """Same workload multiset per scene, with a reproducible randomized order."""
    if len(specs) != 4 or {spec.mode for spec in specs} != set(Mode):
        raise ValueError("HYBRID_DEMO requires exactly IDLE, TENSOR, MEMORY, and MIXED")
    schedule = list(specs)
    seed = int.from_bytes(hashlib.sha256(f"{pattern_seed}:{scene_number}".encode()).digest()[:8], "big")
    random.Random(seed).shuffle(schedule)
    return schedule


def _mode_at(schedule: list[WorkloadSpec], elapsed_s: float) -> Mode:
    cursor = 0.0
    for spec in schedule:
        cursor += spec.segment_s
        if elapsed_s < cursor:
            return spec.mode
    return schedule[-1].mode


def digital_twin_stream(
    schedule: list[WorkloadSpec], follows_schedule: bool, seed: str, sample_period_s: float = 0.2
) -> list[dict]:
    """Generate one deterministic simulated facility-meter trace."""
    rng = random.Random(int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big"))
    duration = sum(spec.segment_s for spec in schedule)
    shortest_segment = min(spec.segment_s for spec in schedule)
    lag_s = min(0.45, shortest_segment * 0.15)
    smoothing_s = min(1.0, shortest_segment * 0.25)
    power, compute, memory = TWIN_TARGETS[Mode.IDLE]
    samples = []
    count = math.ceil(duration / sample_period_s) + 1
    variation = {index: 1 + rng.uniform(-0.045, 0.045) for index in range(len(schedule))}

    def segment_index(t):
        cursor = 0.0
        for index, spec in enumerate(schedule):
            cursor += spec.segment_s
            if t < cursor:
                return index
        return len(schedule) - 1

    for index in range(count):
        elapsed = min(duration, index * sample_period_s)
        delayed = max(0.0, elapsed - lag_s)
        mode = _mode_at(schedule, delayed) if follows_schedule else Mode.IDLE
        target = TWIN_TARGETS[mode]
        scale = variation[segment_index(delayed)] if mode != Mode.IDLE else 1.0
        target = (
            120 + (target[0] - 120) * scale,
            target[1] * scale,
            target[2] * scale,
        )
        alpha = 1 - math.exp(-sample_period_s / smoothing_s)
        power += alpha * (target[0] - power)
        compute += alpha * (target[1] - compute)
        memory += alpha * (target[2] - memory)
        samples.append(
            {
                "time_s": elapsed,
                "facility_power_w": min(700.0, max(0.0, power + rng.uniform(-3.0, 3.0))),
                "gpu_compute_utilization_pct": min(100.0, max(0.0, compute + rng.uniform(-1.6, 1.6))),
                "gpu_memory_utilization_pct": min(100.0, max(0.0, memory + rng.uniform(-1.6, 1.6))),
            }
        )
    return samples


def evaluate_digital_twin(schedule: list[WorkloadSpec], samples: list[dict]) -> dict:
    """Illustrative route-blind comparison; routing ground truth is not an input."""
    expected, observed = [], []
    for sample in samples:
        target = TWIN_TARGETS[_mode_at(schedule, sample["time_s"])]
        expected.extend(((target[0] - 120) / 580, target[1] / 100, target[2] / 100))
        observed.extend(
            (
                (sample["facility_power_w"] - 120) / 580,
                sample["gpu_compute_utilization_pct"] / 100,
                sample["gpu_memory_utilization_pct"] / 100,
            )
        )
    expected_array, observed_array = np.asarray(expected), np.asarray(observed)
    correlation = (
        float(np.corrcoef(expected_array, observed_array)[0, 1]) if np.std(observed_array) > 1e-6 else None
    )
    power_span = max(x["facility_power_w"] for x in samples) - min(x["facility_power_w"] for x in samples)
    matched = correlation is not None and correlation >= 0.45 and power_span >= 80
    return {
        "physical_digital_twin_status": "ECHO MATCHED" if matched else "ECHO MISSING",
        "illustrative_correlation": correlation,
        "illustrative_power_span_w": power_span,
        "illustrative_rule": "correlation >= 0.45 and simulated power span >= 80 W",
        "production_physical_verification": False,
        "production_trust_gate_eligible": False,
        "calibration_threshold": None,
        "routing_label_received": False,
        "disclosure": TWIN_LABEL,
    }


class HybridDemoRuntime:
    def __init__(
        self,
        worker: HTTPWorker,
        telemetry: DemoTelemetryHub,
        specs: list[WorkloadSpec],
        capture_path: Path,
        *,
        facility_name: str = "Declared facility H100",
        facility_region: str = "US Northeast",
        helper_name: str = "Remote helper H100",
        helper_region: str = "Iceland / Europe",
    ):
        if len(specs) != 4 or {spec.mode for spec in specs} != set(Mode):
            raise ValueError("hybrid demo needs one specification for each workload mode")
        if len({spec.segment_s for spec in specs}) != 1 or len({spec.deadline_s for spec in specs}) != 1:
            raise ValueError("hybrid demo modes require equivalent segment durations and deadlines")
        self.worker, self.telemetry, self.specs = worker, telemetry, specs
        self.capture_path = capture_path
        self.reference = WorkloadEngine("cpu")
        self.facility_name, self.facility_region = facility_name, facility_region
        self.helper_name, self.helper_region = helper_name, helper_region
        self.mode = HybridRunMode.LIVE_HYBRID
        self.pattern_seed = "echo-demo-001"
        self.scene_number = 0
        self.selected_route = "unknown"
        self.status = "READY"
        self.error = None
        self.current_scene = None
        self.last_scene = None
        self.events = []
        self.capture_scenes = []
        self.task: asyncio.Task | None = None
        self.stop_event = asyncio.Event()
        self.lock = asyncio.Lock()
        self.scene_started_mono = None

    def _event(self, event: str, **details):
        row = {"time_ns": time.time_ns(), "event": event, **details}
        self.events.append(row)
        self.events = self.events[-500:]
        return row

    def state(self):
        telemetry = self.telemetry.snapshot()
        elapsed = time.monotonic() - self.scene_started_mono if self.scene_started_mono is not None else 0
        if (
            self.scene_started_mono is None
            and self.current_scene
            and self.status
            in {
                "COMPLETE",
                "REPLAY_COMPLETE",
            }
        ):
            elapsed = self.current_scene.get("total_duration_s", 0)
        return {
            "system_mode": "HYBRID_DEMO",
            "run_mode": self.mode.value,
            "disclosure": HYBRID_DISCLOSURE,
            "declared_facility_meter_label": TWIN_LABEL,
            "remote_facility_meter_label": REMOTE_LABEL,
            "raw_host_telemetry_label": RAW_NVML_LABEL,
            "simulated_remote_independent_evidence": False,
            "replay_label": REPLAY_LABEL if self.mode == HybridRunMode.REPLAY else None,
            "production_receipts_enabled": False,
            "production_physical_verification": "DISABLED_IN_HYBRID_DEMO",
            "challenge_bank_used": False,
            "workload_specifications_frozen": False,
            "reference_strategy": "on-demand laptop CPU before seed issuance",
            "worker_route_trust": "stage ground truth only",
            "selected_route": self.selected_route,
            "status": self.status,
            "error": self.error,
            "pattern_seed": self.pattern_seed,
            "scene_elapsed_s": elapsed,
            "facility": {"name": self.facility_name, "region": self.facility_region},
            "helper": {"name": self.helper_name, "region": self.helper_region},
            "connections": {
                "facility": bool(
                    telemetry["declared_site"] and telemetry["declared_site"][-1].get("endpoint_reachable")
                ),
                "helper": bool(
                    telemetry["remote_helper"] and telemetry["remote_helper"][-1].get("endpoint_reachable")
                ),
            },
            "current_scene": self.current_scene,
            "last_scene": self.last_scene,
            "telemetry": telemetry,
            "events": self.events[-30:],
        }

    async def set_route(self, route: str):
        if route not in {"local", "forwarded"}:
            raise ValueError("route must be local or forwarded")
        if self.mode != HybridRunMode.REPLAY:
            self._event("route_drain_started", requested_route=route)
            await self.worker.quiesce()
            self._event("route_drain_completed", requested_route=route)
            response = await self.worker.set_route(route)
            if response.get("route") != route:
                raise RuntimeError("facility did not confirm requested route")
        self.selected_route = route
        self._event("route_confirmed", route=route, trust="stage_ground_truth_only")

    async def _live_scene(self, route: str):
        await self.set_route(route)
        scene_number = self.scene_number
        self.scene_number += 1
        schedule = randomized_schedule(self.specs, self.pattern_seed, scene_number)
        scene_id = uuid.uuid4().hex
        declared_twin = digital_twin_stream(
            schedule,
            route == "local",
            f"{self.pattern_seed}:{scene_number}:{route}:declared",
        )
        remote_twin = digital_twin_stream(
            schedule,
            route == "forwarded",
            f"{self.pattern_seed}:{scene_number}:{route}:remote",
        )
        self.current_scene = {
            "scene_id": scene_id,
            "route_stage_ground_truth": route,
            "route_display": "Executed locally" if route == "local" else "Forwarded US → Iceland",
            "schedule": [spec.mode.value for spec in schedule],
            "segment_duration_s": schedule[0].segment_s,
            "total_duration_s": sum(spec.segment_s for spec in schedule),
            "declared_facility_digital_twin": declared_twin,
            "remote_facility_digital_twin": remote_twin,
            "simulated_remote_independent_evidence": False,
            "computations": [],
            "current_segment": None,
            "headline": "LIVE CHALLENGE IN PROGRESS",
        }
        self.scene_started_mono = time.monotonic()
        self.status, self.error = "RUNNING", None
        scene_telemetry_start = time.time_ns()
        self._event("scene_started", scene_id=scene_id, route=route)
        all_valid, all_on_time = True, True
        for index, spec in enumerate(schedule):
            if self.stop_event.is_set():
                raise asyncio.CancelledError
            seed = secrets.token_bytes(32)
            self.current_scene["current_segment"] = index
            self.current_scene["computations"].append(
                {
                    "mode": spec.mode.value,
                    "status": "REFERENCE_PREPARING",
                    "fresh_challenge_issued": False,
                    "digest_valid": None,
                    "deadline_met": None,
                    "facility_request_sent": False,
                    "helper_response_received": False,
                }
            )
            expected, _ = await asyncio.to_thread(self.reference.execute, seed, spec)
            challenge = Challenge(
                challenge_id=uuid.uuid4().hex,
                epoch_id=scene_id,
                seed=seed.hex(),
                spec=spec,
            )
            record = self.current_scene["computations"][-1]
            record.update(
                status="ISSUED",
                fresh_challenge_issued=True,
                facility_request_sent=True,
            )
            self._event(
                "fresh_challenge_issued",
                scene_id=scene_id,
                challenge_id=challenge.challenge_id,
                mode=spec.mode.value,
            )
            segment_started = time.monotonic()
            try:
                response = await self.worker.execute(challenge)
                elapsed = self.worker.last_transport["auditor_round_trip_s"]
                digest_valid = response.digest == expected
                deadline_met = elapsed <= spec.deadline_s
                record.update(
                    status="VERIFIED" if digest_valid else "INCORRECT",
                    digest_valid=digest_valid,
                    deadline_met=deadline_met,
                    end_to_end_s=elapsed,
                    worker_backend=response.backend,
                    worker_timings=response.timings.model_dump(mode="json"),
                    confirmed_worker_route=route,
                    helper_response_received=route == "forwarded",
                )
                all_valid &= digest_valid
                all_on_time &= deadline_met
                self._event(
                    "challenge_completed",
                    challenge_id=challenge.challenge_id,
                    digest_valid=digest_valid,
                    deadline_met=deadline_met,
                    end_to_end_s=elapsed,
                    confirmed_worker_route=route,
                    helper_response_received=route == "forwarded",
                )
            except Exception as failure:
                record.update(
                    status="WORKER_UNAVAILABLE",
                    digest_valid=False,
                    deadline_met=False,
                    error=f"{type(failure).__name__}: {failure}"[:500],
                )
                all_valid = all_on_time = False
                raise RuntimeError(
                    f"{route} worker path failed during {spec.mode.value}: {failure}"
                ) from failure
            await asyncio.sleep(max(0.0, spec.segment_s - (time.monotonic() - segment_started)))
        # The route-blind detector receives only the expected schedule and the
        # simulated declared-facility meter. Route and remote-demo ground truth
        # remain in the controller/display path.
        detector = evaluate_digital_twin(schedule, declared_twin)
        contradiction = (
            all_valid and all_on_time and detector["physical_digital_twin_status"] == "ECHO MISSING"
        )
        headline = (
            DECISIVE_HEADLINE
            if contradiction
            else (
                "CONSISTENT WITH EXECUTION AT DECLARED SITE"
                if all_valid and all_on_time and detector["physical_digital_twin_status"] == "ECHO MATCHED"
                else "DEMONSTRATION INCOMPLETE"
            )
        )
        scene = {
            **self.current_scene,
            "current_segment": len(schedule),
            "computational_status": "VERIFIED" if all_valid else "FAILED",
            "performance_status": "COMPLETED BEFORE DEMO DEADLINE" if all_on_time else "DEADLINE FAILED",
            "digital_twin_result": detector,
            "headline": headline,
            "live_execution_evidence": {
                "fresh_challenges_issued": sum(
                    bool(row["fresh_challenge_issued"]) for row in self.current_scene["computations"]
                ),
                "facility_requests_sent": sum(
                    bool(row["facility_request_sent"]) for row in self.current_scene["computations"]
                ),
                "helper_responses_received": sum(
                    bool(row["helper_response_received"]) for row in self.current_scene["computations"]
                ),
                "route_confirmed": route,
                "digest_checks_required": True,
                "laptop_round_trip_measured": True,
            },
            "raw_facility_nvml": [
                row
                for row in self.telemetry.snapshot()["declared_site"]
                if row["received_at_ns"] >= scene_telemetry_start
            ],
            "raw_helper_nvml": [
                row
                for row in self.telemetry.snapshot()["remote_helper"]
                if row["received_at_ns"] >= scene_telemetry_start
            ],
            "finished_at_ns": time.time_ns(),
        }
        self.last_scene, self.current_scene = scene, scene
        self.status = "COMPLETE"
        self.scene_started_mono = None
        self._event("scene_completed", scene_id=scene_id, headline=headline)
        if self.mode == HybridRunMode.CAPTURE:
            self.capture_scenes.append(scene)
            self._write_capture()
        return scene

    def _write_capture(self):
        write_json(
            self.capture_path,
            {
                "kind": "hybrid-demo-capture-v1",
                "disclosure": HYBRID_DISCLOSURE,
                "replay_disclosure": REPLAY_LABEL,
                "production_receipt": False,
                "production_physical_verification": False,
                "pattern_seed": self.pattern_seed,
                "scenes": self.capture_scenes,
                "events": self.events,
            },
        )

    async def _replay_scene(self, index: int):
        if not self.capture_path.is_file():
            raise RuntimeError("no captured demonstration is available")
        import json

        capture = json.loads(self.capture_path.read_text(encoding="utf-8"))
        scenes = capture.get("scenes", [])
        if index >= len(scenes):
            raise RuntimeError("captured demonstration does not contain the requested scene")
        scene = scenes[index]
        self.current_scene = scene
        self.status, self.error = "REPLAYING", None
        self.scene_started_mono = time.monotonic()
        self._event("replay_started", captured_scene_id=scene.get("scene_id"))
        await asyncio.sleep(scene["total_duration_s"])
        self.last_scene = scene
        self.status = "REPLAY_COMPLETE"
        self.scene_started_mono = None
        self._event("replay_completed", captured_scene_id=scene.get("scene_id"))
        return scene

    async def run_selected_scene(self):
        if self.mode == HybridRunMode.REPLAY:
            route_index = 1 if self.selected_route == "forwarded" else 0
            return await self._replay_scene(route_index)
        route = self.selected_route if self.selected_route in {"local", "forwarded"} else "local"
        return await self._live_scene(route)

    async def run_auto(self):
        if self.mode == HybridRunMode.REPLAY:
            await self._replay_scene(0)
            await asyncio.sleep(1)
            await self._replay_scene(1)
            return
        await self._live_scene("local")
        await asyncio.sleep(1)
        await self._live_scene("forwarded")

    async def reset(self):
        replaying = self.mode == HybridRunMode.REPLAY
        self.stop_event.set()
        if self.task and not self.task.done():
            await self.worker.quiesce()
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        self.stop_event = asyncio.Event()
        if not replaying:
            await self.set_route("local")
        else:
            self.selected_route = "local"
        self.mode = HybridRunMode.LIVE_HYBRID
        self.pattern_seed = "echo-demo-001"
        self.scene_number = 0
        self.status, self.error = "READY", None
        self.current_scene = self.last_scene = None
        self.events = []
        self.scene_started_mono = None

    async def run_guarded(self, auto: bool):
        try:
            if auto:
                await self.run_auto()
            else:
                await self.run_selected_scene()
        except asyncio.CancelledError:
            self.status = "RESET"
        except Exception as failure:
            self.error = f"{type(failure).__name__}: {failure}"[:500]
            self.status = "CONNECTION_ERROR"
            self.scene_started_mono = None
            self._event("demo_failed", error=self.error)


def create_hybrid_app(runtime: HybridDemoRuntime) -> FastAPI:
    """Create a localhost controller with no production receipt or meter endpoints."""

    @asynccontextmanager
    async def lifespan(app):
        runtime.telemetry.start()
        try:
            try:
                health = await runtime.worker.health()
                runtime.selected_route = health.get("route", "unknown")
            except Exception as failure:
                runtime.error = f"facility unavailable: {failure}"[:500]
            yield
        finally:
            runtime.stop_event.set()
            if runtime.task:
                runtime.task.cancel()
                await asyncio.gather(runtime.task, return_exceptions=True)
            await asyncio.gather(runtime.worker.quiesce(), return_exceptions=True)
            await runtime.telemetry.close()
            await runtime.worker.close()

    app = FastAPI(title="Compute Echo HYBRID_DEMO", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return HYBRID_DASHBOARD_HTML

    @app.get("/health")
    async def health():
        state = runtime.state()
        return {
            "status": runtime.status,
            "system_mode": "HYBRID_DEMO",
            "disclosure": HYBRID_DISCLOSURE,
            "connections": state["connections"],
            "selected_route": runtime.selected_route,
            "production_receipts_enabled": False,
        }

    @app.get("/api/state")
    async def state():
        return runtime.state()

    def idle_required():
        if runtime.task and not runtime.task.done():
            raise HTTPException(409, "a scene is active; reset or wait for it to drain")

    @app.post("/api/route/{route}")
    async def route(route: str):
        idle_required()
        try:
            await runtime.set_route(route)
        except Exception as failure:
            raise HTTPException(503, f"route change failed: {failure}") from failure
        return {"route": runtime.selected_route, "actual_worker_route_confirmed": True}

    @app.post("/api/start", status_code=202)
    async def start():
        idle_required()
        runtime.task = asyncio.create_task(runtime.run_guarded(False))
        return {"started": True, "run_mode": runtime.mode.value}

    @app.post("/api/auto", status_code=202)
    async def auto():
        idle_required()
        runtime.task = asyncio.create_task(runtime.run_guarded(True))
        return {"started": True, "sequence": ["local", "forwarded"]}

    @app.post("/api/reset")
    async def reset():
        await runtime.reset()
        return runtime.state()

    @app.post("/api/mode/{mode}")
    async def mode(mode: str):
        idle_required()
        try:
            selected = HybridRunMode(mode.replace("_", " ").upper())
        except ValueError as failure:
            raise HTTPException(422, "mode must be LIVE_HYBRID, CAPTURE, or REPLAY") from failure
        if selected == HybridRunMode.REPLAY and not runtime.capture_path.is_file():
            raise HTTPException(409, "no captured demonstration is available")
        runtime.mode = selected
        if selected == HybridRunMode.CAPTURE:
            runtime.capture_scenes = []
        runtime.status = "READY"
        return {
            "run_mode": runtime.mode.value,
            "replay_label": REPLAY_LABEL if selected == HybridRunMode.REPLAY else None,
        }

    @app.post("/api/pattern/new")
    async def new_pattern():
        idle_required()
        runtime.pattern_seed = "echo-" + secrets.token_hex(4)
        runtime.scene_number = 0
        return {"pattern_seed": runtime.pattern_seed}

    @app.post("/api/pattern")
    async def set_pattern(request: PatternRequest):
        idle_required()
        runtime.pattern_seed = request.seed
        runtime.scene_number = 0
        return {"pattern_seed": runtime.pattern_seed}

    return app

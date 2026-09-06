import inspect

import httpx

from compute_echo.hybrid import (
    HYBRID_DISCLOSURE,
    RAW_NVML_LABEL,
    REMOTE_LABEL,
    REPLAY_LABEL,
    TWIN_LABEL,
    HybridDemoRuntime,
    HybridRunMode,
    create_hybrid_app,
    digital_twin_stream,
    evaluate_digital_twin,
    randomized_schedule,
)
from compute_echo.hybrid_dashboard import HYBRID_DASHBOARD_HTML
from compute_echo.models import Mode, PipelineTimes, WorkerResult, WorkloadSpec
from compute_echo.workloads import WorkloadEngine


def tiny_specs():
    return [
        WorkloadSpec(
            mode=mode,
            operations=1,
            m=16,
            n=16,
            k=16,
            lanes=1,
            words_per_lane=16,
            memory_steps=1,
            segment_s=0.03,
            deadline_s=0.02,
        )
        for mode in Mode
    ]


class FakeHub:
    def start(self):
        pass

    async def close(self):
        pass

    def snapshot(self):
        return {
            "provenance_label": "HOST GPU TELEMETRY — PROOF OF CONCEPT",
            "production_trust_gate_eligible": False,
            "declared_site": [],
            "remote_helper": [],
            "stream_summaries": {},
            "helper_disclosure": "ATTACK GROUND TRUTH — NOT NORMALLY AVAILABLE TO AUDITOR",
        }


class FakeWorker:
    def __init__(self, fail=False):
        self.route = "local"
        self.calls = []
        self.fail = fail
        self.last_transport = None
        self.engine = WorkloadEngine("cpu")

    async def health(self):
        return {"status": "ready", "route": self.route, "backend": "cuda"}

    async def quiesce(self):
        self.calls.append("quiesce")

    async def set_route(self, route):
        self.calls.append("route:" + route)
        if self.fail:
            raise httpx.ConnectError("fixture worker unavailable")
        self.route = route
        return {"route": route}

    async def execute(self, challenge):
        self.calls.append("execute:" + self.route)
        digest, _ = self.engine.execute(bytes.fromhex(challenge.seed), challenge.spec)
        self.last_transport = {"auditor_round_trip_s": 0.001}
        return WorkerResult(
            challenge_id=challenge.challenge_id,
            epoch_id=challenge.epoch_id,
            digest=digest,
            backend="cuda:H100 fixture",
            operations_completed=0 if challenge.spec.mode == Mode.IDLE else challenge.spec.operations,
            timings=PipelineTimes(execution_s=0.0005, total_worker_s=0.0008),
        )

    async def close(self):
        self.calls.append("close")


class IncorrectDigestWorker(FakeWorker):
    async def execute(self, challenge):
        result = await super().execute(challenge)
        return result.model_copy(update={"digest": "00" * 32})


def test_digital_twin_is_deterministic_bounded_and_detector_is_route_blind():
    specs = [
        WorkloadSpec.model_validate({**spec.model_dump(), "segment_s": 4, "deadline_s": 3})
        for spec in tiny_specs()
    ]
    schedule = randomized_schedule(specs, "fixed", 0)
    assert [x.mode for x in schedule] == [x.mode for x in randomized_schedule(specs, "fixed", 0)]
    active = digital_twin_stream(schedule, True, "fixed:active")
    idle = digital_twin_stream(schedule, False, "fixed:idle")
    assert active == digital_twin_stream(schedule, True, "fixed:active")
    assert max(x["facility_power_w"] for x in active) <= 700
    assert evaluate_digital_twin(schedule, active)["physical_digital_twin_status"] == "ECHO MATCHED"
    missing = evaluate_digital_twin(schedule, idle)
    assert missing["physical_digital_twin_status"] == "ECHO MISSING"
    assert missing["production_physical_verification"] is False
    assert set(inspect.signature(evaluate_digital_twin).parameters) == {"schedule", "samples"}


async def test_live_scenes_verify_fresh_digests_and_real_route_changes_drain(tmp_path):
    worker = FakeWorker()
    runtime = HybridDemoRuntime(worker, FakeHub(), tiny_specs(), tmp_path / "capture.json")
    local = await runtime._live_scene("local")
    forwarded = await runtime._live_scene("forwarded")
    assert local["computational_status"] == forwarded["computational_status"] == "VERIFIED"
    assert local["headline"] == "CONSISTENT WITH EXECUTION AT DECLARED SITE"
    assert forwarded["headline"] == "CORRECT ANSWERS — WRONG PHYSICAL SITE"
    assert all(row["fresh_challenge_issued"] and row["digest_valid"] for row in local["computations"])
    assert all(row["facility_request_sent"] for row in forwarded["computations"])
    assert all(row["helper_response_received"] for row in forwarded["computations"])
    assert not any(row["helper_response_received"] for row in local["computations"])
    assert forwarded["live_execution_evidence"]["helper_responses_received"] == 4
    assert forwarded["simulated_remote_independent_evidence"] is False
    assert local["digital_twin_result"]["physical_digital_twin_status"] == "ECHO MATCHED"
    local_us_span = max(x["facility_power_w"] for x in local["declared_facility_digital_twin"]) - min(
        x["facility_power_w"] for x in local["declared_facility_digital_twin"]
    )
    local_remote_span = max(x["facility_power_w"] for x in local["remote_facility_digital_twin"]) - min(
        x["facility_power_w"] for x in local["remote_facility_digital_twin"]
    )
    forwarded_us_span = max(x["facility_power_w"] for x in forwarded["declared_facility_digital_twin"]) - min(
        x["facility_power_w"] for x in forwarded["declared_facility_digital_twin"]
    )
    forwarded_remote_span = max(
        x["facility_power_w"] for x in forwarded["remote_facility_digital_twin"]
    ) - min(x["facility_power_w"] for x in forwarded["remote_facility_digital_twin"])
    assert local_us_span > local_remote_span
    assert forwarded_remote_span > forwarded_us_span
    assert max(x["facility_power_w"] for x in local["declared_facility_digital_twin"]) > max(
        x["facility_power_w"] for x in local["remote_facility_digital_twin"]
    )
    assert max(x["facility_power_w"] for x in forwarded["remote_facility_digital_twin"]) > max(
        x["facility_power_w"] for x in forwarded["declared_facility_digital_twin"]
    )
    assert worker.calls.index("quiesce") < worker.calls.index("route:local")
    assert "execute:local" in worker.calls and "execute:forwarded" in worker.calls
    events = [row["event"] for row in runtime.events]
    assert events.index("route_drain_started") < events.index("route_drain_completed")
    assert events.index("route_drain_completed") < events.index("route_confirmed")


async def test_capture_replay_reset_and_no_production_endpoints(tmp_path):
    worker = FakeWorker()
    runtime = HybridDemoRuntime(worker, FakeHub(), tiny_specs(), tmp_path / "capture.json")
    app = create_hybrid_app(runtime)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://hybrid"
        ) as client:
            assert (await client.get("/epochs/abc/receipt")).status_code == 404
            assert (await client.post("/meter/samples", json={})).status_code == 404
            state = (await client.get("/api/state")).json()
            assert state["production_receipts_enabled"] is False
            assert state["production_physical_verification"] == "DISABLED_IN_HYBRID_DEMO"
            assert state["simulated_remote_independent_evidence"] is False
            paths = {route.path for route in app.routes}
            assert not any("receipt" in path or "physical-verification" in path for path in paths)
            assert (await client.post("/api/mode/CAPTURE")).status_code == 200
            runtime.mode = HybridRunMode.CAPTURE
            await runtime._live_scene("local")
            await runtime._live_scene("forwarded")
            assert runtime.capture_path.exists()
            assert (await client.post("/api/mode/REPLAY")).status_code == 200
            calls_before_replay = list(worker.calls)
            assert (await client.post("/api/route/forwarded")).status_code == 200
            await runtime.run_selected_scene()
            assert worker.calls == calls_before_replay
            assert runtime.state()["replay_label"] == REPLAY_LABEL
            runtime.pattern_seed = "changed"
            await runtime.reset()
            assert runtime.pattern_seed == "echo-demo-001"
            assert runtime.mode == HybridRunMode.LIVE_HYBRID
            assert runtime.current_scene is None and runtime.status == "READY"


async def test_worker_outage_is_explicit(tmp_path):
    runtime = HybridDemoRuntime(FakeWorker(fail=True), FakeHub(), tiny_specs(), tmp_path / "capture.json")
    app = create_hybrid_app(runtime)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://hybrid") as client:
        response = await client.post("/api/route/forwarded")
        assert response.status_code == 503
        assert "route change failed" in response.json()["detail"]


async def test_live_hybrid_requires_real_digest_validation(tmp_path):
    worker = IncorrectDigestWorker()
    runtime = HybridDemoRuntime(worker, FakeHub(), tiny_specs(), tmp_path / "capture.json")
    assert runtime.mode == HybridRunMode.LIVE_HYBRID
    scene = await runtime._live_scene("forwarded")
    assert scene["computational_status"] == "FAILED"
    assert scene["headline"] == "DEMONSTRATION INCOMPLETE"
    assert all(row["status"] == "INCORRECT" for row in scene["computations"])
    assert "route:forwarded" in worker.calls


def test_every_judge_facing_disclosure_is_literal_in_dashboard():
    for label in (
        HYBRID_DISCLOSURE,
        TWIN_LABEL,
        REMOTE_LABEL,
        REPLAY_LABEL,
        RAW_NVML_LABEL,
    ):
        assert label in HYBRID_DASHBOARD_HTML


def test_prominent_facility_graphs_are_simulated_and_nvml_is_diagnostic_only():
    stage, diagnostic = HYBRID_DASHBOARD_HTML.split("<details>", 1)
    assert TWIN_LABEL in stage
    assert REMOTE_LABEL in stage
    assert RAW_NVML_LABEL not in stage
    assert "rawFacilityChart" not in stage and "rawHelperChart" not in stage
    assert "RAW HOST TELEMETRY" in diagnostic
    assert RAW_NVML_LABEL in diagnostic
    assert "rawFacilityChart" in diagnostic and "rawHelperChart" in diagnostic
    assert "independent facility meter nor detector evidence" in diagnostic
    for label in (TWIN_LABEL, REMOTE_LABEL):
        assert "MEASURED" not in label
        assert "LIVE TELEMETRY" not in label

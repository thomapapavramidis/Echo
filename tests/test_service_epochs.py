import asyncio
import json
import uuid

import httpx
import pytest
from cryptography.exceptions import InvalidSignature
from fastapi import HTTPException

from compute_echo.bank import ChallengeBank
from compute_echo.epochs import EpochRunner, balanced_schedule
from compute_echo.meter import MeterConfig, MeterStore
from compute_echo.models import Challenge, Mode, WorkloadSpec
from compute_echo.receipt import verify_receipt
from compute_echo.service import ProcessLock, create_auditor_app
from compute_echo.transport import HTTPWorker
from compute_echo.worker import WorkerRuntime, create_worker_app
from compute_echo.workloads import WorkloadEngine

TOKEN = "test-worker-token-never-use-in-production"
CONTROL = "test-control-token-never-use-in-production"
ADMIN = "test-admin-token-never-use-in-production"
METER = "test-meter-token-never-use-in-production"


async def in_process_worker(tmp_path, name, helper=None):
    runtime = WorkerRuntime("cpu", tmp_path / (name + ".sqlite3"), helper)
    app = create_worker_app(runtime, TOKEN, CONTROL)
    transport = HTTPWorker("http://" + name, TOKEN, CONTROL)
    await transport.client.aclose()
    transport.client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://" + name,
        headers={"Authorization": "Bearer " + TOKEN},
    )
    return runtime, transport, app


def setup_runner(tmp_path, worker, repeats=1):
    config = MeterConfig(
        meter_id="absent", provenance="simulation", installation="No real meter in this test"
    )
    meter = MeterStore(tmp_path / "meter.sqlite3", config)
    bank = ChallengeBank(tmp_path / "private")
    specs = [
        WorkloadSpec(
            mode=m,
            m=32,
            n=32,
            k=32,
            lanes=4,
            words_per_lane=32,
            memory_steps=8,
            segment_s=0.08,
            deadline_s=0.07,
        )
        for m in Mode
    ]
    for spec in specs:
        bank.generate(spec, 6 * repeats, WorkloadEngine())
    return EpochRunner(bank, worker, meter, tmp_path / "runs"), specs


async def test_full_local_and_forwarded_epochs_use_different_seeds(tmp_path):
    helper, helper_http, _ = await in_process_worker(tmp_path, "helper")
    facility, facility_http, _ = await in_process_worker(tmp_path, "facility", helper_http)
    runner, specs = setup_runner(tmp_path, facility_http)
    try:
        await facility_http.set_route("local")
        local, finding = await runner.run(specs, 1, washout_s=0, timing_only=True)
        await facility_http.set_route("forwarded")
        forwarded, forwarded_finding = await runner.run(specs, 1, washout_s=0, timing_only=True)
        assert local.complete and forwarded.complete
        assert all(s.digest_match and s.deadline_met for e in (local, forwarded) for s in e.segments)
        assert finding["physical_response"] == forwarded_finding["physical_response"] == "INSUFFICIENT"

        def seeds(epoch):
            events = [
                json.loads(x)
                for x in (runner.runs / epoch.epoch_id / "events.jsonl").read_text().splitlines()
            ]
            return {e["challenge"]["seed"] for e in events if e.get("type") == "challenge_issued"}

        assert seeds(local).isdisjoint(seeds(forwarded))
        assert sorted(s.spec_key for s in local.segments) == sorted(s.spec_key for s in forwarded.segments)
        assert all("route" not in s.model_dump() for s in local.segments)
    finally:
        await facility_http.close()
        await facility.close()
        await helper.close()


async def test_worker_rejects_replays_and_unauthorized_control(tmp_path, small_spec):
    runtime, worker, app = await in_process_worker(tmp_path, "worker")
    challenge = Challenge(
        challenge_id=uuid.uuid4().hex, epoch_id=uuid.uuid4().hex, seed="01" * 32, spec=small_spec
    )
    try:
        result = await worker.execute(challenge)
        assert result.digest == WorkloadEngine().execute(bytes.fromhex(challenge.seed), small_spec)[0]
        with pytest.raises(httpx.HTTPStatusError):
            await worker.execute(challenge)
        response = await worker.client.post("/control/route", json={"route": "forwarded"})
        assert response.status_code == 401
        response = await worker.client.post(
            "/worker/task",
            json=challenge.model_dump(mode="json"),
            headers={"Authorization": "Bearer incorrect"},
        )
        assert response.status_code == 401
    finally:
        await worker.close()
        await runtime.close()


async def test_stop_during_computation_retires_seed_and_closes_epoch(tmp_path):
    class SlowWorker:
        def __init__(self):
            self.started = asyncio.Event()
            self.cancelled = []

        async def execute(self, challenge):
            self.started.set()
            await asyncio.sleep(30)

        async def cancel(self, challenge_id):
            self.cancelled.append(challenge_id)

        async def quiesce(self):
            pass

    worker = SlowWorker()
    runner, specs = setup_runner(tmp_path, worker)
    before = sum(runner.bank.available(s) for s in specs)
    task = asyncio.create_task(runner.run(specs, 1, washout_s=0, timing_only=True))
    await worker.started.wait()
    runner.stop()
    epoch, finding = await task
    assert not epoch.complete and epoch.stop_reason == "OPERATOR_STOP"
    assert epoch.segments[0].outcome == "INTERRUPTED"
    assert len(worker.cancelled) == 1
    assert sum(runner.bank.available(s) for s in specs) == before - 1
    assert finding["verdict"] == "INCONCLUSIVE"


async def test_missing_meter_prevents_any_challenge_issuance(tmp_path):
    runtime, worker, _ = await in_process_worker(tmp_path, "worker")
    runner, specs = setup_runner(tmp_path, worker)
    try:
        before = sum(runner.bank.available(s) for s in specs)
        epoch, finding = await runner.run(specs, 1, washout_s=0, timing_only=False)
        assert not epoch.complete and epoch.stop_reason == "SENSOR_LOST"
        assert not epoch.segments
        assert before == sum(runner.bank.available(s) for s in specs)
        assert finding["verdict"] == "INCONCLUSIVE"
    finally:
        await worker.close()
        await runtime.close()


async def test_auditor_auth_epochs_transition_receipt_and_tamper(tmp_path):
    runtime, worker, _ = await in_process_worker(tmp_path, "worker")
    runner, specs = setup_runner(tmp_path, worker, repeats=2)
    app = create_auditor_app(runner, ADMIN, METER, tmp_path / "private", "test site", "test installation")
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://auditor"
        ) as client:
            assert (await client.get("/health")).status_code == 401
            client.headers["Authorization"] = "Bearer " + ADMIN
            body = {
                "specs": [s.model_dump(mode="json") for s in specs],
                "route": "local",
                "repeats": 1,
                "provisional_washout_s": 0,
                "timing_only": True,
            }
            assert (await client.post("/epochs/start", json=body)).status_code == 202
            await asyncio.sleep(0.03)
            assert (await client.post("/epochs/start", json=body)).status_code == 409
            assert (await client.post("/epochs/transition", json=body)).status_code == 202
            for _ in range(200):
                if runner.status.get("state") == "COMPLETE":
                    break
                await asyncio.sleep(0.02)
            assert runner.status["state"] == "COMPLETE", runner.status
            epoch_id = runner.status["epoch_id"]
            receipt_response = await client.post(f"/epochs/{epoch_id}/receipt")
            assert receipt_response.status_code == 200, receipt_response.text
            receipt = receipt_response.json()
            trusted = receipt["payload"]["public_key_id"]
            assert verify_receipt(receipt, trusted, runner.runs / epoch_id)
            receipt["payload"]["finding"]["verdict"] = "FORGED"
            with pytest.raises(InvalidSignature):
                verify_receipt(receipt, trusted)
            assert (await client.get("/epochs/not-a-valid-id")).status_code == 404
            # Admin cannot masquerade as meter, and worker responses cannot provide readings.
            assert (await client.post("/meter/samples", json={})).status_code == 401
    await runtime.close()


def test_single_auditor_process_lock(tmp_path):
    first, second = ProcessLock(tmp_path / "owner.lock"), ProcessLock(tmp_path / "owner.lock")
    first.acquire()
    try:
        with pytest.raises(RuntimeError):
            second.acquire()
    finally:
        first.release()
    second.acquire()
    second.release()


def test_schedule_balanced_and_bounded():
    specs = [WorkloadSpec(mode=m) for m in Mode]
    schedule = balanced_schedule(specs, 4)
    assert len(schedule) == 24
    assert sum(s.mode == Mode.IDLE for s in schedule) == 12
    assert all(sum(s.mode != Mode.IDLE for s in schedule[i : i + 3]) < 3 for i in range(len(schedule) - 2))
    assert [s.mode for s in schedule].count(Mode.MEMORY) == 4


async def test_sensor_loss_during_active_challenge_stops_and_retires(tmp_path):
    import time

    from compute_echo.models import MeterSample

    class HangingWorker:
        def __init__(self):
            self.started = asyncio.Event()
            self.cancelled = False

        async def execute(self, challenge):
            self.started.set()
            await asyncio.sleep(30)

        async def cancel(self, challenge_id):
            self.cancelled = True

        async def quiesce(self):
            pass

    worker = HangingWorker()
    runner, specifications = setup_runner(tmp_path, worker)
    specs = [
        WorkloadSpec.model_validate({**s.model_dump(), "segment_s": 1, "deadline_s": 0.9})
        for s in specifications
    ]
    for spec in specs:
        runner.bank.generate(spec, 3, WorkloadEngine())
    runner.meter = MeterStore(
        tmp_path / "independent-meter.sqlite3",
        MeterConfig(
            meter_id="test-external",
            provenance="external",
            installation="synthetic external policy fixture",
            source_freshness="device_sequence",
            max_sample_age_s=0.08,
        ),
    )
    now = time.time_ns()
    runner.meter.ingest(
        MeterSample(meter_id="test-external", sequence=1, device_sequence=1, acquired_at_ns=now, watts=100)
    )
    epoch, finding = await runner.run(specs, 1, washout_s=0)
    assert worker.cancelled
    assert epoch.stop_reason == "SENSOR_LOST"
    assert epoch.segments[0].outcome == "INTERRUPTED"
    assert finding["physical_response"] == "INSUFFICIENT"
    assert runner.bank.recover_pending() == 0


async def test_worker_enforces_own_deadline_and_drains(tmp_path, small_spec, monkeypatch):
    import threading
    import time

    from compute_echo.workloads import CancelledWork

    runtime = WorkerRuntime("cpu", tmp_path / "worker.sqlite3")
    started = threading.Event()

    def slow(challenge, cancel):
        started.set()
        while not cancel.is_set():
            time.sleep(0.005)
        raise CancelledWork("stopped")

    monkeypatch.setattr(runtime, "calculate", slow)
    spec = WorkloadSpec.model_validate({**small_spec.model_dump(), "deadline_s": 0.06})
    challenge = Challenge(challenge_id=uuid.uuid4().hex, epoch_id=uuid.uuid4().hex, seed="02" * 32, spec=spec)
    task = asyncio.create_task(runtime.execute(challenge))
    try:
        with pytest.raises(HTTPException) as exc:
            await task
        assert exc.value.status_code == 408
        assert started.is_set()
        await asyncio.sleep(0)
        assert not runtime.active
    finally:
        await runtime.close()


async def test_disconnected_auditor_does_not_leave_infinite_work(tmp_path, small_spec, monkeypatch):
    import time

    from compute_echo.workloads import CancelledWork

    runtime = WorkerRuntime("cpu", tmp_path / "worker.sqlite3")

    def slow(challenge, cancel):
        while not cancel.is_set():
            time.sleep(0.005)
        raise CancelledWork("stopped after disconnected auditor")

    monkeypatch.setattr(runtime, "calculate", slow)
    spec = WorkloadSpec.model_validate({**small_spec.model_dump(), "deadline_s": 0.08})
    challenge = Challenge(challenge_id=uuid.uuid4().hex, epoch_id=uuid.uuid4().hex, seed="03" * 32, spec=spec)
    request = asyncio.create_task(runtime.execute(challenge))
    try:
        await asyncio.sleep(0.02)
        request.cancel()
        await asyncio.gather(request, return_exceptions=True)
        assert runtime.active  # HTTP cancellation did not pretend GPU work stopped.
        for _ in range(50):
            if not runtime.active:
                break
            await asyncio.sleep(0.01)
        assert not runtime.active
    finally:
        await runtime.close()


async def test_cancelled_epoch_task_retires_pending_seed_and_saves_incomplete_evidence(tmp_path):
    class WaitingWorker:
        def __init__(self):
            self.started = asyncio.Event()

        async def execute(self, challenge):
            self.started.set()
            await asyncio.sleep(30)

        async def cancel(self, challenge_id):
            pass

        async def quiesce(self):
            pass

    worker = WaitingWorker()
    runner, specs = setup_runner(tmp_path, worker)
    task = asyncio.create_task(runner.run(specs, 1, washout_s=0, timing_only=True))
    await worker.started.wait()
    task.cancel()
    evidence, finding = await task
    assert evidence.stop_reason == "TASK_CANCELLED"
    assert not evidence.complete
    assert finding["verdict"] == "INCONCLUSIVE"
    assert runner.bank.recover_pending() == 0
    assert (runner.runs / evidence.epoch_id / "evidence.json").exists()

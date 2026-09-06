"""Auditor API and routing control plane; no dashboard or public bank endpoints."""

from __future__ import annotations

import asyncio
import hmac
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import Field, model_validator

from .dashboard import DASHBOARD_HTML
from .demo import DemoRuntime
from .epochs import EpochRunner
from .models import MeterSample, StrictModel, WorkloadSpec, canonical
from .receipt import issue_receipt, load_or_create_key
from .storage import EventLog


class StartRequest(StrictModel):
    specs: list[WorkloadSpec] = Field(min_length=2, max_length=4)
    repeats: int = Field(default=4, ge=1, le=100)
    provisional_washout_s: float | None = Field(default=None, ge=0, le=120)
    timing_only: bool = False
    route: Literal["local", "forwarded"]

    @model_validator(mode="after")
    def schedule_contract(self):
        modes = [spec.mode for spec in self.specs]
        if len(set(modes)) != len(modes) or not {"IDLE", "TENSOR"} <= set(modes):
            raise ValueError("unique modes including IDLE and TENSOR required")
        if (
            len({spec.segment_s for spec in self.specs}) != 1
            or len({spec.deadline_s for spec in self.specs}) != 1
        ):
            raise ValueError("all modes must share segment duration and deadline")
        return self


class ProcessLock:
    """OS-released lock prevents two auditors from issuing/recovering one bank."""

    def __init__(self, path: Path):
        self.path, self.file = path, None

    def acquire(self):
        import os

        self.file = self.path.open("a+b")
        self.file.write(b"0")
        self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError("another auditor owns this bank") from None

    def release(self):
        if self.file:
            self.file.close()


def create_auditor_app(
    runner: EpochRunner,
    admin_token: str,
    meter_token: str,
    private: Path,
    site: str,
    site_assurance: str,
    demo: DemoRuntime | None = None,
) -> FastAPI:
    if len(admin_token) < 24 or len(meter_token) < 24 or admin_token == meter_token:
        raise ValueError("distinct admin/meter secrets of at least 24 characters required")
    active: asyncio.Task | None = None
    control_lock = asyncio.Lock()
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    owner = ProcessLock(private / "auditor.lock")
    labels = EventLog(private / "evaluation-labels.jsonl")

    def authenticate(token):
        def dependency(authorization: str = Header(default="")):
            if not hmac.compare_digest(authorization, "Bearer " + token):
                raise HTTPException(401, "invalid credential")

        return dependency

    admin, meter_auth = authenticate(admin_token), authenticate(meter_token)

    @asynccontextmanager
    async def lifespan(app):
        owner.acquire()
        runner.bank.recover_pending()
        if demo:
            demo.telemetry.start()
            try:
                health = await runner.worker.health()
                demo.current_route = health.get("route", "unknown")
            except Exception:
                demo.current_route = "unknown"
        try:
            yield
        finally:
            runner.stop()
            try:
                if active:
                    await asyncio.gather(active, return_exceptions=True)
                await runner.worker.quiesce()
            finally:
                if demo:
                    await demo.telemetry.close()
                if hasattr(runner.worker, "close"):
                    await runner.worker.close()
                owner.release()

    app = FastAPI(title="Compute Echo Auditor", lifespan=lifespan)

    @app.get("/health", dependencies=[Depends(admin)])
    async def health():
        return {
            "status": runner.status,
            "meter": runner.meter.health(),
            "physical_calibration_validated": bool(runner.calibration and runner.calibration.validated),
        }

    @app.post("/meter/samples", dependencies=[Depends(meter_auth)])
    async def ingest(sample: MeterSample):
        try:
            runner.meter.ingest(sample)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        return {"accepted": True, "meter_health": runner.meter.health()}

    async def run_epoch(request: StartRequest):
        try:
            evidence, _ = await runner.run(
                request.specs,
                request.repeats,
                washout_s=request.provisional_washout_s,
                timing_only=request.timing_only,
            )
            if demo:
                demo.last_evidence = evidence
            # Persist separately; these labels never enter EpochEvidence or detect().
            labels.append({"epoch_id": evidence.epoch_id, "route": request.route, "time_ns": time.time_ns()})
        except Exception as error:
            runner.status = {"state": "ERROR", "error_type": type(error).__name__, "message": str(error)}

    @app.post("/epochs/start", status_code=202, dependencies=[Depends(admin)])
    async def start(request: StartRequest):
        nonlocal active
        async with control_lock:
            if active and not active.done():
                raise HTTPException(409, "epoch active; use /epochs/transition to close it first")
            await runner.worker.set_route(request.route)
            if demo:
                demo.current_route = request.route
            runner.status = {"state": "STARTING"}
            active = asyncio.create_task(run_epoch(request))
        return {"state": "STARTING"}

    @app.post("/epochs/transition", status_code=202, dependencies=[Depends(admin)])
    async def transition(request: StartRequest):
        nonlocal active
        async with control_lock:
            runner.stop()
            if active:
                await asyncio.gather(active, return_exceptions=True)
            await runner.worker.quiesce()
            await runner.worker.set_route(request.route)
            if demo:
                demo.current_route = request.route
            runner.status = {"state": "STARTING"}
            active = asyncio.create_task(run_epoch(request))
        return {"state": "STARTING", "previous_epoch_closed": True}

    @app.post("/epochs/stop", dependencies=[Depends(admin)])
    async def stop():
        async with control_lock:
            runner.stop()
            if active:
                await asyncio.gather(active, return_exceptions=True)
        return runner.status

    if demo:

        @app.get("/dashboard", response_class=HTMLResponse)
        async def dashboard():
            return DASHBOARD_HTML

        @app.get("/demo/state", dependencies=[Depends(admin)])
        async def demo_state():
            return demo.state(runner)

        @app.post("/demo/route/{route}", dependencies=[Depends(admin)])
        async def demo_route(route: Literal["local", "forwarded"]):
            nonlocal active
            async with control_lock:
                if active and not active.done():
                    raise HTTPException(409, "epoch active; use the transition control")
                response = await runner.worker.set_route(route)
                # State changes only after the worker confirms the actual route.
                demo.current_route = response.get("route", route)
            return {"route": demo.current_route, "worker_confirmed": True}

        def demo_request(route: str):
            return StartRequest(
                specs=demo.specs,
                repeats=demo.repeats,
                provisional_washout_s=demo.washout_s,
                timing_only=True,
                route=route,
            )

        @app.post("/demo/run", status_code=202, dependencies=[Depends(admin)])
        async def demo_run():
            if demo.current_route not in {"local", "forwarded"}:
                raise HTTPException(409, "set and confirm a worker route first")
            return await start(demo_request(demo.current_route))

        @app.post("/demo/transition/{route}", status_code=202, dependencies=[Depends(admin)])
        async def demo_transition(route: Literal["local", "forwarded"]):
            return await transition(demo_request(route))

    def directory(epoch_id: str):
        import re

        if not re.fullmatch("[a-f0-9]{32}", epoch_id):
            raise HTTPException(404, "unknown epoch")
        path = runner.runs / epoch_id
        if not path.is_dir():
            raise HTTPException(404, "unknown epoch")
        return path

    @app.get("/epochs/{epoch_id}", dependencies=[Depends(admin)])
    async def epoch(epoch_id: str):
        path = directory(epoch_id)
        if not (path / "finding.json").exists():
            return {"state": "COLLECTING", "verdict": None}
        return {
            "evidence": json.loads((path / "evidence.json").read_bytes()),
            "finding": json.loads((path / "finding.json").read_bytes()),
        }

    @app.get("/epochs/{epoch_id}/events", dependencies=[Depends(admin)])
    async def events(epoch_id: str):
        log = EventLog(directory(epoch_id) / "events.jsonl")

        async def stream():
            sent = 0
            while True:
                rows = log.read()
                for row in rows[sent:]:
                    yield b"data: " + canonical(row) + b"\n\n"
                sent = len(rows)
                if rows and rows[-1].get("state") in {"COMPLETE", "INTERRUPTED"}:
                    break
                await asyncio.sleep(0.25)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/epochs/{epoch_id}/receipt", dependencies=[Depends(admin)])
    async def receipt(epoch_id: str):
        path = directory(epoch_id)
        if active and not active.done() and runner.current_id == epoch_id:
            raise HTTPException(409, "epoch must be closed before signing")
        if (path / "receipt.json").exists():
            return json.loads((path / "receipt.json").read_bytes())
        if not (path / "finding.json").exists():
            raise HTTPException(409, "epoch evidence is incomplete")
        return issue_receipt(
            path,
            load_or_create_key(private / "signing.key"),
            site,
            site_assurance,
            runner.calibration.calibration_id if runner.calibration else None,
        )

    @app.get("/epochs/{epoch_id}/receipt", dependencies=[Depends(admin)])
    async def get_receipt(epoch_id: str):
        path = directory(epoch_id) / "receipt.json"
        if not path.exists():
            raise HTTPException(404, "receipt not issued")
        return json.loads(path.read_bytes())

    return app

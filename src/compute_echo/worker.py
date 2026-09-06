"""Single-device worker and optional forwarding proxy. Contains no answer bank."""

from __future__ import annotations

import asyncio
import concurrent.futures
import hmac
import sqlite3
import threading
from contextlib import asynccontextmanager, closing
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse

from .models import Challenge, Mode, StrictModel, WorkerResult
from .safety import GPUSafetyGuard, SafetyStop
from .telemetry import NVMLDemoSensor
from .transport import HTTPWorker
from .workloads import CancelledWork, WorkloadEngine


class RouteRequest(StrictModel):
    route: Literal["local", "forwarded"]


class WorkerRuntime:
    def __init__(
        self,
        backend: str,
        ledger: Path,
        helper: HTTPWorker | None = None,
        max_temperature_c: float = 85,
        telemetry: NVMLDemoSensor | None = None,
        profile_path: Path | None = None,
    ):
        self.backend, self.helper, self.ledger = backend, helper, ledger
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(ledger)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS issued (id TEXT PRIMARY KEY, seed TEXT UNIQUE NOT NULL)")
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="echo-worker")
        self.telemetry_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="echo-nvml"
        )
        self.telemetry = telemetry
        self.profile_path = profile_path
        self.engine = None
        self.safety_guard = GPUSafetyGuard(max_temperature_c) if backend == "cuda" else None
        self.active: dict[str, tuple[threading.Event, asyncio.Task]] = {}
        self.lock = asyncio.Lock()
        self.route = "local"

    def calculate(self, challenge: Challenge, cancel: threading.Event):
        if self.engine is None:
            self.engine = WorkloadEngine(self.backend, self.safety_guard)
        digest, timings = self.engine.execute(bytes.fromhex(challenge.seed), challenge.spec, cancel)
        return WorkerResult(
            challenge_id=challenge.challenge_id,
            epoch_id=challenge.epoch_id,
            digest=digest,
            backend=self.engine.identity,
            timings=timings,
            operations_completed=0 if challenge.spec.mode == Mode.IDLE else challenge.spec.operations,
        )

    async def execute(self, challenge: Challenge):
        import hashlib

        async with self.lock:
            if self.active:
                raise HTTPException(409, "worker busy; concurrent challenges would contaminate epochs")
            try:
                with closing(sqlite3.connect(self.ledger)) as db, db:
                    db.execute(
                        "INSERT INTO issued VALUES(?,?)",
                        (challenge.challenge_id, hashlib.sha256(bytes.fromhex(challenge.seed)).hexdigest()),
                    )
            except sqlite3.IntegrityError as error:
                raise HTTPException(409, "challenge or seed already issued") from error
            cancel = threading.Event()

            async def job():
                if self.route == "forwarded":
                    if self.helper is None:
                        raise HTTPException(503, "no helper configured")
                    return await self.helper.execute(challenge)
                future = asyncio.get_running_loop().run_in_executor(
                    self.pool, self.calculate, challenge, cancel
                )
                try:
                    return await asyncio.wait_for(asyncio.shield(future), challenge.spec.deadline_s)
                except asyncio.TimeoutError as error:
                    cancel.set()
                    # Keep the worker occupied until the actual device work drains,
                    # even when the auditor connection disappeared.
                    await asyncio.gather(asyncio.shield(future), return_exceptions=True)
                    raise HTTPException(408, "worker execution deadline exceeded") from error

            task = asyncio.create_task(job())
            self.active[challenge.challenge_id] = (cancel, task)

            def finished(completed):
                self.active.pop(challenge.challenge_id, None)
                # The request may have disconnected. Retrieve a terminal exception
                # so a failed orphan job does not become an unobserved task error.
                if not completed.cancelled():
                    completed.exception()

            task.add_done_callback(finished)
        try:
            # Client disconnection must not make an executing CUDA job invisible.
            return await asyncio.shield(task)
        except CancelledWork as error:
            raise HTTPException(409, "work interrupted") from error
        except SafetyStop as error:
            raise HTTPException(503, str(error)) from error

    async def cancel(self, challenge_id: str):
        entry = self.active.get(challenge_id)
        if entry:
            event, task = entry
            event.set()
            if self.route == "forwarded" and self.helper:
                await self.helper.cancel(challenge_id)
            try:
                await asyncio.wait_for(asyncio.shield(task), 15)
            except (CancelledWork, HTTPException, SafetyStop):
                pass
            except asyncio.TimeoutError as error:
                raise HTTPException(503, "worker has not quiesced; do not start another epoch") from error

    async def quiesce(self):
        for challenge_id in list(self.active):
            await self.cancel(challenge_id)
        if self.helper:
            await self.helper.quiesce()

    async def close(self):
        await self.quiesce()
        if self.helper:
            await self.helper.close()
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.telemetry_pool.shutdown(wait=True, cancel_futures=True)
        if self.telemetry:
            self.telemetry.close()

    async def warmup(self):
        from .models import WorkloadSpec

        def run():
            self.engine = WorkloadEngine(self.backend, self.safety_guard)
            for mode in Mode:
                self.engine.execute(
                    bytes(32),
                    WorkloadSpec(mode=mode, m=32, n=32, k=32, lanes=8, words_per_lane=64, memory_steps=32),
                )

        await asyncio.get_running_loop().run_in_executor(self.pool, run)


def create_worker_app(runtime: WorkerRuntime, token: str, control_token: str) -> FastAPI:
    if len(token) < 24 or len(control_token) < 24 or token == control_token:
        raise ValueError("distinct worker/control secrets of at least 24 characters required")

    def auth(authorization: str = Header(default="")):
        if not hmac.compare_digest(authorization, "Bearer " + token):
            raise HTTPException(401, "invalid worker credential")

    def control_auth(authorization: str = Header(default="")):
        if not hmac.compare_digest(authorization, "Bearer " + control_token):
            raise HTTPException(401, "invalid control credential")

    @asynccontextmanager
    async def lifespan(app):
        await runtime.warmup()
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(title="Compute Echo Worker", lifespan=lifespan)

    @app.get("/health", dependencies=[Depends(auth)])
    async def health():
        return {
            "status": "busy" if runtime.active else "ready",
            "backend": runtime.backend,
            "route": runtime.route,
            "nvml_demo_available": runtime.telemetry is not None,
            "host_profile_available": bool(runtime.profile_path and runtime.profile_path.is_file()),
        }

    @app.get("/telemetry/nvml-demo", dependencies=[Depends(auth)])
    async def telemetry():
        if runtime.telemetry is None:
            raise HTTPException(503, "NVML_DEMO telemetry is not configured")
        return await asyncio.get_running_loop().run_in_executor(
            runtime.telemetry_pool, runtime.telemetry.sample
        )

    @app.get("/diagnostics/host-profile", dependencies=[Depends(auth)])
    async def host_profile():
        if runtime.profile_path is None or not runtime.profile_path.is_file():
            raise HTTPException(404, "host profile has not been produced")
        return FileResponse(runtime.profile_path, media_type="application/json")

    @app.post("/worker/task", response_model=WorkerResult, dependencies=[Depends(auth)])
    async def task(challenge: Challenge):
        return await runtime.execute(challenge)

    @app.post("/worker/cancel/{challenge_id}", dependencies=[Depends(auth)])
    async def cancel(challenge_id: str):
        await runtime.cancel(challenge_id)
        return {"quiescent": not runtime.active}

    @app.post("/worker/quiesce", dependencies=[Depends(auth)])
    async def quiesce():
        async with runtime.lock:
            await runtime.quiesce()
        return {"quiescent": not runtime.active}

    @app.post("/control/route", dependencies=[Depends(control_auth)])
    async def route(request: RouteRequest):
        async with runtime.lock:
            await runtime.quiesce()
            if request.route == "forwarded" and runtime.helper is None:
                raise HTTPException(409, "forwarding requires a configured unmetered helper")
            runtime.route = request.route
        return {"route": runtime.route}

    return app

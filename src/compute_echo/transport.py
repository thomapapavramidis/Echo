"""Auditor-measured transport timings; worker diagnostics are not timing authority."""

from __future__ import annotations

import time
from typing import Protocol

import httpx

from .models import Challenge, WorkerResult, canonical


class WorkerTransport(Protocol):
    async def execute(self, challenge: Challenge) -> WorkerResult: ...
    async def cancel(self, challenge_id: str) -> None: ...
    async def quiesce(self) -> None: ...


class HTTPWorker:
    def __init__(self, url: str, token: str, control_token: str | None = None):
        self.control_token = control_token
        self.client = httpx.AsyncClient(
            base_url=url.rstrip("/"),
            timeout=125,
            headers={"Authorization": "Bearer " + token},
            trust_env=False,
        )
        self.last_transport = None

    async def execute(self, challenge: Challenge) -> WorkerResult:
        self.last_transport = None
        trace_times = {}

        async def trace(name, info):
            trace_times[name] = time.perf_counter()

        start = time.perf_counter()
        response = await self.client.post(
            "/worker/task",
            content=canonical(challenge),
            headers={"Content-Type": "application/json"},
            extensions={"trace": trace},
        )
        received = time.perf_counter()
        response.raise_for_status()
        result = WorkerResult.model_validate(response.json())
        body_start = trace_times.get("http11.receive_response_body.started")
        body_end = trace_times.get("http11.receive_response_body.complete")
        self.last_transport = {
            "auditor_round_trip_s": received - start,
            "response_body_receive_s": body_end - body_start
            if body_start is not None and body_end is not None
            else None,
            "request_bytes": len(canonical(challenge)),
            "response_bytes": len(response.content),
            "one_way_transmission_s": None,
            "note": "response receive time includes buffering; one-way network time is not inferred",
        }
        return result

    async def cancel(self, challenge_id: str):
        response = await self.client.post("/worker/cancel/" + challenge_id, timeout=15)
        response.raise_for_status()

    async def quiesce(self):
        response = await self.client.post("/worker/quiesce", timeout=30)
        response.raise_for_status()
        if response.json().get("quiescent") is not True:
            raise RuntimeError("worker did not confirm quiescence")

    async def set_route(self, route: str):
        if self.control_token is None:
            raise ValueError("worker control credential required for routing changes")
        response = await self.client.post(
            "/control/route",
            json={"route": route},
            timeout=30,
            headers={"Authorization": "Bearer " + self.control_token},
        )
        response.raise_for_status()
        return response.json()

    async def health(self):
        response = await self.client.get("/health")
        response.raise_for_status()
        return response.json()

    async def close(self):
        await self.client.aclose()

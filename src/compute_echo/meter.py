"""Independent sensor ingestion, durable replay protection, and collectors.

Collector sequence numbers alone do not establish device measurement cadence.
The meter/adapter must expose a genuine device update counter or measurement time.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Literal

import httpx
from pydantic import Field

from .models import MeterSample, StrictModel, canonical


class MeterConfig(StrictModel):
    meter_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    provenance: Literal["external", "simulation", "host_telemetry"] = "external"
    installation: str = Field(min_length=10, max_length=2000)
    max_sample_age_s: float = Field(default=5, gt=0, le=120)
    max_clock_skew_s: float = Field(default=0.25, ge=0, le=5)
    max_watts: float = Field(default=1000, gt=0, le=1_000_000)
    source_freshness: Literal["device_sequence", "device_timestamp", "unknown"] = "unknown"


class MeterStore:
    def __init__(self, path: Path, config: MeterConfig):
        self.path, self.config = path, config
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS meter_config (id TEXT PRIMARY KEY, config TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS samples (
                  meter_id TEXT, sequence INTEGER, acquired_ns INTEGER, received_ns INTEGER,
                  device_sequence INTEGER, device_time_ns INTEGER, payload TEXT NOT NULL,
                  PRIMARY KEY(meter_id,sequence));
            """)
            old = db.execute("SELECT config FROM meter_config WHERE id=?", (config.meter_id,)).fetchone()
            if old and old[0] != canonical(config).decode():
                raise ValueError("meter configuration changed: use a new meter identity/storage")
            db.execute(
                "INSERT OR IGNORE INTO meter_config VALUES(?,?)",
                (config.meter_id, canonical(config).decode()),
            )

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        with closing(db), db:
            yield db

    def ingest(self, sample: MeterSample, now_ns: int | None = None):
        now_ns = time.time_ns() if now_ns is None else now_ns
        cfg = self.config
        if sample.meter_id != cfg.meter_id:
            raise ValueError("unregistered meter")
        age = (now_ns - sample.acquired_at_ns) / 1e9
        if age < -cfg.max_clock_skew_s or age > cfg.max_sample_age_s:
            raise ValueError("stale sample or clock outside configured skew bound")
        if sample.device_time_ns is not None:
            device_age = (sample.acquired_at_ns - sample.device_time_ns) / 1e9
            if device_age < -cfg.max_clock_skew_s or device_age > cfg.max_sample_age_s:
                raise ValueError("stale or future device measurement")
        if cfg.source_freshness == "device_sequence" and sample.device_sequence is None:
            raise ValueError("device update counter required")
        if cfg.source_freshness == "device_timestamp" and sample.device_time_ns is None:
            raise ValueError("device measurement timestamp required")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute(
                "SELECT sequence,acquired_ns,device_sequence,device_time_ns FROM samples "
                "WHERE meter_id=? ORDER BY sequence DESC LIMIT 1",
                (cfg.meter_id,),
            ).fetchone()
            if previous:
                if sample.sequence <= previous[0] or sample.acquired_at_ns <= previous[1]:
                    raise ValueError("replayed or out-of-order collector sample")
                if cfg.source_freshness == "device_sequence" and sample.device_sequence <= previous[2]:
                    raise ValueError("device has not produced a fresh measurement")
                if cfg.source_freshness == "device_timestamp" and sample.device_time_ns <= previous[3]:
                    raise ValueError("device has not produced a fresh measurement")
            db.execute(
                "INSERT INTO samples VALUES(?,?,?,?,?,?,?)",
                (
                    cfg.meter_id,
                    sample.sequence,
                    sample.acquired_at_ns,
                    now_ns,
                    sample.device_sequence,
                    sample.device_time_ns,
                    canonical(sample).decode(),
                ),
            )
        # Preserve the actual out-of-limit reading before the scheduler stops work.

    def latest(self) -> MeterSample | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM samples WHERE meter_id=? ORDER BY sequence DESC LIMIT 1",
                (self.config.meter_id,),
            ).fetchone()
        return MeterSample.model_validate_json(row[0]) if row else None

    def samples(self, start_ns: int, end_ns: int) -> list[MeterSample]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT payload FROM samples WHERE meter_id=? AND acquired_ns>=? "
                "AND acquired_ns<? ORDER BY acquired_ns",
                (self.config.meter_id, start_ns, end_ns),
            ).fetchall()
        return [MeterSample.model_validate_json(row[0]) for row in rows]

    def health(self, now_ns: int | None = None):
        sample = self.latest()
        now_ns = time.time_ns() if now_ns is None else now_ns
        if sample is None or (now_ns - sample.acquired_at_ns) / 1e9 > self.config.max_sample_age_s:
            return "SENSOR_LOST"
        if sample.quality != "GOOD":
            return "SENSOR_INVALID"
        if sample.watts > self.config.max_watts:
            return "POWER_LIMIT"
        return "OK"


async def collect_http(source_url: str, auditor_url: str, token: str, interval_s: float = 0.5):
    """Read a normalized meter adapter. Never invent a device timestamp/counter.

    Source JSON must implement MeterSample. A device-specific adapter must map its
    actual refresh marker; polling a register repeatedly is not fresh measurement.
    """
    import asyncio

    async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
        while True:
            response = await client.get(source_url)
            response.raise_for_status()
            sample = MeterSample.model_validate(response.json())
            result = await client.post(
                auditor_url.rstrip("/") + "/meter/samples",
                json=sample.model_dump(),
                headers={"Authorization": "Bearer " + token},
            )
            # Duplicate hardware readings are deliberately rejected, not counted again.
            if result.status_code != 409:
                result.raise_for_status()
            await asyncio.sleep(interval_s)


async def collect_into_store(source_url: str, store: MeterStore, interval_s: float = 0.5):
    """Collector owned by the CLI auditor, for a standalone physics-run process."""
    import asyncio

    async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
        while True:
            try:
                response = await client.get(source_url)
                response.raise_for_status()
                store.ingest(MeterSample.model_validate(response.json()))
            except (httpx.HTTPError, ValueError):
                # Never synthesize replacement samples. Staleness automatically
                # stops an epoch; the actual collector outage remains an absence.
                pass
            await asyncio.sleep(interval_s)


def serial_samples(port: str, baudrate: int = 115200):
    """Normalized JSONL from an independent serial meter/adapter; no fake readings."""
    import serial

    with serial.Serial(port, baudrate, timeout=2) as connection:
        while True:
            line = connection.readline(4096)
            if line:
                yield MeterSample.model_validate(json.loads(line))

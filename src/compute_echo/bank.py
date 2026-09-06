"""Private offline reference bank with irreversible, transactional issuance.

SQLite WAL + synchronous FULL ensures retirement commits before a seed leaves this
module. There is deliberately no reset/unissue/delete API. Backups must never be
restored as an issuance bank: rollback of private state could resurrect seeds.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
import uuid
from contextlib import closing, contextmanager
from pathlib import Path

from cryptography.fernet import Fernet

from .models import Challenge, Outcome, WorkerResult, WorkloadSpec, canonical
from .workloads import WorkloadEngine


class BankEmpty(RuntimeError):
    pass


class ChallengeBank:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = directory / "bank.sqlite3"
        keypath = directory / "bank.key"
        try:
            fd = os.open(keypath, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "wb") as f:
                f.write(Fernet.generate_key())
                f.flush()
                os.fsync(f.fileno())
        self.cipher = Fernet(keypath.read_bytes())
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS challenges (
                  id TEXT PRIMARY KEY, seed_hash TEXT UNIQUE NOT NULL,
                  spec_key TEXT NOT NULL, payload BLOB NOT NULL,
                  reference_backend TEXT NOT NULL, generated_ns INTEGER NOT NULL,
                  issued_ns INTEGER, epoch_id TEXT,
                  outcome TEXT NOT NULL DEFAULT 'UNUSED', completion_json TEXT
                );
                CREATE INDEX IF NOT EXISTS available ON challenges(spec_key, issued_ns);
                CREATE TRIGGER IF NOT EXISTS never_unissue BEFORE UPDATE OF issued_ns ON challenges
                  WHEN OLD.issued_ns IS NOT NULL AND NEW.issued_ns IS NOT OLD.issued_ns
                  BEGIN SELECT RAISE(ABORT, 'issued seeds cannot be resurrected'); END;
                CREATE TRIGGER IF NOT EXISTS never_delete BEFORE DELETE ON challenges
                  BEGIN SELECT RAISE(ABORT, 'seed retirement records are permanent'); END;
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        with closing(db), db:
            yield db

    def generate(self, spec: WorkloadSpec, count: int, engine: WorkloadEngine):
        if not 1 <= count <= 100_000:
            raise ValueError("count must be between 1 and 100000")
        # This function is CLI-only. Never expose it on the untrusted worker API.
        for _ in range(count):
            seed = secrets.token_bytes(32)
            digest, timings = engine.execute(seed, spec)
            payload = self.cipher.encrypt(
                canonical(
                    {
                        "seed": seed.hex(),
                        "spec": spec.model_dump(mode="json"),
                        "expected_digest": digest,
                        "reference_timings": timings.model_dump(),
                    }
                )
            )
            with self.connect() as db:
                db.execute(
                    "INSERT INTO challenges(id,seed_hash,spec_key,payload,reference_backend,generated_ns) "
                    "VALUES(?,?,?,?,?,?)",
                    (
                        uuid.uuid4().hex,
                        hashlib.sha256(seed).hexdigest(),
                        spec.key,
                        payload,
                        engine.identity,
                        time.time_ns(),
                    ),
                )

    def available(self, spec: WorkloadSpec) -> int:
        with self.connect() as db:
            return db.execute(
                "SELECT count(*) FROM challenges WHERE spec_key=? AND issued_ns IS NULL", (spec.key,)
            ).fetchone()[0]

    def issue(self, spec: WorkloadSpec, epoch_id: str) -> Challenge:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM challenges WHERE spec_key=? AND issued_ns IS NULL "
                "ORDER BY generated_ns,id LIMIT 1",
                (spec.key,),
            ).fetchone()
            if row is None:
                raise BankEmpty(f"no unused challenges for {spec.mode}/{spec.key[:12]}")
            data = json.loads(self.cipher.decrypt(row["payload"]))
            challenge = Challenge(
                challenge_id=row["id"], epoch_id=epoch_id, seed=data["seed"], spec=data["spec"]
            )
            db.execute(
                "UPDATE challenges SET issued_ns=?,epoch_id=?,outcome=? WHERE id=?",
                (time.time_ns(), epoch_id, Outcome.PENDING, row["id"]),
            )
        # Transaction committed, even if caller crashes before sending this value.
        return challenge

    def complete(
        self,
        challenge: Challenge,
        result: WorkerResult | None,
        elapsed_s: float,
        failure: Outcome | None = None,
    ) -> dict:
        if elapsed_s < 0:
            raise ValueError("elapsed time cannot be negative")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM challenges WHERE id=?", (challenge.challenge_id,)).fetchone()
            if row is None or row["epoch_id"] != challenge.epoch_id or row["outcome"] != Outcome.PENDING:
                raise ValueError("unknown, unissued, or already completed challenge")
            data = json.loads(self.cipher.decrypt(row["payload"]))
            deadline = elapsed_s <= WorkloadSpec(**data["spec"]).deadline_s
            match = None
            if result is not None:
                expected_ops = 0 if challenge.spec.mode == "IDLE" else challenge.spec.operations
                match = (
                    result.challenge_id == challenge.challenge_id
                    and result.epoch_id == challenge.epoch_id
                    and result.operations_completed == expected_ops
                    and hmac.compare_digest(result.digest, data["expected_digest"])
                )
                outcome = (
                    Outcome.INCORRECT
                    if not match
                    else (Outcome.PASS if deadline else Outcome.DEADLINE_FAILED)
                )
            else:
                outcome = failure or Outcome.DEADLINE_FAILED
                if outcome not in {Outcome.DEADLINE_FAILED, Outcome.INTERRUPTED, Outcome.ERROR}:
                    raise ValueError("invalid no-response outcome")
            value = {
                "outcome": outcome.value,
                "digest_match": match,
                "deadline_met": deadline and result is not None,
                "elapsed_s": elapsed_s,
            }
            db.execute(
                "UPDATE challenges SET outcome=?,completion_json=? WHERE id=?",
                (outcome, canonical(value).decode(), challenge.challenge_id),
            )
            return value

    def recover_pending(self):
        """Call only when starting the single auditor, before accepting new epochs."""
        with self.connect() as db:
            return db.execute(
                "UPDATE challenges SET outcome=? WHERE outcome=?", (Outcome.INTERRUPTED, Outcome.PENDING)
            ).rowcount

    def interrupt_epoch(self, epoch_id: str):
        with self.connect() as db:
            return db.execute(
                "UPDATE challenges SET outcome=? WHERE epoch_id=? AND outcome=?",
                (Outcome.INTERRUPTED, epoch_id, Outcome.PENDING),
            ).rowcount

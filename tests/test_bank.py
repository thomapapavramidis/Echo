import concurrent.futures
import sqlite3
import uuid

import pytest

from compute_echo.bank import BankEmpty, ChallengeBank
from compute_echo.models import Outcome, WorkerResult
from compute_echo.workloads import WorkloadEngine


def answer(challenge):
    digest, timings = WorkloadEngine().execute(bytes.fromhex(challenge.seed), challenge.spec)
    return WorkerResult(
        challenge_id=challenge.challenge_id,
        epoch_id=challenge.epoch_id,
        digest=digest,
        timings=timings,
        backend="test:cpu",
        operations_completed=challenge.spec.operations,
    )


def test_issue_is_durable_and_private(tmp_path, small_spec):
    bank = ChallengeBank(tmp_path)
    bank.generate(small_spec, 1, WorkloadEngine())
    challenge = bank.issue(small_spec, uuid.uuid4().hex)
    reopened = ChallengeBank(tmp_path)
    assert reopened.available(small_spec) == 0
    with pytest.raises(BankEmpty):
        reopened.issue(small_spec, uuid.uuid4().hex)
    assert "expected_digest" not in challenge.model_dump()
    with bank.connect() as db:
        payload = db.execute("SELECT payload FROM challenges").fetchone()[0]
    assert challenge.seed.encode() not in payload
    assert reopened.recover_pending() == 1
    assert reopened.available(small_spec) == 0


def test_atomic_concurrent_issuance(tmp_path, small_spec):
    bank = ChallengeBank(tmp_path)
    bank.generate(small_spec, 8, WorkloadEngine())

    def issue(_):
        try:
            return bank.issue(small_spec, uuid.uuid4().hex).seed
        except BankEmpty:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(issue, range(16)))
    issued = [v for v in values if v is not None]
    assert len(issued) == len(set(issued)) == 8


@pytest.mark.parametrize(
    "scenario,expected",
    [
        ("valid", "PASS"),
        ("late", "DEADLINE_FAILED"),
        ("missing", "DEADLINE_FAILED"),
        ("bad", "INCORRECT"),
        ("interrupt", "INTERRUPTED"),
    ],
)
def test_terminal_outcomes_are_separate(tmp_path, small_spec, scenario, expected):
    bank = ChallengeBank(tmp_path)
    bank.generate(small_spec, 1, WorkloadEngine())
    challenge = bank.issue(small_spec, uuid.uuid4().hex)
    result = answer(challenge)
    if scenario == "bad":
        result = result.model_copy(update={"digest": "0" * 64})
    if scenario in {"missing", "interrupt"}:
        result = None
    elapsed = 2 if scenario in {"late", "missing"} else 0.1
    value = bank.complete(
        challenge, result, elapsed, Outcome.INTERRUPTED if scenario == "interrupt" else None
    )
    assert value["outcome"] == expected
    if scenario == "late":
        assert value["digest_match"] is True and not value["deadline_met"]
    with pytest.raises(ValueError):
        bank.complete(challenge, result, elapsed)
    assert bank.available(small_spec) == 0


def test_replay_and_wrong_epoch_rejected(tmp_path, small_spec):
    bank = ChallengeBank(tmp_path)
    bank.generate(small_spec, 2, WorkloadEngine())
    old = bank.issue(small_spec, uuid.uuid4().hex)
    old_result = answer(old)
    bank.complete(old, old_result, 0.1)
    new = bank.issue(small_spec, uuid.uuid4().hex)
    # Even relabeling a previous digest as the new task cannot make it correct.
    replay = old_result.model_copy(update={"challenge_id": new.challenge_id, "epoch_id": new.epoch_id})
    assert bank.complete(new, replay, 0.1)["outcome"] == "INCORRECT"
    assert old.seed != new.seed


def test_database_cannot_resurrect_or_delete_issued_seed(tmp_path, small_spec):
    bank = ChallengeBank(tmp_path)
    bank.generate(small_spec, 1, WorkloadEngine())
    bank.issue(small_spec, uuid.uuid4().hex)
    with bank.connect() as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("UPDATE challenges SET issued_ns=NULL")
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("DELETE FROM challenges")

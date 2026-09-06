import json
from types import SimpleNamespace

import pytest

from compute_echo.receipt import issue_receipt, load_or_create_key, verify_receipt
from compute_echo.safety import GPUSafetyGuard, SafetyStop
from compute_echo.storage import EventLog, write_json


def test_overtemperature_or_missing_telemetry_stops(monkeypatch):
    import compute_echo.safety as safety

    for output in ("90", "not supported"):
        monkeypatch.setattr(
            safety.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=output)
        )
        with pytest.raises(SafetyStop):
            GPUSafetyGuard()()
    monkeypatch.setattr(safety.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="50"))
    GPUSafetyGuard()()


def test_receipt_rejects_untrusted_issuer_and_tampered_files(tmp_path, make_epoch):
    epoch = make_epoch()
    write_json(tmp_path / "evidence.json", epoch)
    write_json(tmp_path / "finding.json", {"verdict": "INCONCLUSIVE"})
    EventLog(tmp_path / "events.jsonl").append({"state": "COMPLETE"})
    receipt = issue_receipt(
        tmp_path, load_or_create_key(tmp_path / "private" / "key"), "fixture", "synthetic test", None
    )
    key = receipt["payload"]["public_key_id"]
    assert verify_receipt(receipt, key, tmp_path)
    with pytest.raises(ValueError):
        verify_receipt(receipt, "untrusted")
    evidence = json.loads((tmp_path / "evidence.json").read_bytes())
    evidence["samples"][0]["watts"] += 50
    write_json(tmp_path / "evidence.json", evidence)
    with pytest.raises(ValueError, match="evidence hash"):
        verify_receipt(receipt, key, tmp_path)

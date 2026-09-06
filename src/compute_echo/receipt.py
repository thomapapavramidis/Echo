"""Ed25519 evidence receipts with a precisely specified canonical encoding."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .models import canonical


def load_or_create_key(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.exists():
        key = Ed25519PrivateKey.generate()
        raw = key.private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()
        )
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
    return Ed25519PrivateKey.from_private_bytes(path.read_bytes())


def issue_receipt(
    directory: Path, key: Ed25519PrivateKey, site: str, site_assurance: str, calibration_id: str | None
):
    files = ["evidence.json", "finding.json", "events.jsonl"]
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in files}
    evidence = json.loads((directory / "evidence.json").read_bytes())
    finding = json.loads((directory / "finding.json").read_bytes())
    public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    payload = {
        "schema_version": "echo-receipt-v1",
        "method": "echo-int-v1/settled-segment-linear-response-v1",
        "audit_id": evidence["epoch_id"],
        "issued_at_ns": time.time_ns(),
        "time_interval_ns": [evidence["started_at_ns"], evidence["ended_at_ns"]],
        "registered_site": site,
        "source_of_site_assurance": site_assurance,
        "calibration_id": calibration_id,
        "meter_id": evidence["meter_id"],
        "provenance": evidence["provenance"],
        "finding": finding,
        "hardware_enforcement": False,
        "device_identity_level": "software_registered",
        "expiry_policy": "point-in-time observation; no continued availability assurance",
        "evidence_sha256": hashes,
        "public_key_id": hashlib.sha256(public).hexdigest(),
        "assumptions": [
            "trusted offline reference bank",
            "independent correctly installed meter",
            "documented clock alignment",
            "validated test configuration",
            "no guarantee against adaptive local dummy loads",
        ],
        "signature_scope": "issuer and evidence integrity; not scientific correctness",
    }
    receipt = {
        "payload": payload,
        "public_key": base64.b64encode(public).decode(),
        "signature": base64.b64encode(key.sign(canonical(payload))).decode(),
    }
    path = directory / "receipt.json"
    with path.open("xb") as f:  # A completed receipt cannot be overwritten accidentally.
        f.write(canonical(receipt) + b"\n")
        f.flush()
        os.fsync(f.fileno())
    return receipt


def verify_receipt(receipt: dict, trusted_key_id: str, directory: Path | None = None):
    public = base64.b64decode(receipt["public_key"], validate=True)
    key_id = hashlib.sha256(public).hexdigest()
    if key_id != trusted_key_id or receipt["payload"]["public_key_id"] != key_id:
        raise ValueError("untrusted receipt issuer")
    Ed25519PublicKey.from_public_bytes(public).verify(
        base64.b64decode(receipt["signature"], validate=True), canonical(receipt["payload"])
    )
    if directory is not None:
        for name, expected in receipt["payload"]["evidence_sha256"].items():
            if name not in {"evidence.json", "finding.json", "events.jsonl"}:
                raise ValueError("unexpected evidence filename")
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
                raise ValueError("evidence hash mismatch: " + name)
    return True

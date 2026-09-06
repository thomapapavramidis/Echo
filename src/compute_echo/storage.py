"""Durable local evidence files. Private bank data must never be event payloads."""

import json
import os
import threading
from pathlib import Path

from .models import canonical


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as f:
        f.write(canonical(value) + b"\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


class EventLog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def append(self, value):
        with self.lock, self.path.open("ab") as f:
            f.write(canonical(value) + b"\n")
            f.flush()
            os.fsync(f.fileno())

    def read(self):
        with self.lock:
            if not self.path.exists():
                return []
            return [json.loads(line) for line in self.path.read_bytes().splitlines() if line]

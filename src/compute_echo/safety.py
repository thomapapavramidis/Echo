"""Local operational guard. Host diagnostics are never physical audit evidence."""

import os
import subprocess
import time


class SafetyStop(RuntimeError):
    pass


class GPUSafetyGuard:
    def __init__(self, max_temperature_c: float = 85, device: int = 0):
        self.max_temperature_c, self.device = max_temperature_c, device
        self.last_check = -float("inf")

    def __call__(self):
        now = time.monotonic()
        if now - self.last_check < 1:
            return
        self.last_check = now
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.device}",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if result.returncode != 0:
                raise ValueError("temperature query failed")
            temperature = float(result.stdout.strip())
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            raise SafetyStop(
                "GPU temperature unavailable; stop until safety telemetry is restored"
            ) from error
        if temperature >= self.max_temperature_c:
            raise SafetyStop("GPU temperature limit reached")

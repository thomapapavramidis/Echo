"""Real HTTP integration on one host. This is NOT a remote-site electrical test.

Launches two temporary workers, checks fresh local and forwarded results through
actual TCP sockets, writes diagnostics, and shuts both workers down. No simulated
power is generated. All modes are exercised; optional CUDA uses this host's GPU.
"""

import argparse
import asyncio
import os
import secrets
import socket
import subprocess
import sys
import uuid
from pathlib import Path

from compute_echo.bank import ChallengeBank
from compute_echo.epochs import EpochRunner
from compute_echo.meter import MeterConfig, MeterStore
from compute_echo.models import Mode, WorkloadSpec
from compute_echo.physics import pipeline_report
from compute_echo.storage import write_json
from compute_echo.transport import HTTPWorker
from compute_echo.workloads import WorkloadEngine


def unused_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def main(args):
    directory = Path("runs") / ("loopback-" + uuid.uuid4().hex)
    directory.mkdir(parents=True)
    helper_port, facility_port = unused_port(), unused_port()
    helper_token, facility_token, control = [secrets.token_hex(32) for _ in range(3)]
    processes, logs = [], []
    env = {**os.environ, "CUPY_CACHE_DIR": str(Path(".cupy").resolve())}
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    def launch(name, port, token, helper=False):
        log = (directory / (name + ".log")).open("w")
        logs.append(log)
        command = [
            sys.executable,
            "-m",
            "compute_echo.cli",
            "worker",
            "--backend",
            args.backend,
            "--ledger",
            str(directory / (name + ".sqlite3")),
            "--port",
            str(port),
        ]
        if helper:
            command += ["--helper-url", f"http://127.0.0.1:{helper_port}"]
        process = subprocess.Popen(
            command,
            env={
                **env,
                "ECHO_WORKER_TOKEN": token,
                "ECHO_CONTROL_TOKEN": control,
                "ECHO_HELPER_TOKEN": helper_token,
            },
            stdout=log,
            stderr=log,
            creationflags=flags,
        )
        processes.append(process)

    worker = HTTPWorker(f"http://127.0.0.1:{facility_port}", facility_token, control)
    helper_http = HTTPWorker(f"http://127.0.0.1:{helper_port}", helper_token, control)
    try:
        launch("helper", helper_port, helper_token)
        launch("facility", facility_port, facility_token, True)
        for transport in (helper_http, worker):
            for _ in range(120):
                try:
                    response = await transport.client.get("/health", timeout=1)
                    if response.status_code == 200:
                        break
                except Exception:
                    pass
                if any(p.poll() is not None for p in processes):
                    raise RuntimeError(f"worker exited; inspect {directory}")
                await asyncio.sleep(0.25)
            else:
                raise RuntimeError(f"worker startup failed; inspect {directory}")
        specifications = [
            WorkloadSpec(
                mode=mode,
                m=32,
                n=48,
                k=64,
                lanes=8,
                words_per_lane=64,
                memory_steps=32,
                operations=2,
                segment_s=0.4,
                deadline_s=0.35,
            )
            for mode in Mode
        ]
        bank = ChallengeBank(directory / "private")
        for spec in specifications:
            bank.generate(spec, 3 * args.runs_per_route * 2, WorkloadEngine())
        meter = MeterStore(
            directory / "meter.sqlite3",
            MeterConfig(
                meter_id="absent", provenance="simulation", installation="No meter: loopback timing test"
            ),
        )
        runner = EpochRunner(bank, worker, meter, directory / "epochs")
        reports = {}
        all_pass = True
        for route in ("local", "forwarded"):
            await worker.set_route(route)
            epochs = []
            for _ in range(args.runs_per_route):
                epoch, finding = await runner.run(specifications, 1, washout_s=0, timing_only=True)
                epochs.append(epoch)
                passed = epoch.complete and all(s.digest_match and s.deadline_met for s in epoch.segments)
                all_pass &= passed
                print(
                    f"{route} epoch={epoch.epoch_id} computational_pass={passed} physical={finding['physical_response']}",
                    flush=True,
                )
            reports[route] = {"epochs": [e.epoch_id for e in epochs], "pipeline": pipeline_report(epochs)}
        report = {
            "kind": "same-host-loopback-computational-integration",
            "backend": args.backend,
            "real_meter_used": False,
            "separate_helper_machine": False,
            "all_computational_targets_passed": all_pass,
            "runs_per_route": args.runs_per_route,
            "specifications": [s.model_dump(mode="json") for s in specifications],
            "routes": reports,
            "raw_run_directory": str(directory),
            "physical_acceptance": "NOT_MEASURED",
        }
        write_json(args.output, report)
        if not all_pass:
            raise RuntimeError("computational smoke test failed; inspect saved report")
    finally:
        try:
            await worker.quiesce()
        except Exception:
            pass
        await worker.close()
        await helper_http.close()
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for log in logs:
            log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--runs-per-route", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("runs/loopback-report.json"))
    asyncio.run(main(parser.parse_args()))

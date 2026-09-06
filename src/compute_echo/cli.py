"""Command-line operations for reference generation, physics runs, and services."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import secrets
import subprocess
import sys
from pathlib import Path

from .bank import ChallengeBank
from .calibration import calibrate
from .demo import DemoRuntime
from .deployment import load_env_file, probe_readiness, runtime_config, static_readiness
from .detector import Calibration
from .epochs import EpochRunner
from .evaluation import evaluate
from .hybrid import HybridDemoRuntime, create_hybrid_app
from .meter import MeterConfig, MeterStore, collect_http, collect_into_store
from .models import EpochEvidence, Mode, WorkloadSpec
from .physics import MeterProfile, characterize, pipeline_report
from .profiling import run_host_profile
from .receipt import verify_receipt
from .safety import GPUSafetyGuard
from .service import ProcessLock, create_auditor_app
from .storage import EventLog, write_json
from .telemetry import DemoTelemetryHub, NVMLDemoSensor, RemoteTelemetryStream, TelemetryRole
from .transport import HTTPWorker
from .worker import WorkerRuntime, create_worker_app
from .workloads import WorkloadEngine


def read(path):
    return json.loads(Path(path).read_bytes())


def specs(path):
    return [WorkloadSpec.model_validate(s) for s in read(path)]


def evidence(paths):
    return [
        EpochEvidence.model_validate(read(Path(p) / "evidence.json" if Path(p).is_dir() else p))
        for p in paths
    ]


def secret(name):
    value = os.environ.get(name, "")
    if len(value) < 24:
        raise ValueError(f"{name} must be set to a secret of at least 24 characters")
    return value


def doctor(output: Path, cuda: bool):
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "gpu": None,
        "real_meter_acceptance": "NOT_MEASURED",
        "external_meter": "NOT_CONFIGURED",
        "separate_unmetered_helper": "NOT_CONFIGURED",
        "cuda_digest_checks": None,
        "nvidia_smi_inventory_fields": [
            "uuid",
            "name",
            "memory.total MiB",
            "driver_version",
            "power.limit W",
            "mig.mode.current",
        ],
        "framework_versions": None,
        "cuda_versions": None,
    }
    try:
        process = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,memory.total,driver_version,power.limit,mig.mode.current",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        report["gpu"] = process.stdout.strip() if process.returncode == 0 else "unavailable"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    if cuda:
        try:
            framework = {}
            try:
                import torch

                framework["pytorch_version"] = torch.__version__
                framework["pytorch_cuda_build"] = torch.version.cuda
                framework["pytorch_cuda_available"] = torch.cuda.is_available()
                framework["pytorch_device_name"] = (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                )
                framework["pytorch_device_capability"] = (
                    list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
                )
            except ImportError:
                framework["pytorch"] = "not installed"
            report["framework_versions"] = framework
            cpu, gpu = WorkloadEngine("cpu"), WorkloadEngine("cuda")
            report["cuda_versions"] = {
                "cupy_version": gpu.cp.__version__,
                "cuda_runtime_version": gpu.cp.cuda.runtime.runtimeGetVersion(),
                "cuda_driver_version": gpu.cp.cuda.runtime.driverGetVersion(),
            }
            checks = []
            for mode in Mode:
                spec = WorkloadSpec(
                    mode=mode, m=32, n=48, k=64, operations=2, lanes=8, words_per_lane=64, memory_steps=300
                )
                expected, _ = cpu.execute(bytes(range(32)), spec)
                actual, timing = gpu.execute(bytes(range(32)), spec)
                checks.append(
                    {
                        "mode": mode.value,
                        "matches_cpu": expected == actual,
                        "cuda_pipeline_s": timing.model_dump(),
                    }
                )
            report["cuda_digest_checks"] = checks
        except Exception as error:
            report["cuda_error"] = f"{type(error).__name__}: {error}"
    write_json(output, report)
    print(json.dumps(report, indent=2))


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("doctor", help="inspect hardware; optionally check CPU/CUDA known answers")
    p.add_argument("--output", type=Path, default=Path("runs/doctor.json"))
    p.add_argument("--cuda", action="store_true")
    p = commands.add_parser("workload-profile", help="trusted local workload tuning; not a physical audit")
    p.add_argument("--specs", required=True)
    p.add_argument("--backend", choices=["cpu", "cuda"], required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser(
        "host-profile", help="bank-free CUDA workload profile with non-independent NVML telemetry"
    )
    p.add_argument("--specs", required=True)
    p.add_argument("--role", choices=[role.value for role in TelemetryRole], required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--poll-interval", type=float, default=0.1)
    p.add_argument("--baseline", type=float, default=2)
    p.add_argument("--cooldown", type=float, default=3)
    p.add_argument("--max-run", type=float, default=30)
    p.add_argument("--max-temperature", type=float, default=82)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser(
        "bank-generate", help="TRUSTED EQUIPMENT ONLY: create private single-use challenges"
    )
    p.add_argument("--private", type=Path, default=Path("private"))
    p.add_argument("--specs", required=True)
    p.add_argument("--count-per-mode", type=int, required=True)
    p.add_argument("--backend", choices=["cpu", "cuda"], required=True)
    p = commands.add_parser("worker", help="serve a local executor or forwarding facility")
    p.add_argument("--backend", choices=["cpu", "cuda"], required=True)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--helper-url")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8101)
    p.add_argument("--telemetry-role", choices=[role.value for role in TelemetryRole])
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--profile-path", type=Path)
    p = commands.add_parser("runpod-demo", help="serve the laptop auditor and plain POC dashboard")
    p.add_argument("--env-file", type=Path, default=Path(".env.runpod"))
    p = commands.add_parser(
        "hybrid-demo", help="serve the transparent live-compute plus simulated-meter dashboard"
    )
    p.add_argument("--env-file", type=Path, default=Path(".env.hybrid"))
    p = commands.add_parser("deployment-readiness", help="report missing two-Pod configuration")
    p.add_argument("--env-file", type=Path, default=Path(".env.runpod"))
    p.add_argument("--probe", action="store_true", help="query configured live endpoints")
    for name in ("auditor", "physics-run"):
        p = commands.add_parser(name)
        p.add_argument("--private", type=Path, default=Path("private"))
        p.add_argument("--runs", type=Path, default=Path("runs"))
        p.add_argument("--meter-config", required=True)
        p.add_argument("--worker-url", required=True)
        p.add_argument("--calibration")
        if name == "auditor":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8100)
            p.add_argument("--site", required=True)
            p.add_argument("--site-assurance", required=True)
        else:
            p.add_argument("--specs", required=True)
            p.add_argument("--repeats", type=int, default=4)
            p.add_argument("--route", choices=["local", "forwarded"], required=True)
            p.add_argument("--provisional-washout", type=float)
            p.add_argument("--timing-only", action="store_true")
            p.add_argument("--meter-source-url", help="normalized independent HTTP meter adapter")
    p = commands.add_parser("collect-http")
    p.add_argument("--source-url", required=True)
    p.add_argument("--auditor-url", required=True)
    p.add_argument("--interval", type=float, default=0.5)
    p = commands.add_parser("collect-serial")
    p.add_argument("--port", required=True)
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--auditor-url", required=True)
    p = commands.add_parser("characterize")
    p.add_argument("epochs", nargs="+")
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("calibrate")
    p.add_argument("--training", nargs="+", required=True)
    p.add_argument("--validation", nargs="+", required=True)
    p.add_argument("--profile", required=True)
    p.add_argument("--hardware-configuration", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("evaluate")
    p.add_argument("epochs", nargs="+")
    p.add_argument("--labels", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("verify-receipt")
    p.add_argument("receipt", type=Path)
    p.add_argument("--trusted-key-id", required=True)
    p.add_argument("--evidence-directory", type=Path)
    return root


def main():
    args = parser().parse_args()
    if args.command == "doctor":
        doctor(args.output, args.cuda)
    elif args.command == "host-profile":
        report = run_host_profile(
            specs(args.specs),
            TelemetryRole(args.role),
            args.output,
            repeats=args.repeats,
            poll_interval_s=args.poll_interval,
            baseline_s=args.baseline,
            cooldown_s=args.cooldown,
            max_run_s=args.max_run,
            max_temperature_c=args.max_temperature,
        )
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "backend": report["backend"],
                    "records": len(report["records"]),
                    "telemetry_available_samples": report["telemetry_available_samples"],
                    "telemetry_unavailable_samples": report["telemetry_unavailable_samples"],
                    "workload_specifications_frozen": False,
                    "calibration_thresholds_selected": False,
                },
                indent=2,
            )
        )
        if report["fatal_error"]:
            raise SystemExit(4)
    elif args.command == "workload-profile":
        if not 1 <= args.repeats <= 100:
            raise ValueError("repeats must be 1..100")
        engine = WorkloadEngine(args.backend, GPUSafetyGuard() if args.backend == "cuda" else None)
        records = []
        for spec in specs(args.specs):
            # Explicit discarded warmup, followed by fresh random held-out seeds.
            engine.execute(secrets.token_bytes(32), spec)
            for index in range(args.repeats):
                _, timing = engine.execute(secrets.token_bytes(32), spec)
                record = {
                    "mode": spec.mode.value,
                    "spec": spec.model_dump(mode="json"),
                    "replicate": index,
                    "pipeline": timing.model_dump(),
                    "worker_completes_before_deadline": timing.total_worker_s < spec.deadline_s,
                    "execution_fraction_of_segment": timing.execution_s / spec.segment_s,
                }
                records.append(record)
                print(
                    f"{spec.mode} replicate={index} worker_s={timing.total_worker_s:.4f} "
                    f"execution_s={timing.execution_s:.4f}",
                    flush=True,
                )
        write_json(
            args.output,
            {
                "kind": "trusted-local-workload-profile",
                "backend": engine.identity,
                "external_meter": False,
                "network_included": False,
                "reference_comparison": False,
                "records": records,
            },
        )
    elif args.command == "bank-generate":
        bank = ChallengeBank(args.private)
        engine = WorkloadEngine(args.backend, GPUSafetyGuard() if args.backend == "cuda" else None)
        for spec in specs(args.specs):
            bank.generate(spec, args.count_per_mode, engine)
            print(f"{spec.mode}: {bank.available(spec)} unused challenges; spec={spec.key}")
    elif args.command == "worker":
        import uvicorn

        helper = HTTPWorker(args.helper_url, secret("ECHO_HELPER_TOKEN")) if args.helper_url else None
        telemetry = (
            NVMLDemoSensor(TelemetryRole(args.telemetry_role), args.gpu_index)
            if args.telemetry_role
            else None
        )
        runtime = WorkerRuntime(
            args.backend, args.ledger, helper, telemetry=telemetry, profile_path=args.profile_path
        )
        app = create_worker_app(runtime, secret("ECHO_WORKER_TOKEN"), secret("ECHO_CONTROL_TOKEN"))
        uvicorn.run(app, host=args.host, port=args.port, access_log=False)
    elif args.command == "deployment-readiness":
        if args.env_file.exists():
            load_env_file(args.env_file, override=True)
        report = static_readiness()
        if args.probe and report["ready_to_probe_pods"]:
            report["live_probe"] = asyncio.run(probe_readiness(runtime_config()))
        elif args.probe:
            report["live_probe"] = {"skipped": True, "reason": "static configuration incomplete"}
        print(json.dumps(report, indent=2))
        if not report["ready_to_probe_pods"]:
            raise SystemExit(2)
        if args.probe and not report["live_probe"].get("ready_for_workload_tuning", False):
            raise SystemExit(3)
    elif args.command == "runpod-demo":
        import uvicorn

        load_env_file(args.env_file, override=True)
        config = runtime_config()
        bank = ChallengeBank(config["private"])
        meter = MeterStore(
            config["private"] / "host-telemetry-not-a-meter.sqlite3",
            MeterConfig(
                meter_id="nvml-demo-not-independent",
                provenance="host_telemetry",
                installation="Host NVML diagnostic only; not auditor-controlled external measurement",
                source_freshness="unknown",
            ),
        )
        worker = HTTPWorker(
            config["site_url"], config["site_worker_token"], config["site_control_token"]
        )
        telemetry = DemoTelemetryHub(
            RemoteTelemetryStream(
                TelemetryRole.DECLARED_SITE,
                config["site_url"],
                config["site_worker_token"],
                config["polling_s"],
                config["private"] / "declared-site-nvml-demo.jsonl",
            ),
            RemoteTelemetryStream(
                TelemetryRole.REMOTE_HELPER,
                config["helper_url"],
                config["helper_worker_token"],
                config["polling_s"],
                config["private"] / "helper-attack-ground-truth-nvml-demo.jsonl",
            ),
        )
        runner = EpochRunner(bank, worker, meter, config["runs"])
        demo = DemoRuntime(
            telemetry, config["specs"], config["repeats"], config["washout_s"]
        )
        app = create_auditor_app(
            runner,
            config["admin_token"],
            config["meter_token"],
            config["private"],
            "declared RunPod GPU Pod",
            "host telemetry POC; deployment validation pending",
            demo,
        )
        uvicorn.run(app, host=config["host"], port=config["port"], access_log=False)
    elif args.command == "hybrid-demo":
        import uvicorn

        load_env_file(args.env_file, override=True)
        site_url = os.environ.get("SITE_API_URL", "").strip()
        helper_url = os.environ.get("HELPER_API_URL", "").strip()
        if not site_url.startswith("https://") or not helper_url.startswith("https://"):
            raise ValueError("SITE_API_URL and HELPER_API_URL must be configured HTTPS endpoints")
        site_worker_token = secret("SITE_WORKER_TOKEN")
        site_control_token = secret("SITE_CONTROL_TOKEN")
        helper_worker_token = secret("HELPER_WORKER_TOKEN")
        if len({site_worker_token, site_control_token, helper_worker_token}) != 3:
            raise ValueError("hybrid worker and control credentials must be distinct")
        private = Path(os.environ.get("HYBRID_PRIVATE_DIR", "private/hybrid-demo"))
        hybrid_specs = specs(
            os.environ.get("HYBRID_WORKLOAD_SPECS", "config/workloads.hybrid-demo.json")
        )
        polling_s = float(os.environ.get("SENSOR_POLL_INTERVAL_S", "0.25"))
        worker = HTTPWorker(site_url, site_worker_token, site_control_token)
        telemetry = DemoTelemetryHub(
            RemoteTelemetryStream(
                TelemetryRole.DECLARED_SITE,
                site_url,
                site_worker_token,
                polling_s,
                private / "facility-live-host-telemetry.jsonl",
            ),
            RemoteTelemetryStream(
                TelemetryRole.REMOTE_HELPER,
                helper_url,
                helper_worker_token,
                polling_s,
                private / "helper-private-ground-truth.jsonl",
            ),
        )
        runtime = HybridDemoRuntime(
            worker,
            telemetry,
            hybrid_specs,
            Path(os.environ.get("HYBRID_CAPTURE_PATH", "runs/hybrid-demo/capture.json")),
            facility_name=os.environ.get("FACILITY_NAME", "NVIDIA H100 80GB HBM3"),
            facility_region=os.environ.get("FACILITY_REGION", "US Northeast"),
            helper_name=os.environ.get("HELPER_NAME", "NVIDIA H100 80GB HBM3"),
            helper_region=os.environ.get("HELPER_REGION", "Iceland / Europe"),
        )
        app = create_hybrid_app(runtime)
        # Browser control endpoints contain no worker credentials and stay loopback-only.
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=int(os.environ.get("HYBRID_DASHBOARD_PORT", "8100")),
            access_log=False,
        )
    elif args.command in {"auditor", "physics-run"}:
        bank = ChallengeBank(args.private)
        meter = MeterStore(
            args.private / "meter.sqlite3", MeterConfig.model_validate(read(args.meter_config))
        )
        worker = HTTPWorker(args.worker_url, secret("ECHO_WORKER_TOKEN"), secret("ECHO_CONTROL_TOKEN"))
        model = Calibration.model_validate(read(args.calibration)) if args.calibration else None
        runner = EpochRunner(bank, worker, meter, args.runs, model)
        if args.command == "auditor":
            import uvicorn

            app = create_auditor_app(
                runner,
                secret("ECHO_ADMIN_TOKEN"),
                secret("ECHO_METER_TOKEN"),
                args.private,
                args.site,
                args.site_assurance,
            )
            uvicorn.run(app, host=args.host, port=args.port, access_log=False)
        else:

            async def run():
                lock = ProcessLock(args.private / "auditor.lock")
                lock.acquire()
                collector = None
                try:
                    bank.recover_pending()
                    if args.meter_source_url:
                        collector = asyncio.create_task(collect_into_store(args.meter_source_url, meter))
                        for _ in range(50):
                            if meter.health() == "OK":
                                break
                            await asyncio.sleep(0.1)
                    await worker.set_route(args.route)
                    epoch, finding = await runner.run(
                        specs(args.specs),
                        args.repeats,
                        washout_s=args.provisional_washout,
                        timing_only=args.timing_only,
                    )
                    EventLog(args.private / "evaluation-labels.jsonl").append(
                        {"epoch_id": epoch.epoch_id, "route": args.route}
                    )
                    print(
                        json.dumps(
                            {
                                "epoch_id": epoch.epoch_id,
                                "finding": finding,
                                "pipeline": pipeline_report([epoch]),
                            },
                            indent=2,
                        )
                    )
                finally:
                    if collector:
                        collector.cancel()
                        await asyncio.gather(collector, return_exceptions=True)
                    await worker.close()
                    lock.release()

            asyncio.run(run())
    elif args.command == "collect-http":
        if args.interval <= 0:
            raise ValueError("poll interval must be positive")
        asyncio.run(
            collect_http(args.source_url, args.auditor_url, secret("ECHO_METER_TOKEN"), args.interval)
        )
    elif args.command == "collect-serial":
        import httpx

        from .meter import serial_samples

        with httpx.Client(timeout=5, trust_env=False) as client:
            for sample in serial_samples(args.port, args.baudrate):
                response = client.post(
                    args.auditor_url.rstrip("/") + "/meter/samples",
                    json=sample.model_dump(),
                    headers={"Authorization": "Bearer " + secret("ECHO_METER_TOKEN")},
                )
                if response.status_code != 409:
                    response.raise_for_status()
    elif args.command == "characterize":
        report = characterize(evidence(args.epochs))
        write_json(args.output, report)
        print(json.dumps(report, indent=2))
    elif args.command == "calibrate":
        raw = read(args.profile)
        profile = MeterProfile.model_validate(raw.get("profile", raw))
        model = calibrate(
            evidence(args.training), evidence(args.validation), profile, args.hardware_configuration
        )
        if args.output.exists():
            raise ValueError("calibrations are immutable; choose a new output filename")
        write_json(args.output, model)
        print(
            f"calibration={model.calibration_id} validated={model.validated} categories={model.category_map}"
        )
    elif args.command == "evaluate":
        labels = EventLog(Path(args.labels)).read()
        report = evaluate(
            evidence(args.epochs),
            {x["epoch_id"]: x["route"] for x in labels},
            Calibration.model_validate(read(args.calibration)),
        )
        write_json(args.output, report)
        print(json.dumps(report, indent=2))
    elif args.command == "verify-receipt":
        verify_receipt(read(args.receipt), args.trusted_key_id, args.evidence_directory)
        print("Signature, trusted issuer, and requested evidence hashes verified.")


if __name__ == "__main__":
    main()

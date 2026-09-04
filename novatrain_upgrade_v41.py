from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

VERSION = "4.1.0"
TASK_NAME = "NovaTrain-X4-GitHub-Account-Trainer"
PAYLOADS = (
    ("worker.py.gz.b64", "worker.py", "72f98fefaaa1e62e54fb860a2f91f56335d439f90f8c984af6b71ff6ead08b68"),
    ("telemetry.py.gz.b64", "telemetry.py", "337164c0d4384289f9e09a05a4dd3f078f7ba30068f82a60ba50028f79524157"),
    ("updater.py.gz.b64", "updater.py", "56ff6d3133b3b92eab86e8d40f5e8e59bf640dcd53040a52932473396c293a5d"),
)


def find_install_root() -> Path:
    candidates: list[Path] = []
    env = os.environ.get("NOVATRAIN_INSTALL_ROOT")
    if env:
        candidates.append(Path(env))
    if os.name == "nt":
        candidates.append(Path(r"C:\NovaTrainX4"))
        local = os.environ.get("LOCALAPPDATA")
        if local:
            candidates.append(Path(local) / "NovaTrainX4")
    for root in candidates:
        if (root / "package" / "tools" / "novatrain_x4" / "worker.py").is_file():
            return root
    raise RuntimeError("NovaTrain installation root was not found")


def atomic_write(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".new")
    tmp.write_bytes(data)
    os.replace(tmp, target)


def decode_payload(path: Path, expected_sha256: str) -> bytes:
    try:
        raw = base64.b64decode("".join(path.read_text(encoding="ascii").split()), validate=True)
        data = gzip.decompress(raw)
    except Exception as exc:
        raise RuntimeError(f"invalid NovaTrain upgrade payload: {path.name}") from exc
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(f"NovaTrain upgrade hash mismatch: {path.name}")
    return data


def schedule_worker_restart() -> bool:
    if os.name != "nt":
        return False
    script = (
        "$ErrorActionPreference='SilentlyContinue'; "
        "Start-Sleep -Seconds 120; "
        f"Stop-ScheduledTask -TaskName '{TASK_NAME}'; "
        "Start-Sleep -Seconds 3; "
        f"Start-ScheduledTask -TaskName '{TASK_NAME}'"
    )
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
    try:
        subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-WindowStyle", "Hidden", "-Command", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
            close_fds=True,
        )
        return True
    except OSError:
        return False


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    source_dir = repo_root / ".novatrain" / "releases" / VERSION
    install = find_install_root()
    package_dir = install / "package" / "tools" / "novatrain_x4"
    worker_root = install / "worker"
    backup = install / "backups" / f"pre-{VERSION}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    backup.mkdir(parents=True, exist_ok=True)

    installed: list[str] = []
    for payload_name, target_name, expected in PAYLOADS:
        source = source_dir / payload_name
        if not source.is_file():
            raise RuntimeError(f"upgrade payload missing: {source}")
        data = decode_payload(source, expected)
        target = package_dir / target_name
        if target.is_file():
            shutil.copy2(target, backup / target_name)
        atomic_write(target, data)
        installed.append(str(target))

    config = {
        "version": 1,
        "update": {
            "enabled": True,
            "repo": "lil-fahad/Test",
            "path": ".novatrain/worker_update.json",
            "branch": "main"
        }
    }
    worker_root.mkdir(parents=True, exist_ok=True)
    config_path = worker_root / "config.json"
    tmp = config_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, config_path)

    system_python = install / "venv" / "Scripts" / "python.exe" if os.name == "nt" else install / "venv" / "bin" / "python"
    smoke_version = None
    smoke_capabilities = None
    if system_python.is_file():
        proc = subprocess.run(
            [str(system_python), "-m", "py_compile", *(str(package_dir / target) for _, target, _ in PAYLOADS)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "installed NovaTrain code failed syntax validation")
        smoke_env = os.environ.copy()
        smoke_env["PYTHONPATH"] = str(install / "package")
        version_proc = subprocess.run(
            [str(system_python), "-m", "tools.novatrain_x4.worker", "--version"],
            env=smoke_env, capture_output=True, text=True, check=False, timeout=30,
        )
        if version_proc.returncode != 0 or version_proc.stdout.strip() != VERSION:
            raise RuntimeError(version_proc.stderr.strip() or f"NovaTrain smoke test returned {version_proc.stdout.strip()!r}")
        smoke_version = version_proc.stdout.strip()
        cap_proc = subprocess.run(
            [str(system_python), "-m", "tools.novatrain_x4.worker", "--root", str(worker_root), "--capabilities"],
            env=smoke_env, capture_output=True, text=True, check=False, timeout=60,
        )
        if cap_proc.returncode == 0:
            try:
                smoke_capabilities = json.loads(cap_proc.stdout)
            except json.JSONDecodeError:
                smoke_capabilities = {"raw": cap_proc.stdout[-4000:]}

    restart_scheduled = schedule_worker_restart()
    out = repo_root / ".novatrain" / "output"
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "ok": True,
        "version": VERSION,
        "installed": installed,
        "backup": str(backup),
        "update_config": str(config_path),
        "restart_scheduled": restart_scheduled,
        "smoke_version": smoke_version,
        "smoke_capabilities": smoke_capabilities,
        "message": "NovaTrain-X4.1 installed; scheduled worker will restart automatically."
    }
    (out / "upgrade-v41.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("NOVATRAIN_PROGRESS " + json.dumps({"phase": "upgrade", "progress": 1.0, "version": VERSION}), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

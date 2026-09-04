from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from .github_control import authenticated_login, fetch_authorized_manifest, fetch_file


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    os.replace(tmp, path)


def _safe_target(package_dir: Path, value: str) -> Path:
    rel = Path(value)
    if rel.is_absolute() or ".." in rel.parts or rel.suffix != ".py":
        raise RuntimeError(f"unsafe NovaTrain update target: {value}")
    target = (package_dir / rel).resolve()
    root = package_dir.resolve()
    if target != root and root not in target.parents:
        raise RuntimeError(f"NovaTrain update escaped package directory: {value}")
    return target


def maybe_apply_update(worker_root: Path) -> dict[str, object] | None:
    config_path = worker_root / "config.json"
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    update = config.get("update") if isinstance(config, dict) else None
    if not isinstance(update, dict) or not bool(update.get("enabled", True)):
        return None
    repo = str(update.get("repo", "")).strip()
    path = str(update.get("path", ".novatrain/worker_update.json")).strip()
    branch = str(update.get("branch", "main")).strip() or "main"
    if not repo or "/" not in repo:
        return None

    login = authenticated_login()
    snapshot = fetch_authorized_manifest(repo, path, branch, login)
    if snapshot is None:
        return None
    try:
        spec = json.loads(snapshot.content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid NovaTrain update manifest: {exc}") from exc
    if not isinstance(spec, dict) or spec.get("schema") != 1:
        raise RuntimeError("unsupported NovaTrain update manifest schema")
    version = str(spec.get("version", "")).strip()
    files = spec.get("files")
    if not version or not isinstance(files, list) or not files:
        raise RuntimeError("NovaTrain update manifest is incomplete")

    state_path = worker_root / "update-state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    except Exception:
        state = {}
    if state.get("commit") == snapshot.commit_sha and state.get("version") == version:
        return None

    package_dir = Path(__file__).resolve().parent
    staged: list[tuple[Path, bytes]] = []
    for item in files:
        if not isinstance(item, dict):
            raise RuntimeError("invalid file entry in NovaTrain update manifest")
        source = str(item.get("source", "")).strip()
        target_value = str(item.get("target", "")).strip()
        expected = str(item.get("sha256", "")).strip().lower()
        if not source or not expected or len(expected) != 64:
            raise RuntimeError("NovaTrain update file is missing source or sha256")
        target = _safe_target(package_dir, target_value)
        blob = fetch_file(repo, source, snapshot.commit_sha)
        if blob is None:
            raise RuntimeError(f"NovaTrain update source missing: {source}")
        encoding = str(item.get("encoding", "raw")).strip().lower()
        if encoding == "gzip-base64":
            try:
                blob = gzip.decompress(base64.b64decode(b"".join(blob.split()), validate=True))
            except Exception as exc:
                raise RuntimeError(f"unable to decode NovaTrain update payload: {source}") from exc
        elif encoding != "raw":
            raise RuntimeError(f"unsupported NovaTrain update encoding: {encoding}")
        actual = hashlib.sha256(blob).hexdigest()
        if actual != expected:
            raise RuntimeError(f"NovaTrain update hash mismatch for {source}")
        staged.append((target, blob))

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup = worker_root / "backups" / f"pre-{version}-{stamp}"
    backup.mkdir(parents=True, exist_ok=True)
    for target, blob in staged:
        if target.is_file():
            shutil.copy2(target, backup / target.name)
        _atomic_write(target, blob)

    state_payload = {
        "version": version,
        "commit": snapshot.commit_sha,
        "applied_at": datetime.now(UTC).isoformat(),
        "files": [target.name for target, _ in staged],
    }
    _atomic_write(state_path, (json.dumps(state_payload, indent=2) + "\n").encode("utf-8"))
    return state_payload

from __future__ import annotations

import json
import math
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

STRUCTURED_PREFIX = "NOVATRAIN_PROGRESS "


def _float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def gpu_snapshot() -> dict[str, object] | None:
    """Best-effort NVIDIA telemetry with no extra Python dependency."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    rows: list[dict[str, object]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        name, temp, util, used, total, power = parts[:6]
        row: dict[str, object] = {"name": name}
        for key, raw in (
            ("temperature_c", temp),
            ("utilization_percent", util),
            ("memory_used_mb", used),
            ("memory_total_mb", total),
            ("power_w", power),
        ):
            value = _float(raw)
            if value is not None:
                row[key] = round(value, 2)
        rows.append(row)
    return {"gpus": rows} if rows else None


_GENERIC_EPOCH = re.compile(r"(?i)\bepoch\s*[:=]?\s*(\d+)\s*(?:/|of)\s*(\d+)")
_PYDICT_EPOCH = re.compile(r"['\"]epoch['\"]\s*:\s*(\d+)")
_PYDICT_VAL_ACC = re.compile(r"['\"]val_accuracy['\"]\s*:\s*([0-9.eE+-]+)")
_PYDICT_TRAIN_LOSS = re.compile(r"['\"]train_loss['\"]\s*:\s*([0-9.eE+-]+)")
_PYDICT_TRAIN_ACC = re.compile(r"['\"]train_accuracy['\"]\s*:\s*([0-9.eE+-]+)")


@dataclass
class ProgressTracker:
    started_monotonic: float = field(default_factory=time.monotonic)
    last_publish_monotonic: float = 0.0
    last_signature: tuple[object, ...] | None = None
    best_accuracy: float | None = None
    latest: dict[str, object] = field(default_factory=dict)

    def parse_line(self, line: str) -> dict[str, object] | None:
        text = line.strip()
        if not text:
            return None
        if text.startswith(STRUCTURED_PREFIX):
            raw = text[len(STRUCTURED_PREFIX):].strip()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return None
            if not isinstance(payload, dict):
                return None
            return self._normalize(dict(payload))

        payload: dict[str, object] = {}
        m = _GENERIC_EPOCH.search(text)
        if m:
            payload["epoch"] = int(m.group(1))
            payload["epochs_total"] = int(m.group(2))
        else:
            m = _PYDICT_EPOCH.search(text)
            if m:
                payload["epoch"] = int(m.group(1))
        for regex, key in (
            (_PYDICT_VAL_ACC, "val_accuracy"),
            (_PYDICT_TRAIN_LOSS, "train_loss"),
            (_PYDICT_TRAIN_ACC, "train_accuracy"),
        ):
            match = regex.search(text)
            if match:
                value = _float(match.group(1))
                if value is not None:
                    payload[key] = value
        if not payload:
            return None
        first = text.split(maxsplit=1)[0]
        if first and len(first) <= 80 and first.lower() not in {"epoch", "train", "training"}:
            payload["model"] = first
        return self._normalize(payload)

    def _normalize(self, payload: dict[str, object]) -> dict[str, object]:
        epoch = payload.get("epoch")
        total_epochs = payload.get("epochs_total") or payload.get("total_epochs")
        model_index = payload.get("model_index")
        models_total = payload.get("models_total")
        progress = _float(payload.get("progress"))
        if progress is None and isinstance(epoch, int) and isinstance(total_epochs, int) and total_epochs > 0:
            if isinstance(model_index, int) and isinstance(models_total, int) and models_total > 0:
                progress = ((model_index - 1) + (epoch / total_epochs)) / models_total
            else:
                progress = epoch / total_epochs
        if progress is not None:
            progress = min(1.0, max(0.0, progress))
            payload["progress"] = round(progress, 6)
            payload["progress_percent"] = round(progress * 100, 2)
            if progress > 0:
                elapsed = max(0.001, time.monotonic() - self.started_monotonic)
                eta = elapsed * (1.0 - progress) / progress
                payload["eta_seconds"] = int(max(0.0, eta))
        accuracy = _float(payload.get("val_accuracy") or payload.get("accuracy"))
        if accuracy is not None:
            if self.best_accuracy is None or accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
            payload["best_accuracy_seen"] = self.best_accuracy
        self.latest = payload
        return payload

    def should_publish(self, payload: dict[str, object], min_interval_seconds: float = 20.0) -> bool:
        signature = (
            payload.get("phase"),
            payload.get("model"),
            payload.get("model_index"),
            payload.get("epoch"),
            payload.get("epochs_total") or payload.get("total_epochs"),
        )
        now = time.monotonic()
        milestone_changed = signature != self.last_signature
        if milestone_changed or (now - self.last_publish_monotonic) >= min_interval_seconds:
            self.last_signature = signature
            self.last_publish_monotonic = now
            return True
        return False

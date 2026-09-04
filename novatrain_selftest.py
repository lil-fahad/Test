from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path


def main() -> int:
    started = time.time()
    # Small deterministic compute workload: enough to prove actual execution on the worker.
    checksum = 0
    for i in range(200_000):
        checksum = (checksum * 33 + i) % 1_000_000_007

    out = Path('.novatrain/output')
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        'ok': True,
        'request_id': os.environ.get('NOVATRAIN_REQUEST_ID'),
        'repository': os.environ.get('NOVATRAIN_REPOSITORY'),
        'commit_sha': os.environ.get('NOVATRAIN_COMMIT_SHA'),
        'device': os.environ.get('NOVATRAIN_DEVICE'),
        'gpu_count': os.environ.get('NOVATRAIN_GPU_COUNT'),
        'bf16': os.environ.get('NOVATRAIN_BF16'),
        'fp16': os.environ.get('NOVATRAIN_FP16'),
        'tf32': os.environ.get('NOVATRAIN_TF32'),
        'python': sys.version,
        'platform': platform.platform(),
        'checksum': checksum,
        'elapsed_seconds': round(time.time() - started, 6),
    }
    (out / 'selftest-result.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

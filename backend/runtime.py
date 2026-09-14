"""Identity captured when this service process imports its application code."""

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _identity():
    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for path in sorted((root / "backend").rglob("*.py")):
        if "tests" not in path.relative_to(root / "backend").parts:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                capture_output=True, text=True, timeout=3, check=True)
        commit = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"commit": commit, "backend_sha256": digest.hexdigest(),
            "loaded_at": datetime.now(timezone.utc).isoformat()}


RUNTIME_IDENTITY = _identity()

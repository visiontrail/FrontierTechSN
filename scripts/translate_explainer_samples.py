#!/usr/bin/env python3
"""Translate the known legacy sample labels without invalidating their media.

Dry run by default; pass --apply to back up and update the matching records.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile


ACTIVE = {"writing_script", "generating_audio", "generating_images", "generating_motion", "rendering"}
TITLES = {
    "exp-263d1cd695764133a89bdf7a": (
        "长篇图解 · 实际生成链路验证", "Explainer · End-to-end generation check"),
    "exp-f10de11e918f494d80c3f6c0": (
        "长篇图解 · 静态风格实测样片", "Explainer · Still illustration sample"),
    "exp-10db520fb0cc46ad8ef2a276": (
        "UI QA · 图解工作台交互验证", "UI QA · Explainer Studio interaction check"),
}
RESEARCH_SAMPLE = "exp-10db520fb0cc46ad8ef2a276"
LOG_SAMPLE = "exp-4180997e27eb4590ae854cce"


def translated_project(project: dict) -> dict:
    """Only replace exact historical labels in their known sample records."""
    result = deepcopy(project)
    project_id = project.get("id")
    titles = TITLES.get(project_id)
    if titles and project.get("title") == titles[0]:
        old, new = titles
        result["title"] = new
        if project_id == RESEARCH_SAMPLE and "research" in project:
            result["research"] = "".join(
                new + line[len(old):] if line.rstrip("\r\n") == old else line
                for line in project["research"].splitlines(keepends=True)
            )
    if project_id == LOG_SAMPLE and "logs" in project:
        result["logs"] = [line.replace("DeepSeek 深度求索", "DeepSeek") for line in project["logs"]]
    if result != project:
        if project.get("status") in ACTIVE:
            raise ValueError(f"{project_id}: generation is active; wait before translating")
        revision = project.get("revision")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise ValueError(f"{project_id}: invalid revision")
        result["revision"] = revision + 1
    return result


def _check_unchanged(path: Path, original: bytes) -> None:
    if path.read_bytes() != original:
        raise ValueError(f"{path.parent.name}: project changed concurrently; no overwrite")


def _replace_unchanged(path: Path, original: bytes, replacement: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".translation-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(replacement)
            handle.flush()
            os.fsync(handle.fileno())
        _check_unchanged(path, original)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def migrate(outputs_dir: Path, *, apply: bool = False) -> dict:
    pending = []
    for project_id in (*TITLES, LOG_SAMPLE):
        path = outputs_dir / "explainers" / project_id / "project.json"
        if not path.exists():
            continue
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError(f"{project_id}: symlinked project paths are unsupported")
        original = path.read_bytes()
        project = json.loads(original)
        if project.get("id") != project_id:
            raise ValueError(f"{project_id}: saved project ID does not match its directory")
        updated = translated_project(project)
        if updated != project:
            fields = [key for key in updated if updated[key] != project.get(key)]
            replacement = json.dumps(updated, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
            pending.append((path, original, replacement, fields))

    report = {"apply": apply, "changes": [{"id": path.parent.name, "fields": fields}
                                          for path, _, _, fields in pending], "backup_dir": None}
    if not apply or not pending:
        return report

    for path, original, _, _ in pending:
        _check_unchanged(path, original)
    backup_root = outputs_dir / "explainer-validation"
    if any(part.is_symlink() for part in (backup_root, *backup_root.parents)):
        raise ValueError("Symlinked backup paths are unsupported")
    backup_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    backup_dir = Path(tempfile.mkdtemp(prefix=f"english-sample-labels-{stamp}-", dir=backup_root))
    report["backup_dir"] = str(backup_dir)
    for path, original, _, _ in pending:
        (backup_dir / f"{path.parent.name}.json").write_bytes(original)
    for path, original, replacement, _ in pending:
        _replace_unchanged(path, original, replacement)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Back up and apply the exact sample translations")
    parser.add_argument("--outputs-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    arguments = parser.parse_args()
    try:
        report = migrate(arguments.outputs_dir, apply=arguments.apply)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"Translation stopped: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

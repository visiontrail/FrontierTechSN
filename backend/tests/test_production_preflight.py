import json
import subprocess

import pytest
from PIL import Image, ImageDraw

from backend.pipeline import assembler


def _inspection(**overrides):
    return {
        "schemaVersion": 1, "duration": 10, "samples": [1.0, 5.25],
        "ok": True, "errorCount": 0, "warningCount": 0, "infoCount": 0,
        "issues": [], **overrides,
    }


def test_inspect_uses_explicit_browser_timestamps_and_preserves_warnings(tmp_path, monkeypatch):
    warning = {"severity": "warning", "selector": "#caption", "time": 5.25,
               "code": "text_overflow", "message": "Small visual overflow"}
    payload = _inspection(issues=[warning], warningCount=1)
    commands = []

    def run(**kwargs):
        commands.append(kwargs["command"])
        return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

    monkeypatch.setattr(assembler, "run_capture_logged", run)
    path = tmp_path / "production_preflight.json"
    clean, findings = assembler.inspect_project(tmp_path, timestamps=[5.25, 1, 5.25], report_path=path)
    assert clean and findings == []
    assert commands[0][-3:] == ["--json", "--at", "1.000,5.250"]
    report = json.loads(path.read_text())
    assert report["inspection"]["issues"] == [warning]
    assert report["requested_timestamps"] == [1, 5.25]


@pytest.mark.parametrize("payload,code", [
    (_inspection(), 1),
    (_inspection(samples=[]), 0),
    (_inspection(samples=[1]), 0),
    (_inspection(duration=0), 0),
    (_inspection(schemaVersion=2), 0),
    (_inspection(errorCount=1), 0),
    (_inspection(issues=[{"severity": "error", "message": "Clipped text"}]), 0),
    (_inspection(issues="none"), 0),
    (_inspection(ok=False), 0),
    ("not json", 0),
])
def test_inspect_never_accepts_failed_or_incomplete_evidence(tmp_path, monkeypatch, payload, code):
    monkeypatch.setattr(assembler, "run_capture_logged", lambda **kwargs: subprocess.CompletedProcess(
        [], code, json.dumps(payload) if isinstance(payload, dict) else payload, "browser error",
    ))
    path = tmp_path / "production_preflight.json"
    clean, findings = assembler.inspect_project(tmp_path, timestamps=[1, 5.25], report_path=path)
    assert not clean and findings
    assert json.loads(path.read_text())["passed"] is False


def test_inspect_timeout_persists_a_failed_report(tmp_path, monkeypatch):
    def run(**kwargs):
        raise subprocess.TimeoutExpired(kwargs["command"], 900)

    monkeypatch.setattr(assembler, "run_capture_logged", run)
    path = tmp_path / "production_preflight.json"
    clean, findings = assembler.inspect_project(tmp_path, timestamps=[1], report_path=path)
    assert not clean and "timed out" in findings[0]
    assert json.loads(path.read_text())["passed"] is False


def _write_frame(path, *, blank=False):
    frame = Image.new("RGB", (160, 90), "white")
    if not blank:
        ImageDraw.Draw(frame).rectangle((30, 20, 120, 70), fill="navy")
    frame.save(path)


def test_snapshots_disable_cloud_vision_and_bind_fresh_pixels(tmp_path, monkeypatch):
    previous = tmp_path / "snapshots"
    previous.mkdir()
    (previous / "old.png").write_bytes(b"prior evidence")
    commands = []

    def run(**kwargs):
        commands.append(kwargs["command"])
        folder = tmp_path / "snapshots"
        folder.mkdir()
        _write_frame(folder / "frame-00-at-1.0s.png")
        _write_frame(folder / "frame-01-at-5.3s.png")
        return subprocess.CompletedProcess([], 0,
            "   snapshots/frame-00-at-1.0s.png\n   snapshots/frame-01-at-5.3s.png", "")

    monkeypatch.setattr(assembler, "run_capture_logged", run)
    path = tmp_path / "pixels.json"
    clean, findings = assembler.snapshot_project(tmp_path, timestamps=[1, 5.25], report_path=path)
    assert clean and not findings
    assert commands[0][-2:] == ["--describe", "false"]
    report = json.loads(path.read_text())
    assert (tmp_path / report["previous_snapshots"] / "old.png").read_bytes() == b"prior evidence"
    assert len(report["frames"]) == 2
    assert all(len(frame["sha256"]) == 64 and frame["width"] == 160 for frame in report["frames"])


@pytest.mark.parametrize("failure", ["blank", "missing", "exit", "runtime", "player", "corrupt", "timestamp"])
def test_snapshot_evidence_fails_closed(tmp_path, monkeypatch, failure):
    def run(**kwargs):
        folder = tmp_path / "snapshots"
        folder.mkdir()
        path = folder / "frame-00-at-1.0s.png"
        if failure != "missing":
            _write_frame(path, blank=failure == "blank")
        if failure == "corrupt":
            path.write_bytes(b"bad png")
        warning = {"runtime": "Runtime did not become render-ready",
                   "player": "No player API"}.get(failure, "")
        filename = "frame-00-at-5.0s.png" if failure == "timestamp" else path.name
        return subprocess.CompletedProcess([], 1 if failure == "exit" else 0,
            f"snapshots/{filename}", warning)

    monkeypatch.setattr(assembler, "run_capture_logged", run)
    path = tmp_path / "pixels.json"
    clean, findings = assembler.snapshot_project(tmp_path, timestamps=[1], report_path=path)
    assert not clean and findings
    assert json.loads(path.read_text())["passed"] is False

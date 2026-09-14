"""Durable task spans; wall time is a union, never a sum of parallel work."""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

_recorder = contextvars.ContextVar("pipeline_timing", default=None)
_parent = contextvars.ContextVar("pipeline_span", default=None)


def union_seconds(intervals):
    end = None
    total = 0.0
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


class Recorder:
    def __init__(self, directory):
        from backend.runtime import RUNTIME_IDENTITY
        self.runtime = RUNTIME_IDENTITY
        self.directory = Path(directory)
        self.path = self.directory / "logs" / "timing.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event):
        if event.get("parent") is None and event["event"] == "start":
            event = {**event, "runtime": self.runtime}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def report(self):
        starts, spans = {}, []
        for line in self.path.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue  # an interrupted trailing write cannot invent a duration
            if event["event"] == "start":
                starts[event["id"]] = event
            elif event["id"] in starts:
                start = starts.pop(event["id"])
                spans.append({**start, **event, "start": start["at"], "end": event["at"]})
        stages = {}
        for name in sorted({s["name"] for s in spans}):
            selected = [s for s in spans if s["name"] == name]
            stages[name] = {
                "wall_seconds": union_seconds((s["start"], s["end"]) for s in selected),
                "work_seconds": sum(s["seconds"] for s in selected),
                "attempts": len(selected),
                "failed_attempts": sum(s["status"] != "ok" for s in selected),
            }
        roots = [s for s in spans if s["parent"] is None]
        active = union_seconds((s["start"], s["end"]) for s in roots)
        elapsed = max((s["end"] for s in roots), default=0) - min((s["start"] for s in roots), default=0)
        # Partition the wall clock. A wait overlapping useful work is reported
        # separately; neither nested spans nor concurrent branches are added twice.
        points = sorted({p for s in spans for p in (s["start"], s["end"])})
        categories = {}
        for left, right in zip(points, points[1:]):
            live = [s for s in spans if s["start"] <= left and s["end"] >= right]
            parents = {s["parent"] for s in live}
            leaves = [s for s in live if s["id"] not in parents]
            kinds = {s["kind"] for s in leaves}
            if not kinds:
                kind = "pause_or_recovery_gap"
            elif len(kinds) == 1:
                kind = next(iter(kinds))
            else:
                kind = "parallel_overlap"
            categories[kind] = categories.get(kind, 0.0) + right - left
        report = {
            "schema_version": 1,
            "runtime": self.runtime,
            "wall_seconds": elapsed,
            "active_wall_seconds": active,
            "pause_or_recovery_gap_seconds": max(0.0, elapsed - active),
            "wall_partition_seconds": categories,
            "stages": stages,
            "unfinished_spans": list(starts.values()),
            "notes": ["Processing spans include uninstrumented local and external work; they are not CPU time.",
                      "Failed attempt work is already included in stage totals.",
                      "Unfinished spans after abrupt shutdown are retained as unknown, never counted as completed work."],
        }
        path = self.directory / "timing_report.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(path)
        return report


@contextmanager
def span(name, kind="processing"):
    recorder = _recorder.get()
    if recorder is None:
        yield
        return
    identifier = uuid4().hex
    parent = _parent.get()
    started = time.monotonic()
    recorder.write({"event": "start", "id": identifier, "parent": parent,
                    "name": name, "kind": kind, "at": time.time()})
    token = _parent.set(identifier)
    status = "ok"
    try:
        yield
    except BaseException as exc:
        status = type(exc).__name__
        raise
    finally:
        recorder.write({"event": "end", "id": identifier, "parent": parent,
                        "name": name, "kind": kind, "at": time.time(),
                        "seconds": time.monotonic() - started, "status": status})
        _parent.reset(token)


def timed(name, kind="processing", *, task_entry=False):
    def decorate(function):
        @functools.wraps(function)
        async def wrapped(*args, **kwargs):
            token = None
            if task_entry and _recorder.get() is None:
                from backend import config
                task = inspect.signature(function).bind(*args, **kwargs).arguments["task"]
                directory = task.output_dir or config.OUTPUTS_DIR / task.id
                token = _recorder.set(Recorder(directory))
            try:
                with span(name, kind):
                    return await function(*args, **kwargs)
            finally:
                if token is not None:
                    try:
                        _recorder.get().report()
                    finally:
                        _recorder.reset(token)
        return wrapped
    return decorate

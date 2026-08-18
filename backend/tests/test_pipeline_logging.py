from datetime import datetime, timezone

from backend.pipeline.orchestrator import format_pipeline_log


def test_format_pipeline_log_uses_event_timestamp():
    timestamp = datetime(2026, 8, 12, 15, 37, 58, tzinfo=timezone.utc)

    line = format_pipeline_log("20260812-153758-75ce33", "Stage 1: Extracting", timestamp)

    assert line == (
        "[20260812-15:37:58] [20260812-153758-75ce33] Stage 1: Extracting"
    )


def test_format_pipeline_log_retains_task_id():
    timestamp = datetime(2026, 8, 12, 15, 37, 58, tzinfo=timezone.utc)

    line = format_pipeline_log("20260812-153758-75ce33", "Processing task", timestamp)

    assert "[20260812-153758-75ce33]" in line

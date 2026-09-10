import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.models import SourceType, TaskConfig, TaskResponse, TaskStatus
from backend.routers import tasks as tasks_router


@pytest.mark.parametrize("shot_count", [8, 16])
def test_footage_retry_accepts_complete_long_edition_plan(tmp_path, shot_count):
    script = tmp_path / "script.txt"
    script.write_text("Existing approved narration.", encoding="utf-8")
    task = TaskResponse(
        id="edition-retry",
        created_at="2026-09-10T00:00:00+00:00",
        updated_at="2026-09-10T00:00:00+00:00",
        source_type=SourceType.NEWS_DAILY,
        status=TaskStatus.FAILED,
        config=TaskConfig(),
        output_dir=str(tmp_path),
        script_path=str(script),
    )
    queries = [f"Story {index} actual product demonstration" for index in range(shot_count)]
    app = FastAPI()
    app.include_router(tasks_router.router)
    with (
        patch.object(tasks_router.db, "get_task", AsyncMock(return_value=task)),
        patch.object(tasks_router, "_queue_worker_mode", AsyncMock()) as queue,
        TestClient(app) as client,
    ):
        response = client.post("/api/tasks/edition-retry/footage/acquire", json={"queries": queries})

    assert response.status_code == 200, response.text
    queue.assert_awaited_once()
    payload = json.loads(queue.call_args.kwargs["marker_payload"])
    assert payload == {"resume_status": "failed", "queries": queries}
    assert queue.call_args.kwargs["claim_status"] == TaskStatus.SOURCING

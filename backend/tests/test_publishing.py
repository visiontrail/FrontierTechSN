import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from backend.pipeline.opencli import OpenCLIError
from backend.publishing import _upload_local_media


def test_upload_permission_failure_is_actionable_without_retry():
    browser = AsyncMock(side_effect=OpenCLIError(
        'OpenCLI browser failed: {"code":-32000,"message":"Not allowed"}'
    ))

    with patch("backend.publishing._browser", browser):
        with pytest.raises(OpenCLIError, match="Allow access to file URLs"):
            asyncio.run(
                _upload_local_media(
                    "test-session",
                    "input[name=Filedata]",
                    "/tmp/video.mp4",
                    timeout=300,
                )
            )

    assert browser.await_count == 1
    assert browser.await_args.kwargs["attempts"] == 1


def test_upload_transient_failure_uses_remaining_retry_budget():
    browser = AsyncMock(side_effect=[
        OpenCLIError("temporary browser lease failure"),
        "uploaded",
    ])

    with patch("backend.publishing._browser", browser):
        result = asyncio.run(
            _upload_local_media(
                "test-session",
                "input[name=Filedata]",
                "/tmp/video.mp4",
                timeout=300,
            )
        )

    assert result == "uploaded"
    assert browser.await_count == 2
    assert browser.await_args_list[0].kwargs["attempts"] == 1
    assert browser.await_args_list[1].kwargs["attempts"] == 9

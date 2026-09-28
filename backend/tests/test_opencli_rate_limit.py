import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from backend.pipeline import opencli
from backend.pipeline import opencli_rate_limit as pacing


@pytest.fixture
def image_checkpoint(tmp_path):
    path = tmp_path / "image-request.json"
    path.write_text(json.dumps({
        "stage": "image_wait_failed",
        "prompt": "Generate an image of: an orbital data center",
        "conversationUrl": "https://chatgpt.com/c/owned-request-123",
        "beforeUrls": [],
    }), encoding="utf-8")
    return path


@pytest.mark.parametrize("joined", [False, True])
@pytest.mark.parametrize("stage", ["image_wait_failed", "recovery_requested"])
def test_saved_chatgpt_image_recovery_skips_generation_slot(image_checkpoint, joined, stage):
    prior = json.loads(image_checkpoint.read_text(encoding="utf-8"))
    prior["stage"] = stage
    image_checkpoint.write_text(json.dumps(prior), encoding="utf-8")
    option = (
        [f"--state-file={image_checkpoint}"] if joined
        else ["--state-file", str(image_checkpoint)]
    )
    assert not pacing.is_rate_limited_command([
        "chatgpt", "image", "an orbital data center", *option,
    ])
    # A checkpoint is not an exemption for other generation commands.
    assert pacing.is_rate_limited_command(["chatgpt", "ask", "prompt", *option])
    assert pacing.is_rate_limited_command(["gemini", "image", "prompt", *option])


@pytest.mark.parametrize("contents", [
    "", "{invalid", "null", "false", "0", "[]", "{}",
    json.dumps({"stage": "waiting"}),
    json.dumps({"stage": [], "prompt": "owned", "conversationUrl": "https://chatgpt.com/c/owned-request-123"}),
])
def test_invalid_image_state_still_reserves_generation_slot(tmp_path, contents):
    checkpoint = tmp_path / "image-request.json"
    checkpoint.write_text(contents, encoding="utf-8")
    assert pacing.is_rate_limited_command([
        "chatgpt", "image", "prompt", "--state-file", str(checkpoint),
    ])


@pytest.mark.parametrize(("field", "value"), [
    ("stage", "preparing"),
    ("stage", "submitting"),
    ("stage", "unknown"),
    ("prompt", ""),
    ("prompt", "   "),
    ("prompt", None),
    ("conversationUrl", ""),
    ("conversationUrl", None),
    ("conversationUrl", "https://chatgpt.com/"),
    ("conversationUrl", "https://chatgpt.com/c/short"),
    ("conversationUrl", "https://chatgpt.com/c/owned-request-123/other"),
    ("conversationUrl", "https://example.com/c/owned-request-123"),
])
def test_incomplete_image_state_still_reserves_generation_slot(image_checkpoint, field, value):
    prior = json.loads(image_checkpoint.read_text(encoding="utf-8"))
    prior[field] = value
    image_checkpoint.write_text(json.dumps(prior), encoding="utf-8")
    assert pacing.is_rate_limited_command([
        "chatgpt", "image", "prompt", "--state-file", str(image_checkpoint),
    ])


def test_missing_ambiguous_or_symlink_image_state_keeps_generation_pacing(tmp_path, image_checkpoint):
    symlink = tmp_path / "linked.json"
    symlink.symlink_to(image_checkpoint)
    for option in (
        [],
        ["--state-file"],
        ["--state-file="],
        ["--state-file", str(tmp_path / "missing.json")],
        ["--state-file", str(tmp_path)],
        ["--state-file", str(symlink)],
        ["--state-file", str(image_checkpoint), "--state-file=missing.json"],
        ["--", "--state-file", str(image_checkpoint)],
    ):
        assert pacing.is_rate_limited_command(["chatgpt", "image", "prompt", *option])


def test_wrapper_image_recovery_still_obeys_provider_access_cooldown(image_checkpoint):
    with (
        patch.object(pacing, "wait_for_opencli_cooldown") as cooldown,
        patch.object(pacing, "wait_for_opencli_web_slot") as reserve,
        patch.object(pacing, "wait_for_opencli_generation_quiet_period") as quiet,
    ):
        assert pacing.main([
            "chatgpt", "image", "an orbital data center",
            "--state-file", str(image_checkpoint),
        ]) == 0
    cooldown.assert_called_once_with("chatgpt")
    reserve.assert_not_called()
    quiet.assert_not_called()


def test_backend_image_recovery_waits_for_access_cooldown_only(image_checkpoint):
    process = AsyncMock(returncode=0)
    process.communicate.return_value = (b"[]", b"")
    with (
        patch.object(opencli, "runtime_mode", return_value="bridge"),
        patch.object(opencli, "_environment", return_value={}),
        patch.object(opencli, "opencli_cooldown_remaining", side_effect=[30, 0, 0]) as remaining,
        patch.object(opencli, "wait_for_opencli_web_slot") as reserve,
        patch.object(opencli, "wait_for_opencli_generation_quiet_period") as quiet,
        patch.object(opencli.asyncio, "sleep", AsyncMock()) as sleep,
        patch.object(opencli.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)) as launch,
    ):
        asyncio.run(opencli.run_opencli([
            "chatgpt", "image", "an orbital data center",
            "--state-file", str(image_checkpoint),
        ], timeout=1))
    sleep.assert_awaited_once_with(30.0)
    assert remaining.call_count == 3
    reserve.assert_not_called()
    quiet.assert_not_called()
    launch.assert_awaited_once()
    assert "OPENCLI_WEB_REQUEST_SLOT_RESERVED" not in launch.await_args.kwargs["env"]

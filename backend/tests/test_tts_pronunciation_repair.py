"""Keep failed model decisions available without weakening plan acceptance."""

import json
from unittest.mock import AsyncMock

import pytest

from backend.pipeline import digester, tts_pronunciation as pronunciation


def decision(spoken):
    return {"id": 0, "source": "H", "action": "spell", "spoken": spoken,
            "reason": "Letter in the H3 model identifier."}


def save_attempt(directory, text, response, **changes):
    record = {
        "request": {"source_text": text, "candidates": pronunciation.candidates(text)},
        "response": response, "validation_error": "An old, nonspecific error",
    }
    record.update(changes)
    path = directory / "tts_pronunciation_attempt_previous.json"
    path.write_text(json.dumps(record))
    return path


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(digester, "_resolve_provider", AsyncMock(
        return_value=("https://example.invalid/v1", "model", "test-secret"),
    ))
    chat = AsyncMock(return_value=json.dumps({"decisions": [decision("aitch")]}))
    monkeypatch.setattr(digester, "_chat", chat)
    return chat


@pytest.mark.asyncio
async def test_failed_run_reuses_response_as_feedback_and_revalidates_error(tmp_path, model):
    text = "The H3 model launched."
    raw = json.dumps({"decisions": [decision("H")]})
    path = save_attempt(tmp_path, text, raw)
    before = path.read_bytes()
    result = await pronunciation.prepare_pronunciation(
        text, tmp_path, policy=pronunciation.policy_contract(), emit=lambda _: None,
    )
    payload = json.loads(model.await_args.args[1])
    assert payload["previous_response"] == raw
    assert "Candidate 0 (H), spoken 'H'" in payload["validation_error"]
    assert "must differ from source" in payload["validation_error"]
    assert result["synthesis_text"] == "The aitch 3 model launched."
    assert path.read_bytes() == before
    assert model.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["source", "candidates", "corrupt", "valid", "schema"])
async def test_unrelated_or_unusable_attempts_do_not_seed_repair(tmp_path, model, defect):
    text = "The H3 model launched."
    raw = json.dumps({"decisions": [decision("aitch" if defect == "valid" else "H")]})
    path = save_attempt(tmp_path, text, raw)
    data = json.loads(path.read_text())
    if defect == "source":
        data["request"]["source_text"] = "Another H3 report."
    elif defect == "candidates":
        data["request"]["candidates"] = []
    elif defect == "schema":
        data = []
    path.write_text("{" if defect == "corrupt" else json.dumps(data))
    await pronunciation.prepare_pronunciation(
        text, tmp_path, policy=pronunciation.policy_contract(), emit=lambda _: None,
    )
    assert "previous_response" not in json.loads(model.await_args.args[1])


@pytest.mark.asyncio
async def test_each_repair_gets_previous_response_and_still_fails_closed(tmp_path, model):
    raw = json.dumps({"decisions": [decision("H")]})
    model.return_value = raw
    with pytest.raises(RuntimeError, match="explicit pronunciation hint"):
        await pronunciation.prepare_pronunciation(
            "The H3 model launched.", tmp_path,
            policy=pronunciation.policy_contract(), emit=lambda _: None,
        )
    assert model.await_count == pronunciation.MAX_PLAN_ATTEMPTS
    assert "previous_response" not in json.loads(model.await_args_list[0].args[1])
    assert json.loads(model.await_args_list[1].args[1])["previous_response"] == raw
    assert not (tmp_path / "tts_pronunciation.json").exists()


@pytest.mark.asyncio
async def test_submitted_job_blocks_feedback_replanning(tmp_path, model):
    text = "The H3 model launched."
    save_attempt(tmp_path, text, json.dumps({"decisions": [decision("H")]}))
    checkpoint = tmp_path / "tts_input_part_001.request-original.json"
    checkpoint.write_text('{"job_id":"original"}')
    with pytest.raises(RuntimeError, match="preserving their request identities"):
        await pronunciation.prepare_pronunciation(
            text, tmp_path, policy=pronunciation.policy_contract(), emit=lambda _: None,
        )
    model.assert_not_awaited()
    assert checkpoint.read_text() == '{"job_id":"original"}'

"""Model-boundary and source-integrity checks for contextual pronunciation."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from backend import config
from backend.pipeline import digester, tts_pronunciation as pronunciation


def decisions_for(text: str, spellings: dict[str, str] | None = None) -> list[dict]:
    spellings = spellings or {}
    return [
        {
            "id": candidate["id"],
            "source": candidate["source"],
            "action": "spell" if candidate["source"] in spellings else "keep",
            "spoken": spellings.get(candidate["source"], candidate["source"]),
            "reason": "Contextual reading selected by the mocked model.",
        }
        for candidate in pronunciation.candidates(text)
    ]


@pytest.fixture
def mock_model(monkeypatch):
    resolve = AsyncMock(return_value=("https://primary.invalid/v1", "primary-model", "test-secret"))
    chat = AsyncMock()
    monkeypatch.setattr(digester, "_resolve_provider", resolve)
    monkeypatch.setattr(digester, "_chat", chat)
    return resolve, chat


def test_candidate_occurrences_include_unlisted_names_and_embedded_initialisms():
    text = "QZV interviewed OpenAI about GPUs, GPT-4, U.S. labs and QZV's results."
    found = pronunciation.candidates(text)
    assert [row["source"] for row in found] == ["QZV", "AI", "GPU", "GPT", "U.S.", "QZV"]
    assert [row["id"] for row in found] == list(range(len(found)))
    assert all(text[row["start"]:row["end"]] == row["source"] for row in found)


def test_dotted_candidates_accept_optional_final_period_with_exact_spans():
    text = "U.S, U.S.; U.S.A! (U.S.A.) US USA CNBC. NASA?"
    found = pronunciation.candidates(text)
    assert [row["source"] for row in found] == [
        "U.S", "U.S.", "U.S.A", "U.S.A.", "US", "USA", "CNBC", "NASA",
    ]
    assert [row["id"] for row in found] == list(range(len(found)))
    assert all(text[row["start"]:row["end"]] == row["source"] for row in found)
    assert [text[row["end"]:row["end"] + 1] for row in found] == [
        ",", ";", "!", ")", " ", " ", ".", "?",
    ]


def test_dotted_candidate_rendering_preserves_surrounding_source_punctuation():
    text = 'The "U.S", (U.S.A), and CNBC: all replied!'
    _, edits = pronunciation.validate_decisions(
        text, decisions_for(text, {"U.S": "you ess", "U.S.A": "you ess ay", "CNBC": "C-N-B-C"}),
    )
    assert pronunciation.render_span(text, 0, len(text), edits) == (
        'The "you ess", (you ess eigh), and C-N-B-C: all replied!'
    )


def test_urls_and_email_addresses_are_not_pronunciation_candidates():
    text = (
        "https://CNBC.example/AI?q=GPU#GPT WWW.example.com/TSMC "
        "CNBC@example.com and person+AI@GPU.example. CNBC.com CNBC reports."
    )
    assert [row["source"] for row in pronunciation.candidates(text)] == ["CNBC"]


def test_dotted_initialisms_inside_urls_emails_and_domains_remain_protected():
    text = (
        "https://example.com/U.S.A?q=U.S www.example.com/U.S. "
        "U.S.example U.S.A.example.org/U.S reader@U.S.example "
        "U.S and U.S.A remain candidates."
    )
    assert [row["source"] for row in pronunciation.candidates(text)] == ["U.S", "U.S.A"]


def test_decisions_are_contextual_per_occurrence_and_leave_canonical_source_intact():
    text = "US officials spoke. The sign said HELP US. NASA listened."
    decisions = decisions_for(text)
    decisions[0].update(action="spell", spoken="you ess")
    validated, edits = pronunciation.validate_decisions(text, list(reversed(decisions)))
    assert [row["id"] for row in validated] == list(range(len(decisions)))
    assert len(edits) == 1
    assert pronunciation.render_span(text, 0, len(text), edits) == (
        "you ess officials spoke. The sign said HELP US. NASA listened."
    )
    assert text == "US officials spoke. The sign said HELP US. NASA listened."


@pytest.mark.parametrize("source, spoken", [
    ("QZV", "cue zee vee"),
    ("CNBC", "C-N-B-C"),
    ("CNBC", "c-n-b-c"),
    ("API", "eigh pee eye"),
    ("API", "a.p.i"),
    ("GPU", "gee pee you"),
    ("U.S", "you ess"),
    ("U.S.", "you ess"),
    ("U.S.A", "you ess eigh"),
    ("U.S.A.", "you ess eigh"),
    ("IEEE", "I triple E"),
    ("BBC", "double bee see"),
    ("WWW", "W-W-W"),
    ("WSJ", "double you ess jay"),
])
def test_generic_alphabet_accepts_complete_letter_readings(source, spoken):
    text = f"The {source} team reported the result."
    validated, edits = pronunciation.validate_decisions(text, decisions_for(text, {source: spoken}))
    assert validated[0]["spoken"] == spoken
    assert pronunciation.render_span(text, 0, len(text), edits) == f"The {spoken} team reported the result."


@pytest.mark.parametrize("spoken", [
    "see en bee", "see bee en see", "see en bee see eye",
    "Consumer News Business Channel", "<say-as>CNBC</say-as>",
    "siː ɛn biː siː", "CNBC", " C-N-B-C", "C-N-B-C ", "",
])
def test_spelling_rejects_changed_letters_and_unbounded_rewrites(spoken):
    text = "CNBC reported $42 billion in 2026."
    with pytest.raises(ValueError):
        pronunciation.validate_decisions(text, decisions_for(text, {"CNBC": spoken}))


@pytest.mark.parametrize("defect", [
    "missing", "extra", "duplicate", "unknown_id", "boolean_id", "wrong_source",
    "keep_changed", "bad_action", "missing_reason", "not_list", "not_object",
])
def test_decision_validation_requires_exact_complete_candidate_coverage(defect):
    text = "QZV spoke with NASA."
    rows = decisions_for(text, {"QZV": "cue zee vee"})
    if defect == "missing":
        rows.pop()
    elif defect == "extra":
        rows.append(deepcopy(rows[0]))
    elif defect == "duplicate":
        rows[1]["id"] = rows[0]["id"]
    elif defect == "unknown_id":
        rows[1]["id"] = 42
    elif defect == "boolean_id":
        rows[0]["id"] = False
    elif defect == "wrong_source":
        rows[0]["source"] = "XYZ"
    elif defect == "keep_changed":
        rows[1]["spoken"] = "National Aeronautics and Space Administration"
    elif defect == "bad_action":
        rows[0]["action"] = "rewrite"
    elif defect == "missing_reason":
        rows[0]["reason"] = "   "
    elif defect == "not_list":
        rows = {"decisions": rows}
    elif defect == "not_object":
        rows[0] = "QZV"
    with pytest.raises(ValueError):
        pronunciation.validate_decisions(text, rows)


def test_rendering_preserves_digits_suffixes_punctuation_and_protected_text():
    text = "QZV's GPUs cost $42.50; GPT-4 rose 12% in 2026. https://QZV.example/GPU"
    rows = decisions_for(text, {"QZV": "cue zee vee", "GPU": "gee pee you", "GPT": "gee pee tee"})
    _, edits = pronunciation.validate_decisions(text, rows)
    assert pronunciation.render_span(text, 0, len(text), edits) == (
        "cue zee vee's gee pee yous cost $42.50; gee pee tee-4 rose 12% in 2026. "
        "https://QZV.example/GPU"
    )


def test_embedded_letter_runs_get_readable_synthesis_boundaries():
    text = "OpenAI ships GPT4 tools for GPUs and AI2 research."
    rows = decisions_for(text, {"AI": "ay eye", "GPT": "gee pee tee", "GPU": "gee pee you"})
    _, edits = pronunciation.validate_decisions(text, rows)
    assert pronunciation.render_span(text, 0, len(text), edits) == (
        "Open eigh eye ships gee pee tee 4 tools for gee pee yous and eigh eye 2 research."
    )


def test_ambiguous_a_letter_name_is_canonicalized_without_adding_letters():
    text = "OpenAI uses AI and API tools."
    rows = decisions_for(text, {"AI": "ay eye", "API": "ay pee eye"})
    validated, edits = pronunciation.validate_decisions(text, rows)
    assert [row["spoken"] for row in validated] == ["eigh eye", "eigh eye", "eigh pee eye"]
    assert pronunciation.render_span(text, 0, len(text), edits) == (
        "Open eigh eye uses eigh eye and eigh pee eye tools."
    )
    assert pronunciation.validate_decisions(text, validated) == (validated, edits)


def test_render_span_rejects_splitting_inside_an_edit():
    text = "The QZV result."
    _, edits = pronunciation.validate_decisions(text, decisions_for(text, {"QZV": "cue zee vee"}))
    with pytest.raises(ValueError, match="cannot be split"):
        pronunciation.render_span(text, 0, text.index("QZV") + 1, edits)
    with pytest.raises(ValueError, match="cannot be split"):
        pronunciation.render_span(text, text.index("QZV") + 1, len(text), edits)


def test_normalization_retains_physical_line_boundaries():
    assert pronunciation.normalize_text("  First  line.\n\n Second\tline.  ") == "First line.\n\nSecond line."


@pytest.mark.asyncio
async def test_ai_sees_full_context_and_saved_plan_reuses_without_model_access(tmp_path, mock_model):
    resolve, chat = mock_model
    text = "QZV is the new lab's letter-based identifier. NASA collaborated."
    rows = decisions_for(text, {"QZV": "cue zee vee"})

    async def answer(_prompt, content, **kwargs):
        payload = json.loads(content)
        assert payload["source_text"] == text
        assert payload["candidates"] == pronunciation.candidates(text)
        assert kwargs["enable_skills"] is False
        assert kwargs["disable_thinking"] is True
        kwargs["route_selected"]("https://backup.invalid/v1", "backup-model", "backup-secret")
        return json.dumps({"decisions": rows})

    chat.side_effect = answer
    policy = pronunciation.policy_contract()
    first = await pronunciation.prepare_pronunciation(text, tmp_path, policy=policy, emit=lambda _: None)
    assert first["synthesis_text"] == "cue zee vee is the new lab's letter-based identifier. NASA collaborated."
    assert first["route"] == {"endpoint": "https://backup.invalid/v1", "model": "backup-model"}
    assert first["requested_route"] == {"endpoint": "https://primary.invalid/v1", "model": "primary-model"}
    assert first["identity"]["source_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert first["synthesis_text_sha256"] == hashlib.sha256(first["synthesis_text"].encode()).hexdigest()
    assert json.loads((tmp_path / "tts_pronunciation.json").read_text()) == first
    assert (tmp_path / "tts_synthesis.txt").read_text() == first["synthesis_text"]
    evidence = "\n".join(path.read_text() for path in tmp_path.glob("*.json"))
    assert "test-secret" not in evidence and "backup-secret" not in evidence
    resolve.side_effect = AssertionError("An accepted plan must not resolve the model again")
    chat.side_effect = AssertionError("An accepted plan must not rerun the model")
    second = await pronunciation.prepare_pronunciation(text, tmp_path, policy=policy, emit=lambda _: None)
    assert second == first
    assert chat.await_count == resolve.await_count == 1


@pytest.mark.asyncio
async def test_no_candidates_persists_identity_without_network(tmp_path, mock_model):
    resolve, chat = mock_model
    text = "The spacecraft reached orbit with six payloads."
    result = await pronunciation.prepare_pronunciation(
        text, tmp_path, policy=pronunciation.policy_contract(), emit=lambda _: None,
    )
    assert result["decisions"] == result["edits"] == []
    assert result["synthesis_text"] == text
    assert (tmp_path / "tts_pronunciation.json").is_file()
    resolve.assert_not_awaited()
    chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_ai_output_gets_one_bounded_repair_with_feedback(tmp_path, mock_model):
    _, chat = mock_model
    text = "QZV announced its result."
    chat.side_effect = [
        json.dumps({"decisions": []}),
        "```json\n" + json.dumps({"decisions": decisions_for(text, {"QZV": "cue zee vee"})}) + "\n```",
    ]
    result = await pronunciation.prepare_pronunciation(
        text, tmp_path, policy=pronunciation.policy_contract(), emit=lambda _: None,
    )
    assert result["edits"][0]["spoken"] == "cue zee vee"
    assert chat.await_count == 2
    first_payload = json.loads(chat.await_args_list[0].args[1])
    repaired_payload = json.loads(chat.await_args_list[1].args[1])
    assert "validation_error" not in first_payload
    assert repaired_payload["validation_error"]
    assert first_payload["source_text"] == repaired_payload["source_text"] == text
    attempts = [json.loads(path.read_text()) for path in tmp_path.glob("tts_pronunciation_attempt_*.json")]
    assert len(attempts) == 2
    assert sum("validation_error" in row for row in attempts) == 1


@pytest.mark.asyncio
async def test_exhausted_invalid_plans_cannot_publish_a_ready_plan(tmp_path, mock_model):
    _, chat = mock_model
    chat.return_value = json.dumps({"decisions": []})
    with pytest.raises(RuntimeError, match="failed validation"):
        await pronunciation.prepare_pronunciation(
            "QZV announced results.", tmp_path, policy=pronunciation.policy_contract(), emit=lambda _: None,
        )
    assert chat.await_count == pronunciation.MAX_PLAN_ATTEMPTS
    assert not (tmp_path / "tts_pronunciation.json").exists()
    assert not (tmp_path / "tts_synthesis.txt").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["missing", "corrupt", "tampered", "source_changed", "policy_changed"])
async def test_submitted_job_checkpoint_blocks_replanning_and_is_preserved(tmp_path, mock_model, defect):
    resolve, chat = mock_model
    text = "QZV announced results."
    chat.return_value = json.dumps({"decisions": decisions_for(text, {"QZV": "cue zee vee"})})
    policy = pronunciation.policy_contract()
    await pronunciation.prepare_pronunciation(text, tmp_path, policy=policy, emit=lambda _: None)
    plan_path = tmp_path / "tts_pronunciation.json"
    checkpoint_path = tmp_path / "tts_input_part_001.request-0.json"
    checkpoint_path.write_text('{"identity":"existing","idempotency_key":"keep-this","job_id":"job-1"}')
    original_checkpoint = checkpoint_path.read_bytes()
    if defect == "missing":
        plan_path.unlink()
    elif defect == "corrupt":
        plan_path.write_text("broken json")
    elif defect == "tampered":
        saved = json.loads(plan_path.read_text())
        saved["synthesis_text"] = "Different source content."
        plan_path.write_text(json.dumps(saved))
    elif defect == "source_changed":
        text += " A new statement."
    elif defect == "policy_changed":
        policy = {**policy, "version": policy["version"] + 1}
    preserved_plan = plan_path.read_bytes() if plan_path.exists() else None
    resolve.reset_mock()
    chat.reset_mock()
    with pytest.raises(RuntimeError, match="preserving"):
        await pronunciation.prepare_pronunciation(text, tmp_path, policy=policy, emit=lambda _: None)
    assert checkpoint_path.read_bytes() == original_checkpoint
    assert (plan_path.read_bytes() if plan_path.exists() else None) == preserved_plan
    resolve.assert_not_awaited()
    chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_source_change_before_any_jobs_can_create_a_new_plan(tmp_path, mock_model):
    _, chat = mock_model
    first_text, next_text = "QZV announced results.", "QZV announced new results."
    chat.side_effect = [
        json.dumps({"decisions": decisions_for(text, {"QZV": "cue zee vee"})})
        for text in (first_text, next_text)
    ]
    policy = pronunciation.policy_contract()
    first = await pronunciation.prepare_pronunciation(first_text, tmp_path, policy=policy, emit=lambda _: None)
    second = await pronunciation.prepare_pronunciation(next_text, tmp_path, policy=policy, emit=lambda _: None)
    assert first["identity"] != second["identity"]
    assert first["plan_sha256"] != second["plan_sha256"]
    assert second["synthesis_text"] == "cue zee vee announced new results."
    assert chat.await_count == 2


def test_policy_contract_changes_when_prompt_changes(tmp_path, monkeypatch):
    original_prompt = (config.PROMPTS_DIR / "tts_pronunciation.txt").read_text()
    monkeypatch.setattr(config, "PROMPTS_DIR", tmp_path)
    prompt_path = tmp_path / "tts_pronunciation.txt"
    prompt_path.write_text(original_prompt)
    original = pronunciation.policy_contract()
    prompt_path.write_text(original_prompt + "\nAdditional pronunciation instruction.\n")
    assert pronunciation.policy_contract()["prompt_sha256"] != original["prompt_sha256"]


@pytest.mark.asyncio
async def test_prompt_changed_after_policy_snapshot_blocks_model_call(tmp_path, monkeypatch, mock_model):
    resolve, chat = mock_model
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    original_prompt = (config.PROMPTS_DIR / "tts_pronunciation.txt").read_text()
    (prompt_dir / "tts_pronunciation.txt").write_text(original_prompt)
    monkeypatch.setattr(config, "PROMPTS_DIR", prompt_dir)
    policy = pronunciation.policy_contract()
    (prompt_dir / "tts_pronunciation.txt").write_text(original_prompt + "\nChanged instruction.\n")
    with pytest.raises(RuntimeError, match="prompt changed"):
        await pronunciation.prepare_pronunciation("QZV reported results.", tmp_path / "audio", policy=policy, emit=lambda _: None)
    resolve.assert_not_awaited()
    chat.assert_not_awaited()

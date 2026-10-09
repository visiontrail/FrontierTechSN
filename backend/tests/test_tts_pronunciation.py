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
async def test_cited_word_references_are_source_matched_and_older_plans_do_not_gain_them(tmp_path, mock_model):
    _, chat = mock_model
    source = 'The GUI displays the application window.'
    chat.return_value = json.dumps({'decisions': [
        {'id': 0, 'source': 'GUI', 'action': 'word', 'spoken': 'gui,',
         'phones': ['G', 'UW', 'IY'], 'reason': 'Use the cited conventional word pronunciation.'},
    ]})
    plan = await pronunciation.prepare_pronunciation(
        source, tmp_path / 'current', policy=pronunciation.policy_contract(), emit=lambda _: None)
    payload = json.loads(chat.call_args.args[1])
    references = payload['cited_word_phone_references']
    assert len(references) == 1
    assert references[0]['source'] == 'GUI' and references[0]['id'] == 0
    assert references[0]['variants'] == [['G', 'UW', 'IY']]
    assert references[0]['source_url'] == 'https://www.oxfordlearnersdictionaries.com/definition/english/gui'
    assert plan['synthesis_text'] == 'The gui, displays the application window.'
    chat.reset_mock()
    chat.return_value = json.dumps({'decisions': [
        {'id': 0, 'source': 'GUI', 'action': 'keep', 'spoken': 'GUI',
         'reason': 'Preserve the frozen previous policy.'},
    ]})
    await pronunciation.prepare_pronunciation(
        source, tmp_path / 'old', policy=pronunciation.policy_contract(policy_version=12), emit=lambda _: None)
    assert 'cited_word_phone_references' not in json.loads(chat.call_args.args[1])


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
    original_prompt = pronunciation.prompt_path().read_text()
    monkeypatch.setattr(config, "PROMPTS_DIR", tmp_path)
    prompt_path = tmp_path / pronunciation.prompt_path().name
    prompt_path.write_text(original_prompt)
    original = pronunciation.policy_contract()
    prompt_path.write_text(original_prompt + "\nAdditional pronunciation instruction.\n")
    assert pronunciation.policy_contract()["prompt_sha256"] != original["prompt_sha256"]


@pytest.mark.asyncio
async def test_prompt_changed_after_policy_snapshot_blocks_model_call(tmp_path, monkeypatch, mock_model):
    resolve, chat = mock_model
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    original_prompt = pronunciation.prompt_path().read_text()
    (prompt_dir / pronunciation.prompt_path().name).write_text(original_prompt)
    monkeypatch.setattr(config, "PROMPTS_DIR", prompt_dir)
    policy = pronunciation.policy_contract()
    (prompt_dir / pronunciation.prompt_path().name).write_text(original_prompt + "\nChanged instruction.\n")
    with pytest.raises(RuntimeError, match="prompt changed"):
        await pronunciation.prepare_pronunciation("QZV reported results.", tmp_path / "audio", policy=policy, emit=lambda _: None)
    resolve.assert_not_awaited()
    chat.assert_not_awaited()


def test_numeric_models_include_every_isolated_capital_without_editing_digits():
    text='A2A and B2B use C3I, 2D and 4G beside GPT4o.'
    found=pronunciation.candidates(text)
    assert [r['source'] for r in found]==['A','A','B','B','C','I','D','G','GPT']
    assert len({(r['start'],r['end']) for r in found})==len(found)
    rows=decisions_for(text,{'A':'eigh','B':'bee','C':'see','I':'eye','D':'dee','G':'gee','GPT':'gee pee tee'})
    _,edits=pronunciation.validate_decisions(text,rows)
    rendered=pronunciation.render_span(text,0,len(text),edits)
    assert rendered=='eigh 2 eigh and bee 2 bee use see 3 eye, 2 dee and 4 gee beside gee pee tee 4o.'
    assert text=='A2A and B2B use C3I, 2D and 4G beside GPT4o.'


def test_numeric_candidates_remain_excluded_from_urls_domains_and_email_addresses():
    text='https://A2A.example/A2A person+A2A@B2B.example A2A.example A2A'
    assert [r['source'] for r in pronunciation.candidates(text)]==['A','A']


def test_legacy_policy_retains_original_candidate_set_and_hash():
    current=pronunciation.policy_contract()
    legacy=pronunciation.policy_contract(policy_version=3)
    assert pronunciation.compatible_policy(legacy,current)
    assert pronunciation.candidates('A2A uses API.',policy_version=3)[0]['source']=='API'
    assert [r['source'] for r in pronunciation.candidates('A2A uses API.')]==['A','A','API']
    identity={'source_sha256':hashlib.sha256(b'A2A').hexdigest(),'language':'en','policy':legacy}
    plan=pronunciation._plan('A2A',[],identity)
    assert plan['synthesis_text']=='A2A' and plan['decisions']==[]
    with pytest.raises(ValueError,match='every candidate'):
        pronunciation.validate_decisions('A2A',[])


@pytest.mark.asyncio
async def test_legacy_saved_plan_keeps_submitted_numeric_request_identity(tmp_path,mock_model):
    _resolve,chat=mock_model
    current=pronunciation.policy_contract();legacy=pronunciation.policy_contract(policy_version=3)
    text='A2A is available.'
    identity={'source_sha256':hashlib.sha256(text.encode()).hexdigest(),'language':'en','policy':legacy}
    plan=pronunciation._plan(text,[],identity)
    pronunciation._write_json(tmp_path/'tts_pronunciation.json',plan)
    checkpoint=tmp_path/'tts_input_part_001.request-0.json';checkpoint.write_text('{"job_id":"original","idempotency_key":"retained"}')
    before=checkpoint.read_bytes()
    saved=await pronunciation.prepare_pronunciation(text,tmp_path,policy=legacy,emit=lambda _m:None)
    assert saved==plan and checkpoint.read_bytes()==before
    chat.assert_not_awaited()


@pytest.mark.parametrize('field',['prompt_sha256','validation_alphabet_sha256','version'])
def test_legacy_compatibility_does_not_accept_changed_prompt_or_alphabet(field):
    current=pronunciation.policy_contract();legacy=pronunciation.policy_contract(policy_version=3)
    legacy[field]='changed'
    assert not pronunciation.compatible_policy(legacy,current)


def test_policy5_keeps_old_prompt_resources_and_uses_explicit_names_beside_n():
    legacy3=pronunciation.policy_contract(policy_version=3)
    legacy4=pronunciation.policy_contract(policy_version=4)
    latest=pronunciation.policy_contract()
    assert latest['version']==pronunciation.IMPLEMENTED_POLICY_VERSION
    assert legacy3['prompt_sha256']==legacy4['prompt_sha256']!=latest['prompt_sha256']
    assert pronunciation.compatible_policy(legacy3,latest) and pronunciation.compatible_policy(legacy4,latest)
    prompt=pronunciation.prompt_path().read_text()
    assert 'cue N ex' in prompt and 'A must remain "eigh"' in prompt
    assert 'double ess H' in prompt
    source='WLAN and SSH are present.'
    _,prepared=pronunciation.validate_decisions(source,decisions_for(source,{'WLAN':'double you ell eigh N','SSH':'double ess H'}))
    assert pronunciation.render_span(source,0,len(source),prepared)=='double you ell eigh N and double ess H are present.'
    assert source=='WLAN and SSH are present.'


@pytest.mark.parametrize('version',[3,4])
@pytest.mark.asyncio
async def test_old_policy_can_plan_without_using_the_new_prompt(tmp_path,mock_model,version):
    _resolve,chat=mock_model
    source='A2A is available.'
    policy=pronunciation.policy_contract(policy_version=version)
    if version==3:chat.side_effect=AssertionError('No candidates under original v3')
    else:chat.return_value=json.dumps({'decisions':decisions_for(source,{'A':'eigh'})})
    plan=await pronunciation.prepare_pronunciation(source,tmp_path,policy=policy,emit=lambda _m:None)
    assert plan['identity']['policy']==policy
    if version==4:
        assert chat.await_args.args[0]==pronunciation.prompt_path(policy_version=4).read_text()
        assert 'ENTIRE candidate' in chat.await_args.args[0]


def test_policy6_keeps_v5_bytes_and_disambiguates_m_without_changing_source_letters():
    old=pronunciation.policy_contract(policy_version=5);latest=pronunciation.policy_contract(policy_version=6)
    assert latest['version']==6 and old['prompt_sha256']!=latest['prompt_sha256']
    assert pronunciation.compatible_policy(old,latest)
    assert pronunciation.prompt_path(policy_version=5).name=='tts_pronunciation_v5.txt'
    assert pronunciation.prompt_path(policy_version=6).name=='tts_pronunciation_v6.txt'
    text='The report mentions CMAME today.'
    _,edits=pronunciation.validate_decisions(text,decisions_for(text,{'CMAME':'see M eigh M ee'}))
    assert pronunciation.render_span(text,0,len(text),edits)=='The report mentions see M eigh M ee today.'
    assert 'standalone M' in pronunciation.prompt_path(policy_version=6).read_text()
    assert 'N/M/R rules' in pronunciation.prompt_path(policy_version=6).read_text()


def test_single_m_hint_can_be_distinct_while_retaining_its_model_digit():
    text='The report mentions M2 today.'
    _,edits=pronunciation.validate_decisions(text,decisions_for(text,{'M':'M.'}))
    assert pronunciation.render_span(text,0,len(text),edits)=='The report mentions M. 2 today.'
    assert text=='The report mentions M2 today.'


def test_policy6_keeps_r_as_the_same_letter_in_ocr_and_numeric_fragments():
    text='OCR and R2 remain source identifiers.'
    _,edits=pronunciation.validate_decisions(text,decisions_for(text,{'OCR':'oh see R','R':'R.'}))
    assert pronunciation.render_span(text,0,len(text),edits)=='oh see R and R. 2 remain source identifiers.'
    assert text=='OCR and R2 remain source identifiers.'
    assert 'Prefer standalone R' in pronunciation.prompt_path(policy_version=6).read_text()


def test_policy8_preserves_word_policy7_and_uses_f_without_changing_other_letters():
    old = pronunciation.policy_contract(policy_version=7)
    latest = pronunciation.policy_contract(policy_version=8)
    assert old['prompt_sha256'] != latest['prompt_sha256']
    assert pronunciation.compatible_policy(old, latest)
    assert pronunciation.prompt_path(policy_version=7).name == 'tts_pronunciation_v7.txt'
    assert 'standalone F' in pronunciation.prompt_path().read_text()
    source = 'The report mentions PDF today.'
    _, edits = pronunciation.validate_decisions(source, decisions_for(source, {'PDF': 'pee dee F'}))
    assert pronunciation.render_span(source, 0, len(source), edits) == 'The report mentions pee dee F today.'
    source = 'F2 is unchanged.'
    _, edits = pronunciation.validate_decisions(source, decisions_for(source, {'F': 'F.'}))
    assert pronunciation.render_span(source, 0, len(source), edits) == 'F. 2 is unchanged.'


def test_policy9_preserves_all_submitted_policies_and_disambiguates_only_the_y_hint():
    latest = pronunciation.policy_contract(policy_version=9)
    assert latest['version'] == 9
    for version in range(3, 9):
        legacy = pronunciation.policy_contract(policy_version=version)
        assert pronunciation.compatible_policy(legacy, latest)
        assert not pronunciation.compatible_policy({**legacy, 'prompt_sha256': 'changed'}, latest)
    prompt = pronunciation.prompt_path().read_text()
    assert 'standalone Y' in prompt and 'A as "eigh" and I as "eye"' in prompt
    assert pronunciation.prompt_path(policy_version=8).name == 'tts_pronunciation_v8.txt'
    assert 'standalone Y' not in pronunciation.prompt_path(policy_version=8).read_text()
    source = 'DIY and OpenAI appear today.'
    _, edits = pronunciation.validate_decisions(
        source, decisions_for(source, {'DIY': 'dee eye Y', 'AI': 'eigh eye'}))
    assert pronunciation.render_span(source, 0, len(source), edits) == 'dee eye Y and Open eigh eye appear today.'
    assert source == 'DIY and OpenAI appear today.'


def test_single_y_hint_retains_its_letter_and_model_digit():
    source = 'Y2 is available.'
    _, edits = pronunciation.validate_decisions(source, decisions_for(source, {'Y': 'Y.'}))
    assert pronunciation.render_span(source, 0, len(source), edits) == 'Y. 2 is available.'


def test_policy11_mixed_case_and_camel_boundaries_preserve_exact_source_spans():
    source = 'AIBase PhD eSIM PCIe SoC APIs GPUs OpenAI ChatGPT APIClient MSc BSc iOS macOS'
    found = pronunciation.candidates(source)
    assert [row['source'] for row in found] == [
        'AI', 'PhD', 'eSIM', 'PCIe', 'SoC', 'API', 'GPU', 'AI', 'GPT', 'API',
        'MSc', 'BSc', 'iOS', 'OS',
    ]
    assert all(source[row['start']:row['end']] == row['source'] for row in found)
    assert [row['id'] for row in found] == list(range(len(found)))
    _, edits = pronunciation.validate_decisions(source, decisions_for(source, {
        'AI': 'eigh eye', 'PhD': 'pee aitch dee', 'eSIM': 'ee ess eye M',
        'PCIe': 'pee see eye ee', 'SoC': 'ess oh see',
    }))
    rendered = pronunciation.render_span(source, 0, len(source), edits)
    assert rendered.startswith('eigh eye Base pee aitch dee ee ess eye M pee see eye ee ess oh see')
    assert 'APIs GPUs' in rendered and 'AIBase' in source


def test_policy11_preserves_every_previous_candidate_contract_and_protected_url():
    source = 'AIBase PhD eSIM PCIe SoC'
    for version in range(3, 11):
        old = pronunciation.policy_contract(policy_version=version)
        assert pronunciation.compatible_policy(old)
        assert [row['source'] for row in pronunciation.candidates(source, policy_version=version)] == [
            'AIB', 'SIM', 'PCI',
        ]
        legacy_text = 'AIBase uses APIs and GPUs.'
        legacy_decisions = [
            {**row, 'action': 'spell' if row['source'] == 'AIB' else 'keep',
             'spoken': 'eigh eye bee' if row['source'] == 'AIB' else row['source'],
             'reason': 'Retain the frozen legacy decision.'}
            for row in pronunciation.candidates(legacy_text, policy_version=version)
        ]
        _, edits = pronunciation.validate_decisions(
            legacy_text, legacy_decisions, policy_version=version,
        )
        assert pronunciation.render_span(legacy_text, 0, len(legacy_text), edits) == (
            'eigh eye beease uses APIs and GPUs.'
        )
    text = 'See https://example.com/AIBase/PhD and AIBase@example.com. PhD remains.'
    assert [row['source'] for row in pronunciation.candidates(text)] == ['PhD']
    assert pronunciation.policy_contract()['candidate_boundary_version'] == 1
    assert [row['source'] for row in pronunciation.candidates('APIservice CSVfile JSONParser AIService')] == [
        'API', 'CSV', 'JSON', 'AI',
    ]


@pytest.mark.asyncio
async def test_source_bound_audio_feedback_is_advisory_and_never_rebinds_a_submitted_plan(tmp_path, mock_model):
    _resolve, chat = mock_model
    source = 'The report mentions DIY today.'
    chat.return_value = json.dumps({'decisions': decisions_for(source, {'DIY': 'dee eye Y'})})
    feedback = {'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                'observations': [{'source': 'DIY', 'failure': 'extra why after closing words'}]}
    plan = await pronunciation.prepare_pronunciation(
        source, tmp_path, policy=pronunciation.policy_contract(), emit=lambda _m: None,
        audio_feedback=feedback)
    assert json.loads(chat.await_args.args[1])['audio_feedback'] == feedback
    assert plan['synthesis_text'] == 'The report mentions dee eye Y today.'
    checkpoint = tmp_path / 'tts_input_part_001.request-0.json'
    checkpoint.write_text('{"job_id":"original","idempotency_key":"original-key"}')
    original = checkpoint.read_bytes()
    updated = {**feedback, 'observations': [{'source': 'DIY', 'failure': 'different observation'}]}
    chat.reset_mock()
    reused = await pronunciation.prepare_pronunciation(
        source, tmp_path, policy=pronunciation.policy_contract(), emit=lambda _m: None,
        audio_feedback=updated)
    assert reused == plan and checkpoint.read_bytes() == original
    chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_audio_feedback_for_another_source_is_rejected_before_model_or_job_changes(tmp_path, mock_model):
    _resolve, chat = mock_model
    with pytest.raises(ValueError, match='complete canonical source'):
        await pronunciation.prepare_pronunciation(
            'DIY today.', tmp_path, policy=pronunciation.policy_contract(), emit=lambda _m: None,
            audio_feedback={'source_sha256': 'another-source'})
    chat.assert_not_awaited()
    assert not list(tmp_path.glob('tts_input_part_*.request-*.json'))


def v14_decisions(text, spellings=None):
    spellings = spellings or {}
    return [{**row, 'action': 'number' if row.get('kind') == 'number' else
             'spell' if row['source'] in spellings else 'keep',
             'spoken': pronunciation.numeric_cue(row['source']) if row.get('kind') == 'number' else
             spellings.get(row['source'], row['source']), 'reason': 'Exact source reading.'}
            for row in pronunciation.candidates(text, policy_version=14)]


def programming_decisions(text):
    return [{**row, 'action': 'programming' if row.get('kind') == 'programming' else 'keep',
             'spoken': pronunciation.PROGRAMMING_CUES[row['source']] if row.get('kind') == 'programming' else row['source'],
             'reason': 'Programming language name in the sentence.'}
            for row in pronunciation.candidates(text, policy_version=15)]


def test_policy16_short_mixed_cloud_acronyms_are_complete_and_not_embedded_or_protected():
    text = 'PaaS IaaS SoC PhD use https://example.com/PaaS and XPaaS and PaaServices.'
    found = pronunciation.candidates(text, policy_version=16)
    assert [r['source'] for r in found] == ['PaaS', 'IaaS', 'SoC', 'PhD', 'XP']
    assert [r['id'] for r in found] == list(range(5))
    for version in range(3, 16):
        assert not any(r['source'] in {'PaaS', 'IaaS'} for r in pronunciation.candidates(text, policy_version=version))
    assert pronunciation.POLICY_VERSION == 13


@pytest.mark.parametrize(('name', 'phones'), [('PaaS', ['P', 'AE', 'S']), ('IaaS', ['AY', 'AE', 'Z'])])
def test_policy16_cited_cloud_word_hint_preserves_source_and_binds_hint_version(name, phones):
    text = f'The {name} service works.'
    row = pronunciation.candidates(text, policy_version=16)[0]
    _, edits = pronunciation.validate_decisions(text, [{**row, 'action': 'word', 'spoken': name+',',
        'phones': phones, 'reason': 'Cited cloud-service word reading.'}], policy_version=16)
    assert pronunciation.render_span(text, 0, len(text), edits) == f'The {name}, service works.'
    assert edits[0]['word_hint_version'] == 5
    assert pronunciation.word_hint_reference(edits[0])['word_hint_version'] == 5
    assert pronunciation.compatible_policy(pronunciation.policy_contract(policy_version=16))


def test_policy16_cloud_spelling_retains_repeated_lowercase_letters():
    text = 'The IaaS service works.'
    row = pronunciation.candidates(text, policy_version=16)[0]
    decision = {**row, 'action': 'spell', 'spoken': 'eye double eigh ess', 'reason': 'Published conventional letter reading.'}
    _, edits = pronunciation.validate_decisions(text, [decision], policy_version=16)
    assert pronunciation.render_span(text, 0, len(text), edits) == 'The eye double eigh ess service works.'
    with pytest.raises(ValueError, match='every source letter'):
        pronunciation.validate_decisions(text, [{**decision, 'spoken': 'eye eigh ess'}], policy_version=16)


@pytest.mark.parametrize('name', ['AQuA', 'ASCiI', 'ABcdE', 'ABCdeF'])
def test_policy17_upper_prefix_mixed_candidates_are_complete_and_non_overlapping(name):
    text = f'The report mentions {name} today.'
    found = pronunciation.candidates(text, policy_version=17)
    assert len(found) == 1 and found[0]['source'] == name
    assert text[found[0]['start']:found[0]['end']] == name
    assert pronunciation.compatible_policy(pronunciation.policy_contract(policy_version=17))
    assert pronunciation.POLICY_VERSION == 13


def test_policy17_mixed_candidate_keeps_last_letter_in_spelled_mode():
    text = 'The AQuA result is available.'
    row = pronunciation.candidates(text, policy_version=17)[0]
    decision = {**row,'action':'spell','spoken':'eigh cue you eigh','reason':'All four source letters.'}
    _, edits = pronunciation.validate_decisions(text, [decision], policy_version=17)
    assert pronunciation.render_span(text, 0, len(text), edits) == 'The eigh cue you eigh result is available.'
    with pytest.raises(ValueError, match='every source letter'):
        pronunciation.validate_decisions(text, [{**decision,'spoken':'eigh cue you'}], policy_version=17)


def test_policy17_complete_mixed_word_is_eligible_for_existing_dictionary_hint():
    text = 'The report mentions AQuA today.'
    row = pronunciation.candidates(text, policy_version=17)[0]
    _, edits = pronunciation.validate_decisions(text, [{**row,'action':'word','spoken':'aqua,',
        'phones':['AE','K','W','AH'],'reason':'Complete source dictionary word reading.'}], policy_version=17)
    assert pronunciation.render_span(text, 0, len(text), edits) == 'The report mentions aqua, today.'
    assert edits[0]['source'] == 'AQuA' and edits[0]['word_hint_version'] == 5


def test_policy18_cited_full_word_replaces_only_its_partial_capital_candidate():
    text='The report mentions PostgreSQL and OpenAI today.'
    assert [r['source'] for r in pronunciation.candidates(text,policy_version=18)]==['PostgreSQL','AI']
    assert [r['source'] for r in pronunciation.candidates(text,policy_version=17)]==['SQL','AI']
    assert pronunciation.POLICY_VERSION==13
    assert pronunciation.compatible_policy(pronunciation.policy_contract(policy_version=18))


def test_policy18_official_postgresql_cue_preserves_canonical_source_and_binds_reference6():
    text='The report mentions PostgreSQL today.'
    row=pronunciation.candidates(text,policy_version=18)[0]
    phones=['P','OW','S','T','G','R','EH','S','K','Y','UW','EH','L']
    _,edits=pronunciation.validate_decisions(text,[{**row,'action':'sound','spoken':'post gress cue ell',
        'phones':phones,'reason':'Official complete-name pronunciation.'}],policy_version=18)
    assert pronunciation.render_span(text,0,len(text),edits)=='The report mentions post gress cue ell today.'
    assert edits[0]['word_hint_version']==6 and edits[0]['source']=='PostgreSQL'
    assert text=='The report mentions PostgreSQL today.'


def test_policy18_cited_word_never_changes_url_email_or_a_larger_name():
    text='https://example.com/PostgreSQL PostgreSQL@example.com MyPostgreSQL remain.'
    assert not any(r['source']=='PostgreSQL' for r in pronunciation.candidates(text,policy_version=18))


def test_policy19_storage_nas_cue_keeps_word_mode_and_canonical_source():
    text = 'The report mentions NAS today.'
    decision = {'id': 0, 'source': 'NAS', 'action': 'sound', 'spoken': 'nass',
                'phones': ['N', 'AE', 'S'], 'reason': 'Contextual storage word reading.'}
    _, edits = pronunciation.validate_decisions(text, [decision], policy_version=19)
    assert pronunciation.render_span(text, 0, len(text), edits) == 'The report mentions nass today.'
    assert edits[0]['word_hint_version'] == 7 and edits[0]['word_hint_type'] == 'homophone'
    assert pronunciation.word_hint_reference(edits[0])['word_hint_version'] == 7
    assert pronunciation.compatible_policy(pronunciation.policy_contract(policy_version=19))
    assert pronunciation.POLICY_VERSION == 13
    with pytest.raises(ValueError, match='independent complete-source'):
        pronunciation.validate_decisions(text, [decision], policy_version=18)


@pytest.mark.asyncio
async def test_policy19_planner_receives_cited_storage_word_reference(tmp_path, mock_model):
    _, chat = mock_model
    chat.return_value = json.dumps({'decisions': [{'id': 0, 'source': 'NAS',
        'action': 'sound', 'spoken': 'nass', 'phones': ['N', 'AE', 'S'],
        'reason': 'Use the cited storage word reading.'}]})
    plan = await pronunciation.prepare_pronunciation('The report mentions NAS today.', tmp_path,
        policy=pronunciation.policy_contract(policy_version=19), emit=lambda _: None)
    payload = json.loads(chat.await_args.args[1])
    assert payload['cited_word_phone_references'][0]['source'] == 'NAS'
    assert payload['cited_word_phone_references'][0]['variants'] == [['N', 'AE', 'S']]
    assert plan['edits'][0]['word_hint_version'] == 7


@pytest.mark.asyncio
async def test_policy20_planner_and_audio_reference_use_same_cited_whole_word(tmp_path, mock_model):
    _, chat = mock_model
    phones = ['EH', 'B', 'IH', 'T', 'D', 'AH']
    chat.return_value = json.dumps({'decisions': [{'id': 0, 'source': 'EBITDA',
        'action': 'sound', 'spoken': 'ebb it duh', 'phones': phones,
        'reason': 'Retain the cited complete word reading.'}]})
    source = 'The report mentions EBITDA today.'
    plan = await pronunciation.prepare_pronunciation(source, tmp_path,
        policy=pronunciation.policy_contract(policy_version=20), emit=lambda _: None)
    payload = json.loads(chat.await_args.args[1])
    assert payload['cited_word_phone_references'][0]['variants'] == [phones]
    assert plan['synthesis_text'] == 'The report mentions ebb it duh today.'
    assert plan['edits'][0]['word_hint_version'] == 8
    assert pronunciation.compatible_policy(pronunciation.policy_contract(policy_version=20))
    assert pronunciation.POLICY_VERSION == 13


def test_policy17_does_not_capture_mixed_names_inside_urls_or_larger_words():
    text = 'https://example.com/AQuA XAQuA AQuAClient remain.'
    assert not any(row['source']=='AQuA' for row in pronunciation.candidates(text, policy_version=17))
    assert pronunciation.candidates('AQuA', policy_version=16)[0]['source'] == 'AQu'


@pytest.mark.asyncio
async def test_policy17_cannot_rebind_submitted_partial_mixed_plan(tmp_path, mock_model):
    _resolve, chat = mock_model
    source = 'The report mentions AQuA today.'
    chat.return_value = json.dumps({'decisions':[{**row,'action':'keep','spoken':row['source'],
        'reason':'Native word reading.'} for row in pronunciation.candidates(source,policy_version=16)]})
    await pronunciation.prepare_pronunciation(source,tmp_path,policy=pronunciation.policy_contract(policy_version=16),emit=lambda _:None)
    checkpoint=tmp_path/'tts_input_part_001.request-0.json';checkpoint.write_text('{"identity":"original","idempotency_key":"same","job_id":"original-job"}')
    original_checkpoint=checkpoint.read_bytes();original_plan=(tmp_path/'tts_pronunciation.json').read_bytes()
    with pytest.raises(RuntimeError,match='preserving their request identities'):
        await pronunciation.prepare_pronunciation(source,tmp_path,policy=pronunciation.policy_contract(policy_version=17),emit=lambda _:None)
    assert checkpoint.read_bytes()==original_checkpoint and (tmp_path/'tts_pronunciation.json').read_bytes()==original_plan


@pytest.mark.parametrize(('name', 'cue', 'phones'), [
    ('PaaS', 'pass', ['P', 'AE', 'S']), ('IaaS', 'eye as', ['AY', 'AE', 'Z']),
])
def test_policy16_phonetic_word_cue_is_versioned_and_canonical_text_unchanged(name, cue, phones):
    text = f'The {name} service works.'
    row = pronunciation.candidates(text, policy_version=16)[0]
    _, edits = pronunciation.validate_decisions(text, [{**row, 'action':'sound', 'spoken':cue,
        'phones':phones, 'reason':'Independent cited word reading with identical ordinary dictionary cue.'}], policy_version=16)
    assert pronunciation.render_span(text, 0, len(text), edits) == f'The {cue} service works.'
    assert text == f'The {name} service works.'
    assert pronunciation.word_hint_reference(edits[0])['word_hint_type'] == 'homophone'


def test_policy15_programming_cues_retain_both_operators_and_source():
    text = 'The programmer reads a C++ example and a C# example.'
    _, edits = pronunciation.validate_decisions(text, programming_decisions(text), policy_version=15)
    assert pronunciation.render_span(text, 0, len(text), edits) == (
        'The programmer reads a see plus plus example and a see sharp example.')
    assert text == 'The programmer reads a C++ example and a C# example.'
    assert [(row['source'], row['kind']) for row in edits] == [('C++', 'programming'), ('C#', 'programming')]
    for version in range(3, 15):
        assert not pronunciation.candidates(text, policy_version=version)
    assert pronunciation.POLICY_VERSION == 13


@pytest.mark.parametrize('wrong', ['see plus', 'see plus plus plus', 'cee plus plus', 'C++'])
def test_policy15_programming_cue_rejects_symbol_changes(wrong):
    text = 'The programmer reads a C++ example.'
    decisions = programming_decisions(text)
    decisions[0]['spoken'] = wrong
    with pytest.raises(ValueError, match='every named symbol'):
        pronunciation.validate_decisions(text, decisions, policy_version=15)


@pytest.mark.parametrize('wrong', ['see hashtag', 'see pound', 'see', 'see sharp sharp'])
def test_policy15_sharp_cue_rejects_symbol_changes(wrong):
    text = 'The programmer reads a C# example.'
    decisions = programming_decisions(text)
    decisions[0]['spoken'] = wrong
    with pytest.raises(ValueError, match='every named symbol'):
        pronunciation.validate_decisions(text, decisions, policy_version=15)


def test_policy15_programming_names_avoid_protected_and_embedded_source():
    text = 'https://example.com/C++ https://example.com/C# coder@example.com XC++ C+++ C## C++x c++ remain.'
    assert not any(row.get('kind') == 'programming'
                   for row in pronunciation.candidates(text, policy_version=15))


@pytest.mark.asyncio
async def test_policy15_programming_plan_is_frozen_and_cannot_rebind_submitted_plan(tmp_path, mock_model):
    _resolve, chat = mock_model
    text = 'The programmer reads a C++ example.'
    old_policy = pronunciation.policy_contract(policy_version=14)
    old = await pronunciation.prepare_pronunciation(text, tmp_path, policy=old_policy, emit=lambda _: None)
    checkpoint = tmp_path / 'tts_input_part_001.request-0.json'
    checkpoint.write_text('{"identity":"old","idempotency_key":"same","job_id":"original"}')
    original_checkpoint = checkpoint.read_bytes()
    original_plan = (tmp_path / 'tts_pronunciation.json').read_bytes()
    policy = pronunciation.policy_contract(policy_version=15)
    with pytest.raises(RuntimeError, match='preserving their request identities'):
        await pronunciation.prepare_pronunciation(text, tmp_path, policy=policy, emit=lambda _: None)
    assert checkpoint.read_bytes() == original_checkpoint
    assert (tmp_path / 'tts_pronunciation.json').read_bytes() == original_plan
    assert old['synthesis_text'] == text
    fresh = tmp_path / 'new-input'
    chat.return_value = json.dumps({'decisions': programming_decisions(text)})
    plan = await pronunciation.prepare_pronunciation(text, fresh, policy=policy, emit=lambda _: None)
    assert plan['synthesis_text'] == 'The programmer reads a see plus plus example.'
    assert await pronunciation.prepare_pronunciation(text, fresh, policy=policy, emit=lambda _: None) == plan
    assert pronunciation.compatible_policy(policy)


def test_policy14_numeric_cues_preserve_source_and_all_legacy_policy_contracts(tmp_path):
    text = 'The GPT-5.2 and TLS1.3 models use GD01 and GPT4o.'
    old = {version: pronunciation.policy_contract(policy_version=version) for version in range(3, 14)}
    _, edits = pronunciation.validate_decisions(text, v14_decisions(text, {'GPT': 'gee pee tee', 'TLS': 'tee ell ess', 'GD': 'gee dee'}), policy_version=14)
    rendered = pronunciation.render_span(text, 0, len(text), edits)
    assert 'gee pee tee-five point two' in rendered
    assert 'tee ell ess one point three' in rendered
    assert 'gee dee zero one' in rendered
    assert 'gee pee tee four o' in rendered
    assert text == 'The GPT-5.2 and TLS1.3 models use GD01 and GPT4o.'
    for version in range(3, 14):
        assert pronunciation.policy_contract(policy_version=version) == old[version]
        assert pronunciation.compatible_policy(old[version])
        assert not any(row.get('kind') == 'number' for row in pronunciation.candidates(text, policy_version=version))
    assert pronunciation.POLICY_VERSION == 13
    assert pronunciation.policy_contract(policy_version=14)['numeric_cue_version'] == 1


@pytest.mark.parametrize(('source', 'wrong'), [
    ('5.2', 'five twenty two'), ('1.3', 'one three'), ('01', 'one'),
    ('1.03', 'one point three'), ('1.30', 'one point three'), ('4', 'four oh'),
])
def test_policy14_numeric_cue_rejects_value_precision_and_letter_changes(source, wrong):
    text = f'The GPT{source} model is ready.'
    decisions = v14_decisions(text)
    row = next(row for row in decisions if row.get('kind') == 'number')
    row['spoken'] = wrong
    with pytest.raises(ValueError, match='exact source value'):
        pronunciation.validate_decisions(text, decisions, policy_version=14)


def test_policy14_numbers_inside_urls_and_unbounded_versions_are_not_transformed():
    text = 'https://example.com/GPT5.2 and GPT1.2.3 and GPT1234567 remain.'
    assert not any(row.get('kind') == 'number' for row in pronunciation.candidates(text, policy_version=14))
    with pytest.raises(ValueError, match='Unsupported'):
        pronunciation.numeric_cue('1234567')


@pytest.mark.asyncio
async def test_policy14_numeric_plan_cache_is_frozen_before_any_job(tmp_path, mock_model):
    _resolve, chat = mock_model
    text = 'The report mentions TLS1.3 today.'
    decisions = v14_decisions(text, {'TLS': 'tee ell ess'})
    chat.return_value = json.dumps({'decisions': decisions})
    policy = pronunciation.policy_contract(policy_version=14)
    plan = await pronunciation.prepare_pronunciation(text, tmp_path, policy=policy, emit=lambda _: None)
    assert plan['synthesis_text'] == 'The report mentions tee ell ess one point three today.'
    again = await pronunciation.prepare_pronunciation(text, tmp_path, policy=policy, emit=lambda _: None)
    assert again == plan and chat.await_count == 1


@pytest.mark.asyncio
async def test_policy14_cannot_rebind_a_submitted_policy13_numeric_plan(tmp_path, mock_model):
    _resolve, chat = mock_model
    text = 'The TLS1.3 report is ready.'
    chat.return_value = json.dumps({'decisions': decisions_for(text, {'TLS': 'tee ell ess'})})
    old = await pronunciation.prepare_pronunciation(text, tmp_path, policy=pronunciation.policy_contract(policy_version=13), emit=lambda _: None)
    checkpoint = tmp_path / 'tts_input_part_001.request-0.json'
    checkpoint.write_text('{"identity":"old-request","idempotency_key":"old-key","job_id":"old-job"}')
    before = checkpoint.read_bytes()
    plan_before = (tmp_path / 'tts_pronunciation.json').read_bytes()
    chat.reset_mock()
    with pytest.raises(RuntimeError, match='preserving.*request identities'):
        await pronunciation.prepare_pronunciation(text, tmp_path, policy=pronunciation.policy_contract(policy_version=14), emit=lambda _: None)
    assert checkpoint.read_bytes() == before
    assert (tmp_path / 'tts_pronunciation.json').read_bytes() == plan_before
    assert old['synthesis_text'] == 'The tee ell ess 1.3 report is ready.'
    chat.assert_not_awaited()

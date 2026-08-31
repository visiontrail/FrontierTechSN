import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from backend.pipeline import model_router, tts


def _words(text: str) -> list[dict]:
    return [
        {"text": word, "start": index * 0.2, "end": index * 0.2 + 0.1}
        for index, word in enumerate(text.split())
    ]


def test_medium_asr_verdict_requires_identical_close_non_alphanumeric_evidence():
    expected = tts._lexical_tokens(
        "DeepTech China reports that venture firm Andreessen Horowitz announced."
    )
    corroborated = (
        "Deep Tech China reports that venture firm Andreasen Horowitz announced"
    )

    assert tts._medium_asr_verdict_is_corroborated(
        expected,
        corroborated,
        corroborated,
    )
    assert not tts._medium_asr_verdict_is_corroborated(
        expected,
        corroborated,
        corroborated.replace("Andreasen", "Andres and"),
    )
    assert not tts._medium_asr_verdict_is_corroborated(
        tts._lexical_tokens("Venture firm a16z announced a fund."),
        "Venture firm A6EZ announced a fund",
        "Venture firm A6EZ announced a fund",
    )


@pytest.mark.asyncio
async def test_llm_adjudication_persists_high_confidence_asr_only_approval(tmp_path):
    source = "Parts as made in Taiwan. The source is Nikkei Asia."
    normal = _words("Parts is made in Taiwan The source is Nikkei Asia")
    slower = _words("Parts as made in Taiwan The source is Nikkei Asia")
    report = tts._orpheus_transcript_report(source, normal)
    expected_indexes = list(range(len(tts._lexical_tokens(source))))

    async def chat(*_args, route_selected=None, **_kwargs):
        assert route_selected is not None
        route_selected("https://models.example/v1", "judge-model", "secret")
        return json.dumps(
            {
                "decision": "approve_asr_error",
                "all_source_tokens_accounted_for": True,
                "accounted_source_token_indexes": expected_indexes,
                "confidence": "high",
                "reason": "The slower transcript recovers the as/is homophone exactly.",
            }
        )

    with (
        patch(
            "backend.pipeline.digester._resolve_provider",
            AsyncMock(return_value=("https://models.example/v1", "judge-model", "secret")),
        ),
        patch("backend.pipeline.digester._chat", AsyncMock(side_effect=chat)),
        patch(
            "backend.pipeline.model_router.resolve_model_routes",
            AsyncMock(
                return_value=(
                    model_router.ModelRoute(
                        slot="standalone",
                        provider_id=None,
                        provider_type="test",
                        provider_name="Test judge",
                        endpoint="https://models.example/v1",
                        model="judge-model",
                        api_key="secret",
                    ),
                )
            ),
        ),
    ):
        result = await tts._adjudicate_orpheus_asr_mismatch(
            source,
            normal,
            slower,
            report,
            tmp_path,
            emit=lambda _message: None,
        )

    assert result is not None
    assert result["decision"] == "approve_asr_error"
    evidence = json.loads((tmp_path / "llm_asr_adjudication.json").read_text())
    assert evidence["status"] == "approved"
    assert evidence["route"] == {
        "endpoint": "https://models.example/v1",
        "model": "judge-model",
    }
    assert "secret" not in json.dumps(evidence)


@pytest.mark.asyncio
async def test_llm_adjudication_prompt_treats_equivalent_number_format_as_evidence(
    tmp_path,
):
    source = (
        "Finally, Axios reports chip design startup Agentrys raised "
        "twenty-four point five million."
    )
    normal = _words(
        "Finally Axios reports chip design startup Agentries raised 24.5 million"
    )
    slower = _words(
        "Finally Axios reports chip design startup Agentries raised $24.5 million"
    )
    report = tts._orpheus_transcript_report(source, normal)
    expected_indexes = list(range(len(tts._lexical_tokens(source))))

    async def chat(system_prompt, *_args, route_selected=None, **_kwargs):
        assert "mismatch trigger, not as ground truth" in system_prompt
        assert "twenty-four point five" in system_prompt
        assert route_selected is not None
        route_selected("https://models.example/v1", "judge-model", "secret")
        return json.dumps(
            {
                "decision": "approve_asr_error",
                "all_source_tokens_accounted_for": True,
                "accounted_source_token_indexes": expected_indexes,
                "confidence": "high",
                "reason": "Both transcripts contain the exact 24.5 million value and full sentence.",
            }
        )

    with (
        patch(
            "backend.pipeline.digester._resolve_provider",
            AsyncMock(return_value=("https://models.example/v1", "judge-model", "secret")),
        ),
        patch("backend.pipeline.digester._chat", AsyncMock(side_effect=chat)),
        patch(
            "backend.pipeline.model_router.resolve_model_routes",
            AsyncMock(
                return_value=(
                    model_router.ModelRoute(
                        slot="standalone",
                        provider_id=None,
                        provider_type="test",
                        provider_name="Test judge",
                        endpoint="https://models.example/v1",
                        model="judge-model",
                        api_key="secret",
                    ),
                )
            ),
        ),
    ):
        result = await tts._adjudicate_orpheus_asr_mismatch(
            source,
            normal,
            slower,
            report,
            tmp_path,
            emit=lambda _message: None,
        )

    assert result is not None
    assert result["decision"] == "approve_asr_error"


@pytest.mark.asyncio
async def test_llm_adjudication_accepts_corroborated_medium_name_spelling(tmp_path):
    source = (
        "DeepTech China reports that venture firm Andreessen Horowitz announced."
    )
    transcript = _words(
        "Deep Tech China reports that venture firm Andreasen Horowitz announced"
    )
    report = tts._orpheus_transcript_report(source, transcript)
    expected_indexes = list(range(len(tts._lexical_tokens(source))))

    async def chat(*_args, route_selected=None, **_kwargs):
        assert route_selected is not None
        route_selected("https://models.example/v1", "judge-model", "secret")
        return json.dumps(
            {
                "decision": "approve_asr_error",
                "all_source_tokens_accounted_for": True,
                "accounted_source_token_indexes": expected_indexes,
                "confidence": "medium",
                "reason": "Both decodes contain the full sentence; only the proper-name spelling differs.",
            }
        )

    with (
        patch(
            "backend.pipeline.digester._resolve_provider",
            AsyncMock(return_value=("https://models.example/v1", "judge-model", "secret")),
        ),
        patch("backend.pipeline.digester._chat", AsyncMock(side_effect=chat)),
        patch(
            "backend.pipeline.model_router.resolve_model_routes",
            AsyncMock(
                return_value=(
                    model_router.ModelRoute(
                        slot="standalone",
                        provider_id=None,
                        provider_type="test",
                        provider_name="Test judge",
                        endpoint="https://models.example/v1",
                        model="judge-model",
                        api_key="secret",
                    ),
                )
            ),
        ),
    ):
        result = await tts._adjudicate_orpheus_asr_mismatch(
            source,
            transcript,
            transcript,
            report,
            tmp_path,
            emit=lambda _message: None,
        )

    assert result is not None
    assert result["confidence"] == "medium"
    assert result["medium_confidence_corroborated"] is True
    evidence = json.loads((tmp_path / "llm_asr_adjudication.json").read_text())
    assert evidence["status"] == "approved"


@pytest.mark.asyncio
async def test_llm_adjudication_rejects_incomplete_token_accounting(tmp_path):
    source = "The source is Nikkei Asia."
    normal = _words("The source is Nikkei")
    slower = _words("The source is Nikkei")
    report = tts._orpheus_transcript_report(source, normal)

    with (
        patch(
            "backend.pipeline.digester._resolve_provider",
            AsyncMock(return_value=("https://models.example/v1", "judge-model", "secret")),
        ),
        patch(
            "backend.pipeline.digester._chat",
            AsyncMock(
                return_value=json.dumps(
                    {
                        "decision": "approve_asr_error",
                        "all_source_tokens_accounted_for": True,
                        "accounted_source_token_indexes": [0, 1, 2],
                        "confidence": "high",
                        "reason": "Incomplete evidence must not pass.",
                    }
                )
            ),
        ),
        patch(
            "backend.pipeline.model_router.resolve_model_routes",
            AsyncMock(
                return_value=(
                    model_router.ModelRoute(
                        slot="standalone",
                        provider_id=None,
                        provider_type="test",
                        provider_name="Test judge",
                        endpoint="https://models.example/v1",
                        model="judge-model",
                        api_key="secret",
                    ),
                )
            ),
        ),
    ):
        result = await tts._adjudicate_orpheus_asr_mismatch(
            source,
            normal,
            slower,
            report,
            tmp_path,
            emit=lambda _message: None,
        )

    assert result is None
    evidence = json.loads((tmp_path / "llm_asr_adjudication.json").read_text())
    assert evidence["status"] == "rejected"


@pytest.mark.asyncio
async def test_orpheus_verifier_escalates_mismatch_to_model_when_enabled(tmp_path):
    source = "Parts as made in Taiwan."
    normal = _words("Parts is made in Taiwan")
    slower = _words("Parts as made in Taiwan")
    decision = {
        "decision": "approve_asr_error",
        "confidence": "high",
        "reason": "Slower ASR recovers the exact wording.",
        "normal_speed_transcript": "Parts is made in Taiwan",
        "slower_speed_transcript": "Parts as made in Taiwan",
        "evidence_path": "llm_asr_adjudication.json",
        "route": {"model": "judge-model"},
    }
    messages: list[str] = []

    with (
        patch(
            "backend.pipeline.av_sync.ensure_word_transcript",
            AsyncMock(return_value=(normal, {"passed": True})),
        ),
        patch.object(
            tts,
            "_transcribe_orpheus_at_speed",
            AsyncMock(return_value=(slower, {"passed": True})),
        ),
        patch.object(
            tts,
            "_adjudicate_orpheus_asr_mismatch",
            AsyncMock(return_value=decision),
        ) as adjudicator,
    ):
        report = await tts._verify_orpheus_part(
            Path("part.wav"),
            source,
            tmp_path,
            emit=messages.append,
            adjudicate_asr=True,
        )

    assert report["verified"] is True
    assert report["verification_mode"] == "llm_asr_adjudication"
    adjudicator.assert_awaited_once()
    assert any("two-level ASR plus model" in message for message in messages)


@pytest.mark.asyncio
async def test_orpheus_verifier_keeps_mismatch_strict_without_adjudication(tmp_path):
    source = "The source is Nikkei Asia."
    normal = _words("The source is Nikkei")

    with patch(
        "backend.pipeline.av_sync.ensure_word_transcript",
        AsyncMock(return_value=(normal, {"passed": True})),
    ):
        with pytest.raises(tts.TtsIntegrityError):
            await tts._verify_orpheus_part(
                Path("part.wav"),
                source,
                tmp_path,
                emit=lambda _message: None,
            )

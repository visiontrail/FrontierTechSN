"""Production-shaped OpenCLI envelopes for all three visual review contracts."""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from backend.pipeline import multimodal_review, web_footage
from backend.pipeline.opencli import OpenCLIError, OpenCLIResult
from backend.pipeline.review_response import ReviewResponseError


PREVIEW = {
    "image_received": True, "suitable": True, "confidence": .9,
    "selected_window": 0, "visible_content": "A robot arm moving components",
    "reason": "The industrial robot matches the narrated demonstration",
}
CONTRACTS = [
    (PREVIEW, lambda text: web_footage._preview_response_payload(text, batch=False)),
    ({"results": [{"candidate_id": 0, **PREVIEW}]},
     lambda text: web_footage._preview_response_payload(text, batch=True)),
    ({"image_received": True, "reviews": [{"id": "scene-01", "score": 90}]},
     multimodal_review._response_payload),
]


def wrapped(payload):
    return json.dumps([{"response": json.dumps(payload)}])


@pytest.mark.parametrize("payload,parse", CONTRACTS)
@pytest.mark.parametrize("shape", ["raw", "fenced", "response", "Response", "array", "dict", "stderr"])
def test_each_review_contract_accepts_actual_cli_response_shapes(payload, parse, shape):
    text = json.dumps(payload)
    variants = {
        "raw": text,
        "fenced": f"```json\n{text}\n```",
        "response": json.dumps({"response": text}),
        "Response": json.dumps({"Response": text}),
        "array": wrapped(payload),
        "dict": json.dumps({"response": payload}),
        "stderr": "plain stdout\n" + wrapped(payload),
    }
    assert parse(variants[shape]) == payload


@pytest.mark.parametrize("payload,parse", CONTRACTS)
@pytest.mark.parametrize("shape", ["raw", "string", "rows", "dicts"])
def test_never_selects_one_of_multiple_complete_answers(payload, parse, shape):
    text = json.dumps(payload)
    variants = {
        "raw": text + "\n" + text,
        "string": json.dumps({"response": text + "\n" + text}),
        "rows": json.dumps([{"response": text}, {"response": text}]),
        "dicts": json.dumps([payload, payload]),
    }
    with pytest.raises(ReviewResponseError):
        parse(variants[shape])


@pytest.mark.parametrize("payload,parse", CONTRACTS)
@pytest.mark.parametrize("invalid", ["", "[NO RESPONSE]", '{"image_received": true,', '{}'])
def test_missing_or_incomplete_answer_cannot_pass(payload, parse, invalid):
    with pytest.raises(ReviewResponseError):
        parse(json.dumps([{"response": invalid}]))


@pytest.mark.parametrize("payload,parse", CONTRACTS)
def test_contracts_are_not_interchangeable(payload, parse):
    for other, _ in CONTRACTS:
        if other != payload:
            with pytest.raises(ReviewResponseError):
                parse(wrapped(other))


def prepared_preview(tmp_path):
    folder = tmp_path / "preview"
    folder.mkdir()
    sheet = folder / "contact-sheet.jpg"
    Image.new("RGB", (160, 90), "blue").save(sheet)
    window = folder / "window-0"
    window.mkdir()
    (window / "preview.mp4").write_bytes(b"retained preview")
    return {"folder": folder, "sheet": sheet,
            "intervals": [{"start_seconds": 5, "end_seconds": 20}]}


def result(payload, *, stderr=False):
    return OpenCLIResult((), 0, "" if stderr else wrapped(payload), wrapped(payload) if stderr else "")


def test_single_preview_recovers_malformed_reply_and_retains_every_attempt(tmp_path):
    prepared = prepared_preview(tmp_path)
    candidate = {"source_page_url": "https://youtu.be/test", "title": "Robot demo"}
    responses = [result({"image_received": True, "reviews": []}), result(PREVIEW, stderr=True)]
    async def run():
        with (patch.object(web_footage, "_prepare_candidate_preview", AsyncMock(return_value=prepared)) as prepare,
              patch.object(web_footage, "run_opencli", AsyncMock(side_effect=responses)) as ask):
            review = await web_footage._analyze_candidate_preview(candidate, "Industrial robots", tmp_path)
        assert review["suitable"] is True
        assert review["start_seconds"] == 5
        assert review["reviewed_preview_sha256"] == web_footage._sha256(prepared["folder"] / "window-0/preview.mp4")
        assert prepare.await_count == 1
        assert ask.await_count == 2
        assert "previous response did not satisfy" in ask.await_args_list[1].args[0][2]
    asyncio.run(run())
    attempts = sorted(prepared["folder"].glob("review-attempts/*/attempt-*.json"))
    assert len(attempts) == 2
    first, second = [json.loads(path.read_text()) for path in attempts]
    assert first["failure_kind"] == "response_contract"
    assert first["stdout"] == responses[0].stdout
    assert second["stderr"] == responses[1].stderr
    assert second["status"] == "reviewed"


@pytest.mark.parametrize("defect", ["transport", "exit", "provider_generation", "malformed", "missing_image", "bad_window", "nan_confidence"])
def test_preview_retries_are_bounded_and_never_mark_unavailable_as_rejected(tmp_path, defect):
    prepared = prepared_preview(tmp_path)
    response = {
        "transport": OpenCLIError("Browser connection unavailable"),
        "exit": OpenCLIResult((), 1, "", "browser connection failed"),
        "provider_generation": OpenCLIResult((), 1, "", "Gemini generation failed: Sorry, something went wrong. Please try your request again."),
        "malformed": result({"image_received": True, "reviews": []}),
        "missing_image": result({**PREVIEW, "image_received": False}),
        "bad_window": result({**PREVIEW, "selected_window": 99}),
        "nan_confidence": result({**PREVIEW, "confidence": float("nan")}),
    }[defect]
    ask = AsyncMock(side_effect=response) if isinstance(response, Exception) else AsyncMock(return_value=response)
    async def run():
        with (patch.object(web_footage, "_prepare_candidate_preview", AsyncMock(return_value=prepared)),
              patch.object(web_footage, "run_opencli", ask),
              pytest.raises(web_footage.WebFootageReviewUnavailable) as error):
            await web_footage._analyze_candidate_preview({"title": "Demo"}, "Robots", tmp_path)
        expected_kind = ("transport" if defect in {"transport", "exit"}
                         else "provider_generation" if defect == "provider_generation"
                         else "response_contract")
        assert error.value.failure_kind == expected_kind
        assert "Restore the browser connection" not in str(error.value)
        assert ask.await_count == web_footage._preview_attempt_budget()
    asyncio.run(run())
    assert not (prepared["folder"] / "review.json").exists()
    assert len(list(prepared["folder"].glob("review-attempts/*/attempt-*.json"))) == web_footage._preview_attempt_budget()


@pytest.mark.parametrize("verdict", [
    {**PREVIEW, "suitable": False}, {**PREVIEW, "confidence": .64},
    # Original Silver Lake response recovered from its Gemini conversation:
    # a valid rejection deliberately has no selected interval.
    {**PREVIEW, "suitable": False, "confidence": .68, "selected_window": None},
])
def test_explicit_negative_review_is_not_retried(tmp_path, verdict):
    prepared = prepared_preview(tmp_path)
    async def run():
        with (patch.object(web_footage, "_prepare_candidate_preview", AsyncMock(return_value=prepared)),
              patch.object(web_footage, "run_opencli", AsyncMock(return_value=result(verdict))) as ask,
              pytest.raises(web_footage.WebFootageError, match="review rejected") as error):
            await web_footage._analyze_candidate_preview({"title": "Demo"}, "Robots", tmp_path)
        assert not isinstance(error.value, web_footage.WebFootageReviewUnavailable)
        assert ask.await_count == 1
    asyncio.run(run())
    assert json.loads((prepared["folder"] / "review.json").read_text()) == verdict


def test_batch_preview_recovers_wrapper_and_preserves_independent_rejections(tmp_path):
    prepared = prepared_preview(tmp_path)
    payload = {"results": [{"candidate_id": 1, **PREVIEW, "suitable": False},
                           {"candidate_id": 0, **PREVIEW}]}
    async def run():
        with (patch.object(web_footage, "_prepare_candidate_preview", AsyncMock(return_value=prepared)),
              patch.object(web_footage, "run_opencli", AsyncMock(side_effect=[
                  result({"image_received": True, "reviews": []}), result(payload),
              ])) as ask):
            reviews = await web_footage._analyze_preview_batch([
                ({"source_page_url": "https://youtu.be/a"}, "Robots"),
                ({"source_page_url": "https://youtu.be/b"}, "Robots"),
            ], tmp_path)
        assert ask.await_count == 2
        assert reviews[0]["suitable"] is True
        assert type(reviews[1]) is web_footage.WebFootageError
    asyncio.run(run())


def test_failed_batch_falls_back_to_saved_individual_pixels_and_narration(tmp_path):
    previews = {}
    for index in range(2):
        root = tmp_path / str(index)
        root.mkdir()
        previews[str(index)] = prepared_preview(root)
    requests = [({"source_page_url": str(i), "title": f"Subject {i}"}, f"Narration {i}")
                for i in range(2)]
    responses = [
        OpenCLIResult((), 1, "", "Gemini generation failed"),
        OpenCLIResult((), 1, "", "Gemini generation failed"),
        result("Sorry, something went wrong. Please try your request again."),
        result(PREVIEW), result({**PREVIEW, "suitable": False}),
    ]
    async def run():
        with (patch.object(web_footage, "_prepare_candidate_preview", AsyncMock(
                  side_effect=lambda candidate, *args: previews[candidate["source_page_url"]])) as prepare,
              patch.object(web_footage, "run_opencli", AsyncMock(side_effect=responses)) as ask):
            reviews = await web_footage._analyze_preview_batch(requests, tmp_path)
        assert prepare.await_count == 2
        assert ask.await_count == 5
        assert reviews[0]["suitable"] is True
        assert type(reviews[1]) is web_footage.WebFootageError
        for index, call in enumerate(ask.await_args_list[3:]):
            command = call.args[0]
            assert f"Narration: Narration {index}" in command[2]
            assert f"Narration {1-index}" not in command[2]
            assert command[command.index("--file") + 1] == str(previews[str(index)]["sheet"])
    asyncio.run(run())
    records = [json.loads(p.read_text()) for p in tmp_path.glob("**/review-attempts/*/attempt-*.json")]
    assert len(records) == 5
    assert sum(r["status"] == "reviewed" for r in records) == 2


def test_partial_batch_recovers_only_undecided_rows(tmp_path):
    prepared = prepared_preview(tmp_path)
    batch = {"results": [
        {"candidate_id": 0, **PREVIEW},
        {"candidate_id": 1, **PREVIEW, "suitable": False},
        {"candidate_id": 2, **PREVIEW, "confidence": .4},
    ]}
    async def run():
        with (patch.object(web_footage, "_prepare_candidate_preview", AsyncMock(return_value=prepared)),
              patch.object(web_footage, "run_opencli", AsyncMock(side_effect=[result(batch), result(PREVIEW)])) as ask):
            reviews = await web_footage._analyze_preview_batch([
                ({"source_page_url": str(i)}, f"Narration {i}") for i in range(4)
            ], tmp_path)
        assert ask.await_count == 2
        assert reviews[0]["suitable"] is True
        assert type(reviews[1]) is type(reviews[2]) is web_footage.WebFootageError
        assert reviews[3]["suitable"] is True
        assert "Narration: Narration 3" in ask.await_args_list[1].args[0][2]
    asyncio.run(run())


@pytest.mark.parametrize("budget", [2, 6, 10])
def test_batch_and_individual_failures_share_bounded_budget(tmp_path, budget):
    prepared = prepared_preview(tmp_path)
    async def run():
        with (patch.object(web_footage.config, "WEB_FOOTAGE_PREVIEW_MAX_ATTEMPTS", budget),
              patch.object(web_footage, "_prepare_candidate_preview", AsyncMock(return_value=prepared)),
              patch.object(web_footage, "run_opencli", AsyncMock(side_effect=OpenCLIError("offline"))) as ask):
            reviews = await web_footage._analyze_preview_batch([
                ({"source_page_url": "test"}, "Robots"),
            ], tmp_path)
        assert ask.await_count == budget
        assert isinstance(reviews[0], web_footage.WebFootageReviewUnavailable)
        assert not (prepared["folder"] / "review.json").exists()
    asyncio.run(run())


def test_cancelling_individual_recovery_does_not_start_more_reviews(tmp_path):
    prepared = prepared_preview(tmp_path)
    async def run():
        with (patch.object(web_footage, "_prepare_candidate_preview", AsyncMock(return_value=prepared)),
              patch.object(web_footage, "_request_preview_review", AsyncMock(side_effect=[
                  web_footage.WebFootageReviewUnavailable("offline"), asyncio.CancelledError(),
              ])) as ask):
            with pytest.raises(asyncio.CancelledError):
                await web_footage._analyze_preview_batch([
                    ({"source_page_url": str(i)}, "Robots") for i in range(2)
                ], tmp_path)
        assert ask.await_count == 2
    asyncio.run(run())

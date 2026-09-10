import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from backend.pipeline import multimodal_review as review
from backend.pipeline.opencli import OpenCLIResult
from backend.pipeline.review_response import ReviewResponseError


def result(value):
    return OpenCLIResult((), 0, json.dumps(value), "")


def test_unfinished_reply_recovers_the_original_negative_verdict(tmp_path):
    prompt = 'Review this image.\nReturn JSON.'
    payload = {"image_received": True, "reviews": [{"id": "scene-09", "score": 68,
               "verdict": "partial", "issues": ["Missing CPU thesis"]}]}
    receipt = result([{"conversationId": "6aa23493-b274-83ec-a248-e34aa77a2410",
                       "response": '{"image_received":true,"reviews":'}])
    rows = [{"Role": "User", "Text": prompt.replace('\n', ' ')},
            {"Role": "Assistant", "Text": json.dumps(payload), "Generating": False, "StableSeconds": 12}]
    call = AsyncMock(return_value=result(rows))
    with patch.object(review, "run_opencli", call):
        recovered = asyncio.run(review._review_result_payload(
            receipt, provider="chatgpt", prompt=prompt, sheet=tmp_path / 'sheet.jpg',
            timeout=120, phase="initial", batch_index=2, attempt=1, log=None,
        ))
    assert recovered == payload
    assert call.await_count == 1
    assert call.await_args.args[0][:3] == ["chatgpt", "detail", "6aa23493-b274-83ec-a248-e34aa77a2410"]
    assert (tmp_path / 'response-initial-02-attempt-01-chatgpt-detail.json').is_file()


@pytest.mark.parametrize('defect', ['wrong_prompt', 'generating', 'unstable', 'extra_reply', 'ambiguous', 'missing_id'])
def test_recovery_cannot_change_ownership_or_select_a_favorable_answer(tmp_path, defect):
    payload = {"image_received": True, "reviews": []}
    receipt = {"conversationId": "6aa23493-b274-83ec-a248-e34aa77a2410", "response": '{"image_received":'}
    rows = [{"Role": "User", "Text": 'Review'},
            {"Role": "Assistant", "Text": json.dumps(payload), "Generating": False, "StableSeconds": 12}]
    if defect == 'wrong_prompt': rows[0]['Text'] = 'Another request'
    if defect == 'generating': rows[1]['Generating'] = True
    if defect == 'unstable': rows[1]['StableSeconds'] = 2
    if defect == 'extra_reply': rows.append(dict(rows[1]))
    if defect == 'ambiguous': receipt['response'] = json.dumps(payload) + json.dumps(payload)
    if defect == 'missing_id': receipt.pop('conversationId')
    call = AsyncMock(return_value=result(rows))
    with patch.object(review, "run_opencli", call), pytest.raises(ReviewResponseError):
        asyncio.run(review._review_result_payload(
            result([receipt]), provider="chatgpt", prompt='Review', sheet=tmp_path / 'sheet.jpg',
            timeout=120, phase="initial", batch_index=2, attempt=1, log=None,
        ))
    assert call.await_count == (0 if defect in {'ambiguous', 'missing_id'} else 1)

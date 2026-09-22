"""Plan failures must not turn into permanent, unfilmable search slots."""
import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from backend.pipeline import footage, web_footage


@pytest.mark.parametrize('query', ['Qwen3.8-Max training demo', 'Model4.5 benchmark dashboard', 'GPT-5.2 coding demo'])
def test_query_sanitizer_preserves_exact_versions(query):
    assert footage._sanitize_query(query) == query


def test_plan_parser_extracts_one_contract_and_rejects_conflicting_answers():
    answer = json.dumps({'queries': [{'query': 'chip factory', 'purpose': 'Factory production'}]})
    assert footage._parse_plan('Here is the plan:\n```json\n' + answer + '\n```', None)[0]['query'] == 'chip factory'
    with pytest.raises(ValueError, match='exactly one'):
        footage._parse_plan(answer + '\n' + answer, None)


def test_invalid_plan_gets_one_schema_correction_before_fallback():
    story = 'Workers assemble processors at a chip factory.'
    valid = json.dumps({'queries': [{'query': 'chip factory workers', 'purpose': story, 'script_excerpt': story}]})
    chat = AsyncMock(side_effect=['I would use some factory footage.', valid])
    with (patch.object(footage, '_resolve_provider', AsyncMock(return_value=('endpoint', 'model', 'key'))),
          patch.object(footage, '_chat', chat)):
        plan, planner = asyncio.run(footage.plan_footage_queries(
            title='Test', script=story, count=None, provider_id=None, ai_endpoint=None, ai_model=None,
        ))
    assert planner == 'ai:model'
    assert plan[0]['query'] == 'chip factory workers'
    assert chat.await_count == 2


def test_invalid_correction_is_bounded_and_fallback_keeps_subject_action_order():
    story = 'Acme engineers assemble processors before Bloomberg interviews Reuters analysts.'
    chat = AsyncMock(return_value='No JSON available.')
    with (patch.object(footage, '_resolve_provider', AsyncMock(return_value=('endpoint', 'model', 'key'))),
          patch.object(footage, '_chat', chat)):
        plan, planner = asyncio.run(footage.plan_footage_queries(
            title='Test', script=story, count=None, provider_id=None, ai_endpoint=None, ai_model=None,
        ))
    assert chat.await_count == 2
    assert planner == 'deterministic-fallback'
    assert plan[0]['query'].startswith('Acme engineers assemble processors')


def repair_fixture(tmp_path):
    story = 'Acme claims its model will improve itself. Engineers train models on servers.'
    unrelated = 'Othercorp launches a telephone.'
    old = {'query': 'Acme recursive self-improvement', 'purpose': story,
           'script_excerpt': 'Acme claims its model will improve itself.'}
    kept = {'query': 'telephone factory', 'script_excerpt': unrelated, 'purpose': unrelated}
    clip = {'id': 'clip-01', 'plan_query': kept['query'], 'sha256': 'unchanged', 'analysis': {'suitable': True}}
    manifest = {'queries': [kept, old], 'clips': [clip], 'requested_clip_count': 2,
                'errors': [{'query': old['query'], 'stage': 'web-download-edit', 'message': 'Rejected unrelated talking head'}]}
    proposal = {'query': 'model training servers', 'purpose': 'Illustrative servers used for model training',
                'script_excerpt': 'Engineers train models on servers.'}
    return story + '\n' + unrelated, old, manifest, proposal


def revise(tmp_path, script, old, manifest):
    return asyncio.run(footage._revise_unfilled_shots(
        task_dir=tmp_path, manifest=manifest, pending=[old], script=script,
        provider_id=None, ai_endpoint=None, ai_model=None, log=None,
    ))


def test_shot_revision_keeps_verified_clips_quota_and_audit(tmp_path):
    script, old, manifest, proposal = repair_fixture(tmp_path)
    previous_clip = dict(manifest['clips'][0])
    chat = AsyncMock(return_value=json.dumps({'queries': [proposal]}))
    with (patch.object(footage, '_resolve_provider', AsyncMock(return_value=('endpoint', 'model', 'key'))),
          patch.object(footage, '_chat', chat)):
        assert revise(tmp_path, script, old, manifest) == [proposal]
        assert revise(tmp_path, script, proposal, manifest) == [proposal]
    assert chat.await_count == 1
    assert manifest['clips'] == [previous_clip]
    assert manifest['requested_clip_count'] == 2
    assert manifest['shot_revisions'][0]['previous'] == old
    assert manifest['errors'][0]['query'] == old['query']
    assert footage.read_manifest(tmp_path)['queries'][-1] == proposal


@pytest.mark.parametrize('defect', ['other_story', 'invented', 'fragment', 'empty', 'repeated', 'two'])
def test_unsafe_or_missing_replacement_does_not_waive_slot(tmp_path, defect):
    script, old, manifest, proposal = repair_fixture(tmp_path)
    if defect == 'other_story':
        proposal['script_excerpt'] = 'Othercorp launches a telephone.'
    elif defect == 'invented':
        proposal['script_excerpt'] = 'Acme opens a factory.'
    elif defect == 'fragment':
        proposal['script_excerpt'] = 'train models on servers'
    elif defect == 'repeated':
        proposal['query'] = old['query']
    proposals = [] if defect == 'empty' else [proposal] * (2 if defect == 'two' else 1)
    with (patch.object(footage, '_resolve_provider', AsyncMock(return_value=('endpoint', 'model', 'key'))),
          patch.object(footage, '_chat', AsyncMock(return_value=json.dumps({'queries': proposals})))):
        assert revise(tmp_path, script, old, manifest) == [old]
    assert manifest['queries'][-1] == old
    assert manifest['requested_clip_count'] == 2
    assert 'error' in manifest['shot_revisions'][0]


def test_revised_shot_must_pass_scout_and_outage_never_triggers_revision(tmp_path):
    script, old, manifest, proposal = repair_fixture(tmp_path)
    async def scout(**kwargs):
        assert kwargs['query_plan'] == [proposal]
        # A new search plan alone cannot count as a verified clip.
        assert len(kwargs['manifest']['clips']) == 1
        raise web_footage.WebFootageReviewUnavailable('browser unavailable')
    with (patch.object(footage, '_resolve_provider', AsyncMock(return_value=('endpoint', 'model', 'key'))),
          patch.object(footage, '_chat', AsyncMock(return_value=json.dumps({'queries': [proposal]}))),
          patch.object(web_footage, 'supplement_web_footage', AsyncMock(side_effect=scout))):
        with pytest.raises(web_footage.WebFootageReviewUnavailable):
            asyncio.run(footage._complete_web_plan(
                task_dir=tmp_path, manifest=manifest, query_plan=[old], target_total=2,
                orientation='landscape', script=script, log=None,
                provider_id=None, ai_endpoint=None, ai_model=None,
            ))
    assert len(manifest['clips']) == 1
    assert len(manifest['shot_revisions']) == 1


def test_fresh_exhausted_search_gets_one_new_shot_and_retains_count(tmp_path):
    script, old, manifest, proposal = repair_fixture(tmp_path)
    manifest['errors'] = []
    calls = []
    async def scout(**kwargs):
        calls.append(kwargs['query_plan'])
        result = kwargs['manifest']
        if len(calls) == 1:
            result.update(status='partial', missing_queries=[old])
            result['errors'].append({'query': old['query'], 'stage': 'web-selection'})
        else:
            assert kwargs['query_plan'] == [proposal]
            result['clips'].append({'id': 'clip-02', 'plan_query': proposal['query']})
            result.update(status='ready', missing_queries=[])
        return result
    with (patch.object(footage, '_resolve_provider', AsyncMock(return_value=('endpoint', 'model', 'key'))),
          patch.object(footage, '_chat', AsyncMock(return_value=json.dumps({'queries': [proposal]}))) as chat,
          patch.object(web_footage, 'supplement_web_footage', AsyncMock(side_effect=scout))):
        result = asyncio.run(footage._complete_web_plan(
            task_dir=tmp_path, manifest=manifest, query_plan=[old], target_total=2,
            orientation='landscape', script=script, log=None,
            provider_id=None, ai_endpoint=None, ai_model=None,
        ))
    assert result['status'] == 'ready'
    assert result['requested_clip_count'] == len(result['clips']) == 2
    assert calls == [[old], [proposal]]
    assert chat.await_count == 1


@pytest.mark.parametrize('outage', [web_footage.WebFootageSearchUnavailable, web_footage.WebFootageReviewUnavailable])
def test_fresh_provider_outage_does_not_replan(tmp_path, outage):
    script, old, manifest, _ = repair_fixture(tmp_path)
    manifest['errors'] = []
    with (patch.object(footage, '_revise_unfilled_shots', AsyncMock()) as revise_shots,
          patch.object(web_footage, 'supplement_web_footage', AsyncMock(side_effect=outage('unavailable')))):
        with pytest.raises(outage):
            asyncio.run(footage._complete_web_plan(
                task_dir=tmp_path, manifest=manifest, query_plan=[old], target_total=2,
                orientation='landscape', script=script, log=None,
                provider_id=None, ai_endpoint=None, ai_model=None,
            ))
    revise_shots.assert_not_awaited()

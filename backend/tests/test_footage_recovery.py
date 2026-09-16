"""Regressions for story binding, safe hybrid resume, and independent frame verdicts."""
import asyncio
import hashlib
import json
import sys
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from backend.pipeline import footage, web_footage
from backend.pipeline.opencli import OpenCLIError, OpenCLIResult


@pytest.fixture(autouse=True)
def isolate_search_repair_provider(monkeypatch):
    monkeypatch.setattr(footage, '_resolve_provider', AsyncMock(return_value=('endpoint', 'model', 'key')))
    monkeypatch.setattr(footage, '_chat', AsyncMock(return_value='{"queries":[]}'))


def test_generic_visual_query_stays_in_its_intended_story():
    script = (
        'WAIC CONNECT brought AI companies to an exhibition in Malaysia.\n'
        'Vaire Computing builds chips that recycle energy as heat.\n'
        'NASA helped transform American food inspection. '
        'E. coli in Jack in the Box hamburgers accelerated adoption of HACCP.'
    )
    plan = footage._distinct_grounded_plan([
        {'query': 'technology conference exhibition hall', 'purpose': 'WAIC CONNECT Malaysia AI showcase'},
        {'query': 'silicon wafer microchip closeup', 'purpose': 'Vaire Computing chips recycle heat'},
        {'query': 'hamburger fast food', 'purpose': 'Jack in the Box E. coli outbreak'},
    ], script, None, allow_same_scene=True)
    assert 'WAIC' in plan[0]['script_excerpt']
    assert 'Vaire' in plan[1]['script_excerpt']
    assert plan[2]['script_excerpt'].startswith('E. coli')
    assert 'hamburgers' in plan[2]['script_excerpt']


def test_exact_planner_excerpt_is_preserved_but_wrong_story_excerpt_is_rebound():
    script = 'WAIC CONNECT holds an AI exhibition in Malaysia.\nVaire builds energy recycling chips.'
    raw = json.dumps({'queries': [{
        'query': 'exhibition hall', 'purpose': 'WAIC CONNECT exhibition',
        'script_excerpt': 'Vaire builds energy recycling chips.',
    }]})
    parsed = footage._parse_plan(raw, None)
    assert parsed[0]['script_excerpt'].startswith('Vaire')
    bound = footage._distinct_grounded_plan(parsed, script, None)
    assert bound[0]['script_excerpt'].startswith('WAIC')
    assert footage._distinct_grounded_plan(bound, script, None) == bound


def saved_hybrid(root):
    (root / 'footage').mkdir()
    script = 'WAIC CONNECT hosts an exhibition.\nVaire builds energy recycling chips.'
    queries = [
        {'query': 'exhibition hall', 'purpose': 'WAIC CONNECT exhibition',
         'script_excerpt': 'WAIC CONNECT hosts an exhibition.'},
        {'query': 'microchip closeup', 'purpose': 'Vaire energy recycling chips',
         'script_excerpt': 'Vaire builds energy recycling chips.'},
    ]
    clips = []
    for index, shot in enumerate(queries):
        relative = f'footage/clip-{index + 1:02}.mp4'
        body = f'reviewed content {index}'.encode()
        (root / relative).write_bytes(body)
        clips.append({**shot, 'query': shot['query'] + ' stock footage',
                      'id': f'clip-{index + 1:02}', 'local_path': relative,
                      'sha256': hashlib.sha256(body).hexdigest(),
                      'platform': 'youtube', 'source_page_url': f'https://youtu.be/{index}',
                      'analysis': {'analyzer': 'gemini-web-contact-sheet', 'image_received': True,
                                   'suitable': True, 'confidence': .9, 'visible_content': 'Actual footage'}})
    manifest = {'provider_id': 'hybrid-youtube', 'orientation': 'landscape',
                'selection_mode': 'ai', 'requested_clip_count': 2, 'queries': queries,
                'clips': clips, 'errors': [{'query': 'exhibition hall', 'message': 'old rejection'}]}
    footage._write_manifest(root, manifest)
    return script, manifest


def resume(root, script):
    return footage._resume_web_manifest(root, provider='hybrid', script=script,
                                        orientation='landscape', clip_count=None)


def test_legacy_hybrid_resume_keeps_artifacts_plan_count_and_history(tmp_path):
    script, old = saved_hybrid(tmp_path)
    result = resume(tmp_path, script)
    assert len(result['clips']) == 2
    assert result['requested_clip_count'] == 2
    assert result['errors'] == old['errors']
    assert result['clips'][0]['plan_query'] == 'exhibition hall'
    assert result['binding_version'] == 2
    assert list((tmp_path / 'footage/history').glob('manifest-*.json'))
    again = resume(tmp_path, script)
    assert again['clips'] == result['clips']


@pytest.mark.parametrize('defect', ['checksum', 'missing', 'rejected', 'misbound', 'outside'])
def test_resume_invalidates_only_the_unusable_clip(tmp_path, defect):
    script, manifest = saved_hybrid(tmp_path)
    bad = manifest['clips'][1]
    if defect == 'checksum':
        (tmp_path / bad['local_path']).write_bytes(b'changed')
    elif defect == 'missing':
        (tmp_path / bad['local_path']).unlink()
    elif defect == 'rejected':
        bad['analysis']['suitable'] = False
    elif defect == 'misbound':
        bad['script_excerpt'] = manifest['queries'][0]['script_excerpt']
    elif defect == 'outside':
        bad['local_path'] = '../outside.mp4'
    footage._write_manifest(tmp_path, manifest)
    result = resume(tmp_path, script)
    assert [clip['id'] for clip in result['clips']] == ['clip-01']
    assert result['requested_clip_count'] == 2
    assert result['invalidated_clips'][0]['id'] == 'clip-02'


def test_changed_script_cannot_reuse_previous_review(tmp_path):
    script, _ = saved_hybrid(tmp_path)
    resume(tmp_path, script)
    assert resume(tmp_path, script + ' A newly added claim.') is None


@pytest.mark.parametrize('defect', [None, 'checksum', 'rejected', 'missing_image', 'missing_rights', 'link_only'])
def test_publisher_preview_resume_preserves_review_and_rights_requirements(tmp_path, defect):
    script, manifest = saved_hybrid(tmp_path)
    clip = manifest['clips'][1]
    clip.update(platform='publisher', provider_id='publisher-direct',
                source_page_url='https://example.org/research/demo',
                review_required=True, rights_status='review_required',
                license='Rights not verified — human review required')
    if defect == 'checksum':
        (tmp_path / clip['local_path']).write_bytes(b'changed')
    elif defect == 'rejected':
        clip['analysis']['suitable'] = False
    elif defect == 'missing_image':
        clip['analysis']['image_received'] = False
    elif defect == 'missing_rights':
        clip.pop('rights_status')
    elif defect == 'link_only':
        clip['analysis']['analyzer'] = 'gemini-web'
    footage._write_manifest(tmp_path, manifest)
    result = resume(tmp_path, script)
    assert len(result['clips']) == (1 if defect else 2)
    if not defect:
        assert result['clips'][1]['rights_status'] == 'review_required'
        assert result['clips'][1]['license'] == clip['license']


def test_batch_verdicts_are_mapped_by_id_and_missing_or_duplicate_results_fail(tmp_path):
    async def prepare(candidate, excerpt, task_dir):
        folder = task_dir / candidate['source_page_url']
        folder.mkdir()
        sheet = folder / 'sheet.jpg'
        Image.new('RGB', (160, 90), 'blue').save(sheet)
        return {'folder': folder, 'sheet': sheet,
                'intervals': [{'start_seconds': 5, 'end_seconds': 20}]}

    def verdict(index, **overrides):
        return {'candidate_id': index, 'image_received': True, 'suitable': True,
                'confidence': .9, 'selected_window': 0,
                'visible_content': f'Content for candidate {index}', **overrides}

    async def run():
        requests = [({'source_page_url': str(i), 'title': str(i)}, f'narration {i}') for i in range(4)]
        # Out-of-order results, two contradictory results for 1, no result for 3.
        response = {'results': [verdict(2, confidence=.64), verdict(1), verdict(0), verdict(1, suitable=False)]}
        with (patch.object(web_footage, '_prepare_candidate_preview', AsyncMock(side_effect=prepare)),
              patch.object(web_footage, 'run_opencli', AsyncMock(return_value=OpenCLIResult(
                  (), 0, json.dumps(response), ''))) as ask):
            results = await web_footage._analyze_preview_batch(requests, tmp_path)
        assert ask.await_count == 1
        assert results[0]['visible_content'] == 'Content for candidate 0'
        assert all(isinstance(results[i], Exception) for i in (1, 2, 3))
        assert (tmp_path / results[0]['batch_evidence']).is_file()
    asyncio.run(run())


def test_completed_resume_does_not_replan_or_review_again(tmp_path):
    script, _ = saved_hybrid(tmp_path)
    path = tmp_path / 'script.txt'
    path.write_text(script)
    async def run():
        with (patch.object(footage, 'plan_footage_queries', AsyncMock()) as planner,
              patch.object(web_footage, 'search_youtube', AsyncMock()) as search,
              patch.object(web_footage, '_analyze_preview_batch', AsyncMock()) as review):
            result = await footage.acquire_footage(
                media_provider='hybrid', task_id='test', task_dir=tmp_path, title='test',
                script_path=path, clip_count=None, orientation='landscape', license_policy='open_only',
                provider_id=None, ai_endpoint=None, ai_model=None,
            )
        assert result['status'] == 'ready'
        planner.assert_not_awaited()
        search.assert_not_awaited()
        review.assert_not_awaited()
    asyncio.run(run())


def test_scout_reviews_four_missing_shots_with_one_request(tmp_path):
    plan = [{'query': f'query {i}', 'purpose': f'story {i}', 'script_excerpt': f'Narration {i}.'} for i in range(4)]
    manifest = {'provider_id': 'youtube-web', 'clips': [], 'errors': [], 'url_inspection_unavailable': True}
    async def search(query):
        return [{'source_page_url': f'https://youtu.be/{query[-1]}', 'title': query,
                 'platform': 'youtube', 'duration_seconds': 60}]
    async def download(candidate, raw_dir, analysis):
        path = raw_dir / f"{candidate['source_page_url'][-1]}.mp4"
        path.write_bytes(b'source')
        return path, True
    async def trim(source, target, analysis, orientation):
        target.write_bytes(source.read_bytes() + target.name.encode())
    async def run():
        with (patch.object(web_footage, 'search_youtube', AsyncMock(side_effect=search)),
              patch.object(web_footage, '_analyze_preview_batch', AsyncMock(return_value=[
                  {'start_seconds': 5, 'end_seconds': 20, 'suitable': True, 'confidence': .9,
                   'analyzer': 'test'} for _ in plan])) as review,
              patch.object(web_footage, '_download_youtube', AsyncMock(side_effect=download)),
              patch.object(web_footage, '_trim', AsyncMock(side_effect=trim)),
              patch.object(web_footage, '_probe', AsyncMock(return_value={'duration_seconds': 15, 'width': 1280, 'height': 720})),
              patch.object(web_footage, '_evidence_frames', AsyncMock(return_value=['frame-01.jpg']))):
            result = await web_footage.supplement_web_footage(
                task_dir=tmp_path, manifest=manifest, query_plan=plan, target_total=4,
                orientation='landscape', script=' '.join(item['script_excerpt'] for item in plan),
            )
        assert result['status'] == 'ready'
        assert len(result['clips']) == 4
        assert result['missing_queries'] == []
        assert review.await_count == 1
        assert [entry[1] for entry in review.await_args.args[0]] == [shot['script_excerpt'] for shot in plan]
    asyncio.run(run())


def test_scout_consumes_batch_results_before_reviewing_alternate_queries(tmp_path):
    plan = [{'query': f'query {i}', 'purpose': f'story {i}', 'script_excerpt': f'Narration {i}.'} for i in range(4)]
    query_plan = []
    for shot in plan:
        query_plan.extend([shot, {**shot, 'query': shot['query'] + ' alternate', 'plan_query': shot['query']}])
    manifest = {'provider_id': 'youtube-web', 'clips': [], 'errors': [], 'url_inspection_unavailable': True}
    async def search(query):
        return [{'source_page_url': f'https://youtu.be/{query[-1]}', 'title': query,
                 'platform': 'youtube', 'duration_seconds': 60}]
    async def download(candidate, raw_dir, analysis):
        path = raw_dir / f"{candidate['source_page_url'][-1]}.mp4"
        path.write_bytes(b'source')
        return path, True
    async def trim(source, target, analysis, orientation):
        target.write_bytes(source.read_bytes() + target.name.encode())
    async def review_batch(requests, root, **kwargs):
        return [web_footage.WebFootageError('Preview visual review rejected: wrong scene')
                if candidate['title'] == 'query 0' else
                {'start_seconds': 5, 'end_seconds': 20, 'suitable': True, 'confidence': .9, 'analyzer': 'test'}
                for candidate, _ in requests]
    async def run():
        with (patch.object(web_footage, 'search_youtube', AsyncMock(side_effect=search)),
              patch.object(web_footage, '_analyze_preview_batch', AsyncMock(side_effect=review_batch)) as review,
              patch.object(web_footage, '_download_youtube', AsyncMock(side_effect=download)),
              patch.object(web_footage, '_trim', AsyncMock(side_effect=trim)),
              patch.object(web_footage, '_probe', AsyncMock(return_value={'duration_seconds': 15, 'width': 1280, 'height': 720})),
              patch.object(web_footage, '_evidence_frames', AsyncMock(return_value=['frame-01.jpg']))):
            result = await web_footage.supplement_web_footage(
                task_dir=tmp_path, manifest=manifest, query_plan=query_plan, target_total=4,
                orientation='landscape', script=' '.join(item['script_excerpt'] for item in plan),
            )
        assert result['status'] == 'ready'
        assert len(result['clips']) == 4
        assert result['missing_queries'] == []
        assert [len(call.args[0]) for call in review.await_args_list] == [4, 1]
        assert [entry[1] for entry in review.await_args_list[0].args[0]] == [shot['script_excerpt'] for shot in plan]
    asyncio.run(run())


@pytest.mark.parametrize('approved_index', [2, None])
def test_one_remaining_shot_reviews_three_candidates_once(tmp_path, approved_index):
    plan = [{'query': 'convention centre', 'purpose': 'Narrated venue', 'script_excerpt': 'The venue opened.'}]
    manifest = {'provider_id': 'youtube-web', 'clips': [], 'errors': [], 'url_inspection_unavailable': True}
    candidates = [{'source_page_url': f'https://youtu.be/{i}', 'title': f'Centre {i}',
                   'platform': 'youtube', 'duration_seconds': 60} for i in range(3)]
    async def download(candidate, raw_dir, analysis):
        path = raw_dir / 'source.mp4'
        path.write_bytes(b'source')
        return path, True
    async def trim(source, target, analysis, orientation):
        target.write_bytes(b'trimmed')
    async def review_batch(requests, root, **kwargs):
        assert [entry[0]['source_page_url'] for entry in requests] == [c['source_page_url'] for c in candidates]
        return [({'start_seconds': 5, 'end_seconds': 20, 'suitable': True, 'confidence': .9,
                  'analyzer': 'test'} if i == approved_index else
                 web_footage.WebFootageError('Preview visual review rejected: unrelated event'))
                for i in range(3)]
    async def run():
        with (patch.object(web_footage, 'search_youtube', AsyncMock(return_value=candidates)),
              patch.object(web_footage, '_analyze_preview_batch', AsyncMock(side_effect=review_batch)) as review,
              patch.object(web_footage, '_download_youtube', AsyncMock(side_effect=download)),
              patch.object(web_footage, '_trim', AsyncMock(side_effect=trim)),
              patch.object(web_footage, '_probe', AsyncMock(return_value={'duration_seconds': 15, 'width': 1280, 'height': 720})),
              patch.object(web_footage, '_evidence_frames', AsyncMock(return_value=['frame-01.jpg']))):
            result = await web_footage.supplement_web_footage(
                task_dir=tmp_path, manifest=manifest, query_plan=plan, target_total=1,
                orientation='landscape', script='The venue opened.',
            )
        assert review.await_count == 1
        assert len(result['clips']) == (1 if approved_index is not None else 0)
        assert len(result['rejected_candidates']) == (2 if approved_index is not None else 3)
        if approved_index is not None:
            assert result['clips'][0]['source_page_url'] == candidates[approved_index]['source_page_url']
            assert result['status'] == 'ready'
        else:
            assert result['status'] != 'ready'
    asyncio.run(run())


def test_review_connection_failure_does_not_reject_or_replan_the_candidate(tmp_path):
    plan = [{'query': 'venue building', 'purpose': 'Narrated venue', 'script_excerpt': 'The venue opened.'}]
    manifest = {'provider_id': 'youtube-web', 'clips': [], 'errors': [], 'url_inspection_unavailable': True}
    candidate = {'source_page_url': 'https://youtu.be/venue', 'title': 'Venue building', 'duration_seconds': 30}
    async def run():
        with (patch.object(web_footage, 'search_youtube', AsyncMock(return_value=[candidate])),
              patch.object(web_footage, '_analyze_preview_batch', AsyncMock(return_value=[
                  web_footage.WebFootageReviewUnavailable('OpenCLI connection failed')]))):
            with pytest.raises(web_footage.WebFootageReviewUnavailable, match='previews and verified clips were retained'):
                await web_footage.supplement_web_footage(
                    task_dir=tmp_path, manifest=manifest, query_plan=plan, target_total=1,
                    orientation='landscape', script='The venue opened.',
                )
        assert manifest['status'] == 'review_unavailable'
        assert not manifest.get('rejected_candidates')
        assert not manifest.get('query_replans')
        assert manifest['errors'][0]['stage'] == 'web-review'
        assert 'source_page_url' not in manifest['errors'][0]
    asyncio.run(run())


def test_batch_transport_exception_remains_an_unavailable_review(tmp_path):
    folder = tmp_path / 'prepared'
    folder.mkdir()
    sheet = folder / 'sheet.jpg'
    Image.new('RGB', (160, 90), 'blue').save(sheet)
    prepared = {'folder': folder, 'sheet': sheet,
                'intervals': [{'start_seconds': 3, 'end_seconds': 18}]}
    async def run():
        with (patch.object(web_footage, '_prepare_candidate_preview', AsyncMock(return_value=prepared)),
              patch.object(web_footage, 'run_opencli', AsyncMock(side_effect=OpenCLIError('connection failed')))):
            results = await web_footage._analyze_preview_batch([
                ({'source_page_url': 'https://youtu.be/venue', 'title': 'Venue'}, 'The venue opened.')
            ], tmp_path)
        assert isinstance(results[0], web_footage.WebFootageReviewUnavailable)
    asyncio.run(run())


def test_prepared_preview_reuse_checks_identity_and_every_artifact(tmp_path):
    candidate = {'source_page_url': 'https://youtu.be/venue', 'duration_seconds': 30}
    folder = tmp_path / 'footage/evidence/previews' / hashlib.sha256(candidate['source_page_url'].encode()).hexdigest()[:16]
    window = folder / 'window-0'
    window.mkdir(parents=True)
    for path in (folder / 'contact-sheet.jpg', window / 'frames.jpg'):
        Image.new('RGB', (160, 90), 'blue').save(path)
    (window / 'preview.mp4').write_bytes(b'verified download bytes')
    prepared = {'folder': folder, 'sheet': folder / 'contact-sheet.jpg',
                'intervals': [{'start_seconds': 3, 'end_seconds': 18}]}
    web_footage._save_prepared_preview(candidate, prepared, tmp_path)
    async def run():
        with patch.object(web_footage, '_download_youtube', AsyncMock()) as download:
            result = await web_footage._prepare_candidate_preview(candidate, 'A new narration binding.', tmp_path)
        assert result == prepared
        download.assert_not_awaited()
    asyncio.run(run())
    assert web_footage._load_prepared_preview({**candidate, 'duration_seconds': 31}, folder, tmp_path) is None
    (window / 'preview.mp4').write_bytes(b'tampered')
    assert web_footage._load_prepared_preview(candidate, folder, tmp_path) is None


def test_resume_does_not_blacklist_legacy_browser_transport_failures(tmp_path):
    script, manifest = saved_hybrid(tmp_path)
    manifest['errors'] = [{'stage': 'web-download-edit', 'query': 'query',
                           'source_page_url': 'https://youtu.be/undecided',
                           'message': 'OpenCLI gemini ask failed: sendCommand: max attempts exhausted'}]
    (tmp_path / 'footage/manifest.json').write_text(json.dumps(manifest))
    result = resume(tmp_path, script)
    assert result['errors'][0]['stage'] == 'web-review'
    assert result['errors'][0]['pending_source_page_url'].endswith('/undecided')
    assert 'source_page_url' not in result['errors'][0]


def test_redownload_refreshes_an_existing_raw_interval(tmp_path):
    path = tmp_path / 'video.mp4'
    path.write_bytes(b'old interval')
    async def command(args, **kwargs):
        assert '--force-overwrites' in args
        assert '--no-mtime' in args
        path.write_bytes(b'new requested interval')
        return 0, '', ''
    async def run():
        with patch.object(web_footage, '_run_command', AsyncMock(side_effect=command)):
            media, sectioned = await web_footage._download_youtube(
                {'source_page_url': 'https://youtu.be/video', 'duration_seconds': 60},
                tmp_path, {'start_seconds': 20, 'end_seconds': 35},
            )
        assert media == path and sectioned
        assert media.read_bytes() == b'new requested interval'
    asyncio.run(run())


def test_search_repair_outage_uses_grounded_queries_without_more_provider_calls(tmp_path):
    errors = [{'query': query, 'source_page_url': f'https://youtu.be/{query}{i}'}
              for query in ['generic chip', 'generic cloud'] for i in range(3)]
    manifest = {'provider_id': 'youtube-web', 'clips': [], 'errors': errors}
    plan = [{'query': 'generic chip', 'purpose': 'Vaire Computing energy recycling chips', 'script_excerpt': 'Vaire builds chips.'},
            {'query': 'generic cloud', 'purpose': 'Gimlet Labs multi silicon inference cloud', 'script_excerpt': 'Gimlet builds an inference cloud.'}]
    async def run():
        with (patch.object(footage, '_resolve_provider', AsyncMock(return_value=('endpoint', 'model', 'key'))),
              patch.object(footage, '_chat', AsyncMock(side_effect=TimeoutError)) as chat,
              patch.object(web_footage, 'search_youtube', AsyncMock(return_value=[])) as search):
            result = await web_footage.supplement_web_footage(
                task_dir=tmp_path, manifest=manifest, query_plan=plan, target_total=2,
                orientation='landscape', script='Vaire builds chips. Gimlet builds an inference cloud.',
            )
        assert chat.await_count == 1
        assert any(call.args[0].startswith('Vaire Computing') for call in search.await_args_list)
        assert any(call.args[0].startswith('Gimlet Labs') for call in search.await_args_list)
        assert result['status'] == 'no_results'
        assert len(result['missing_queries']) == 2
        assert all(item['fallback'] == 'grounded-purpose' for item in result['query_replans'].values())
    asyncio.run(run())


def test_search_metadata_rejects_single_generic_anchor_before_visual_review():
    weak = {'title': 'Virtual Conference Showcase', 'description': 'An unrelated event',
            'query_relevance_score': 111}
    assert not web_footage._metadata_matches_query(weak, 'WAIC CONNECT Malaysia gathering AI showcase')
    strong = {'title': 'WAIC CONNECT in Malaysia', 'query_relevance_score': 300}
    assert web_footage._metadata_matches_query(strong, 'WAIC CONNECT Malaysia gathering AI showcase')
    assert web_footage._metadata_matches_query(
        {'title': 'E. Coli Outbreak Investigation', 'query_relevance_score': 300},
        'E coli outbreak hamburgers',
    )


@pytest.mark.parametrize('excerpt,expected', [
    ('Bloomberg reported on September 15 that Altera, a chipmaker backed by Silver Lake and Intel, '
     'has filed confidentially for an initial public offering.', 'Altera chipmaker'),
    ("The Chinese-language outlet Machine Heart relays a Quanta Magazine interview with Sarah Demers, "
     "Yale's physics chair and a participant in the Mu2e experiment at Fermi National Accelerator Laboratory.",
     'Mu2e experiment'),
    ("For example, the firm says Tomahawk cruise missiles run roughly two to four million dollars each, "
     "while commercially built alternatives like Anduril's Barracuda family cost about one-fifth to one-tenth as much.",
     'Tomahawk cruise missiles'),
])
def test_outage_queries_keep_narrated_entities_without_publisher_or_date(excerpt, expected):
    shot = {'script_excerpt': excerpt, 'purpose': 'One physicist or a September essay is in the news.'}
    queries = footage._fallback_search_queries(shot, excluded=set())
    assert expected in queries
    assert all(2 <= len(query.split()) <= 6 for query in queries)
    assert not any(word in ' '.join(queries) for word in ['Bloomberg', 'September', 'Machine Heart', 'Quanta'])
    next_queries = footage._fallback_search_queries(shot, excluded={q.casefold() for q in queries})
    assert not set(queries) & set(next_queries)


def test_outage_retry_excludes_failed_fallbacks_and_skips_fulfilled_plans(tmp_path):
    shot = {'query': 'defense essay', 'purpose': 'September essay venture firm Andreessen Horowitz',
            'script_excerpt': "Tomahawk cruise missiles cost millions while Anduril's Barracuda family costs less."}
    key = hashlib.sha256((shot['query'] + '\n' + shot['script_excerpt']).encode()).hexdigest()[:16]
    errors = [{'query': shot['query'], 'plan_query': shot['query'], 'stage': 'web-download-edit',
               'source_page_url': f'https://youtu.be/old{i}'} for i in range(3)]
    previous = {'plan_query': shot['query'], 'queries': ['September essay venture'],
                'error': 'TimeoutError', 'fallback': 'grounded-purpose'}
    errors.append({'query': 'September essay venture', 'plan_query': shot['query'], 'stage': 'web-selection'})
    # Even exhausted discovery on an already fulfilled story must not call a provider.
    errors.extend({**error, 'query': 'verified shot', 'plan_query': 'verified shot'} for error in errors[:3])
    retained = {'id': 'clip-01', 'plan_query': 'verified shot'}
    manifest = {'clips': [retained], 'errors': errors, 'query_replans': {key: previous}}

    async def run():
        with (patch.object(footage, '_chat', AsyncMock(side_effect=TimeoutError)) as chat,
              patch.object(web_footage, 'search_youtube', AsyncMock(return_value=[])) as search):
            result = await web_footage.supplement_web_footage(
                task_dir=tmp_path, manifest=manifest, query_plan=[{'query': 'verified shot'}, shot],
                target_total=2, orientation='landscape', script=shot['script_excerpt'],
            )
        assert chat.await_count == 1
        assert result['clips'] == [retained]
        assert result['query_replan_history'] == [{'key': key, **previous}]
        searched = {call.args[0] for call in search.await_args_list}
        assert 'Tomahawk cruise missiles' in searched
        assert 'September essay venture' not in searched
        assert 'verified shot' not in searched
    asyncio.run(run())


@pytest.mark.parametrize('label', ['(Audio Only)', '| Audio Only', '[audio-only]'])
def test_explicit_audio_only_source_never_downloads_or_consumes_a_visual_review(tmp_path, label):
    candidate = {'title': f'SpaceX IPO {label}', 'source_page_url': 'https://youtu.be/audio',
                 'query_relevance_score': 300}
    async def run():
        with (patch.object(web_footage, 'search_youtube', AsyncMock(return_value=[candidate])),
              patch.object(web_footage, '_analyze_preview_batch', AsyncMock()) as review,
              patch.object(web_footage, '_prepare_candidate_preview', AsyncMock()) as prepare):
            result = await web_footage.supplement_web_footage(
                task_dir=tmp_path, manifest={'clips': [], 'errors': [], 'url_inspection_unavailable': True},
                query_plan=[{'query': 'SpaceX IPO', 'purpose': 'SpaceX public debut',
                             'script_excerpt': 'SpaceX made its public debut.'}],
                target_total=1, orientation='landscape', script='SpaceX made its public debut.',
            )
        assert not result['clips']
        assert any('Audio Only' in error['message'] for error in result['errors'])
        prepare.assert_not_awaited()
        review.assert_not_awaited()
    asyncio.run(run())


def test_podcast_or_audio_topic_alone_still_requires_pixel_review():
    for title in ['SpaceX IPO podcast interview', 'SpaceX IPO audio only workflow demo']:
        assert web_footage._metadata_rejection({'title': title}, 'SpaceX IPO') is None


@pytest.mark.parametrize('failure_stage', ['web-download-edit', 'web-selection', 'web-review'])
def test_retry_refreshes_failed_alternatives_but_preserves_pending_review(tmp_path, failure_stage):
    shot = {'query': 'students classroom lesson', 'purpose': 'Students learning basic skills',
            'script_excerpt': 'Technology affects students learning basic skills.'}
    key = hashlib.sha256((shot['query'] + '\n' + shot['script_excerpt']).encode()).hexdigest()[:16]
    original_errors = [{'query': shot['query'], 'plan_query': shot['query'],
                        'source_page_url': f'https://youtu.be/old{i}',
                        'stage': 'web-download-edit', 'message': 'Talking head'} for i in range(3)]
    previous = {'plan_query': shot['query'], 'queries': ['students taking Pisa test'],
                'failures': original_errors.copy()}
    failure = {'query': previous['queries'][0], 'plan_query': shot['query'],
               'script_excerpt': shot['script_excerpt'], 'stage': failure_stage,
               'message': 'No classroom frames' if failure_stage != 'web-review' else 'Browser disconnected'}
    retained = {'id': 'verified-clip', 'plan_query': 'other shot'}
    manifest = {'provider_id': 'hybrid-youtube', 'clips': [retained],
                'errors': [*original_errors, failure], 'query_replans': {key: previous}}

    async def run():
        with (patch.object(footage, '_chat', AsyncMock(return_value=json.dumps({
                  'queries': ['students taking Pisa test', 'students classroom writing exercise']}))) as chat,
              patch.object(web_footage, 'search_youtube', AsyncMock(return_value=[])) as search,
              patch.object(web_footage, '_analyze_preview_batch', AsyncMock()) as review):
            result = await web_footage.supplement_web_footage(
                task_dir=tmp_path, manifest=manifest, query_plan=[shot], target_total=2,
                orientation='landscape', script=shot['script_excerpt'],
            )
        assert result['clips'] == [retained]
        assert result['status'] == 'partial'
        assert failure in result['errors']
        review.assert_not_awaited()
        if failure_stage == 'web-review':
            chat.assert_not_awaited()
            assert not result.get('query_replan_history')
        else:
            assert chat.await_count == 1
            assert result['query_replan_history'] == [{'key': key, **previous}]
            assert result['query_replans'][key]['queries'] == ['students classroom writing exercise']
            assert all(call.args[0] != 'students taking Pisa test' for call in search.await_args_list)
            assert all(error.get('plan_query') == shot['query'] for error in result['errors']
                       if error.get('stage') == 'web-selection')
            prompt = json.loads(chat.await_args.args[1])
            assert prompt['previous_queries_to_avoid'] == ['students taking pisa test']
    asyncio.run(run())


@pytest.mark.parametrize('tampered', [False, True])
def test_approved_preview_is_reused_without_redownload_or_evidence_deletion(tmp_path, tampered):
    preview = tmp_path / 'footage/evidence/approved.mp4'
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b'reviewed video pixels')
    analysis = {'start_seconds': 30, 'end_seconds': 45, 'suitable': True, 'confidence': .9,
                'analyzer': 'gemini-web-contact-sheet', 'reviewed_preview_path': 'footage/evidence/approved.mp4',
                'reviewed_preview_sha256': hashlib.sha256(preview.read_bytes()).hexdigest()}
    if tampered:
        preview.write_bytes(b'changed after review')
    async def trim(source, target, selection, orientation):
        assert selection == {'start_seconds': 0.0, 'end_seconds': 15}
        target.write_bytes(source.read_bytes())
    async def run():
        with (patch.object(web_footage, 'search_youtube', AsyncMock(return_value=[{
                'source_page_url': 'https://youtu.be/test', 'duration_seconds': 60}])),
              patch.object(web_footage, '_analyze_preview_batch', AsyncMock(return_value=[analysis])),
              patch.object(web_footage, '_download_youtube', AsyncMock()) as download,
              patch.object(web_footage, '_trim', AsyncMock(side_effect=trim)),
              patch.object(web_footage, '_probe', AsyncMock(return_value={'duration_seconds': 15, 'width': 1280, 'height': 720})),
              patch.object(web_footage, '_evidence_frames', AsyncMock(return_value=['frame-01.jpg']))):
            result = await web_footage.supplement_web_footage(
                task_dir=tmp_path, manifest={'provider_id': 'youtube-web', 'clips': [], 'errors': [], 'url_inspection_unavailable': True},
                query_plan=[{'query': 'chip factory', 'script_excerpt': 'A chip factory.'}],
                target_total=1, orientation='landscape', script='A chip factory.',
            )
        download.assert_not_awaited()
        assert preview.is_file()
        assert bool(result['clips']) is not tampered
        if not tampered:
            assert (tmp_path / result['clips'][0]['local_path']).read_bytes() == preview.read_bytes()
    asyncio.run(run())


def test_legacy_migration_keeps_the_narration_used_by_an_old_rejection(tmp_path):
    script, manifest = saved_hybrid(tmp_path)
    manifest['queries'][0]['script_excerpt'] = manifest['queries'][1]['script_excerpt']
    manifest['errors'] = [{'query': 'exhibition hall stock footage', 'source_page_url': 'https://youtu.be/old', 'message': 'Not a chip factory'}]
    footage._write_manifest(tmp_path, manifest)
    result = resume(tmp_path, script)
    assert result['queries'][0]['script_excerpt'].startswith('WAIC')
    assert result['errors'][0]['script_excerpt'].startswith('Vaire')


def test_legacy_normalized_clip_metadata_uses_the_saved_render_file(tmp_path):
    path = tmp_path / "footage" / "clip-01-render.mp4"
    path.parent.mkdir()
    path.write_bytes(b"render-video")
    manifest = {"clips": [{"id": "clip-01", "local_path": "footage/clip-01-render.mp4", "duration_seconds": 38.9, "width": 1920, "height": 1280, "render_safe": True}]}
    probe = AsyncMock(return_value={"duration_seconds": 19.34, "width": 1620, "height": 1080})
    with patch.object(web_footage, "_probe", probe):
        result = asyncio.run(footage.normalize_manifest_clips(tmp_path, manifest))
        asyncio.run(footage.normalize_manifest_clips(tmp_path, result))
    probe.assert_awaited_once_with(path.resolve())
    clip = result["clips"][0]
    assert clip["source_duration_seconds"] == 38.9
    assert clip["duration_seconds"] == 19.34
    assert clip["height"] == 1080
    assert path.read_bytes() == b"render-video"


@pytest.mark.parametrize('repaired, legacy_cache', [(False, True), (True, True), (True, False)])
def test_cached_candidate_is_reviewed_before_downloading_spare_alternatives(tmp_path, repaired, legacy_cache):
    candidates = [{'source_page_url': f'https://youtu.be/port-{i}', 'title': 'Los Angeles port containers', 'duration_seconds': 90} for i in range(3)]
    original_query = 'Los Angeles port containers'
    current_query = 'Los Angeles container loading' if repaired else original_query
    retained = {**candidates[2], 'visual_query': original_query, 'visual_purpose': ''}
    if not legacy_cache:
        retained.update(visual_query='Los Angeles dockside cranes', visual_plan_query=original_query)
    cache_path = tmp_path / 'footage/evidence/previews/retained/preview-cache.json'
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(json.dumps({'candidate': retained}))
    review = AsyncMock(return_value=[web_footage.WebFootageReviewUnavailable('Review offline')])
    with (patch.object(web_footage, 'search_youtube', AsyncMock(return_value=candidates[:2])) as search,
          patch.object(web_footage, '_load_prepared_preview', side_effect=lambda candidate, *args: {'sheet': 'cached.jpg'} if candidate['source_page_url'] == candidates[2]['source_page_url'] else None),
          patch.object(web_footage, '_analyze_preview_batch', review),
          pytest.raises(web_footage.WebFootageReviewUnavailable, match='Review offline')):
        asyncio.run(web_footage.supplement_web_footage(
            task_dir=tmp_path, manifest={'provider_id': 'youtube-web', 'clips': [], 'errors': [], 'url_inspection_unavailable': True},
            query_plan=[{'query': current_query, 'plan_query': original_query,
                         'script_excerpt': 'A Southern California port.'}],
            target_total=1, orientation='landscape', script='A Southern California port.',
        ))
    search.assert_awaited_once()
    assert len(review.await_args.args[0]) == 1
    assert review.await_args.args[0][0][0]['source_page_url'] == candidates[2]['source_page_url']
    assert review.await_args.args[0][0][0]['visual_query'] == current_query
    assert review.await_args.args[0][0][0]['visual_plan_query'] == original_query
    assert review.await_args.args[0][0][1] == 'A Southern California port.'
    result = json.loads((tmp_path / 'footage/manifest.json').read_text())
    assert result['status'] == 'review_unavailable'
    assert not any(error.get('source_page_url') for error in result['errors'])


def test_preview_preparation_overlaps_two_sources_and_isolates_failures(tmp_path):
    async def run():
        active = set()
        peak = 0
        started = asyncio.Event()
        release = asyncio.Event()

        async def prepare(candidate, excerpt, task_dir):
            nonlocal peak
            url = candidate['source_page_url']
            assert url not in active
            active.add(url)
            peak = max(peak, len(active))
            if len(active) == 2:
                started.set()
            try:
                await release.wait()
                if url == 'failed':
                    raise web_footage.WebFootageError('download failed')
                folder = task_dir / url
                folder.mkdir(exist_ok=True)
                sheet = folder / 'sheet.jpg'
                Image.new('RGB', (160, 90), 'blue').save(sheet)
                return {'folder': folder, 'sheet': sheet,
                        'intervals': [{'start_seconds': 5, 'end_seconds': 20}]}
            finally:
                active.remove(url)

        async def review(prompt, sheet, prepared, **kwargs):
            assert not active
            assert list(prepared) == [0, 2, 3]
            return {'results': [
                {'candidate_id': i, 'image_received': True, 'suitable': True,
                 'confidence': .9, 'selected_window': 0,
                 'visible_content': f'Actual content {i}', 'reason': 'Relevant'}
                for i in prepared
            ]}

        requests = [({'source_page_url': url}, f'narration {i}')
                    for i, url in enumerate(['same', 'failed', 'same', 'last'])]
        with (patch.object(web_footage, '_prepare_candidate_preview', side_effect=prepare),
              patch.object(web_footage, '_request_preview_review', side_effect=review) as ask):
            job = asyncio.create_task(web_footage._analyze_preview_batch(requests, tmp_path))
            try:
                await asyncio.wait_for(started.wait(), timeout=2)
                assert peak == 2
            finally:
                release.set()
            results = await job
        assert peak == 2
        assert ask.await_count == 1
        assert str(results[1]) == 'download failed'
        assert [results[i]['visible_content'] for i in (0, 2, 3)] == [
            'Actual content 0', 'Actual content 2', 'Actual content 3',
        ]
    asyncio.run(run())


def test_cancelling_preview_batch_drains_downloads_before_returning(tmp_path):
    async def run():
        active = set()
        started = asyncio.Event()

        async def prepare(candidate, *args):
            url = candidate['source_page_url']
            active.add(url)
            if len(active) == 2:
                started.set()
            try:
                await asyncio.Event().wait()
            finally:
                active.remove(url)

        with (patch.object(web_footage, '_prepare_candidate_preview', side_effect=prepare),
              patch.object(web_footage, '_request_preview_review', AsyncMock()) as ask):
            job = asyncio.create_task(web_footage._analyze_preview_batch(
                [({'source_page_url': str(i)}, 'narration') for i in range(4)], tmp_path))
            await asyncio.wait_for(started.wait(), timeout=2)
            job.cancel()
            with pytest.raises(asyncio.CancelledError):
                await job
        assert not active
        ask.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize('stop', ['cancel', 'timeout'])
def test_stopped_footage_command_reaps_its_process(stop):
    async def run():
        created = []
        ready = asyncio.Event()
        spawn = asyncio.create_subprocess_exec

        async def track(*args, **kwargs):
            process = await spawn(*args, **kwargs)
            created.append(process)
            ready.set()
            return process

        with patch.object(web_footage.asyncio, 'create_subprocess_exec', side_effect=track):
            job = asyncio.create_task(web_footage._run_command(
                [sys.executable, '-c', 'import time; time.sleep(30)'],
                timeout=0.05 if stop == 'timeout' else 30,
            ))
            await asyncio.wait_for(ready.wait(), timeout=2)
            if stop == 'cancel':
                job.cancel()
            with pytest.raises(asyncio.CancelledError if stop == 'cancel' else web_footage.WebFootageError):
                await job
        assert created[0].returncode is not None
        assert created[0].returncode != 0
    asyncio.run(run())

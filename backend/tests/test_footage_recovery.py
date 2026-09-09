"""Regressions for story binding, safe hybrid resume, and independent frame verdicts."""
import asyncio
import hashlib
import json
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

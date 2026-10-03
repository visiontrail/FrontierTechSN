"""Preserve evidence framing and equal-weight assertions in directed shots."""

import re

import pytest

from backend.pipeline import scene_kit
from backend.pipeline.video_format import LANDSCAPE, PORTRAIT


@pytest.mark.parametrize('frame', [LANDSCAPE, PORTRAIT])
def test_article_uses_separate_evidence_and_copy_regions(frame):
    plan = scene_kit.ScenePlan(
        id='scene-07', duration=12, headline='Source report', frame=frame,
        media_shots=({'kind': 'article', 'src': '../source.png', 'start': 0,
                      'duration': 12, 'copy': ['The qualified source claim.'],
                      'layout': 'focus', 'credit': 'Source'},),
    )
    html = scene_kit.render_scene(plan)
    assert 'class="clip shot-panel visual article-panel"' in html
    assert '.shot-media.article { width:58%; object-fit:contain;' in html
    assert '.shot-panel.article-panel { left:58%;' in html
    assert '#scene-07-shot-1-media", {scale:' not in html
    assert 'The qualified source claim.' in html and '>Source</div>' in html
    if frame == PORTRAIT:
        assert '.shot-media.article { width:100%; height:52%;' in html
        assert '.shot-panel.article-panel { left:0; top:52%;' in html


@pytest.mark.parametrize('long_side', [0, 1])
def test_paired_copy_uses_one_density_for_both_assertions(long_side):
    copy = ['First qualified assertion.', 'Second qualified assertion.']
    copy[long_side] *= 5
    html = scene_kit.render_scene(scene_kit.ScenePlan(
        id='scene-02', duration=10, headline='Comparison',
        media_shots=({'kind': 'editorial', 'start': 0, 'duration': 10,
                      'copy': copy, 'layout': 'cards'},),
    ))
    assert 'class="shot-grid cards shot-pair" data-pair-density="dense"' in html
    # Both children inherit the same size; the dense first-title exception
    # previously overrode the right card even when pair-wide sizing was set.
    assert '.shot-pair-title.dense .shot-title' not in html
    shared = re.search(r'\.shot-panel \.shot-pair\[data-pair-density\] \.shot-title,.*?\{([^}]+)\}', html)
    assert shared and 'font-size:var(--pair-copy-size)' in shared[1]
    assert 'font-weight:700' in shared[1]
    assert all(text in html for text in copy)

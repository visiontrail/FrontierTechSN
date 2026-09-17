"""Live search results must never start an unbounded footage download."""
import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from yt_dlp.utils import match_filter_func

from backend.pipeline import web_footage


@pytest.mark.parametrize('state', [
    {'is_live': True}, {'is_upcoming': True}, {'live_status': 'is_live'},
    {'live_status': 'is_upcoming'}, {'live_status': 'post_live'},
])
def test_live_candidates_are_rejected_before_download(tmp_path, state):
    candidate = {'title': 'White House LIVE', 'source_page_url': 'https://youtu.be/live', **state}
    assert 'broadcast' in web_footage._metadata_rejection(candidate, 'White House')
    with patch.object(web_footage, '_run_command', AsyncMock()) as command:
        with pytest.raises(web_footage.WebFootageError, match='broadcast'):
            asyncio.run(web_footage._download_youtube(candidate, tmp_path,
                        {'start_seconds': 0, 'end_seconds': 15}))
    command.assert_not_awaited()


def test_search_preserves_broadcast_state_for_selection():
    result = json.dumps({'url': 'https://youtu.be/live', 'title': 'White House LIVE',
                         'is_live': True, 'live_status': 'is_live'})
    with patch.object(web_footage, '_run_command', AsyncMock(return_value=(0, result, ''))):
        candidates = asyncio.run(web_footage.search_youtube('White House'))
    assert candidates[0]['is_live'] is True
    assert candidates[0]['live_status'] == 'is_live'


@pytest.mark.parametrize('state,rejected', [
    ({}, False), ({'is_live': True}, True), ({'is_upcoming': True}, True),
    ({'live_status': 'post_live'}, True), ({'live_status': 'was_live'}, False),
    ({'live_status': 'not_live'}, False),
])
def test_downloader_filter_blocks_unfinished_but_allows_archived_broadcasts(state, rejected):
    # Exercise the installed downloader's parser, not a hand-written imitation.
    predicate = match_filter_func(web_footage.YOUTUBE_FINISHED_VIDEO_ARGS[1])
    assert bool(predicate(state, incomplete=False)) is rejected


def test_live_word_in_archived_title_is_not_a_rejection():
    assert web_footage._metadata_rejection(
        {'title': 'White House LIVE briefing replay', 'live_status': 'was_live'}, 'White House'
    ) is None

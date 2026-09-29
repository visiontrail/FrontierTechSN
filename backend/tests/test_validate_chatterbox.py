"""Validate request identity without synthesizing or rendering media."""

import pytest

from backend import config
from scripts import validate_chatterbox as validation


@pytest.fixture
def preparation(monkeypatch):
    monkeypatch.setattr(validation, "require_profile", lambda: {"width": 3840, "height": 2160, "fps": 30})
    monkeypatch.setattr(validation.music, "resolve_program_music_track", lambda track: None)
    monkeypatch.setattr(validation.outros, "resolve_outro_preset", lambda preset: None)
    monkeypatch.setattr(config, "TTS_DEFAULT_VOICE_1", "Carter")


@pytest.mark.parametrize("model, expected", [("chatterbox-multilingual", "space-en-f1"), ("pocket-tts-en", "alba")])
def test_new_validation_uses_selected_model_default(tmp_path, preparation, model, expected):
    request = validation.prepare(tmp_path, model=model, voice=None, track="morning-briefing", captions=False)
    assert request["voice"] == expected
    assert validation.read_json(tmp_path / "validation_request.json")["voice"] == expected


def test_resume_preserves_existing_validation_voice_after_model_default_change(tmp_path, preparation):
    request = validation.prepare(tmp_path, model="chatterbox-multilingual", voice="jane", track="morning-briefing", captions=False)
    marker = (tmp_path / "validation_request.json").read_bytes()
    resumed = validation.prepare(tmp_path, model="chatterbox-multilingual", voice=None, track="morning-briefing", captions=False)
    assert resumed == request
    assert resumed["voice"] == "jane"
    assert (tmp_path / "validation_request.json").read_bytes() == marker
    with pytest.raises(ValueError, match="another request"):
        validation.prepare(tmp_path, model="chatterbox-multilingual", voice="space-en-f1", track="morning-briefing", captions=False)
    assert (tmp_path / "validation_request.json").read_bytes() == marker

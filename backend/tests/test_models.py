import pytest
from pydantic import ValidationError

from backend.models import DEFAULT_CLOSING_REMARKS, TaskConfig


def test_captions_are_off_by_default():
    assert TaskConfig().captions_enabled is False


def test_audio_review_is_skipped_by_default():
    assert TaskConfig().auto_render is True


def test_task_has_a_spoken_closing_by_default():
    assert TaskConfig().closing_remarks == DEFAULT_CLOSING_REMARKS


def test_closing_remarks_are_trimmed_and_cannot_be_blank():
    assert TaskConfig(closing_remarks="  Thanks for watching.  ").closing_remarks == "Thanks for watching."
    with pytest.raises(ValidationError, match="Closing remarks cannot be blank"):
        TaskConfig(closing_remarks=" \n ")


def test_task_config_defaults_to_landscape_and_four_collages():
    config = TaskConfig()

    assert config.video_orientation == "landscape"
    assert config.footage_orientation == "landscape"
    assert config.collage_broll_enabled is True
    assert config.collage_broll_count == 4
    assert config.opening_style == "editorial_motion"


def test_task_orientation_controls_legacy_footage_orientation():
    config = TaskConfig(video_orientation="portrait", footage_orientation="landscape")

    assert config.video_orientation == "portrait"
    assert config.footage_orientation == "portrait"


def test_legacy_only_portrait_orientation_is_promoted():
    config = TaskConfig(footage_orientation="portrait")

    assert config.video_orientation == "portrait"
    assert config.footage_orientation == "portrait"


def test_tts_uses_the_fast_model_by_default():
    assert TaskConfig().tts_model == "vibevoice-0.5b"


def test_orpheus_accepts_only_its_own_voices():
    config = TaskConfig(tts_model="orpheus-en", voice_1="tara")
    assert config.voice_1 == "tara"
    with pytest.raises(ValidationError, match="unavailable"):
        TaskConfig(tts_model="orpheus-en", voice_1="Carter")


def test_orpheus_is_monologue_only():
    with pytest.raises(ValidationError, match="monologue only"):
        TaskConfig(
            tts_model="orpheus-en",
            script_format="dialogue",
            voice_1="tara",
            voice_2="leah",
        )

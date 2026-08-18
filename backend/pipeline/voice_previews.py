"""Persistent preview generation for voices backed by remote TTS providers."""

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

from backend import config
from backend.pipeline.tts import _generate_orpheus

logger = logging.getLogger(__name__)

PREVIEW_TEXT = "Hello, this is a short preview of my voice."
PREVIEW_MAX_TOKENS = 1024
_locks: dict[tuple[str, str], asyncio.Lock] = {}


async def ensure_voice_preview(voice: str, tts_model: str) -> Path:
    """Return a cached preview, generating it remotely exactly once if absent."""
    model = config.TTS_MODELS.get(tts_model)
    if model is None or voice not in config.voices_for_model(tts_model):
        raise ValueError(f"Unknown voice '{voice}' for TTS model '{tts_model}'")

    existing = config.voice_sample_path(voice, tts_model)
    if existing is not None:
        return existing
    if model.get("kind") != "orpheus_http":
        raise FileNotFoundError(f"No preview sample installed for '{voice}'")

    key = (tts_model, voice)
    lock = _locks.setdefault(key, asyncio.Lock())
    async with lock:
        existing = config.voice_sample_path(voice, tts_model)
        if existing is not None:
            return existing

        destination = config.VOICE_PREVIEW_CACHE_DIR / tts_model / f"{voice}.wav"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temp_dir:
            work_dir = Path(temp_dir)
            script = work_dir / "preview.txt"
            script.write_text(PREVIEW_TEXT, encoding="utf-8")
            generated = await _generate_orpheus(
                str(script),
                str(work_dir / "audio"),
                voice,
                str(model.get("language") or "en"),
                log=None,
                emit=logger.info,
                max_tokens=PREVIEW_MAX_TOKENS,
            )
            # Copy to a sibling temporary file and atomically publish it, so a
            # concurrent list/read can never observe a partial WAV.
            staged = destination.with_suffix(".wav.tmp")
            shutil.copyfile(generated, staged)
            staged.replace(destination)
        return destination


async def preload_remote_voice_previews() -> None:
    """Warm every missing remote preview in the background at app startup."""
    if not config.ORPHEUS_TTS_API_KEY:
        logger.info("Skipping Orpheus voice preview preload: API key is not configured")
        return
    for model_id, model in config.TTS_MODELS.items():
        if model.get("kind") != "orpheus_http":
            continue
        for voice in config.voices_for_model(model_id):
            if config.voice_sample_path(voice, model_id) is not None:
                continue
            try:
                await ensure_voice_preview(voice, model_id)
                logger.info("Cached %s preview for %s", model_id, voice)
            except asyncio.CancelledError:
                raise
            except Exception:
                # One unavailable voice must not prevent the remaining voices
                # or the application itself from starting.
                logger.exception("Could not preload %s preview for %s", model_id, voice)

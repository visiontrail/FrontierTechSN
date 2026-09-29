"""Persistent preview generation for voices backed by remote TTS providers."""

import asyncio
import hashlib
import json
import logging
import shutil
import tempfile
from contextlib import nullcontext
from pathlib import Path

from backend import config
from backend.pipeline.tts import _generate_orpheus, _generate_pocket_tts

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
    if not model.get("remote_preview"):
        raise FileNotFoundError(f"No preview sample installed for '{voice}'")

    key = (tts_model, voice)
    lock = _locks.setdefault(key, asyncio.Lock())
    async with lock:
        existing = config.voice_sample_path(voice, tts_model)
        if existing is not None:
            return existing

        destination = config.voice_preview_cache_dir(tts_model) / f"{voice}.wav"
        destination.parent.mkdir(parents=True, exist_ok=True)
        contract = None
        if model.get("kind") == "chatterbox_http":
            from backend.pipeline.chatterbox import cache_contract

            contract = cache_contract()
            # Preserve the submitted job identity across timeout, cancellation
            # and process restarts; an uncertain remote job must be resumed.
            work = destination.parent / f".{voice}-work"
            work.mkdir(parents=True, exist_ok=True)
            workspace = nullcontext(str(work))
        else:
            workspace = tempfile.TemporaryDirectory(dir=destination.parent)
        with workspace as temp_dir:
            work_dir = Path(temp_dir)
            script = work_dir / "preview.txt"
            script.write_text(PREVIEW_TEXT, encoding="utf-8")
            if model.get("kind") == "orpheus_http":
                generated = await _generate_orpheus(
                    str(script),
                    str(work_dir / "audio"),
                    voice,
                    str(model.get("language") or "en"),
                    log=None,
                    emit=logger.info,
                    max_tokens=PREVIEW_MAX_TOKENS,
                )
            elif model.get("kind") == "pocket_tts_http":
                generated = await _generate_pocket_tts(
                    str(script),
                    str(work_dir / "audio"),
                    voice,
                    str(model.get("language") or "en"),
                    log=None,
                    emit=logger.info,
                )
            elif model.get("kind") == "chatterbox_http":
                from backend.pipeline.chatterbox import _generate_chatterbox

                generated = await _generate_chatterbox(
                    str(script),
                    str(work_dir / "audio"),
                    voice,
                    str(model.get("language") or "en"),
                    log=None,
                    emit=logger.info,
                    verify_text=False,
                )
            else:
                raise FileNotFoundError(f"No remote preview generator for '{tts_model}'")
            if contract is not None:
                manifest = json.loads((work_dir / "audio" / "tts_manifest.json").read_text(encoding="utf-8"))
                if (
                    config.voice_preview_cache_dir(tts_model) != destination.parent
                    or cache_contract() != contract
                    or manifest.get("cache_contract") != contract
                    or manifest.get("voice") != voice
                    or manifest.get("language") != str(model.get("language") or "en")
                ):
                    raise RuntimeError("Voice preview settings changed during generation; reload the voice catalog")
            # Copy to a sibling temporary file and atomically publish it, so a
            # concurrent list/read can never observe a partial WAV.
            staged = destination.with_suffix(".wav.tmp")
            shutil.copyfile(generated, staged)
            if contract is not None:
                with staged.open("rb") as stream:
                    actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual_hash != manifest.get("output_audio_sha256"):
                    staged.unlink()
                    raise RuntimeError("Voice preview audio does not match its synthesis manifest")
            staged.replace(destination)
        return destination


async def preload_remote_voice_previews() -> None:
    """Warm every missing remote preview in the background at app startup."""
    for model_id, model in config.TTS_MODELS.items():
        if not model.get("remote_preview"):
            continue
        if model.get("preload_previews", True) is False:
            logger.info("Skipping eager %s preview preload; previews are generated on demand", model_id)
            continue
        if model.get("kind") == "orpheus_http" and not config.ORPHEUS_TTS_API_KEY:
            logger.info("Skipping Orpheus voice preview preload: API key is not configured")
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

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from backend import config
from backend.models import TtsModelOption, VoiceOption
from backend.pipeline.voice_previews import ensure_voice_preview

router = APIRouter(prefix="/api/voices", tags=["voices"])


@router.get("", response_model=list[VoiceOption])
async def list_voices(tts_model: str | None = Query(default=None)):
    """Selectable voices plus whether each one can be previewed.

    Preview availability is per-model: the 0.5B model substitutes some voices,
    and a substituted preset may not ship a reference WAV.
    """
    return [
        VoiceOption(
            name=name,
            gender=meta["gender"],
            lang=meta["lang"],
            resolved_name=config.resolve_voice(name, tts_model),
            preview_available=config.voice_preview_supported(name, tts_model),
        )
        for name, meta in config.voices_for_model(tts_model).items()
    ]


@router.get("/models", response_model=list[TtsModelOption])
async def list_tts_models():
    return [
        TtsModelOption(
            id=model_id,
            label=model["label"],
            provider=model.get("provider", "Unknown"),
            single_speaker=bool(model.get("single_speaker")),
            is_default=model_id == config.TTS_DEFAULT_MODEL,
        )
        for model_id, model in config.TTS_MODELS.items()
    ]


@router.get("/{voice}/preview")
async def preview_voice(voice: str, tts_model: str | None = Query(default=None)):
    if voice not in config.voices_for_model(tts_model):
        raise HTTPException(status_code=404, detail=f"Unknown voice '{voice}'")

    model_id = tts_model or config.TTS_DEFAULT_MODEL
    sample = config.voice_sample_path(voice, model_id)
    if sample is None and config.TTS_MODELS.get(model_id, {}).get("kind") == "orpheus_http":
        try:
            sample = await ensure_voice_preview(voice, model_id)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Could not generate preview for '{voice}': {exc}",
            ) from exc
    if sample is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No preview sample installed for '{voice}'. Expected a WAV in "
                f"{config.VOICE_SAMPLE_DIR}."
            ),
        )

    # Samples are static per install; let the browser reuse them across clicks.
    return FileResponse(
        sample,
        media_type="audio/wav",
        headers={"Cache-Control": "public, max-age=86400"},
    )

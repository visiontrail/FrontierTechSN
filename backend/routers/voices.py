from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse
from backend import config
from backend.models import TtsModelOption, VoiceOption
from backend.pipeline.voice_previews import ensure_voice_preview

router = APIRouter(prefix="/api/voices", tags=["voices"])


@router.get("", response_model=list[VoiceOption])
async def list_voices(response: Response, tts_model: str | None = Query(default=None)):
    """Selectable voices plus whether each one can be previewed.

    Preview availability is per-model: the 0.5B model substitutes some voices,
    and a substituted preset may not ship a reference WAV.
    """
    response.headers["Cache-Control"] = "no-store"
    return [
        VoiceOption(
            name=name,
            label=meta.get("label"),
            description=meta.get("description"),
            gender=meta["gender"],
            lang=meta["lang"],
            resolved_name=config.resolve_voice(name, tts_model),
            preview_available=config.voice_preview_supported(name, tts_model),
            preview_version=config.voice_preview_version(name, tts_model),
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
            default_voice=config.default_voice_for_model(model_id),
        )
        for model_id, model in config.TTS_MODELS.items()
    ]


@router.get("/{voice}/preview")
async def preview_voice(
    voice: str, tts_model: str | None = Query(default=None), v: str | None = Query(default=None),
):
    if voice not in config.voices_for_model(tts_model):
        raise HTTPException(status_code=404, detail=f"Unknown voice '{voice}'")

    model_id = tts_model or config.TTS_DEFAULT_MODEL
    version = config.voice_preview_version(voice, model_id)
    if version is not None and v is not None and v != version:
        raise HTTPException(status_code=409, detail="Voice preview changed. Reload the voice catalog.")
    sample = config.voice_sample_path(voice, model_id)
    if sample is None and config.TTS_MODELS.get(model_id, {}).get("remote_preview"):
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

    if version != config.voice_preview_version(voice, model_id):
        raise HTTPException(status_code=409, detail="Voice preview changed. Reload the voice catalog.")
    # Versioned URLs keep deployed voice references out of a stale browser
    # cache. Legacy Chatterbox URLs must revalidate after the application update.
    cache_control = "public, max-age=86400"
    if version is not None:
        cache_control = "private, max-age=86400, immutable" if v == version else "no-store"
    return FileResponse(
        sample,
        media_type="audio/wav",
        headers={"Cache-Control": cache_control},
    )

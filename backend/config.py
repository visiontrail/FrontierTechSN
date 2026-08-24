"""Process configuration.

Every value here is a *live* module attribute, not a constant: `.env` (and the
built-in fallbacks) only seed the process, and the Admin console layers its
saved settings on top at import time — see :mod:`backend.settings_store`. That
is why consumers must read ``config.NAME`` at call time rather than binding
``from backend.config import NAME`` at import: a rebound module attribute is
what makes an Admin save reach the next pipeline stage without a restart.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no")


def resolve_project_path(raw: str | os.PathLike) -> Path:
    """Resolve a configured path: absolute as-is, relative under the repo root."""
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


AI_ENDPOINT = os.getenv("AI_ENDPOINT", "http://oneapi.yhroot.com/v1/chat/completions")
AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "glm-4.6-chat")
# HTTP-backend request ceiling. A reasoning model summarising a full transcript
# spends minutes on one completion (measured: ~160s for 1.7k words through a
# proxy gateway), so the old two-minute default aborted calls that were healthy.
AI_TIMEOUT = int(os.getenv("AI_TIMEOUT", "600"))
# yhroot is deliberately treated as a slow, eventually-successful gateway. Nine
# retries after the first attempt gives every provider-backed stage ten chances
# before the edition is failed closed.
AI_MAX_RETRIES = int(os.getenv("AI_MAX_RETRIES", "9"))
AI_RETRY_BASE_SECONDS = float(os.getenv("AI_RETRY_BASE_SECONDS", "4"))
AI_RETRY_MAX_SECONDS = float(os.getenv("AI_RETRY_MAX_SECONDS", "60"))

# Public-footage scouting. Wikimedia Commons needs no key, but asks API clients
# to identify themselves. The byte ceiling prevents an autonomous scout from
# pulling down unexpectedly large archival masters.
FOOTAGE_USER_AGENT = os.getenv(
    "FOOTAGE_USER_AGENT",
    "FrontierTechSN/1.0 (local AI media scout)",
)
FOOTAGE_TIMEOUT = int(os.getenv("FOOTAGE_TIMEOUT", "45"))
FOOTAGE_MAX_BYTES = int(os.getenv("FOOTAGE_MAX_BYTES", str(50 * 1024 * 1024)))

# News stills are planned against exact narrated scenes by ChatGPT through
# OpenCLI, independently researched with Google News/Search through OpenCLI,
# and then downloaded from the open-license Wikimedia Commons inventory. Keep their
# browser and byte ceilings separate from moving footage: one still should never
# inherit a 50 MB archival-video allowance or a ten-attempt social-write retry.
NEWS_IMAGE_MAX_BYTES = int(
    os.getenv("NEWS_IMAGE_MAX_BYTES", str(12 * 1024 * 1024))
)
NEWS_IMAGE_OPENCLI_TIMEOUT = int(os.getenv("NEWS_IMAGE_OPENCLI_TIMEOUT", "45"))
NEWS_IMAGE_OPENCLI_ATTEMPTS = int(os.getenv("NEWS_IMAGE_OPENCLI_ATTEMPTS", "1"))
NEWS_IMAGE_SEARCH_TIMEOUT = int(os.getenv("NEWS_IMAGE_SEARCH_TIMEOUT", "60"))

# Project-local OpenCLI + web-footage expansion.  The binary is deliberately a
# repository wrapper rather than a global npm command, so installing or
# upgrading this feature never changes the operator's global Claude Code setup.
OPENCLI_BIN = resolve_project_path(os.getenv("OPENCLI_BIN", "scripts/opencli.sh"))
OPENCLI_PROFILE = os.getenv("OPENCLI_PROFILE", "").strip()
OPENCLI_SITE_SESSION_NAMESPACE = (
    os.getenv("OPENCLI_SITE_SESSION_NAMESPACE", "frontiertechsn").strip()
    or "frontiertechsn"
)
OPENCLI_TIMEOUT = int(os.getenv("OPENCLI_TIMEOUT", "180"))
OPENCLI_MAX_ATTEMPTS = int(os.getenv("OPENCLI_MAX_ATTEMPTS", "10"))
OPENCLI_RETRY_BASE_SECONDS = float(os.getenv("OPENCLI_RETRY_BASE_SECONDS", "3"))
# Cross-process start-to-start spacing for Gemini and ChatGPT web commands.
# The runtime limiter clamps direct environment overrides to 10–30 seconds too.
OPENCLI_WEB_REQUEST_INTERVAL_SECONDS = int(
    os.getenv("OPENCLI_WEB_REQUEST_INTERVAL_SECONDS", "10")
)
# Daily-news claim review starts with fast Gemini Flash. A failed or malformed
# primary turn falls back to ChatGPT at a deliberately non-Pro reasoning level.
DAILY_NEWS_CHATGPT_REVIEW_MODEL = os.getenv(
    "DAILY_NEWS_CHATGPT_REVIEW_MODEL", "medium"
).strip() or "medium"
DAILY_NEWS_GEMINI_REVIEW_MODEL = os.getenv(
    "DAILY_NEWS_GEMINI_REVIEW_MODEL", "3.7-flash"
).strip() or "3.7-flash"
DAILY_NEWS_WEB_REVIEW_TIMEOUT = int(
    os.getenv("DAILY_NEWS_WEB_REVIEW_TIMEOUT", "90")
)
# OpenCode is the optional autonomous planner for account operations. It calls
# the project-local OpenCLI wrapper through the account-operations skill; the
# deterministic backend still owns image publishing and result persistence.
OPENCODE_BIN = os.getenv("OPENCODE_BIN", "opencode").strip() or "opencode"
OPENCODE_TIMEOUT = int(os.getenv("OPENCODE_TIMEOUT", "600"))
ACCOUNT_OPS_POLL_SECONDS = int(os.getenv("ACCOUNT_OPS_POLL_SECONDS", "10"))
# Video publication is fail-closed. This process-wide kill switch must be
# enabled in addition to a task's own opt-in and the target platform switch.
# Automatic adapters also require an expected account identity so a stale
# browser login can never silently publish to the wrong channel.
VIDEO_AUTO_PUBLISH_ENABLED = _env_bool("VIDEO_AUTO_PUBLISH_ENABLED", "0")
VIDEO_PUBLISH_YOUTUBE_ENABLED = _env_bool("VIDEO_PUBLISH_YOUTUBE_ENABLED", "1")
VIDEO_PUBLISH_YOUTUBE_CHANNEL_NAME = os.getenv(
    "VIDEO_PUBLISH_YOUTUBE_CHANNEL_NAME", ""
).strip()
VIDEO_PUBLISH_YOUTUBE_CHANNEL_ID = os.getenv(
    "VIDEO_PUBLISH_YOUTUBE_CHANNEL_ID", ""
).strip()
VIDEO_PUBLISH_X_ENABLED = _env_bool("VIDEO_PUBLISH_X_ENABLED", "1")
VIDEO_PUBLISH_X_HANDLE = os.getenv("VIDEO_PUBLISH_X_HANDLE", "").strip().lstrip("@")
VIDEO_PUBLISH_APPLE_PODCAST_ENABLED = _env_bool(
    "VIDEO_PUBLISH_APPLE_PODCAST_ENABLED", "1"
)
VIDEO_PUBLISH_APPLE_FEED_TITLE = os.getenv(
    "VIDEO_PUBLISH_APPLE_FEED_TITLE", "Frontier Tech Daily"
).strip()
VIDEO_PUBLISH_APPLE_FEED_AUTHOR = os.getenv(
    "VIDEO_PUBLISH_APPLE_FEED_AUTHOR", "FrontierTechSN"
).strip()
VIDEO_PUBLISH_APPLE_FEED_PUBLIC_BASE_URL = os.getenv(
    "VIDEO_PUBLISH_APPLE_FEED_PUBLIC_BASE_URL", ""
).strip().rstrip("/")
# ChatGPT image generation routinely takes longer than simple browser reads.
THUMBNAIL_CHATGPT_TIMEOUT = int(os.getenv("THUMBNAIL_CHATGPT_TIMEOUT", "360"))
# Collage stills run through ChatGPT Web; motion runs through Gemini Create
# Video. These are browser waits, not API-key-backed provider timeouts.
COLLAGE_CHATGPT_TIMEOUT = int(os.getenv("COLLAGE_CHATGPT_TIMEOUT", "420"))
COLLAGE_GEMINI_TIMEOUT = int(os.getenv("COLLAGE_GEMINI_TIMEOUT", "1800"))
# Veo 3.1 returns one eight-second clip per generation. Keep the provider
# ceiling configurable so future duration changes do not require a code change.
COLLAGE_GEMINI_MAX_SECONDS = int(os.getenv("COLLAGE_GEMINI_MAX_SECONDS", "8"))
WEB_FOOTAGE_ENABLED = _env_bool("WEB_FOOTAGE_ENABLED", "1")
WEB_FOOTAGE_GEMINI_ENABLED = _env_bool("WEB_FOOTAGE_GEMINI_ENABLED", "1")
WEB_FOOTAGE_GEMINI_TIMEOUT = int(os.getenv("WEB_FOOTAGE_GEMINI_TIMEOUT", "120"))
WEB_FOOTAGE_CLIP_SECONDS = int(os.getenv("WEB_FOOTAGE_CLIP_SECONDS", "15"))
WEB_FOOTAGE_CLIP_MIN_SECONDS = int(os.getenv("WEB_FOOTAGE_CLIP_MIN_SECONDS", "10"))
WEB_FOOTAGE_DOWNLOAD_TIMEOUT = int(os.getenv("WEB_FOOTAGE_DOWNLOAD_TIMEOUT", "600"))

# Which backend drives digestion/scriptwriting AI calls:
#   "agent_sdk" (default) — Claude Agent SDK, talking the Anthropic protocol to
#                           a provider gateway (the bundled/system `claude` CLI
#                           is spawned in-process by the SDK).
#   "http"                — the legacy direct OpenAI-compatible HTTP client.
# The provider registry (endpoint/api_key/model) still drives both; for the
# Agent SDK the endpoint is mapped to an Anthropic base URL (see ANTHROPIC_*).
AI_BACKEND = os.getenv("AI_BACKEND", "agent_sdk").strip().lower()

# Anthropic-protocol provider settings for the Claude Agent SDK. When left
# blank, ANTHROPIC_BASE_URL/ANTHROPIC_AUTH_TOKEN are derived from the resolved
# provider (its OpenAI-style endpoint host and api_key). Set them explicitly to
# point at a gateway whose Anthropic route lives on a non-root path (e.g.
# DeepSeek's https://api.deepseek.com/anthropic).
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", "").strip()
ANTHROPIC_AUTH_TOKEN = os.getenv("ANTHROPIC_AUTH_TOKEN", "").strip()
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "").strip()
# Haiku-tier model the CLI uses for cheap background tasks (title/summary);
# point it at the same gateway model when the provider has no separate haiku.
ANTHROPIC_DEFAULT_HAIKU_MODEL = os.getenv("ANTHROPIC_DEFAULT_HAIKU_MODEL", "").strip()
# Optional explicit path to the `claude` CLI. Blank = let the SDK locate it
# (bundled with the wheel, else the first `claude` on PATH).
CLAUDE_CLI_PATH = os.getenv("CLAUDE_CLI_PATH", "").strip()

# The SDK spawns a CLI that owns its own retry ladder, so it needs two ceilings
# rather than the single AI_TIMEOUT the HTTP client uses:
#   AGENT_REQUEST_TIMEOUT bounds one /v1/messages call inside the CLI
#     (API_TIMEOUT_MS). Reasoning models behind a slow gateway routinely need
#     two minutes for a single digestion turn, and a stall of 30s mid-stream is
#     normal there — set this too low and every request is aborted just before
#     it would have finished.
#   AGENT_TURN_TIMEOUT bounds the whole CLI process. This is the ceiling that
#     matters operationally: when a request keeps timing out the CLI silently
#     retries it, so a 120s request budget can and did burn 25 minutes per
#     attempt before exiting 1.
AGENT_REQUEST_TIMEOUT = int(os.getenv("AGENT_REQUEST_TIMEOUT", "600"))
AGENT_TURN_TIMEOUT = int(os.getenv("AGENT_TURN_TIMEOUT", "900"))
# When the Agent SDK transport fails outright (CLI missing, gateway with no
# Anthropic route, process dying), fall back to the OpenAI-compatible client on
# the same provider instead of failing the stage. Costs one extra call on a
# genuine outage; saves a whole pipeline run when only the SDK path is broken.
AI_HTTP_FALLBACK = _env_bool("AI_HTTP_FALLBACK", "1")

# yt-dlp source extraction. start.sh exports sensible defaults for the first
# two before launching (a node runtime it can find, and the browser to borrow
# cookies from), which is why these are read from the environment at all.
YTDLP_JS_RUNTIME = os.getenv("YTDLP_JS_RUNTIME", "").strip()
YTDLP_REMOTE_COMPONENTS = os.getenv("YTDLP_REMOTE_COMPONENTS", "ejs:github").strip()
YTDLP_COOKIES = os.getenv("YTDLP_COOKIES", "").strip()
YTDLP_COOKIES_FROM_BROWSER = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "").strip()

AIWORK_ROOT = resolve_project_path(os.getenv("AIWORK_ROOT", "/Volumes/TP-1TB/AIWork"))
TTS_DEVICE = os.getenv("TTS_DEVICE", "mps")

# VibeVoice synthesis is the slowest stage and its cost scales with script
# length: a full-length monologue is 8192 decode steps, and the step rate decays
# as the KV cache grows, so an hour of wall clock is not by itself evidence of a
# hang. The real watchdog is TTS_STALL_TIMEOUT — the model streams a progress
# line several times a second, so going quiet for minutes means it is wedged,
# whatever the elapsed time. TTS_TIMEOUT is only a coarse backstop.
TTS_TIMEOUT = int(os.getenv("TTS_TIMEOUT", str(6 * 3600)))
TTS_STALL_TIMEOUT = int(os.getenv("TTS_STALL_TIMEOUT", "600"))
# VibeVoice's decode cost can become superlinear late in a long Apple-MPS
# request. This limit applies only to local VibeVoice; Orpheus derives its own
# smaller/larger chunk size from the configured audio-token budget. Keep the old
# env name as a compatibility seed for machines that already set it.
VIBEVOICE_TTS_CHUNK_WORDS = int(
    os.getenv("VIBEVOICE_TTS_CHUNK_WORDS", os.getenv("TTS_CHUNK_WORDS", "350"))
)
TTS_RANDOM_SEED = int(os.getenv("TTS_RANDOM_SEED", "42"))

TTS_DEFAULT_MODEL = os.getenv("TTS_DEFAULT_MODEL", "vibevoice-0.5b")
TTS_DEFAULT_VOICE_1 = os.getenv("TTS_DEFAULT_VOICE_1", "Carter")
TTS_DEFAULT_VOICE_2 = os.getenv("TTS_DEFAULT_VOICE_2", "Alice")

# Remote Orpheus service. The key is intentionally blank in source and is
# managed as a masked secret by Admin -> System (or seeded through .env).
ORPHEUS_TTS_URL = os.getenv("ORPHEUS_TTS_URL", "http://10.60.11.3:8088").rstrip("/")
ORPHEUS_TTS_API_KEY = os.getenv("ORPHEUS_TTS_API_KEY", "")
# Narration is the timing source of truth.  Keep synthesis at the provider's
# natural speed and make scenes follow the measured WAV; accepting a saved/env
# speed override would hide a script-duration mismatch by retiming the voice.
ORPHEUS_TTS_SPEED_PERCENT = 100
# 16,384 is the current external service maximum. It is only a ceiling: the
# client derives a much smaller request budget for each short utterance so a
# bad generation cannot burn through minutes of unrelated audio tokens.
ORPHEUS_TTS_MAX_TOKENS = int(os.getenv("ORPHEUS_TTS_MAX_TOKENS", "16384"))
# Orpheus is trained on utterances, not chapter-sized prompts. Long prompts can
# silently skip their opening while still returning a syntactically valid WAV.
ORPHEUS_TTS_CHUNK_WORDS = int(os.getenv("ORPHEUS_TTS_CHUNK_WORDS", "12"))
ORPHEUS_TTS_N_THREADS = int(os.getenv("ORPHEUS_TTS_N_THREADS", "64"))
ORPHEUS_TTS_POLL_SECONDS = int(os.getenv("ORPHEUS_TTS_POLL_SECONDS", "2"))
ORPHEUS_TTS_REQUEST_TIMEOUT = int(os.getenv("ORPHEUS_TTS_REQUEST_TIMEOUT", "60"))

# Audio/visual alignment. MLX Whisper reads the finished WAV from whichever TTS
# provider the task selected to obtain word timestamps; it never generates or
# replaces speech. Alignment is
# advisory when transcription is unavailable, but measured low script/audio
# coverage is a hard completeness failure and blocks video rendering.
AV_SYNC_LANGUAGE = os.getenv("AV_SYNC_LANGUAGE", "en").strip() or "en"
AV_SYNC_MLX_MODEL = os.getenv(
    "AV_SYNC_MLX_MODEL", "mlx-community/whisper-large-v3-turbo-q4"
).strip()
AV_SYNC_TRANSCRIBE_MAX_RETRIES = int(
    os.getenv("AV_SYNC_TRANSCRIBE_MAX_RETRIES", "2")
)
AV_SYNC_MIN_WORD_COVERAGE_PERCENT = int(
    os.getenv("AV_SYNC_MIN_WORD_COVERAGE_PERCENT", "65")
)
AV_SYNC_MAX_BOUNDARY_UNCERTAINTY_MS = int(
    os.getenv("AV_SYNC_MAX_BOUNDARY_UNCERTAINTY_MS", "3000")
)
# Final rendered-pixel review. OpenCLI uploads labeled keyframe contact sheets
# to the signed-in Gemini web app and compares them with the acoustic scene
# excerpts. This is separate from lexical plan grounding: it judges the actual
# MP4 after the renderer, including any authored HTML and placed B-roll.
AV_SYNC_GEMINI_REVIEW_ENABLED = _env_bool("AV_SYNC_GEMINI_REVIEW_ENABLED", "1")
AV_SYNC_FRAME_MAX_RETRIES = int(os.getenv("AV_SYNC_FRAME_MAX_RETRIES", "2"))
AV_SYNC_GEMINI_BATCH_SIZE = int(os.getenv("AV_SYNC_GEMINI_BATCH_SIZE", "8"))
AV_SYNC_GEMINI_MIN_SCENE_SCORE = int(
    os.getenv("AV_SYNC_GEMINI_MIN_SCENE_SCORE", "70")
)
AV_SYNC_GEMINI_MIN_AVERAGE_SCORE = int(
    os.getenv("AV_SYNC_GEMINI_MIN_AVERAGE_SCORE", "82")
)
AV_SYNC_GEMINI_TIMEOUT = int(os.getenv("AV_SYNC_GEMINI_TIMEOUT", "180"))
AV_SYNC_GEMINI_MAX_RETRIES = int(os.getenv("AV_SYNC_GEMINI_MAX_RETRIES", "2"))

AVAILABLE_VOICES = {
    "Carter": {"gender": "male", "lang": "en"},
    "Frank": {"gender": "male", "lang": "en"},
    "Alice": {"gender": "female", "lang": "en"},
    "Maya": {"gender": "female", "lang": "en"},
    "Mary": {"gender": "female", "lang": "en"},
    "Samuel": {"gender": "male", "lang": "in"},
}

ORPHEUS_EN_VOICES = {
    "tara": {"gender": "female", "lang": "en"},
    "leah": {"gender": "female", "lang": "en"},
    "jess": {"gender": "female", "lang": "en"},
    "leo": {"gender": "male", "lang": "en"},
    "dan": {"gender": "male", "lang": "en"},
    "mia": {"gender": "female", "lang": "en"},
    "zac": {"gender": "male", "lang": "en"},
    "zoe": {"gender": "female", "lang": "en"},
}


def _build_tts_models(root: Path) -> dict[str, dict]:
    """Registry of local and remote TTS model invocation contracts.

    Local entries carry the full invocation contract: which venv to source,
    which project directory to cd into, which inference script to run, and
    which speaker flag the script expects (1.5B uses plural
    --speaker_names, 0.5B uses singular --speaker_name). The 0.5B realtime model
    is single-speaker, so it only ever receives one voice source. Adding a
    future model of an existing provider kind is a data change here.

    It is a function because AIWORK_ROOT is settable from the Admin console:
    every path below has to be rebuilt when the root moves.
    """
    return {
        "vibevoice-1.5b": {
            "label": "1.5B (high quality)",
            "provider": "Microsoft VibeVoice",
            "kind": "local_subprocess",
            "env_script": root / "env_vibevoice_1.5b.sh",
            "project_dir": root / "VibeVoice-1.5B",
            "inference_script": root / "VibeVoice-1.5B" / "demo" / "inference_from_file.py",
            "speaker_flag": "--speaker_names",
            "single_speaker": False,
            "requires_speaker_labels": True,
        },
        "vibevoice-0.5b": {
            "label": "0.5B (fast draft)",
            "provider": "Microsoft VibeVoice",
            "kind": "local_subprocess",
            "env_script": root / "env_vibevoice.sh",
            "project_dir": root / "VibeVoice",
            "inference_script": root / "VibeVoice" / "demo" / "realtime_model_inference_from_file.py",
            "speaker_flag": "--speaker_name",
            "single_speaker": True,
            # Realtime 0.5B ships a smaller preset set than 1.5B. Keep existing
            # tasks portable by substituting the closest English voice rather
            # than letting VibeVoice silently fall back to its first preset
            # (German).
            "voice_aliases": {
                "Alice": "Emma",
                "Maya": "Grace",
                "Mary": "Emma",
            },
        },
        "orpheus-en": {
            "label": "English Q4 (remote CPU)",
            "provider": "Orpheus",
            "kind": "orpheus_http",
            "single_speaker": True,
            "language": "en",
            "voices": ORPHEUS_EN_VOICES,
        },
    }


TTS_MODELS: dict[str, dict] = _build_tts_models(AIWORK_ROOT)

# Voice preview samples. VibeVoice defines each preset by a short reference WAV
# that the model clones, so playing that file previews the timbre a task will
# get without paying for a synthesis run. Only the 1.5B project ships the raw
# WAVs (demo/voices/<lang>-<Name>_<gender>[_bgm].wav); the 0.5B realtime model
# ships pre-encoded .pt embeddings, so its previews resolve through the same
# voice_aliases map used at synthesis time and fall back to the 1.5B WAV.
VOICE_SAMPLE_DIR = AIWORK_ROOT / "VibeVoice-1.5B" / "demo" / "voices"
# Generated previews for remote providers are runtime data, not source assets.
# Keeping them under ignored data/ makes a successful remote synthesis survive
# app restarts without ever entering git.
VOICE_PREVIEW_CACHE_DIR = PROJECT_ROOT / "data" / "voice_previews"


def resolve_voice(voice: str, tts_model: str | None = None) -> str:
    """Map a requested voice to the preset the given model actually uses."""
    model = TTS_MODELS.get(tts_model or TTS_DEFAULT_MODEL, {})
    return model.get("voice_aliases", {}).get(voice, voice)


def voices_for_model(tts_model: str | None = None) -> dict[str, dict]:
    """Return only the voices accepted by the selected synthesis model."""
    model = TTS_MODELS.get(tts_model or TTS_DEFAULT_MODEL, {})
    return model.get("voices", AVAILABLE_VOICES)


def tts_provider_label(tts_model: str | None = None) -> str:
    model = TTS_MODELS.get(tts_model or TTS_DEFAULT_MODEL, {})
    return model.get("provider", "Unknown TTS provider")


def voice_sample_path(voice: str, tts_model: str | None = None):
    """Return the reference WAV for a voice, or None if none is installed.

    `voice` must already be a known preset name — callers validate it against
    the model registry so an arbitrary string never reaches the glob.
    """
    model = TTS_MODELS.get(tts_model or TTS_DEFAULT_MODEL, {})
    if model.get("kind") == "orpheus_http":
        cached = VOICE_PREVIEW_CACHE_DIR / (tts_model or TTS_DEFAULT_MODEL) / f"{voice}.wav"
        return cached if cached.is_file() and cached.stat().st_size >= 44 else None
    if model.get("kind") != "local_subprocess":
        return None
    resolved = resolve_voice(voice, tts_model)
    matches = sorted(VOICE_SAMPLE_DIR.glob(f"*-{resolved}_*.wav"))
    return matches[0] if matches else None


def voice_preview_supported(voice: str, tts_model: str | None = None) -> bool:
    """Whether a voice can be previewed now or generated on first use."""
    model = TTS_MODELS.get(tts_model or TTS_DEFAULT_MODEL, {})
    if voice not in voices_for_model(tts_model):
        return False
    return model.get("kind") == "orpheus_http" or voice_sample_path(voice, tts_model) is not None


HYPERFRAME_DIR = resolve_project_path(os.getenv("HYPERFRAME_DIR", "hyperframe"))

# HyperFrames render tuning. The CLI is pinned and installed locally under
# HYPERFRAME_DIR/node_modules so a render never triggers an on-demand `npx`
# install (the old unpinned `npx hyperframes` re-installed latest every run —
# minutes of silent stall on first/cold runs, and a moving version target).
# Frame-by-frame headless-Chrome capture of a multi-minute 1080p composition is
# inherently heavy (~frames = duration x fps), so the defaults trade frame rate
# and quality for wall-clock time.
# Claude Agent SDK video-production crews. When enabled, agents author the
# per-scene HyperFrames compositions on top of the deterministic drafts; every
# file they write is gated against the runtime contract and reverted to its
# draft if it fails, so turning this off only costs visual variety.
DIRECTOR_ENABLED = _env_bool("DIRECTOR_ENABLED", "1")
# Cap how many scenes are handed to agents (0 = no cap). Each crew is a `claude`
# CLI process and a provider round-trip, so this is the cost/latency dial.
DIRECTOR_MAX_SCENES = int(os.getenv("DIRECTOR_MAX_SCENES", "0"))
# Run `hyperframes inspect` before the render and let the agents repair the
# layout failures it reports. It drives headless Chrome over the timeline, so it
# costs a couple of minutes on a long episode; turning it off only skips the
# layout polish pass.
INSPECT_ENABLED = _env_bool("INSPECT_ENABLED", "1")

HYPERFRAMES_VERSION = os.getenv("HYPERFRAMES_VERSION", "0.6.99")
RENDER_FPS = int(os.getenv("RENDER_FPS", "15"))
RENDER_QUALITY = os.getenv("RENDER_QUALITY", "draft")  # draft | standard | high
RENDER_WORKERS = os.getenv("RENDER_WORKERS", "2")      # integer or "auto"
RENDER_RESOLUTION = os.getenv("RENDER_RESOLUTION", "landscape")  # 1920x1080

OUTPUTS_DIR = resolve_project_path(os.getenv("OUTPUTS_DIR", "outputs"))
UPLOADS_DIR = resolve_project_path(os.getenv("UPLOADS_DIR", "uploads"))
# DB_PATH may point at a custom location; defaults to the repo root.
_db_env = os.getenv("DB_PATH", "")
DB_PATH = resolve_project_path(_db_env) if _db_env else PROJECT_ROOT / "tasks.db"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
ISLA_READER_PROMOTION_SCRIPT = resolve_project_path(
    os.getenv(
        "ISLA_READER_PROMOTION_SCRIPT",
        "/Volumes/TP-1TB/WorspacePlus/IslaProject/Isla-Reader-Promotional/Promotion-Agent/generate-promotion.sh",
    )
)


def rebuild_derived() -> None:
    """Recompute values that hang off other settings, and ensure directories.

    Called after any settings change so that e.g. moving AIWORK_ROOT relocates
    the whole VibeVoice contract (model scripts and voice samples) with it.
    """
    global TTS_MODELS, VOICE_SAMPLE_DIR
    TTS_MODELS = _build_tts_models(AIWORK_ROOT)
    VOICE_SAMPLE_DIR = AIWORK_ROOT / "VibeVoice-1.5B" / "demo" / "voices"
    for directory in (OUTPUTS_DIR, UPLOADS_DIR, DB_PATH.parent):
        directory.mkdir(parents=True, exist_ok=True)


def apply_values(values: dict) -> None:
    """Rebind config attributes from an already-coerced settings mapping.

    :mod:`backend.settings_store` owns validation and coercion; this only lands
    the values on the module and refreshes whatever derives from them.
    """
    module = sys.modules[__name__]
    for key, value in values.items():
        setattr(module, key, value)
    rebuild_derived()


rebuild_derived()

# Layer the Admin console's saved settings over the .env seed. Imported here (at
# the bottom, deferred) because settings_store reads this module for its
# defaults: by now every attribute above exists.
from backend import settings_store as _settings_store  # noqa: E402

_settings_store.apply_saved()

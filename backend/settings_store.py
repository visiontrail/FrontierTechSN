"""Runtime settings registry — the Admin console's source of truth for every
knob that used to live only in ``.env``.

``.env`` still seeds the process at import (see :mod:`backend.config`), but its
values are now *defaults*: whatever the Admin console saves into
``data/settings.json`` is layered on top at startup and re-applied immediately
on every save. Almost every consumer reads ``config.NAME`` at call time, so a
saved change reaches the next pipeline stage without a restart; the few that
cannot (the ``/outputs`` static mount, the SQLite file opened at startup) carry
``restart_required`` and the UI says so.

Adding a knob is a data change: append a :class:`SettingSpec` whose ``key``
matches the attribute in ``config``, and both the API schema and the Admin form
pick it up.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

_STORE_VERSION = 1
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}
_RETIRED_KEYS = {
    "ORPHEUS_TTS_SPEED_PERCENT",
    # Fact check is ChatGPT-only. Keep Gemini settings for video stages, but
    # discard the retired daily-news review override from older stores.
    "DAILY_NEWS_GEMINI_REVIEW_MODEL",
    # Replaced by an allowed current-level range. A persisted legacy value is
    # migrated to the new minimum before this key is pruned.
    "DAILY_NEWS_CHATGPT_REVIEW_MODEL",
    # Provider routing is owned by Admin -> Models. Prune the former System
    # overrides so hidden settings cannot keep winning after the fields move.
    "AI_ENDPOINT",
    "AI_API_KEY",
    "AI_MODEL",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
}

_CHATGPT_REVIEW_LEVEL_ORDER = {
    "medium": 1,
    "high": 2,
    "xhigh": 3,
}


class SettingsError(ValueError):
    """Raised when a submitted setting value is not usable."""


@dataclass(frozen=True)
class SettingGroup:
    id: str
    label: str
    description: str


@dataclass(frozen=True)
class SettingSpec:
    key: str          # matches the attribute name in backend.config
    group: str
    label: str
    type: str         # string | secret | int | bool | choice | path
    description: str = ""
    placeholder: str = ""
    unit: str = ""
    options: tuple[str, ...] = ()
    # Choices that depend on runtime config (installed TTS models, voice presets).
    dynamic_options: Callable[[], tuple[str, ...]] | None = field(default=None, compare=False)
    minimum: int | None = None
    maximum: int | None = None
    # False = an empty submission means "go back to the default" rather than
    # storing a blank (used for paths and other values that must not be empty).
    allow_blank: bool = True
    restart_required: bool = False


def _config():
    """Import backend.config lazily.

    config imports this module at the end of its own import to apply saved
    settings, so a module-level import here would be circular.
    """
    from backend import config

    return config


def _tts_model_options() -> tuple[str, ...]:
    return tuple(sorted(_config().TTS_MODELS))


def _voice_options() -> tuple[str, ...]:
    voices: dict[str, None] = {}
    for model_id in _config().TTS_MODELS:
        voices.update(dict.fromkeys(_config().voices_for_model(model_id)))
    return tuple(voices)


GROUPS: tuple[SettingGroup, ...] = (
    SettingGroup(
        "ai",
        "AI Engine",
        "Execution policy for the provider selected in the Models tab: transport, "
        "retry limits, and HTTP fallback behavior.",
    ),
    SettingGroup(
        "agent_sdk",
        "Claude Agent SDK",
        "Process and timeout controls for the `claude` CLI spawned by the SDK. "
        "Provider endpoint, credentials, and model are managed in the Models tab.",
    ),
    SettingGroup(
        "tts",
        "Voice & TTS",
        "Model-specific safe chunking, deterministic voices, lossless PCM "
        "joining, runtimes, credentials, and inference limits.",
    ),
    SettingGroup(
        "av_sync",
        "Audio / Visual Sync",
        "Word-level acoustic alignment and the quality gate that prevents "
        "estimated or drifting scene timing from reaching the renderer.",
    ),
    SettingGroup(
        "render",
        "Render",
        "HyperFrames capture settings. Frame count is duration x fps, so these "
        "are the main wall-clock dials for a render.",
    ),
    SettingGroup(
        "director",
        "Video Direction",
        "Agent crews that author the per-scene compositions. Everything they "
        "write is checked against the runtime contract and reverted on failure, "
        "so turning this off costs visual variety, never a working render.",
    ),
    SettingGroup(
        "source",
        "Source Extraction",
        "How yt-dlp fetches YouTube sources. start.sh seeds a node runtime and "
        "a cookie browser at launch; these override that.",
    ),
    SettingGroup(
        "footage",
        "Footage Sources",
        "Wikimedia Commons plus the project-local OpenCLI/yt-dlp web scout. "
        "Web-platform clips retain a source ledger and require rights review.",
    ),
    SettingGroup(
        "thumbnail",
        "Viral Thumbnail",
        "Script-driven cover art generated in the signed-in ChatGPT web app. "
        "The editable art-direction formula lives in the Prompts tab.",
    ),
    SettingGroup(
        "collage",
        "Paper-collage B-roll",
        "Agent-planned stills and Gemini Create Video motion generated through "
        "the signed-in project-local OpenCLI web sessions.",
    ),
    SettingGroup(
        "publication",
        "Publishing & Accounts",
        "Fail-closed delivery controls. Automatic publication requires the global "
        "switch, the platform switch, a task-level opt-in, and an exact match "
        "between the configured identity and the signed-in browser account.",
    ),
    SettingGroup(
        "paths",
        "Paths & Storage",
        "Where the app reads and writes. Relative paths resolve under the "
        "project root.",
    ),
)

SPECS: tuple[SettingSpec, ...] = (
    SettingSpec(
        "VIDEO_AUTO_PUBLISH_ENABLED", "publication", "Enable automatic publication", "bool",
        description="Master kill switch for every unattended video publication. "
                    "It is off by default; manual private test publishing remains available.",
    ),
    SettingSpec(
        "VIDEO_PUBLISH_YOUTUBE_ENABLED", "publication", "YouTube adapter", "bool",
        description="Allow YouTube as a publication target. Automatic runs still "
                    "require both channel identity fields below.",
    ),
    SettingSpec(
        "VIDEO_PUBLISH_YOUTUBE_CHANNEL_NAME", "publication", "YouTube channel name", "string",
        placeholder="ByteFront Espresso",
        description="Exact visible channel name expected in YouTube Studio. Automatic "
                    "publishing refuses a blank or mismatched name.",
    ),
    SettingSpec(
        "VIDEO_PUBLISH_YOUTUBE_CHANNEL_ID", "publication", "YouTube channel ID", "string",
        placeholder="UC…",
        description="Stable channel identifier from the YouTube Studio URL. It is checked "
                    "together with the channel name before any video is uploaded.",
    ),
    SettingSpec(
        "VIDEO_PUBLISH_X_ENABLED", "publication", "X adapter", "bool",
        description="Allow x.com as a publication target. The configured handle below "
                    "must match the active X account before the composer opens.",
    ),
    SettingSpec(
        "VIDEO_PUBLISH_X_HANDLE", "publication", "X account handle", "string",
        placeholder="bytefrontespresso",
        description="Expected account handle without the leading @. Automatic publishing "
                    "refuses a blank or mismatched handle.",
    ),
    SettingSpec(
        "VIDEO_PUBLISH_APPLE_PODCAST_ENABLED", "publication", "Podcast RSS adapter", "bool",
        description="Maintain the podcast RSS feed for completed editions. This does not "
                    "submit or update an Apple Podcasts Connect account.",
    ),
    SettingSpec(
        "VIDEO_PUBLISH_APPLE_FEED_TITLE", "publication", "Podcast feed title", "string",
        placeholder="ByteFront Espresso", allow_blank=False,
        description="Public show title written into the generated RSS feed.",
    ),
    SettingSpec(
        "VIDEO_PUBLISH_APPLE_FEED_AUTHOR", "publication", "Podcast feed author", "string",
        placeholder="FrontierTechSN", allow_blank=False,
        description="Author identity written into the RSS and iTunes metadata.",
    ),
    SettingSpec(
        "VIDEO_PUBLISH_APPLE_FEED_PUBLIC_BASE_URL", "publication", "Podcast public base URL", "string",
        placeholder="https://media.example.com",
        description="Public origin used for audio enclosure URLs. Leave blank for local "
                    "preview only; Apple Podcasts submission remains external.",
    ),
    # ── AI engine ────────────────────────────────────────────────────────
    SettingSpec(
        "AI_BACKEND", "ai", "Backend", "choice",
        options=("agent_sdk", "http"),
        description="agent_sdk spawns the `claude` CLI and talks the Anthropic "
                    "protocol; http is the direct OpenAI-compatible client.",
    ),
    SettingSpec(
        "AI_TIMEOUT", "ai", "Request timeout", "int", unit="seconds",
        minimum=1, maximum=3600,
        description="Per-request ceiling for the http backend. The Agent SDK "
                    "has its own two ceilings below, because the CLI it spawns "
                    "retries requests on its own.",
    ),
    SettingSpec(
        "AI_MAX_RETRIES", "ai", "Max retries", "int", unit="attempts",
        minimum=0, maximum=20,
        description="Extra attempts after the first failure.",
    ),
    SettingSpec(
        "AI_RETRY_BASE_SECONDS", "ai", "Retry base delay", "int", unit="seconds",
        minimum=1, maximum=60,
        description="Initial exponential-backoff delay for yhroot provider retries.",
    ),
    SettingSpec(
        "AI_RETRY_MAX_SECONDS", "ai", "Retry delay ceiling", "int", unit="seconds",
        minimum=1, maximum=300,
        description="Maximum pause between yhroot provider attempts.",
    ),
    SettingSpec(
        "AI_YINHE_MIN_REQUEST_INTERVAL_SECONDS", "ai", "Galaxy OneAPI pool spacing", "int",
        unit="seconds", minimum=6, maximum=300,
        description="Always-on minimum spacing between Galaxy OneAPI request starts across "
                    "the complete primary API-key pool. The default seven seconds keeps the "
                    "service below the gateway's ten-request rolling-minute ceiling.",
    ),
    SettingSpec(
        "AI_PRIMARY_MIN_REQUEST_INTERVAL_SECONDS", "ai", "Primary request spacing", "int",
        unit="seconds", minimum=12, maximum=300,
        description="Minimum spacing between request starts for each primary API key. "
                    "The default 15 seconds stays below a five-request-per-minute key limit "
                    "during the configured weekday daytime window.",
    ),
    SettingSpec(
        "AI_PRIMARY_RATE_LIMIT_COOLDOWN_SECONDS", "ai", "Primary 429 cooldown", "int",
        unit="seconds", minimum=60, maximum=600,
        description="Galaxy OneAPI pool-wide pause after the provider reports a rolling-minute "
                    "rate limit. The default 65 seconds clears the full quota window with "
                    "jitter headroom before any primary key is tried again.",
    ),
    SettingSpec(
        "AI_PRIMARY_RATE_LIMIT_MAX_WAITS", "ai", "Primary 429 wait windows", "int",
        unit="windows", minimum=0, maximum=60,
        description="Full quota-cooldown windows that do not consume the primary route's "
                    "normal failure attempts. Untried primary keys rotate first; the backup "
                    "is considered only after the pool and retry ladder are exhausted.",
    ),
    SettingSpec(
        "AI_PRIMARY_RATE_LIMIT_TIMEZONE", "ai", "Primary rate-limit timezone", "string",
        description="IANA timezone used to decide whether OneAPI's proactive weekday "
                    "request spacing is active.",
    ),
    SettingSpec(
        "AI_PRIMARY_RATE_LIMIT_START_HOUR", "ai", "Primary daytime start", "int",
        unit="hour", minimum=0, maximum=23,
        description="Local weekday hour when proactive OneAPI request spacing begins.",
    ),
    SettingSpec(
        "AI_PRIMARY_RATE_LIMIT_END_HOUR", "ai", "Primary daytime end", "int",
        unit="hour", minimum=1, maximum=24,
        description="Local weekday hour when proactive OneAPI request spacing stops. "
                    "Night and weekend calls are not proactively throttled.",
    ),
    SettingSpec(
        "AI_PRIMARY_MAX_RETRIES", "ai", "Primary max retries", "int", unit="attempts",
        minimum=0, maximum=10,
        description="Extra OneAPI turns after the first. The default gives OneAPI four "
                    "long, rate-limited chances before switching to DeepSeek.",
    ),
    SettingSpec(
        "AI_BACKUP_MAX_RETRIES", "ai", "Backup max retries", "int", unit="attempts",
        minimum=0, maximum=10,
        description="Extra DeepSeek turns after failover before the task fails closed.",
    ),
    SettingSpec(
        "AI_HTTP_FALLBACK", "ai", "Fall back to HTTP", "bool",
        description="When the Agent SDK backend fails outright, retry the call "
                    "on the same provider's OpenAI-compatible route instead of "
                    "failing the stage.",
    ),
    # ── Claude Agent SDK ─────────────────────────────────────────────────
    SettingSpec(
        "CLAUDE_CLI_PATH", "agent_sdk", "claude CLI path", "string",
        placeholder="/usr/local/bin/claude",
        description="Blank lets the SDK locate the CLI (bundled, else on PATH).",
    ),
    SettingSpec(
        "AGENT_REQUEST_TIMEOUT", "agent_sdk", "Request timeout", "int", unit="seconds",
        minimum=30, maximum=3600,
        description="Ceiling for one API call inside the CLI (API_TIMEOUT_MS). "
                    "A reasoning model behind a slow gateway can need two "
                    "minutes for a single digestion turn, so a tight value here "
                    "makes the CLI abort requests that were about to succeed.",
    ),
    SettingSpec(
        "AGENT_TURN_TIMEOUT", "agent_sdk", "Turn timeout", "int", unit="seconds",
        minimum=60, maximum=7200,
        description="Wall-clock ceiling for one `claude` process. Pipeline retries "
                    "run outside the CLI so they can obey provider rate limits.",
    ),
    SettingSpec(
        "AGENT_INTERNAL_MAX_RETRIES", "agent_sdk", "CLI internal retries", "int",
        unit="attempts", minimum=0, maximum=3,
        description="Keep at zero for OneAPI: one request per CLI turn lets the shared "
                    "pipeline limiter pace retries instead of sending an 11-request burst.",
    ),
    # ── TTS ──────────────────────────────────────────────────────────────
    SettingSpec(
        "AIWORK_ROOT", "tts", "VibeVoice install root", "path",
        placeholder="/Volumes/TP-1TB/AIWork",
        description="Holds env_vibevoice*.sh and the VibeVoice project "
                    "directories. Moving it relocates every model script and "
                    "voice sample.",
        allow_blank=False,
    ),
    SettingSpec(
        "TTS_DEVICE", "tts", "Device", "choice",
        options=("mps", "cuda", "cpu"),
        description="Torch device passed to the inference script.",
    ),
    SettingSpec(
        "TTS_DEFAULT_MODEL", "tts", "Default model", "choice",
        dynamic_options=_tts_model_options,
        description="Used when a task does not choose one.",
    ),
    SettingSpec(
        "TTS_DEFAULT_VOICE_1", "tts", "Default voice 1", "choice",
        dynamic_options=_voice_options,
        description="Solo host, and the first speaker in a dialogue.",
    ),
    SettingSpec(
        "TTS_DEFAULT_VOICE_2", "tts", "Default voice 2", "choice",
        dynamic_options=_voice_options,
        description="Second speaker, used only by dialogue scripts.",
    ),
    SettingSpec(
        "VIBEVOICE_TTS_CHUNK_WORDS", "tts", "VibeVoice words per chunk", "int",
        unit="words",
        minimum=0, maximum=5000,
        description="Maximum spoken words sent to each local VibeVoice process. "
                    "Every process uses the same voice preset and random seed; "
                    "PCM frames are joined without re-encoding. 0 disables chunking.",
    ),
    SettingSpec(
        "TTS_RANDOM_SEED", "tts", "Deterministic seed", "int",
        minimum=0, maximum=2**31 - 1,
        description="Fixed RNG seed applied to every VibeVoice chunk so its "
                    "diffusion decoder does not change timbre between parts.",
    ),
    SettingSpec(
        "TTS_STALL_TIMEOUT", "tts", "Stall timeout", "int", unit="seconds",
        minimum=30, maximum=24 * 3600,
        description="Seconds of total silence that count as a wedged run. "
                    "VibeVoice prints progress several times a second, so this "
                    "— not elapsed time — is the real hang detector.",
    ),
    SettingSpec(
        "TTS_TIMEOUT", "tts", "Total timeout", "int", unit="seconds",
        minimum=60, maximum=48 * 3600,
        description="Coarse ceiling on one synthesis run. A full-length script "
                    "legitimately decodes for hours.",
    ),
    SettingSpec(
        "ORPHEUS_TTS_URL", "tts", "Orpheus service URL", "string",
        placeholder="http://10.60.11.3:8088",
        description="Base URL of the authenticated Orpheus async TTS service.",
        allow_blank=False,
    ),
    SettingSpec(
        "ORPHEUS_TTS_API_KEY", "tts", "Orpheus API key", "secret",
        description="Sent only as X-API-Key to the configured Orpheus service.",
    ),
    SettingSpec(
        "ORPHEUS_TTS_MAX_TOKENS", "tts", "Orpheus max tokens", "int",
        minimum=28, maximum=16384,
        description="Maximum audio-token ceiling. Each short utterance gets a "
                    "smaller derived budget and a ceiling-truncation check.",
    ),
    SettingSpec(
        "ORPHEUS_TTS_CHUNK_WORDS", "tts", "Orpheus chunk words", "int",
        minimum=8, maximum=80,
        description="Maximum spoken words sent in one Orpheus utterance. Short "
                    "utterances prevent the CPU model from silently skipping text.",
    ),
    SettingSpec(
        "ORPHEUS_TTS_N_THREADS", "tts", "Orpheus CPU threads", "int",
        minimum=1, maximum=64,
        description="CPU thread count requested from the remote Orpheus worker.",
    ),
    SettingSpec(
        "ORPHEUS_TTS_POLL_SECONDS", "tts", "Orpheus poll interval", "int", unit="seconds",
        minimum=1, maximum=60,
    ),
    SettingSpec(
        "ORPHEUS_TTS_REQUEST_TIMEOUT", "tts", "Orpheus request timeout", "int", unit="seconds",
        minimum=5, maximum=600,
        description="HTTP timeout per submit, status, or audio-download request. "
                    "The overall synthesis ceiling remains Total timeout.",
    ),
    SettingSpec(
        "POCKET_TTS_URL", "tts", "Pocket TTS service URL", "string",
        placeholder="http://10.60.11.3:8090",
        description="Base URL of the Docker-hosted Kyutai Pocket TTS service.",
        allow_blank=False,
    ),
    SettingSpec(
        "POCKET_TTS_API_KEY", "tts", "Pocket TTS API key", "secret",
        description="Optional X-API-Key for a protected Pocket TTS gateway. "
                    "Leave blank for the upstream service on a trusted private network.",
    ),
    SettingSpec(
        "POCKET_TTS_MODEL_REVISION", "tts", "Pocket TTS revision", "string",
        description="Exact upstream git revision deployed in Docker. It is part of "
                    "the verified-audio cache identity.",
        allow_blank=False,
    ),
    SettingSpec(
        "POCKET_TTS_CHUNK_WORDS", "tts", "Pocket external chunk words", "int",
        unit="words", minimum=0, maximum=2000,
        description="Maximum words in one application-level Pocket request. Physical "
                    "script lines/stories remain intact; only complete sentences split "
                    "above this limit. 0 keeps every physical line intact.",
    ),
    SettingSpec(
        "POCKET_TTS_REQUEST_TIMEOUT", "tts", "Pocket request timeout", "int",
        unit="seconds", minimum=5, maximum=3600,
        description="HTTP ceiling for one paragraph/story request. Pocket synthesis "
                    "is synchronous and returns a streaming WAV response.",
    ),
    # ── Audio / visual sync ──────────────────────────────────────────────
    SettingSpec(
        "AV_SYNC_LANGUAGE", "av_sync", "Narration language", "string",
        placeholder="en",
        description="ISO language code passed to Whisper.",
        allow_blank=False,
    ),
    SettingSpec(
        "AV_SYNC_MLX_MODEL", "av_sync", "Apple Silicon model", "string",
        placeholder="mlx-community/whisper-large-v3-turbo-q4",
        description="Post-TTS transcription model. It reads the finished narration "
                    "WAV and never generates or replaces narration.",
        allow_blank=False,
    ),
    SettingSpec(
        "AV_SYNC_TRANSCRIBE_MAX_RETRIES", "av_sync", "Transcription retries", "int",
        minimum=0, maximum=5,
        description="Retry MLX Whisper this many times before continuing with "
                    "estimated timing and a final quality warning.",
    ),
    SettingSpec(
        "AV_SYNC_MIN_WORD_COVERAGE_PERCENT", "av_sync", "Minimum word coverage", "int",
        unit="%", minimum=40, maximum=100,
        description="Minimum share of canonical script words matched to the "
                    "recognized acoustic timeline.",
    ),
    SettingSpec(
        "AV_SYNC_MAX_BOUNDARY_UNCERTAINTY_MS", "av_sync", "Maximum boundary uncertainty", "int",
        unit="ms", minimum=250, maximum=10000,
        description="Mark alignment uncertain when a scene boundary is too far "
                    "from the nearest acoustically matched word anchors.",
    ),
    SettingSpec(
        "AV_SYNC_GEMINI_REVIEW_ENABLED", "av_sync", "Gemini rendered-frame review", "bool",
        description="After rendering, upload labeled scene keyframe sheets and "
                    "matching narration to Gemini Web through project-local OpenCLI.",
    ),
    SettingSpec(
        "AV_SYNC_FRAME_MAX_RETRIES", "av_sync", "Keyframe extraction retries", "int",
        minimum=0, maximum=5,
        description="Retry failed final-MP4 keyframe extraction before recording "
                    "a review warning and continuing delivery.",
    ),
    SettingSpec(
        "AV_SYNC_GEMINI_BATCH_SIZE", "av_sync", "Scenes per Gemini sheet", "int",
        unit="scenes", minimum=1, maximum=12,
        description="Number of labeled keyframes combined into one browser upload.",
    ),
    SettingSpec(
        "AV_SYNC_GEMINI_MIN_SCENE_SCORE", "av_sync", "Minimum scene match", "int",
        unit="/100", minimum=40, maximum=100,
        description="Every rendered scene must reach this semantic match score.",
    ),
    SettingSpec(
        "AV_SYNC_GEMINI_MIN_AVERAGE_SCORE", "av_sync", "Minimum average match", "int",
        unit="/100", minimum=40, maximum=100,
        description="The mean match score across the full video must reach this value.",
    ),
    SettingSpec(
        "AV_SYNC_GEMINI_TIMEOUT", "av_sync", "Gemini review timeout", "int",
        unit="seconds", minimum=30, maximum=900,
        description="Per-contact-sheet wait for the signed-in Gemini Web response.",
    ),
    SettingSpec(
        "AV_SYNC_GEMINI_MAX_RETRIES", "av_sync", "Gemini review retries", "int",
        minimum=0, maximum=5,
        description="Bounded retries for a missing, late, or malformed Gemini Web "
                    "response. Exhausted retries become a final warning, not a render block.",
    ),
    # ── Render ───────────────────────────────────────────────────────────
    SettingSpec(
        "HYPERFRAME_DIR", "render", "HyperFrames project", "path",
        placeholder="hyperframe",
        description="Project the render runs in; its node_modules holds the "
                    "pinned CLI, GSAP and Lottie.",
        allow_blank=False,
    ),
    SettingSpec(
        "HYPERFRAMES_VERSION", "render", "CLI version", "string",
        placeholder="0.6.99",
        description="Pinned version used for the `npx` fallback when the local "
                    "CLI is missing. Keep it pinned — `latest` re-installs on "
                    "every cold run.",
        allow_blank=False,
    ),
    SettingSpec(
        "RENDER_RESOLUTION", "render", "Resolution", "choice",
        options=("landscape", "portrait", "square"),
        description="landscape is 1920x1080.",
    ),
    SettingSpec(
        "RENDER_FPS", "render", "Frame rate", "int", unit="fps",
        minimum=1, maximum=60,
        description="Frames captured per second of video — the single biggest "
                    "factor in render time.",
    ),
    SettingSpec(
        "RENDER_QUALITY", "render", "Quality", "choice",
        options=("draft", "standard", "high"),
        description="Encoder preset passed to the CLI.",
    ),
    SettingSpec(
        "RENDER_WORKERS", "render", "Workers", "string",
        placeholder="2",
        description="Parallel capture workers: an integer, or `auto` to let the "
                    "CLI decide.",
        allow_blank=False,
    ),
    # ── Direction ────────────────────────────────────────────────────────
    SettingSpec(
        "DIRECTOR_ENABLED", "director", "Agent direction", "bool",
        description="Off renders the deterministic scene drafts as-is.",
    ),
    SettingSpec(
        "DIRECTOR_MAX_SCENES", "director", "Max directed scenes", "int", unit="scenes",
        minimum=0, maximum=200,
        description="Hand only the first N scenes to agents (0 = all). Each "
                    "crew is a CLI process and a provider round-trip: this is "
                    "the cost/latency dial.",
    ),
    SettingSpec(
        "INSPECT_ENABLED", "director", "Layout inspection", "bool",
        description="Run `hyperframes inspect` before the render and let agents "
                    "repair the layout failures it reports. Costs a headless "
                    "Chrome pass over the timeline.",
    ),
    # ── Source extraction ────────────────────────────────────────────────
    SettingSpec(
        "YTDLP_JS_RUNTIME", "source", "JS runtime", "string",
        placeholder="node:/usr/local/bin/node",
        description="Runtime yt-dlp uses to solve YouTube's player challenges. "
                    "Blank auto-detects node on PATH.",
    ),
    SettingSpec(
        "YTDLP_REMOTE_COMPONENTS", "source", "Remote components", "string",
        placeholder="ejs:github",
        description="Passed to --remote-components. `off` (or blank) disables "
                    "fetching player components entirely.",
    ),
    SettingSpec(
        "YTDLP_COOKIES", "source", "Cookies file", "string",
        placeholder="/path/to/cookies.txt",
        description="Netscape cookie file for age- or bot-gated videos. Takes "
                    "precedence over the browser below.",
    ),
    SettingSpec(
        "YTDLP_COOKIES_FROM_BROWSER", "source", "Cookies from browser", "string",
        placeholder="chrome",
        description="Borrow cookies from a local browser profile (chrome, "
                    "safari, firefox…) when no cookie file is set.",
    ),
    # ── Footage ──────────────────────────────────────────────────────────
    SettingSpec(
        "FOOTAGE_USER_AGENT", "footage", "User agent", "string",
        placeholder="FrontierTechSN/1.0 (local AI media scout)",
        description="Wikimedia asks API clients to identify themselves.",
        allow_blank=False,
    ),
    SettingSpec(
        "FOOTAGE_TIMEOUT", "footage", "Request timeout", "int", unit="seconds",
        minimum=5, maximum=600,
    ),
    SettingSpec(
        "FOOTAGE_MAX_BYTES", "footage", "Max clip size", "int", unit="bytes",
        minimum=1024 * 1024, maximum=2 * 1024 * 1024 * 1024,
        description="Ceiling per download, so an autonomous scout cannot pull "
                    "an archival master. 52428800 = 50 MB.",
    ),
    SettingSpec(
        "OPENCLI_BIN", "footage", "OpenCLI wrapper", "path",
        placeholder="scripts/opencli.sh",
        description="Project-local wrapper. Do not point this at a global skill install.",
        allow_blank=False,
    ),
    SettingSpec(
        "OPENCLI_PROFILE", "footage", "OpenCLI browser profile", "string",
        description="Browser Bridge profile ID or alias. Pin the Ego Lite profile when multiple browsers are connected. "
                    "After a browser-profile import, ensure each OpenCLI extension has a distinct profile ID; "
                    "duplicated IDs cause stale pages and command timeouts. Blank auto-selects the only connected profile.",
    ),
    SettingSpec(
        "OPENCLI_BROWSER_RUNTIME", "footage", "OpenCLI browser runtime", "choice",
        options=("bridge", "isolated-headless"),
        description="Bridge preserves the existing operator Chrome behavior. "
                    "Isolated-headless explicitly starts a dedicated Chrome for "
                    "Testing/Chromium profile and never falls back to operator Chrome.",
        allow_blank=False,
        restart_required=True,
    ),
    SettingSpec(
        "OPENCLI_ISOLATED_CHROME_BIN", "footage", "Isolated Chrome binary", "string",
        placeholder="/path/to/Google Chrome for Testing",
        description="Executable for the dedicated runtime. Use Chrome for Testing "
                    "or Chromium because normal Chrome no longer accepts unpacked "
                    "extensions from command-line flags.",
        restart_required=True,
    ),
    SettingSpec(
        "OPENCLI_ISOLATED_USER_DATA_DIR", "footage", "Isolated Chrome data", "path",
        placeholder="data/opencli-isolated-browser",
        description="Dedicated ignored user-data directory. Human Chrome profile "
                    "directories are rejected.",
        allow_blank=False,
        restart_required=True,
    ),
    SettingSpec(
        "OPENCLI_ISOLATED_EXTENSION_PATH", "footage", "OpenCLI extension directory", "string",
        placeholder="/path/to/unpacked/opencli-extension",
        description="Unpacked extension directory containing OpenCLI's manifest.json.",
        restart_required=True,
    ),
    SettingSpec(
        "OPENCLI_ISOLATED_PROFILE", "footage", "Isolated OpenCLI profile", "string",
        placeholder="frontiertechsn-headless",
        description="Exact Browser Bridge context id or local alias discovered "
                    "during bootstrap. Required in isolated mode so commands can "
                    "never auto-select a human browser profile.",
        restart_required=True,
    ),
    SettingSpec(
        "OPENCLI_ISOLATED_DEBUG_PORT", "footage", "Isolated Chrome debug port", "int",
        unit="port", minimum=1024, maximum=65535,
        description="Loopback-only port used for readiness and process ownership checks.",
        restart_required=True,
    ),
    SettingSpec(
        "OPENCLI_ISOLATED_START_TIMEOUT", "footage", "Isolated Chrome startup", "int",
        unit="seconds", minimum=5, maximum=120,
        description="Maximum wait for both Chrome and the exact isolated Browser "
                    "Bridge profile to become ready.",
        restart_required=True,
    ),
    SettingSpec(
        "OPENCLI_TIMEOUT", "footage", "OpenCLI command timeout", "int", unit="seconds",
        minimum=10, maximum=1800,
    ),
    SettingSpec(
        "OPENCLI_MAX_ATTEMPTS", "footage", "OpenCLI attempts", "int", unit="attempts",
        minimum=1, maximum=20,
        description="Total attempts for ChatGPT, Gemini, YouTube, and X browser-backed stages.",
    ),
    SettingSpec(
        "OPENCLI_RETRY_BASE_SECONDS", "footage", "OpenCLI retry base", "int", unit="seconds",
        minimum=1, maximum=60,
    ),
    SettingSpec(
        "OPENCLI_WEB_REQUEST_INTERVAL_SECONDS", "footage",
        "Gemini / ChatGPT request interval", "int", unit="seconds",
        minimum=180, maximum=600,
        description="Minimum start-to-start spacing shared by all OpenCLI Gemini "
                    "and ChatGPT web commands, including concurrent tasks. "
                    "Choose 180–600 seconds (3–10 minutes).",
    ),
    SettingSpec(
        "DAILY_NEWS_CHATGPT_REVIEW_MIN_LEVEL", "footage",
        "News review minimum ChatGPT level", "choice",
        options=("medium", "high", "xhigh"),
        description="Preferred level attempted first and minimum fallback "
                    "accepted for daily-news Fact check.",
        allow_blank=False,
    ),
    SettingSpec(
        "DAILY_NEWS_CHATGPT_REVIEW_MAX_LEVEL", "footage",
        "News review maximum ChatGPT level", "choice",
        options=("medium", "high", "xhigh"),
        description="Maximum fallback accepted after a failed switch. Pro is "
                    "intentionally unavailable and is rejected.",
        allow_blank=False,
    ),
    SettingSpec(
        "DAILY_NEWS_WEB_REVIEW_TIMEOUT", "footage",
        "News web review timeout", "int", unit="seconds",
        minimum=15, maximum=180,
        description="Per-provider wait for each short daily-news story-review token.",
    ),
    SettingSpec(
        "WEB_FOOTAGE_ENABLED", "footage", "Web footage", "bool",
        description="Allow YouTube discovery in hybrid footage mode.",
    ),
    SettingSpec(
        "WEB_FOOTAGE_GEMINI_ENABLED", "footage", "Gemini web analysis", "bool",
        description="Send public candidate links and matching script excerpts to the logged-in Gemini web app for trim selection.",
    ),
    SettingSpec(
        "WEB_FOOTAGE_GEMINI_TIMEOUT", "footage", "Gemini analysis timeout", "int", unit="seconds",
        minimum=15, maximum=600,
    ),
    SettingSpec(
        "WEB_FOOTAGE_CLIP_SECONDS", "footage", "Web clip max length", "int", unit="seconds",
        minimum=5, maximum=60,
        description="Maximum B-roll clip length downloaded from YouTube. Gemini is asked to aim for close to this duration.",
    ),
    SettingSpec(
        "WEB_FOOTAGE_CLIP_MIN_SECONDS", "footage", "Web clip min length", "int", unit="seconds",
        minimum=3, maximum=30,
        description="Minimum B-roll clip length. Short Gemini intervals are extended to at least this many seconds.",
    ),
    SettingSpec(
        "WEB_FOOTAGE_DOWNLOAD_TIMEOUT", "footage", "Web download timeout", "int", unit="seconds",
        minimum=30, maximum=3600,
    ),
    # ── Thumbnail ─────────────────────────────────────────────────────────
    SettingSpec(
        "THUMBNAIL_CHATGPT_TIMEOUT", "thumbnail", "ChatGPT image timeout", "int",
        unit="seconds", minimum=30, maximum=1800,
        description="Maximum time to wait for ChatGPT Web to generate and export the cover image.",
    ),
    SettingSpec(
        "COLLAGE_CHATGPT_TIMEOUT", "collage", "Collage still timeout", "int",
        unit="seconds", minimum=60, maximum=1800,
        description="Maximum wait per editorial collage still generated through signed-in ChatGPT Web.",
    ),
    SettingSpec(
        "COLLAGE_GEMINI_TIMEOUT", "collage", "Collage video timeout", "int",
        unit="seconds", minimum=120, maximum=3600,
        description="Maximum wait per collage animation generated through Gemini Web Create Video.",
    ),
    SettingSpec(
        "COLLAGE_GEMINI_MAX_SECONDS", "collage", "Gemini max clip length", "int",
        unit="seconds", minimum=4, maximum=60,
        description=(
            "Provider ceiling for one Gemini Create Video result. Each paper-collage clip "
            "targets the shorter of this value and its narration scene."
        ),
    ),
    # ── Paths ────────────────────────────────────────────────────────────
    SettingSpec(
        "OUTPUTS_DIR", "paths", "Outputs directory", "path",
        placeholder="outputs",
        description="Per-task working directory: script, audio, scenes, video.",
        allow_blank=False,
        restart_required=True,
    ),
    SettingSpec(
        "UPLOADS_DIR", "paths", "Uploads directory", "path",
        placeholder="uploads",
        description="Where uploaded EPUB/PDF sources are stored.",
        allow_blank=False,
    ),
    SettingSpec(
        "DB_PATH", "paths", "Task database", "path",
        placeholder="tasks.db",
        description="SQLite file holding tasks and providers.",
        allow_blank=False,
        restart_required=True,
    ),
    SettingSpec(
        "ISLA_READER_PROMOTION_SCRIPT", "paths", "Isla Reader script", "path",
        description="External generator used by the EPUB curated-highlights mode.",
        allow_blank=False,
    ),
)

_SPEC_BY_KEY: dict[str, SettingSpec] = {spec.key: spec for spec in SPECS}

# Defaults captured from config before any override is applied — i.e. what
# .env (or the built-in fallback) produced. Filled in by apply_saved().
_DEFAULTS: dict[str, Any] = {}


def store_path() -> Path:
    return _config().PROJECT_ROOT / "data" / "settings.json"


def _read_store() -> dict[str, Any]:
    path = store_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    values = payload.get("values")
    if not isinstance(values, dict):
        return {}
    legacy_review_level = values.get("DAILY_NEWS_CHATGPT_REVIEW_MODEL")
    if (
        legacy_review_level in _CHATGPT_REVIEW_LEVEL_ORDER
        and "DAILY_NEWS_CHATGPT_REVIEW_MIN_LEVEL" not in values
        and legacy_review_level
        != _DEFAULTS.get(
            "DAILY_NEWS_CHATGPT_REVIEW_MIN_LEVEL",
            _config().DAILY_NEWS_CHATGPT_REVIEW_MIN_LEVEL,
        )
    ):
        values["DAILY_NEWS_CHATGPT_REVIEW_MIN_LEVEL"] = legacy_review_level
    # Prune retired overrides rather than merely hiding them, so they cannot
    # return after restart or keep influencing an older runtime path.
    if any(key in values for key in _RETIRED_KEYS):
        values = {
            key: value for key, value in values.items() if key not in _RETIRED_KEYS
        }
        _write_store(values)
    # TTS_CHUNK_WORDS used to drive every provider. Preserve an existing Admin
    # override as the VibeVoice-only limit after the model-specific migration.
    if (
        "TTS_CHUNK_WORDS" in values
        and "VIBEVOICE_TTS_CHUNK_WORDS" not in values
    ):
        values["VIBEVOICE_TTS_CHUNK_WORDS"] = values["TTS_CHUNK_WORDS"]
    return {k: v for k, v in values.items() if k in _SPEC_BY_KEY}


def _write_store(values: Mapping[str, Any]) -> None:
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"version": _STORE_VERSION, "values": dict(sorted(values.items()))},
        indent=2,
        ensure_ascii=False,
    )
    # Write-then-rename so a crash mid-write cannot leave a truncated store that
    # would silently drop every saved setting on the next start.
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name, suffix=".tmp", delete=False
    ) as handle:
        handle.write(payload)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def spec_options(spec: SettingSpec) -> tuple[str, ...]:
    return spec.dynamic_options() if spec.dynamic_options else spec.options


def coerce(spec: SettingSpec, raw: Any) -> Any:
    """Turn a submitted value into the type ``config`` expects.

    Raises :class:`SettingsError` with a message meant for the Admin UI.
    """
    if spec.type == "bool":
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
        raise SettingsError(f"{spec.label}: expected true or false, got {raw!r}")

    if spec.type == "int":
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError):
            raise SettingsError(f"{spec.label}: expected a whole number, got {raw!r}") from None
        if spec.minimum is not None and value < spec.minimum:
            raise SettingsError(f"{spec.label}: must be at least {spec.minimum}")
        if spec.maximum is not None and value > spec.maximum:
            raise SettingsError(f"{spec.label}: must be at most {spec.maximum}")
        return value

    text = "" if raw is None else str(raw).strip()

    if spec.type == "choice":
        options = spec_options(spec)
        if text not in options:
            raise SettingsError(f"{spec.label}: must be one of {', '.join(options)}")
        return text

    if spec.type == "path":
        if not text:
            raise SettingsError(f"{spec.label}: a path is required")
        return _config().resolve_project_path(text)

    return text


def _store_value(spec: SettingSpec, value: Any) -> Any:
    """Serialize a coerced value for the JSON store."""
    if spec.type in ("bool", "int"):
        return value
    return str(value)


def _is_blank(spec: SettingSpec, raw: Any) -> bool:
    if spec.type == "bool":
        return False
    return str("" if raw is None else raw).strip() == ""


def _blank_resets(spec: SettingSpec) -> bool:
    """Whether a blank submission drops the override instead of storing an empty.

    Numbers, choices and paths have no meaningful empty value, so clearing such
    a field in the Admin form reads as "put it back the way it was".
    """
    return not spec.allow_blank or spec.type in ("int", "choice", "path")


def _resolved(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Defaults with the stored overrides layered on, all coerced."""
    values = dict(_DEFAULTS)
    for key, raw in overrides.items():
        spec = _SPEC_BY_KEY.get(key)
        if spec is None:
            continue
        try:
            values[key] = coerce(spec, raw)
        except SettingsError:
            # A hand-edited or stale store entry must not stop the app from
            # starting; fall back to the .env-seeded default for that key.
            continue
    return values


def _validate_chatgpt_review_range(values: Mapping[str, Any]) -> None:
    minimum = str(values.get("DAILY_NEWS_CHATGPT_REVIEW_MIN_LEVEL", ""))
    maximum = str(values.get("DAILY_NEWS_CHATGPT_REVIEW_MAX_LEVEL", ""))
    if minimum not in _CHATGPT_REVIEW_LEVEL_ORDER:
        raise SettingsError("News review minimum ChatGPT level is invalid")
    if maximum not in _CHATGPT_REVIEW_LEVEL_ORDER:
        raise SettingsError("News review maximum ChatGPT level is invalid")
    if _CHATGPT_REVIEW_LEVEL_ORDER[minimum] > _CHATGPT_REVIEW_LEVEL_ORDER[maximum]:
        raise SettingsError(
            "News review ChatGPT range: minimum level cannot exceed maximum level"
        )


def _tree_fingerprint(root: Path) -> tuple[tuple[str, str, int, str], ...]:
    """Content fingerprint for a small durable directory tree."""
    rows: list[tuple[str, str, int, str]] = []
    try:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise SettingsError(
                    f"Refusing to migrate symlinked podcast state: {path}"
                )
            if path.is_dir():
                rows.append((relative, "dir", 0, ""))
                continue
            if not path.is_file():
                raise SettingsError(
                    f"Refusing to migrate non-file podcast state: {path}"
                )
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            rows.append((relative, "file", path.stat().st_size, digest.hexdigest()))
    except OSError as exc:
        raise SettingsError(f"Could not inspect podcast state under {root}: {exc}") from exc
    return tuple(rows)


def _migrate_podcast_state(current_root: Path, desired_root: Path) -> None:
    """Copy the global podcast ledger when OUTPUTS_DIR changes.

    Task artifacts retain their own persisted roots; the podcast feed is the
    one cross-task artifact served from the configured root and must move as a
    single verified unit before a restart can switch the static mount.
    """
    current_root = current_root.resolve()
    desired_root = desired_root.resolve()
    if current_root == desired_root:
        return
    source = current_root / "podcast"
    destination = desired_root / "podcast"
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise SettingsError(
            "Refusing to migrate podcast state between nested output roots"
        )
    if not source.exists():
        return
    if source.is_symlink() or not source.is_dir():
        raise SettingsError(f"Podcast source is not a regular directory: {source}")
    source_fingerprint = _tree_fingerprint(source)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise SettingsError(
                f"Podcast destination is not a regular directory: {destination}"
            )
        if _tree_fingerprint(destination) != source_fingerprint:
            raise SettingsError(
                "The new outputs directory already contains different podcast "
                "state; reconcile it before changing OUTPUTS_DIR"
            )
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".podcast-migration-", dir=destination.parent)
    )
    try:
        shutil.rmtree(staging)
        shutil.copytree(source, staging)
        if _tree_fingerprint(staging) != source_fingerprint:
            raise SettingsError("Podcast state migration did not verify byte-for-byte")
        os.replace(staging, destination)
    except OSError as exc:
        raise SettingsError(
            f"Could not migrate podcast state to {destination}: {exc}"
        ) from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _apply_live_values(after: Mapping[str, Any]) -> None:
    """Apply hot settings while retaining restart-bound process resources."""
    config = _config()
    live = dict(after)
    for spec in SPECS:
        if spec.restart_required:
            live[spec.key] = getattr(config, spec.key)
    config.apply_values(live)


def apply_saved() -> None:
    """Snapshot the .env defaults, then apply the saved overrides to ``config``.

    The snapshot is taken once, on the first call (from config's own import,
    before anything has been overridden). Later calls only re-apply the store,
    so a stray second call can never promote an override into a "default".
    """
    config = _config()
    global _DEFAULTS
    if not _DEFAULTS:
        _DEFAULTS = {spec.key: getattr(config, spec.key) for spec in SPECS}
    resolved = _resolved(_read_store())
    try:
        _validate_chatgpt_review_range(resolved)
    except SettingsError:
        # A stale or hand-edited store must not make startup unusable. Restore
        # the known-safe built-in/.env range; interactive saves reject the same
        # invalid order below instead of silently changing it.
        resolved["DAILY_NEWS_CHATGPT_REVIEW_MIN_LEVEL"] = _DEFAULTS[
            "DAILY_NEWS_CHATGPT_REVIEW_MIN_LEVEL"
        ]
        resolved["DAILY_NEWS_CHATGPT_REVIEW_MAX_LEVEL"] = _DEFAULTS[
            "DAILY_NEWS_CHATGPT_REVIEW_MAX_LEVEL"
        ]
        _validate_chatgpt_review_range(resolved)
    _migrate_podcast_state(
        Path(config.OUTPUTS_DIR),
        Path(resolved["OUTPUTS_DIR"]),
    )
    config.apply_values(resolved)


def defaults() -> dict[str, Any]:
    return dict(_DEFAULTS)


def restart_required_change_pending(key: str) -> bool:
    spec = _SPEC_BY_KEY.get(key)
    if spec is None or not spec.restart_required:
        return False
    desired = _resolved(_read_store()).get(key)
    return desired != getattr(_config(), key)


def mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}"


def _display(spec: SettingSpec, value: Any) -> Any:
    if spec.type == "secret":
        return ""
    if spec.type == "path":
        return str(value)
    if spec.type in ("bool", "int"):
        return value
    return "" if value is None else str(value)


def schema() -> list[dict[str, Any]]:
    """Groups + fields with their current values, for the Admin console."""
    config = _config()
    overrides = _read_store()
    fields_by_group: dict[str, list[dict[str, Any]]] = {group.id: [] for group in GROUPS}

    for spec in SPECS:
        current = getattr(config, spec.key)
        default = _DEFAULTS.get(spec.key, current)
        entry: dict[str, Any] = {
            "key": spec.key,
            "label": spec.label,
            "type": spec.type,
            "description": spec.description,
            "placeholder": spec.placeholder,
            "unit": spec.unit,
            "options": list(spec_options(spec)),
            "value": _display(spec, current),
            "default": _display(spec, default),
            "is_overridden": spec.key in overrides,
            "restart_required": spec.restart_required,
            "allow_blank": not _blank_resets(spec),
        }
        if spec.type == "secret":
            entry["masked"] = mask_secret(str(current))
            entry["default_masked"] = mask_secret(str(default))
            entry["is_set"] = bool(str(current))
        fields_by_group[spec.group].append(entry)

    return [
        {
            "id": group.id,
            "label": group.label,
            "description": group.description,
            "fields": fields_by_group[group.id],
        }
        for group in GROUPS
    ]


def update(submitted: Mapping[str, Any]) -> list[str]:
    """Validate, persist and apply the submitted values.

    Only the keys present are touched. A blank submission for a field that
    cannot be blank drops the override, which restores the .env/built-in
    default. Returns the changed keys that need a restart to take full effect.
    """
    unknown = [key for key in submitted if key not in _SPEC_BY_KEY]
    if unknown:
        raise SettingsError(f"Unknown setting(s): {', '.join(sorted(unknown))}")

    overrides = _read_store()
    before = _resolved(overrides)
    for key, raw in submitted.items():
        spec = _SPEC_BY_KEY[key]
        if _is_blank(spec, raw) and _blank_resets(spec):
            overrides.pop(key, None)
            continue
        value = coerce(spec, raw)
        if value == _DEFAULTS.get(key):
            # Identical to the default: drop the override rather than freezing a
            # copy of it, so later .env edits still show through.
            overrides.pop(key, None)
        else:
            overrides[key] = _store_value(spec, value)

    after = _resolved(overrides)
    _validate_chatgpt_review_range(after)
    _migrate_podcast_state(
        Path(_config().OUTPUTS_DIR),
        Path(after["OUTPUTS_DIR"]),
    )
    _write_store(overrides)
    _apply_live_values(after)
    return [
        key
        for key in submitted
        if _SPEC_BY_KEY[key].restart_required and before.get(key) != after.get(key)
    ]


def reset(keys: Iterable[str]) -> list[str]:
    """Drop the stored overrides for ``keys``, restoring the .env defaults.

    Returns the reset keys that need a restart to take full effect.
    """
    keys = list(keys)
    unknown = [key for key in keys if key not in _SPEC_BY_KEY]
    if unknown:
        raise SettingsError(f"Unknown setting(s): {', '.join(sorted(unknown))}")
    overrides = _read_store()
    before = _resolved(overrides)
    for key in keys:
        overrides.pop(key, None)
    after = _resolved(overrides)
    _migrate_podcast_state(
        Path(_config().OUTPUTS_DIR),
        Path(after["OUTPUTS_DIR"]),
    )
    _write_store(overrides)
    _apply_live_values(after)
    return [
        key
        for key in keys
        if _SPEC_BY_KEY[key].restart_required and before.get(key) != after.get(key)
    ]

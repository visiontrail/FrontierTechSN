import json
import logging
import re
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from backend import config
from backend.pipeline.extractors.base import ExtractedContent
from backend.pipeline.process_logging import run_capture_logged

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]

# Subtitle languages we ask yt-dlp to fetch. English first, then Simplified and
# Traditional Chinese variants. The downstream summarizer/scriptwriter always
# produces an English podcast, so a non-English track only needs to be readable.
SUB_LANGS = "en,en-US,en-GB,zh,zh-Hans,zh-Hant,zh-CN,zh-TW,zh-HK"

# Order in which we prefer an available track. English is preferred when present
# (no translation needed); otherwise we fall back to any Chinese variant.
LANG_PRIORITY = [
    "en", "en-us", "en-gb",
    "zh-hans", "zh-cn", "zh",
    "zh-hant", "zh-tw", "zh-hk",
]

BOT_CHECK_HINTS = (
    "sign in to confirm",
    "not a bot",
    "cookies-from-browser",
    "cookies for the authentication",
)


def _lang_of(path: Path) -> str:
    """Extract the subtitle language code from a yt-dlp filename like
    "<id>.<lang>.vtt" (e.g. "abc123.zh-Hans.vtt" -> "zh-hans")."""
    parts = path.name.split(".")
    return parts[-2].lower() if len(parts) >= 3 else ""


def _pick_subtitle(vtt_files: list[Path]) -> Path:
    """Choose the best available subtitle file by language priority, falling back
    to any track that was downloaded."""
    def rank(path: Path) -> int:
        lang = _lang_of(path)
        for i, pref in enumerate(LANG_PRIORITY):
            if lang == pref or lang.startswith(pref):
                return i
        return len(LANG_PRIORITY)

    return sorted(vtt_files, key=rank)[0]


def _find_node_runtime() -> str | None:
    configured = config.YTDLP_JS_RUNTIME.strip()
    if configured:
        return configured

    node_path = shutil.which("node")
    if node_path:
        return f"node:{node_path}"

    bundled_node = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
    if bundled_node.exists():
        return f"node:{bundled_node}"

    return None


def _yt_dlp_common_args(*, include_cookies: bool = True) -> list[str]:
    args: list[str] = []

    js_runtime = _find_node_runtime()
    if js_runtime:
        args.extend(["--js-runtimes", js_runtime])

    remote_components = config.YTDLP_REMOTE_COMPONENTS.strip()
    if remote_components.lower() not in {"", "0", "false", "none", "off"}:
        args.extend(["--remote-components", remote_components])

    if include_cookies:
        cookies_file = config.YTDLP_COOKIES.strip()
        cookies_browser = config.YTDLP_COOKIES_FROM_BROWSER.strip()
        if cookies_file:
            args.extend(["--cookies", cookies_file])
        elif cookies_browser:
            args.extend(["--cookies-from-browser", cookies_browser])

    return args


def _is_bot_check(stderr: str | None) -> bool:
    text = (stderr or "").lower()
    return any(hint in text for hint in BOT_CHECK_HINTS)


def _yt_dlp_command(args: list[str], url: str, *, include_cookies: bool = True) -> list[str]:
    return ["yt-dlp", *_yt_dlp_common_args(include_cookies=include_cookies), *args, url]


def _parse_vtt(vtt_text: str) -> str:
    lines = []
    seen = set()
    for line in vtt_text.splitlines():
        line = line.strip()
        if not line or line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if "-->" in line:
            continue
        if re.match(r"^\d+$", line):
            continue
        clean = re.sub(r"<[^>]+>", "", line)
        clean = re.sub(r"\[.*?\]", "", clean).strip()
        if clean and clean not in seen:
            seen.add(clean)
            lines.append(clean)
    return " ".join(lines)


async def extract_youtube(url: str, log: LogCallback | None = None) -> ExtractedContent:
    emit = lambda message: log(message) if log else logger.info(message)
    emit(f"Extracting YouTube subtitles from {url}")

    common_args = _yt_dlp_common_args()
    if any(arg == "--js-runtimes" for arg in common_args):
        emit("yt-dlp JavaScript runtime configured")
    else:
        emit("yt-dlp JavaScript runtime not found; YouTube extraction may be limited")
    if "--remote-components" in common_args:
        emit("yt-dlp remote JavaScript components enabled")
    if "--cookies" in common_args:
        emit("yt-dlp cookies file configured")
    elif "--cookies-from-browser" in common_args:
        emit("yt-dlp browser cookies configured")

    meta_command = _yt_dlp_command(["--dump-json", "--no-download"], url)
    meta_result = run_capture_logged(
        name="yt-dlp metadata",
        command=meta_command,
        logger=logger,
        log=log,
        timeout=60,
        stdout_log_limit=2_000,
    )
    if meta_result.returncode != 0:
        if _is_bot_check(meta_result.stderr):
            raise RuntimeError(
                "yt-dlp metadata failed because YouTube requested sign-in/bot verification. "
                "Set YTDLP_COOKIES_FROM_BROWSER=chrome (or another logged-in browser), "
                "or set YTDLP_COOKIES=/path/to/cookies.txt, then restart ./scripts/start.sh. "
                f"yt-dlp stderr: {meta_result.stderr}"
            )
        raise RuntimeError(f"yt-dlp metadata failed: {meta_result.stderr}")

    meta = json.loads(meta_result.stdout)
    title = meta.get("title", "Unknown")
    channel = meta.get("channel", "")
    duration = meta.get("duration", 0)

    with tempfile.TemporaryDirectory() as tmpdir:
        sub_command = _yt_dlp_command([
            "--write-auto-sub", "--write-sub",
            "--sub-langs", SUB_LANGS,
            "--sub-format", "vtt",
            "--skip-download",
            "-o", f"{tmpdir}/%(id)s.%(ext)s",
        ], url)
        sub_result = run_capture_logged(
            name="yt-dlp subtitles",
            command=sub_command,
            logger=logger,
            log=log,
            timeout=120,
        )

        vtt_files = list(Path(tmpdir).glob("*.vtt"))
        if not vtt_files:
            raise RuntimeError(
                "No English or Chinese subtitles found for this video. "
                f"yt-dlp output: {sub_result.stderr}"
            )

        chosen = _pick_subtitle(vtt_files)
        sub_lang = _lang_of(chosen)
        vtt_text = chosen.read_text(encoding="utf-8")
        text = _parse_vtt(vtt_text)

    if not text.strip():
        raise RuntimeError("Extracted subtitle text is empty")

    emit(f"Extracted {len(text.split())} words from '{title}' (subtitle lang: {sub_lang or 'unknown'})")
    return ExtractedContent(
        source_type="youtube",
        title=title,
        text=text,
        metadata={"channel": channel, "duration": duration, "url": url, "subtitle_lang": sub_lang},
    )

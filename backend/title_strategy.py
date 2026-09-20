"""Shared editorial policy for production titles and upload-copy titles."""

from backend import config

YOUTUBE_TITLE_MAX_CHARS = 100


def with_title_strategy(prompt: str) -> str:
    # Read on every request so Admin edits apply to both generation paths.
    strategy = (config.PROMPTS_DIR / "title_strategy.txt").read_text(encoding="utf-8").strip()
    if not strategy:
        raise ValueError("The shared title strategy must not be empty")
    return f"{prompt}\n\n<shared_title_strategy>\n{strategy}\n</shared_title_strategy>"

"""Persisted next-day backoff; unavailable UI is distinct from proven quota exhaustion."""
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from backend import config


class GeminiVideoDeferred(RuntimeError):
    def __init__(self, reason: str, evidence: str, scene_id: str = ""):
        super().__init__(evidence)
        self.reason = reason
        self.scene_id = scene_id


def availability_reason(error: Exception) -> str | None:
    message = str(error)
    if "GEMINI_VIDEO_QUOTA_EXHAUSTED" in message:
        return "quota_exhausted"
    if "GEMINI_VIDEO_UNAVAILABLE" in message:
        return "video_unavailable"
    if "GEMINI_VIDEO_GENERATION_FAILED" in message:
        return "generation_failed"
    # Navigation, missing controls and uncertain submissions are not terminal
    # video failures. Keep their reconciliation path; never replay unknown work.
    return None


def next_retry_at(stamp: str) -> str:
    zone = ZoneInfo(config.AI_PRIMARY_RATE_LIMIT_TIMEZONE)
    local = datetime.fromisoformat(stamp).astimezone(zone)
    # Retry once on the next calendar day, at the same local time. This also
    # gives rolling limits a full day, rather than hammering just after midnight.
    retry = datetime.combine(local.date() + timedelta(days=1),
                             time(local.hour, local.minute, local.second), zone)
    return retry.astimezone(timezone.utc).isoformat()

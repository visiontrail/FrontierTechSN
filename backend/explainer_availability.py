"""Evidence-based retry timing; a failed generation is not a daily quota."""
import re
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


def next_retry_at(stamp: str, reason: str = "quota_exhausted", *,
                  evidence: str = "", failures: int = 1) -> str:
    zone = ZoneInfo(config.AI_PRIMARY_RATE_LIMIT_TIMEZONE)
    local = datetime.fromisoformat(stamp).astimezone(zone)
    if reason != "quota_exhausted":
        # Bounded exponential cooldown for transient failures, not a claim about
        # the provider's quota. A manual retry can supersede it with new evidence.
        minutes = min(60, (15 if reason == "generation_failed" else 30) * 2 ** min(2, max(0, failures - 1)))
        return (local + timedelta(minutes=minutes)).astimezone(timezone.utc).isoformat()
    relative = re.search(r"(?:try again|retry|resets?|refresh(?:es)?)\s+(?:in|after)\s+(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?)", evidence, re.I)
    clock = re.search(r"(?:resets?|refresh(?:es)?|try again)\s+(?:at\s+)?(\d{1,2}):(\d{2})\s*(AM|PM)?\b", evidence, re.I)
    if relative and 0 < float(relative[1]) <= 168:
        minutes = float(relative[1]) * (60 if relative[2].lower().startswith(("hour", "hr")) else 1)
        retry = local + timedelta(minutes=minutes + 1)
    elif clock and int(clock[1]) < 24 and int(clock[2]) < 60 and (not clock[3] or 1 <= int(clock[1]) <= 12):
        hour = int(clock[1]) % 12 + (12 if clock[3].upper() == "PM" else 0) if clock[3] else int(clock[1])
        retry = datetime.combine(local.date(), time(hour, int(clock[2])), zone)
        if retry <= local or re.search(r"\btomorrow\b", evidence, re.I):
            retry += timedelta(days=1)
        retry += timedelta(minutes=1)
    else:
        midnight = datetime.combine(local.date() + timedelta(days=1), time(0, 5), zone)
        # An explicit daily limit waits for the next day. Otherwise use a
        # bounded check interval, not an invented fixed daily entitlement.
        retry = midnight if re.search(r"\bdaily\b|per day|tomorrow|每天|每日|明天", evidence, re.I) else min(midnight, local + timedelta(hours=5))
        retry = max(retry, local + timedelta(minutes=15))
    return retry.astimezone(timezone.utc).isoformat()

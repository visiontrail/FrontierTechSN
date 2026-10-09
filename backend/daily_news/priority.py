"""Time budget for an hourly check to interrupt an Explainer."""

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


PREEMPTION_TIMEZONE = "Asia/Singapore"
EARLIEST_PREEMPTION_TIME = time(12)
PUBLICATION_DEADLINE = time(20)
PRODUCTION_TIME_BUDGET = timedelta(hours=6)


def hourly_preemption_window(now: datetime | None = None) -> tuple[bool, str]:
    """Permit interruption from noon while the six-hour budget fits before 20:00.

    This is an admission estimate, not a guaranteed production completion time.
    Explicit Schedule and manual requests have their own authorization.
    """
    zone = ZoneInfo(PREEMPTION_TIMEZONE)
    local_now = (now or datetime.now(timezone.utc)).astimezone(zone)
    earliest = datetime.combine(local_now.date(), EARLIEST_PREEMPTION_TIME, zone)
    deadline = datetime.combine(local_now.date(), PUBLICATION_DEADLINE, zone)
    if local_now < earliest:
        return False, "before_noon"
    if local_now + PRODUCTION_TIME_BUDGET > deadline:
        return False, "insufficient_time_before_deadline"
    return True, "within_preemption_window"

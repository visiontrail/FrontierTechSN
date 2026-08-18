from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


def next_daily_run(
    schedule_time: str,
    timezone_name: str,
    *,
    after: datetime | None = None,
) -> str:
    """Return the first daily wall-clock occurrence strictly after ``after``."""
    return next_scheduled_run([schedule_time], timezone_name, after=after)


def next_scheduled_run(
    schedule_times: list[str],
    timezone_name: str,
    *,
    after: datetime | None = None,
) -> str:
    """Return the earliest configured daily occurrence strictly after ``after``."""
    reference = after or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    zone = ZoneInfo(timezone_name)
    local_reference = reference.astimezone(zone)
    candidates: list[datetime] = []
    for schedule_time in schedule_times:
        hour, minute = (int(part) for part in schedule_time.split(":", maxsplit=1))
        candidate = datetime.combine(
            local_reference.date(),
            time(hour=hour, minute=minute),
            tzinfo=zone,
        )
        if candidate <= local_reference:
            candidate += timedelta(days=1)
        candidates.append(candidate)
    if not candidates:
        raise ValueError("At least one schedule time is required")
    return min(candidates).astimezone(timezone.utc).isoformat()


def local_event_date(timezone_name: str, *, at: datetime | None = None) -> str:
    reference = at or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    return reference.astimezone(ZoneInfo(timezone_name)).date().isoformat()

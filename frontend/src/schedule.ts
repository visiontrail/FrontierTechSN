/**
 * Helpers for deferred task start times.
 *
 * A start time travels as a UTC ISO-8601 string but is always shown and picked
 * in the operator's own timezone — the point of the feature is "run it while I
 * am asleep", so local wall-clock is the only reading that matters.
 */

/** Format a Date as the `YYYY-MM-DDTHH:mm` an <input type="datetime-local"> expects. */
export function toLocalInputValue(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0')
  return (
    `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}` +
    `T${pad(date.getHours())}:${pad(date.getMinutes())}`
  )
}

/** Convert a datetime-local value (read as local time) to a UTC ISO string. */
export function localInputToIso(value: string): string | null {
  if (!value) return null
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime()) ? null : parsed.toISOString()
}

/** True while the start time is still ahead of us — the task is parked. */
export function isPendingStart(scheduledAt: string | null | undefined): boolean {
  if (!scheduledAt) return false
  const at = new Date(scheduledAt).getTime()
  return !Number.isNaN(at) && at > Date.now()
}

/** Local wall-clock rendering, e.g. "Jul 28, 01:00". Today drops the date. */
export function formatStart(scheduledAt: string): string {
  const at = new Date(scheduledAt)
  if (Number.isNaN(at.getTime())) return scheduledAt
  const time = at.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
  const sameDay = at.toDateString() === new Date().toDateString()
  if (sameDay) return `today ${time}`
  const day = at.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  return `${day}, ${time}`
}

/** Coarse wait remaining, e.g. "in 5h 20m". Empty once the time has passed. */
export function countdown(scheduledAt: string): string {
  const diff = new Date(scheduledAt).getTime() - Date.now()
  if (Number.isNaN(diff) || diff <= 0) return ''
  const mins = Math.round(diff / 60000)
  if (mins < 60) return `in ${mins}m`
  const hrs = Math.floor(mins / 60)
  const rem = mins % 60
  if (hrs < 24) return rem ? `in ${hrs}h ${rem}m` : `in ${hrs}h`
  const days = Math.floor(hrs / 24)
  return `in ${days}d ${hrs % 24}h`
}

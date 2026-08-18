import { useQuery } from '@tanstack/react-query'
import { Link, useNavigate } from 'react-router-dom'
import { fetchTasks } from '../api'
import type { Task } from '../api'
import { countdown, formatStart, isPendingStart } from '../schedule'

const STATUS_LABELS: Record<string, string> = {
  queued: 'Queued',
  scheduled: 'Scheduled',
  extracting: 'Extracting',
  digesting: 'Digesting',
  titling: 'Generating Title',
  tts: 'Generating Audio',
  composing: 'Composing Video',
  complete: 'Complete',
  failed: 'Failed',
  awaiting_review: 'Awaiting Review',
  sourcing: 'Sourcing Footage',
}

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  return `${Math.floor(hrs / 24)}d ago`
}

function sourceName(task: Task): string {
  return task.source_title || task.source_url || `Untitled ${task.source_type.toUpperCase()} source`
}

function taskName(task: Task): string {
  return task.generated_title || sourceName(task)
}

function formatCreated(iso: string): string {
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(iso))
}

export default function TaskList() {
  const navigate = useNavigate()
  const { data: tasks, isLoading } = useQuery({
    queryKey: ['tasks'],
    queryFn: fetchTasks,
  })

  const counts = (tasks ?? []).reduce(
    (result, task) => {
      const scheduled = task.status === 'queued' && isPendingStart(task.scheduled_at)
      if (scheduled) result.scheduled += 1
      else if (task.status === 'complete') result.complete += 1
      else if (task.status === 'failed') result.failed += 1
      else result.active += 1
      return result
    },
    { active: 0, scheduled: 0, complete: 0, failed: 0 },
  )

  return (
    <section className="tasks-workspace">
      <header className="tasks-command">
        <div className="tasks-title-block">
          <span className="eyebrow">Production queue</span>
          <h1>Tasks</h1>
          <p>Track every source from ingestion through final render.</p>
        </div>

        <dl className="task-totals" aria-label="Task summary">
          <div><dt>Active</dt><dd>{counts.active}</dd></div>
          <div><dt>Scheduled</dt><dd>{counts.scheduled}</dd></div>
          <div><dt>Complete</dt><dd>{counts.complete}</dd></div>
          <div><dt>Failed</dt><dd>{counts.failed}</dd></div>
        </dl>

        <Link to="/new" className="tasks-new">
          <span aria-hidden="true">+</span>
          New Task
        </Link>
      </header>

      <div className="tasks-table-shell">
        {isLoading ? (
          <div className="table-message"><span className="loading-mark" />Loading production queue…</div>
        ) : !tasks?.length ? (
          <div className="table-message table-empty">
            <strong>No tasks yet</strong>
            <span>Create a task to turn a YouTube video, EPUB, or PDF into a podcast video.</span>
            <Link to="/new">Create first task</Link>
          </div>
        ) : (
          <div className="tasks-table-scroll">
            <table className="tasks-table">
              <thead>
                <tr>
                  <th scope="col">Title / Source</th>
                  <th scope="col">Status</th>
                  <th scope="col">Format</th>
                  <th scope="col">Origin</th>
                  <th scope="col">Target</th>
                  <th scope="col">Start</th>
                  <th scope="col">Created</th>
                  <th scope="col"><span className="sr-only">Open task</span></th>
                </tr>
              </thead>
              <tbody>
                {tasks.map((task: Task) => {
                  // A queued task with a future start is parked, not waiting on the worker.
                  const parked = task.status === 'queued' && isPendingStart(task.scheduled_at)
                  const state = parked ? 'scheduled' : task.status
                  const open = () => navigate(`/tasks/${task.id}`)
                  return (
                    <tr
                      key={task.id}
                      tabIndex={0}
                      aria-label={`Open ${taskName(task)}`}
                      onClick={open}
                      onKeyDown={(event) => {
                        if (event.key === 'Enter' || event.key === ' ') {
                          event.preventDefault()
                          open()
                        }
                      }}
                    >
                      <td>
                        <div className="task-source">
                          <span className={`source-glyph ${task.source_type}`} aria-hidden="true">
                            {task.source_type === 'youtube' ? 'YT' : task.source_type === 'topic' ? 'TP' : task.source_type.toUpperCase()}
                          </span>
                          <span className="task-source-copy">
                            <strong title={taskName(task)}>{taskName(task)}</strong>
                            <code>{task.generated_title ? sourceName(task) : task.id.slice(0, 8)}</code>
                          </span>
                        </div>
                      </td>
                      <td><span className={`badge ${state}`}>{STATUS_LABELS[state] || state}</span></td>
                      <td><span className="table-value">{task.source_type.toUpperCase()}</span></td>
                      <td>
                        {task.origin_type === 'content_plan' ? (
                          <span className="task-origin-cell">
                            <strong>Content plan</strong>
                            <small title={task.origin_label || ''}>{task.origin_label || task.origin_id}</small>
                          </span>
                        ) : <span className="table-muted">Manual</span>}
                      </td>
                      <td><span className="table-value">{task.config.target_duration_minutes || '—'} min</span></td>
                      <td>
                        {parked ? (
                          <span className="schedule-cell">
                            <strong>{countdown(task.scheduled_at!)}</strong>
                            <small>{formatStart(task.scheduled_at!)}</small>
                          </span>
                        ) : (
                          <span className="table-muted">Immediate</span>
                        )}
                      </td>
                      <td>
                        <span className="created-cell">
                          <strong>{formatCreated(task.created_at)}</strong>
                          <small>{relativeTime(task.created_at)}</small>
                        </span>
                      </td>
                      <td><span className="row-arrow" aria-hidden="true">→</span></td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  )
}

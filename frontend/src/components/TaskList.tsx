import { useQuery } from '@tanstack/react-query'
import { Link, useNavigate } from 'react-router-dom'
import { fetchTasks } from '../api'
import type { Task } from '../api'
import { fetchExplainers } from '../explainerApi'
import type { ExplainerProject } from '../explainerApi'
import { selectExplainerTasks } from '../explainerTasks'
import { countdown, formatStart, isPendingStart } from '../schedule'
import './TaskList.css'

const STATUS_LABELS: Record<string, string> = {
  queued: 'Queued', scheduled: 'Scheduled', researching: 'Researching', extracting: 'Extracting',
  digesting: 'Digesting', titling: 'Generating Title', reviewing: 'Reviewing', tts: 'Generating Audio',
  composing: 'Composing Video', complete: 'Complete', failed: 'Failed', awaiting_review: 'Awaiting Review',
  sourcing: 'Sourcing Footage', music: 'Generating Music', publishing: 'Publishing',
  draft: 'Draft', writing_script: 'Writing Script', script_ready: 'Script Ready',
  generating_audio: 'Generating Audio', generating_images: 'Generating Images',
  waiting_provider: 'Waiting for Gemini', generating_motion: 'Generating Motion', rendering: 'Rendering', cancelled: 'Cancelled', interrupted: 'Interrupted',
}

interface ProductionRow {
  key: string
  path: string
  title: string
  subtitle: string
  glyph: string
  sourceType: string
  status: string
  progress?: ExplainerProject['progress']
  format: string
  origin: string
  originLabel?: string | null
  duration: number | null
  targetMinutes: number | null
  scheduledAt?: string | null
  queuePosition?: number | null
  createdAt: string
}

function sourceName(task: Task): string {
  return task.source_title || task.source_url || `Untitled ${task.source_type.toUpperCase()} source`
}

function newsRow(task: Task): ProductionRow {
  return {
    key: `task:${task.id}`, path: `/tasks/${task.id}`, title: task.generated_title || sourceName(task),
    subtitle: task.generated_title ? sourceName(task) : task.id.slice(0, 8),
    glyph: task.source_type === 'youtube' ? 'YT' : task.source_type === 'topic' ? 'TP' : task.source_type.toUpperCase(),
    sourceType: task.source_type, format: task.source_type.toUpperCase(),
    status: task.status === 'queued' && isPendingStart(task.scheduled_at) ? 'scheduled' : task.status,
    origin: task.origin_type === 'daily_news' ? 'Morning desk' : task.origin_type === 'content_plan' ? 'Legacy content plan' : 'Manual',
    originLabel: task.origin_label || task.origin_id, duration: task.duration_seconds,
    targetMinutes: task.config.target_duration_minutes, scheduledAt: task.scheduled_at, createdAt: task.created_at,
  }
}

function explainerRow(project: ExplainerProject): ProductionRow {
  return {
    key: `explainer:${project.id}`, path: `/tasks/explainers/${project.id}`,
    title: project.title || 'Untitled explainer',
    subtitle: project.scenes.length ? `${project.scenes.length} storyboard scenes` : 'Research to video',
    glyph: 'EX', sourceType: 'explainer', format: 'Byte By Byte',
    status: project.status === 'completed' ? 'complete' : project.status,
    progress: project.progress, origin: 'Byte By Byte',
    originLabel: project.language === 'zh' ? 'Chinese narration' : 'English narration',
    duration: project.artifacts.duration_seconds ?? null, targetMinutes: project.target_minutes,
    queuePosition: project.queue?.position, createdAt: project.created_at,
  }
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

function formatCreated(iso: string): string {
  return new Intl.DateTimeFormat(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  }).format(new Date(iso))
}

function formatDuration(seconds: number): string {
  const totalSeconds = Math.max(0, Math.round(seconds))
  return `${Math.floor(totalSeconds / 60)}:${String(totalSeconds % 60).padStart(2, '0')}`
}

export default function TaskList() {
  const navigate = useNavigate()
  const news = useQuery({ queryKey: ['tasks'], queryFn: fetchTasks })
  const explainers = useQuery({ queryKey: ['explainers'], queryFn: fetchExplainers })
  const tasks = [
    ...(news.data ?? []).map(newsRow),
    ...selectExplainerTasks(explainers.data?.projects).map(explainerRow),
  ].sort((a, b) => b.createdAt.localeCompare(a.createdAt) || a.key.localeCompare(b.key))
  const isLoading = news.isLoading || explainers.isLoading
  const hasError = news.isError || explainers.isError
  const counts = tasks.reduce((result, task) => {
    if (['scheduled', 'waiting_provider'].includes(task.status)) result.scheduled += 1
    else if (task.status === 'complete') result.complete += 1
    else if (['failed', 'cancelled', 'interrupted', 'script_ready', 'draft'].includes(task.status)) result.attention += 1
    else result.active += 1
    return result
  }, { active: 0, scheduled: 0, complete: 0, attention: 0 })

  return (
    <section className="tasks-workspace">
      <header className="tasks-command">
        <div className="tasks-title-block">
          <span className="eyebrow">Production queue</span>
          <h1>Tasks</h1>
          <p>Track Morning Desk and Explainer videos from queue to completion.</p>
        </div>
        <dl className="task-totals" aria-label="Task summary">
          <div><dt>Active</dt><dd>{counts.active}</dd></div>
          <div><dt>Scheduled</dt><dd>{counts.scheduled}</dd></div>
          <div><dt>Complete</dt><dd>{counts.complete}</dd></div>
          <div><dt>Needs attention</dt><dd>{counts.attention}</dd></div>
        </dl>
        <Link to="/morning-desk" className="tasks-new"><span aria-hidden="true">↗</span>Configure Desk</Link>
      </header>

      {hasError && <div className="tasks-load-error" role="alert">
        <span>Could not load {news.isError && explainers.isError ? 'the production queue' : news.isError ? 'Morning Desk tasks' : 'Explainer tasks'}. Available tasks are shown below.</span>
        <button type="button" onClick={() => { void news.refetch(); void explainers.refetch() }}>Retry</button>
      </div>}

      <div className="tasks-table-shell">
        {isLoading && !tasks.length ? (
          <div className="table-message"><span className="loading-mark" />Loading production queue…</div>
        ) : !tasks.length ? (
          <div className="table-message table-empty">
            <strong>{hasError ? 'Production queue unavailable' : 'No tasks yet'}</strong>
            {!hasError && <><span>Videos queued from Morning Desk and Explainer Studio will appear here.</span>
              <Link to="/explainer-studio">Create an explainer</Link>
              <Link to="/morning-desk">Configure Morning Desk</Link></>}
          </div>
        ) : (
          <div className="tasks-table-scroll">
            <table className="tasks-table">
              <thead><tr>
                <th scope="col">Title / Source</th><th scope="col">Status</th><th scope="col">Format</th>
                <th scope="col">Origin</th><th scope="col">Length</th><th scope="col">Start</th>
                <th scope="col">Created</th><th scope="col"><span className="sr-only">Open task</span></th>
              </tr></thead>
              <tbody>{tasks.map((task) => {
                const open = () => navigate(task.path)
                const hasProgress = task.progress && task.status !== 'complete' && task.status !== 'queued'
                return <tr key={task.key} tabIndex={0} aria-label={`Open ${task.title}`} onClick={open}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); open() }
                  }}>
                  <td><div className="task-source">
                    <span className={`source-glyph ${task.sourceType}`} aria-hidden="true">{task.glyph}</span>
                    <span className="task-source-copy"><strong title={task.title}>{task.title}</strong><code>{task.subtitle}</code></span>
                  </div></td>
                  <td><span className="task-status-cell">
                    <span className={`badge ${task.status}`}>{STATUS_LABELS[task.status] || task.status}</span>
                    {hasProgress && task.progress && <>
                      {task.progress.total > 0 && <small>{task.progress.completed} / {task.progress.total} steps</small>}
                      {task.progress.message && <small title={task.progress.message}>{task.progress.message}</small>}
                    </>}
                  </span></td>
                  <td><span className="table-value">{task.format}</span></td>
                  <td>{task.origin === 'Manual' ? <span className="table-muted">Manual</span> : <span className="task-origin-cell">
                    <strong>{task.origin}</strong><small title={task.originLabel || ''}>{task.originLabel}</small>
                  </span>}</td>
                  <td><span className="task-length-cell">
                    <strong>{task.duration !== null ? `${formatDuration(task.duration)} actual` : 'Not rendered'}</strong>
                    <small>{task.targetMinutes === null ? 'Automatic length' : `${task.targetMinutes} min target`}</small>
                  </span></td>
                  <td>{task.status === 'scheduled' && task.scheduledAt ? <span className="schedule-cell">
                    <strong>{countdown(task.scheduledAt)}</strong><small>{formatStart(task.scheduledAt)}</small>
                  </span> : task.sourceType === 'explainer' && task.status === 'queued' ? <span className="schedule-cell">
                    <strong>{task.queuePosition ? `Queue #${task.queuePosition}` : 'Waiting'}</strong><small>Automatic start</small>
                  </span> : <span className="table-muted">Immediate</span>}</td>
                  <td><span className="created-cell"><strong>{formatCreated(task.createdAt)}</strong><small>{relativeTime(task.createdAt)}</small></span></td>
                  <td><span className="row-arrow" aria-hidden="true">→</span></td>
                </tr>
              })}</tbody>
            </table>
          </div>
        )}
      </div>
    </section>
  )
}

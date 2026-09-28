import { Link } from 'react-router-dom'
import type { ExplainerProject, ExplainerQueue, ExplainerStatus } from '../explainerApi'
import { canQueueExplainer, selectQueuedExplainers, selectTopicPlans } from '../explainerPlanning'

export default function ExplainerTopicPlans({ projects, queue, libraryError, queueError, currentId, pending, statusLabels, hasLocalDraft, onQueue, onRemove, onRetry }: {
  projects?: ExplainerProject[]
  queue?: ExplainerQueue
  libraryError: boolean
  queueError: boolean
  currentId?: string
  pending: string | null
  statusLabels: Record<ExplainerStatus, string>
  hasLocalDraft: (id: string) => boolean
  onQueue: (project: ExplainerProject) => void
  onRemove: (project: ExplainerProject) => void
  onRetry: () => void
}) {
  const waiting = selectQueuedExplainers(queue?.items)
  const waitingIds = new Set(waiting.map((project) => project.id))
  const plans = selectTopicPlans(projects).filter((project) => !waitingIds.has(project.id) && queue?.active?.id !== project.id)
  const active = queue?.active

  return <section className="explainer-planning" aria-label="Topic planning and video queue">
    <div className="explainer-plans-column">
      <div className="explainer-planning-heading"><div><p className="explainer-eyebrow">PLAN AHEAD</p><h2>Topic plans <span>{plans.length}</span></h2></div><a href="#explainer-source-section" className="explainer-planning-jump">Research & direction ↓</a></div>
      <p className="explainer-planning-intro">Keep each idea and its research together. Save a plan, then add another whenever inspiration strikes.</p>
      {libraryError && <div className="explainer-planning-error" role="alert">Could not refresh topic plans. <button type="button" onClick={onRetry}>Retry</button></div>}
      {!projects && !libraryError && <p className="explainer-plan-empty" role="status">Loading topic plans…</p>}
      {projects && !plans.length && <p className="explainer-plan-empty">Your saved topics will appear here, ready to join the queue.</p>}
      <ul className="explainer-plan-list" aria-label="Saved topic plans">
        {plans.map((project) => {
          const localDraft = hasLocalDraft(project.id)
          const ready = canQueueExplainer(project)
          return <li key={project.id} className={currentId === project.id ? 'is-current' : ''}>
            <Link to={`/explainer-studio/${project.id}`} aria-current={currentId === project.id ? 'page' : undefined}>
              <strong>{project.title}</strong><small>{localDraft ? 'Local edits · Open to save' : `${statusLabels[project.status]} · ${project.target_minutes} min · ${project.language === 'zh' ? 'Chinese' : 'English'}`}</small>
            </Link>
            <button type="button" disabled={!!pending || queueError || !queue || localDraft || !ready} onClick={() => onQueue(project)} aria-label={`Add ${project.title} to video queue`} title={localDraft ? 'Open this plan and save its local edits first.' : !ready ? 'Add at least 40 characters of research or complete its scenes.' : 'Generate the full video when its turn arrives.'}>{pending === `queue:${project.id}` ? 'Adding…' : 'Queue →'}</button>
          </li>
        })}
      </ul>
    </div>
    <div className="explainer-queue-column">
      <div className="explainer-planning-heading"><div><p className="explainer-eyebrow">ONE PRODUCTION AT A TIME</p><h2>Video queue <span>{waiting.length}</span></h2></div><span className="explainer-queue-order">FIRST IN, FIRST OUT</span></div>
      <p className="explainer-planning-intro">Topics run in queue order. Due Morning Desk jobs go first when the slot opens; the current production finishes uninterrupted.</p>
      {queueError && <div className="explainer-planning-error" role="alert">Could not refresh the queue. Its state may have changed. <button type="button" onClick={onRetry}>Retry</button></div>}
      {!queue && !queueError && <p className="explainer-plan-empty" role="status">Checking production availability…</p>}
      {queue && <>
        <div className={`explainer-queue-current${active ? ' is-active' : ''}`}>
          <span className="explainer-queue-signal" aria-hidden="true" />
          <div><strong>{active ? `${active.kind === 'morning_desk' ? 'Morning Desk' : 'Explainer Studio'} is running` : 'Ready for the next topic'}</strong>
            {active && <Link to={active.kind === 'morning_desk' ? `/tasks/${active.id}` : `/explainer-studio/${active.id}`}>{active.title || active.id} ↗</Link>}
            <p role="status">{queue.message || (active ? 'Queued videos start after the current production finishes.' : 'Add a topic to start production automatically.')}</p>
          </div>
        </div>
        {!waiting.length && <p className="explainer-queue-empty">No topics waiting.</p>}
        <ol className="explainer-queue-list" aria-label="Queued explainer videos">
          {waiting.map((project, index) => <li key={project.id}>
            <span className="explainer-queue-number" aria-label={`Queue position ${project.queue?.position ?? index + 1}`}>{String(project.queue?.position ?? index + 1).padStart(2, '0')}</span>
            <Link to={`/explainer-studio/${project.id}`} aria-current={currentId === project.id ? 'page' : undefined}><strong>{project.title}</strong><small>{project.queue?.operation === 'script' ? 'Script only' : project.queue?.operation === 'render' ? 'Video from storyboard' : 'Full video'} · {index === 0 ? 'Next in line' : 'Waiting'}</small></Link>
            <button type="button" disabled={!!pending || queueError} onClick={() => onRemove(project)} aria-label={`Remove ${project.title} from queue`}>{pending === `dequeue:${project.id}` ? 'Removing…' : 'Remove'}</button>
          </li>)}
        </ol>
      </>}
    </div>
  </section>
}

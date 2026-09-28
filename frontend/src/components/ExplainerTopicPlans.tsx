import { Link } from 'react-router-dom'
import type { ExplainerProject } from '../explainerApi'
import { canQueueExplainer, selectTopicPlans } from '../explainerPlanning'

export default function ExplainerTopicPlans({ projects, libraryError, queueUnavailable, currentId, pending, hasLocalDraft, onQueue, onRetry }: {
  projects?: ExplainerProject[]
  libraryError: boolean
  queueUnavailable: boolean
  currentId?: string
  pending: string | null
  hasLocalDraft: (id: string) => boolean
  onQueue: (project: ExplainerProject) => void
  onRetry: () => void
}) {
  const plans = selectTopicPlans(projects)
  if (!plans.length && !libraryError) return null

  return <details className="explainer-planning">
    <summary>Topic plans <span>{plans.length}</span></summary>
    <div className="explainer-plans-column">
      <p className="explainer-planning-intro">Saved research that has not entered production. Queued and generated videos are in <Link to="/">Tasks ↗</Link>.</p>
      {libraryError && <div className="explainer-planning-error" role="alert">Could not refresh topic plans. <button type="button" onClick={onRetry}>Retry</button></div>}
      {queueUnavailable && <div className="explainer-planning-error" role="alert">Could not check the video queue. <button type="button" onClick={onRetry}>Retry</button></div>}
      <ul className="explainer-plan-list" aria-label="Saved topic plans">
        {plans.map((project) => {
          const localDraft = hasLocalDraft(project.id)
          const ready = canQueueExplainer(project)
          return <li key={project.id} className={currentId === project.id ? 'is-current' : ''}>
            <Link to={`/explainer-studio/${project.id}`} aria-current={currentId === project.id ? 'page' : undefined}>
              <strong>{project.title}</strong><small>{localDraft ? 'Local edits · Open to save' : `${project.target_minutes} min · ${project.language === 'zh' ? 'Chinese' : 'English'}`}</small>
            </Link>
            <button type="button" disabled={!!pending || queueUnavailable || localDraft || !ready} onClick={() => onQueue(project)} aria-label={`Add ${project.title} to video queue`} title={localDraft ? 'Open this plan and save its local edits first.' : !ready ? 'Add at least 40 characters of research or complete its scenes.' : 'Add this video to Tasks.'}>{pending === `queue:${project.id}` ? 'Adding…' : 'Queue →'}</button>
          </li>
        })}
      </ul>
    </div>
  </details>
}

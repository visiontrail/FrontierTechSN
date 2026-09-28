import { useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { isExplainerRunning } from '../explainerApi'
import type { ExplainerProject, ExplainerStatus } from '../explainerApi'

export default function ExplainerProjectHistory({ projects, error, currentId, statusLabels, onRetry }: {
  projects?: ExplainerProject[]
  error: boolean
  currentId?: string
  statusLabels: Record<ExplainerStatus, string>
  onRetry: () => void
}) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [search, setSearch] = useState('')
  const matches = projects?.filter((project) => project.title.toLowerCase().includes(search.trim().toLowerCase()))

  return <>
    <button type="button" className="explainer-button" aria-haspopup="dialog" onClick={() => dialog.current?.showModal()}>Project history</button>
    <dialog ref={dialog} className="explainer-history" aria-labelledby="explainer-history-title">
      <div className="explainer-history-heading">
        <div><p className="explainer-eyebrow">PICK UP WHERE YOU LEFT OFF</p><h2 id="explainer-history-title">Project history</h2></div>
        <button type="button" aria-label="Close project history" onClick={() => dialog.current?.close()}>×</button>
      </div>
      <p className="explainer-history-intro">Reopen a saved story or an earlier sample. Style previews are also available in Make it your own.</p>
      <label className="explainer-sr-only" htmlFor="explainer-history-search">Search saved projects</label>
      <input id="explainer-history-search" type="search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search saved projects…" />
      {error && <div className="explainer-alert is-error" role="alert"><p>Could not load projects.</p><button type="button" onClick={onRetry}>Retry project history</button></div>}
      {!projects && !error && <p className="explainer-library-empty" role="status">Loading saved projects…</p>}
      {projects?.length === 0 && <p className="explainer-library-empty">Your saved projects will appear here. Start with your research to create the first one.</p>}
      {!!projects?.length && matches?.length === 0 && <p className="explainer-library-empty" role="status">No projects match “{search}”.</p>}
      <nav className="explainer-history-list" aria-label="Saved explainer projects">
        {matches?.map((project) => <Link key={project.id} to={`/explainer-studio/${project.id}`} onClick={() => dialog.current?.close()}
          className={`explainer-project-link${currentId === project.id ? ' is-selected' : ''}`} aria-current={currentId === project.id ? 'page' : undefined}>
          <span className={`explainer-status-dot${isExplainerRunning(project.status) ? ' is-running' : ''}`} />
          <span><strong>{project.title}</strong><small>{statusLabels[project.status] ?? project.status} · {project.scenes.length} {project.scenes.length === 1 ? 'scene' : 'scenes'} · {new Date(project.updated_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</small></span>
          <span className="explainer-history-arrow" aria-hidden="true">↗</span>
        </Link>)}
      </nav>
    </dialog>
  </>
}

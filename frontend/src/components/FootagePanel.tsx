import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  acquireFootage,
  fetchFootage,
  footageFileUrl,
} from '../api'
import type { FootageManifest, Task } from '../api'

function formatBytes(bytes: number) {
  if (!bytes) return '—'
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function statusLabel(manifest: FootageManifest, task: Task) {
  if (task.status === 'sourcing') return 'Scout working'
  const labels: Record<FootageManifest['status'], string> = {
    not_started: 'Waiting',
    planning: 'Planning',
    searching: 'Searching',
    ready: 'Ready',
    partial: 'Partial',
    no_results: 'No results',
  }
  return labels[manifest.status]
}

export default function FootagePanel({ task }: { task: Task }) {
  const queryClient = useQueryClient()
  const { data: manifest, isLoading, isError } = useQuery({
    queryKey: ['footage', task.id],
    queryFn: () => fetchFootage(task.id),
    refetchInterval: ['queued', 'sourcing'].includes(task.status) ? 2000 : false,
  })

  const acquireMutation = useMutation({
    mutationFn: () => acquireFootage(task.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['task', task.id] })
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
      queryClient.invalidateQueries({ queryKey: ['footage', task.id] })
    },
  })

  if (isLoading) {
    return <section className="footage-workbench footage-loading">Loading public footage…</section>
  }
  if (isError || !manifest) {
    return <section className="footage-workbench"><div className="error-box">Footage manifest unavailable.</div></section>
  }

  const canRetry =
    !!task.script_path &&
    ['awaiting_review', 'complete', 'failed'].includes(task.status)
  const acquired = manifest.clips.length
  const requested = manifest.requested_clip_count || task.config.footage_clip_count || 0

  return (
    <section className="footage-workbench">
      <div className="workbench-head">
        <div>
          <span className="eyebrow">Rights-ledgered assets</span>
          <h3>Public Footage Workbench</h3>
          <p>
            {acquired}/{requested} clips prepared from {manifest.provider}
            {manifest.planner && <> · planned by {manifest.planner.replace('ai:', 'AI / ')}</>}
          </p>
        </div>
        <div className="workbench-actions">
          <span className={`scout-status status-${manifest.status}`}>
            <i aria-hidden="true" />
            {statusLabel(manifest, task)}
          </span>
          <button
            className="btn-ghost"
            disabled={!canRetry || acquireMutation.isPending}
            onClick={() => acquireMutation.mutate()}
          >
            {acquireMutation.isPending ? 'Queuing…' : acquired ? 'Re-scout' : 'Scout now'}
          </button>
        </div>
      </div>

      <div className="audit-strip">
        <span><b>Source</b> {manifest.provider}</span>
        <span><b>Policy</b> {manifest.rights_review_required ? 'Human rights review required' : 'Open licenses only'}</span>
        <span><b>Allowlist</b> {manifest.license_allowlist.length ? manifest.license_allowlist.join(' · ') : 'No automatic rights claim'}</span>
      </div>

      {!!manifest.publication_blockers?.length && (
        <div className="error-box">Publication gate: {manifest.publication_blockers.join(' · ')}</div>
      )}

      {manifest.queries.length > 0 && (
        <div className="query-ledger">
          {manifest.queries.map((item, index) => (
            <span key={`${item.query}-${index}`}>{item.query}</span>
          ))}
        </div>
      )}

      {manifest.clips.length > 0 ? (
        <div className="footage-grid">
          {manifest.clips.map((clip, index) => (
            <article className="footage-clip" key={clip.id}>
              <div className="clip-preview">
                <video
                  controls
                  muted
                  playsInline
                  preload="metadata"
                  src={footageFileUrl(task.id, clip.id)}
                />
                <span className="clip-index">{String(index + 1).padStart(2, '0')}</span>
              </div>
              <div className="clip-body">
                <span className="clip-query">{clip.query}</span>
                <h4>{clip.title}</h4>
                <p>{clip.purpose || clip.description || 'Public B-roll candidate'}</p>
                {clip.analysis && (
                  <p className="clip-analysis">
                    <b>{clip.analysis.start_seconds.toFixed(1)}–{clip.analysis.end_seconds.toFixed(1)}s</b>
                    {' · '}{clip.analysis.analyzer.replace('gemini-web-via-opencli', 'Gemini Web')}
                    {' · '}{clip.analysis.reason}
                  </p>
                )}
                <div className="clip-metadata">
                  <span>{clip.width}×{clip.height}</span>
                  <span>{clip.duration_seconds.toFixed(1)}s</span>
                  <span>{formatBytes(clip.bytes)}</span>
                </div>
                <div className="clip-license">
                  <span>{clip.license}</span>
                  <small>{clip.creator}</small>
                </div>
                <a href={clip.source_page_url} target="_blank" rel="noreferrer">
                  Review source & license ↗
                </a>
              </div>
            </article>
          ))}
        </div>
      ) : (
        <div className="footage-empty">
          <span>PUBLIC / B-ROLL</span>
          <h4>
            {task.status === 'sourcing'
              ? 'The agent is searching and recording source rights.'
              : 'No public footage has been downloaded yet.'}
          </h4>
          <p>
            {manifest.rights_review_required
              ? 'Web candidates remain blocked from publication until a human reviews reuse rights.'
              : 'The scout will only retain files with explicit open-license metadata.'}
          </p>
        </div>
      )}

      {manifest.errors.length > 0 && (
        <details className="scout-errors">
          <summary>{manifest.errors.length} scout note{manifest.errors.length === 1 ? '' : 's'}</summary>
          {manifest.errors.map((error, index) => (
            <p key={`${error.stage}-${index}`}>
              {error.query && <b>{error.query}: </b>}{error.message}
            </p>
          ))}
        </details>
      )}

      {acquireMutation.isError && (
        <div className="error-box">{(acquireMutation.error as Error).message}</div>
      )}
    </section>
  )
}

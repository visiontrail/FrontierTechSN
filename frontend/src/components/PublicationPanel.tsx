import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { deleteTestEpisode, fetchPublications, publishTestEpisode, reconcilePublicationSafetyHold } from '../api'

export default function PublicationPanel({
  taskId,
  recoveryRequired = false,
}: {
  taskId: string
  recoveryRequired?: boolean
}) {
  const queryClient = useQueryClient()
  const { data } = useQuery({ queryKey: ['publications', taskId], queryFn: () => fetchPublications(taskId) })
  const publish = useMutation({
    mutationFn: () => publishTestEpisode(taskId),
    onMutate: () => {
      queryClient.setQueryData(['task', taskId], (current: Record<string, unknown> | undefined) => (
        current ? { ...current, status: 'publishing' } : current
      ))
    },
    onSuccess: (manifest) => queryClient.setQueryData(['publications', taskId], manifest),
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['task', taskId] })
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
      queryClient.invalidateQueries({ queryKey: ['publications', taskId] })
    },
  })
  const remove = useMutation({
    mutationFn: () => deleteTestEpisode(taskId),
    onMutate: () => {
      queryClient.setQueryData(['task', taskId], (current: Record<string, unknown> | undefined) => (
        current ? { ...current, status: 'publishing' } : current
      ))
    },
    onSuccess: (manifest) => queryClient.setQueryData(['publications', taskId], manifest),
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['task', taskId] })
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
      queryClient.invalidateQueries({ queryKey: ['publications', taskId] })
    },
  })
  const reconcile = useMutation({
    mutationFn: (confirmedDeletedTestTargets: Array<'youtube' | 'x'> = []) => (
      reconcilePublicationSafetyHold(taskId, confirmedDeletedTestTargets)
    ),
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['task', taskId] })
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
      queryClient.invalidateQueries({ queryKey: ['publications', taskId] })
    },
  })
  const entries = Object.entries(data?.platforms || {})
  const hasPublishedTest = entries.some(([, entry]) => entry?.status === 'published' && entry.test_mode)
  const hasPreviousRevision = entries.some(([, entry]) => entry?.status === 'published' && entry.matches_current_media === false)
  const hasActiveNonTest = entries.some(([, entry]) => entry?.status === 'published' && !entry.test_mode)
  const allCurrentTestTargetsPublished = (['youtube', 'x'] as const).every((platform) => {
    const entry = data?.platforms?.[platform]
    return entry?.status === 'published' && !!entry.test_mode && entry.matches_current_media === true
  })
  const hasPartialCurrentTest = hasPublishedTest && !allCurrentTestTargetsPublished && !hasPreviousRevision
  const busy = publish.isPending || remove.isPending || reconcile.isPending
  const publishBlocked = busy || hasPreviousRevision || hasActiveNonTest || allCurrentTestTargetsPublished

  const resumePublication = () => {
    publish.mutate()
  }

  const reconcileOnly = () => {
    if (!confirm('Check every configured destination first. Clear the safety hold only after confirming there is no unrecorded publication for this exact media.')) return
    reconcile.mutate([])
  }

  const reconcileDeletedTests = () => {
    const targets = (['youtube', 'x'] as const).filter((platform) => {
      const entry = data?.platforms?.[platform]
      return entry?.status === 'published' && entry.test_mode
    })
    if (!targets.length) return
    if (!confirm('Use this only after checking the recorded URLs and confirming those exact test posts were already deleted externally. Mark them deleted in the ledger and clear the safety hold?')) return
    reconcile.mutate([...targets])
  }

  return (
    <section className="detail-panel publication-panel">
      <span className="eyebrow">Distribution ledger</span>
      <h3>Runtime-bound publishing</h3>
      <p className="detail-hint">YouTube and X require the signed-in browser identity to match Admin → Publishing. The matched identity, exact URL, media hash, and deletion receipt are recorded here.</p>
      {recoveryRequired && <p className="detail-hint">Publication stopped with an indeterminate or partial result. Keep this exact media revision and verify every destination. Clear the safety hold only after confirming there is no unrecorded publication; no target is guessed or retried while the outcome is uncertain.</p>}
      {entries.length > 0 && <div className="publication-ledger">{entries.map(([platform, entry]) => entry && (
        <article key={platform}>
          <span>{platform.replace('_', ' ')}</span><strong>{entry.status}{entry.status === 'published' && entry.matches_current_media === false ? ' · previous revision' : ''}</strong>
          {entry.url?.startsWith('http') ? <a href={entry.url} target="_blank" rel="noreferrer">Open receipt ↗</a> : <code>{entry.url}</code>}
        </article>
      ))}</div>}
      {hasPreviousRevision && <p className="detail-hint">The recorded publication belongs to an earlier video revision. Delete the recorded test publication before publishing the current cut.</p>}
      <div className="actions">
        {!recoveryRequired && <button type="button" className="btn-primary" disabled={publishBlocked} onClick={resumePublication}>{publish.isPending ? 'Publishing…' : hasPreviousRevision ? 'Previous revision is published' : hasActiveNonTest ? 'Production publication recorded' : allCurrentTestTargetsPublished ? 'Test publication recorded' : hasPartialCurrentTest ? 'Publish missing test target' : 'Publish private test'}</button>}
        {recoveryRequired && <button type="button" className="btn-ghost" disabled={busy} onClick={reconcileOnly}>{reconcile.isPending ? 'Reconciling…' : 'Clear verified safety hold'}</button>}
        {recoveryRequired && hasPublishedTest && <button type="button" className="btn-danger" disabled={busy} onClick={reconcileDeletedTests}>Confirm recorded test posts already deleted</button>}
        {hasPublishedTest && <button type="button" className="btn-danger" disabled={busy} onClick={() => { if (confirm('Permanently delete the exact YouTube and X test publications recorded for this task?')) remove.mutate() }}>{remove.isPending ? 'Deleting…' : 'Delete test publications'}</button>}
      </div>
      {(publish.isError || remove.isError || reconcile.isError) && <div className="error-box">{String((publish.error || remove.error || reconcile.error) as Error)}</div>}
    </section>
  )
}

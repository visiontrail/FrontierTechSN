import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { deleteTestEpisode, fetchPublications, publishTestEpisode } from '../api'

export default function PublicationPanel({ taskId }: { taskId: string }) {
  const queryClient = useQueryClient()
  const { data } = useQuery({ queryKey: ['publications', taskId], queryFn: () => fetchPublications(taskId) })
  const publish = useMutation({
    mutationFn: () => publishTestEpisode(taskId),
    onSuccess: (manifest) => queryClient.setQueryData(['publications', taskId], manifest),
  })
  const remove = useMutation({
    mutationFn: () => deleteTestEpisode(taskId),
    onSuccess: (manifest) => queryClient.setQueryData(['publications', taskId], manifest),
  })
  const entries = Object.entries(data?.platforms || {})
  const hasPublishedTest = entries.some(([, entry]) => entry?.status === 'published' && entry.test_mode)

  return (
    <section className="detail-panel publication-panel">
      <span className="eyebrow">Distribution ledger</span>
      <h3>Runtime-bound publishing</h3>
      <p className="detail-hint">YouTube and X use whichever browser account is signed in at publish time. The discovered identity, exact URL, media hash, and deletion receipt are recorded here.</p>
      {entries.length > 0 && <div className="publication-ledger">{entries.map(([platform, entry]) => entry && (
        <article key={platform}>
          <span>{platform.replace('_', ' ')}</span><strong>{entry.status}</strong>
          {entry.url?.startsWith('http') ? <a href={entry.url} target="_blank" rel="noreferrer">Open receipt ↗</a> : <code>{entry.url}</code>}
        </article>
      ))}</div>}
      <div className="actions">
        <button type="button" className="btn-primary" disabled={publish.isPending} onClick={() => publish.mutate()}>{publish.isPending ? 'Publishing…' : 'Publish private test'}</button>
        {hasPublishedTest && <button type="button" className="btn-danger" disabled={remove.isPending} onClick={() => { if (confirm('Permanently delete the exact YouTube and X test publications recorded for this task?')) remove.mutate() }}>{remove.isPending ? 'Deleting…' : 'Delete test publications'}</button>}
      </div>
      {(publish.isError || remove.isError) && <div className="error-box">{String((publish.error || remove.error) as Error)}</div>}
    </section>
  )
}

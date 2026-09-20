import { useEffect, useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { fetchSocialCopy, generateSocialCopy, type Task } from '../api'

export default function SocialCopyPanel({ task, scriptDirty }: { task: Task, scriptDirty: boolean }) {
  const client = useQueryClient()
  const queryKey = ['social-copy', task.id, task.updated_at]
  const [copied, setCopied] = useState<string | null>(null)
  const [copyError, setCopyError] = useState<string | null>(null)
  const autoRequested = useRef(false)
  const copyTimer = useRef<number | undefined>(undefined)
  useEffect(() => () => window.clearTimeout(copyTimer.current), [])
  const query = useQuery({
    queryKey,
    queryFn: () => fetchSocialCopy(task.id),
    refetchInterval: (state) => state.state.data?.status === 'generating' ? 2000 : false,
  })
  const generation = useMutation({
    mutationFn: (regenerate: boolean) => generateSocialCopy(task.id, regenerate),
    onSuccess: (data) => client.setQueryData(queryKey, data),
  })
  const { mutate } = generation
  const eligible = !!task.script_path && ['complete', 'awaiting_review', 'failed'].includes(task.status)
  useEffect(() => {
    if (query.data?.status === 'missing' && task.status === 'complete' && eligible && !scriptDirty && !autoRequested.current) {
      autoRequested.current = true
      mutate(false)
    }
  }, [query.data?.status, task.status, eligible, scriptDirty, mutate])

  const data = query.data
  const copy = data?.copy
  const generating = generation.isPending || data?.status === 'generating'
  const stale = data?.stale || scriptDirty
  const error = generation.error?.message || query.error?.message || data?.error

  async function copyText(key: string, value: string) {
    try {
      await navigator.clipboard.writeText(value)
      setCopied(key)
      setCopyError(null)
      window.clearTimeout(copyTimer.current)
      copyTimer.current = window.setTimeout(() => setCopied(null), 1800)
    } catch {
      setCopyError('Clipboard unavailable. Select the text and copy it manually.')
    }
  }

  return (
    <section className="social-copy-panel" aria-labelledby="social-copy-heading">
      <header className="social-copy-header">
        <span className="eyebrow">Ready for your audience</span>
        <h2 id="social-copy-heading">Publishing Copy</h2>
        <p>YouTube title, Show Notes, and X post</p>
        <div className="social-copy-toolbar">
          <span className={`social-copy-state${stale ? ' is-stale' : ''}`} role="status">
            {generating ? 'Writing…' : stale ? 'Script changed' : data?.status === 'failed' ? 'Generation failed' : copy ? 'Copy ready' : 'Not generated yet'}
          </span>
          <button type="button" className="btn-ghost" disabled={!eligible || scriptDirty || generating || query.isPending}
            onClick={() => generation.mutate(!!copy || data?.status === 'failed')}
          >
            {generating ? 'Generating…' : copy ? 'Regenerate' : data?.status === 'failed' ? 'Retry generation' : 'Generate copy'}
          </button>
        </div>
      </header>

      {error && <div className="error-box" role="alert">{error}</div>}
      {copyError && <p className="social-copy-notice" role="alert">{copyError}</p>}
      {stale && <p className="social-copy-notice" role="status">
        {scriptDirty ? 'Save the script, then regenerate the copy to match the current content.' : 'This copy is based on an earlier script. Regenerate it before use.'}
      </p>}
      {task.status !== 'complete' && <p className="social-copy-notice">The video is not complete yet. This copy is based on the saved script to help you prepare for publishing.</p>}

      <div className="social-copy-fields">
        {[
          { key: 'title', label: 'YouTube Title', hint: 'Use as the title when uploading your video', value: copy?.youtube_title || '', count: copy ? `${Array.from(copy.youtube_title).length} / 100` : 'Up to 100 characters', rows: 3 },
          { key: 'notes', label: 'YouTube Show Notes', hint: 'Paste into the video description', value: copy?.youtube_show_notes || '', count: copy ? `${Array.from(copy.youtube_show_notes).length.toLocaleString()} / 5,000` : 'Up to 5,000 characters', rows: 13 },
          { key: 'x', label: 'X Post', hint: 'Lead with the news · Group related ideas into paragraphs · Up to 280 weighted characters', value: copy?.x_post || '', count: copy ? `${copy.x_weighted_length} / 280` : 'One standard post', rows: 6 },
        ].map((field, index) => (
          <div className={`social-copy-field social-copy-${field.key}`} key={field.key}>
            <div className="social-copy-field-heading">
              <label htmlFor={`social-copy-${field.key}`}><span>{String(index + 1).padStart(2, '0')}</span>{field.label}</label>
              <button type="button" className="btn-ghost" disabled={!field.value || stale}
                aria-label={`Copy ${field.label}`} onClick={() => copyText(field.key, field.value)}>
                {copied === field.key ? 'Copied' : 'Copy'}
              </button>
            </div>
            <p className="social-copy-hint">{field.hint}</p>
            <textarea id={`social-copy-${field.key}`} readOnly value={field.value} rows={field.rows}
              placeholder={generating ? 'Writing from this episode’s script…' : eligible ? 'Generate copy to preview and copy it here' : 'Copy can be generated once the script is ready'} />
            <span className="social-copy-count">{field.count}</span>
          </div>
        ))}
      </div>
      <footer className="social-copy-footer">
        <span>Show Notes and X posts written with <a href="https://github.com/blader/humanizer" target="_blank" rel="noreferrer">Humanizer</a></span>
        {data?.generated_at && <time dateTime={data.generated_at}>Updated {new Date(data.generated_at).toLocaleString('en-US')}</time>}
      </footer>
      <span className="sr-only" role="status">{copied ? 'Copy saved to clipboard' : ''}</span>
    </section>
  )
}

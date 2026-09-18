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
        <h2 id="social-copy-heading">发布文案</h2>
        <p>YouTube 标题、Show Notes 与 X 推文</p>
        <div className="social-copy-toolbar">
          <span className={`social-copy-state${stale ? ' is-stale' : ''}`} role="status">
            {generating ? '正在编写…' : stale ? '脚本已变更' : data?.status === 'failed' ? '生成失败' : copy ? '文案已就绪' : '等待生成'}
          </span>
          <button type="button" className="btn-ghost" disabled={!eligible || scriptDirty || generating || query.isPending}
            onClick={() => generation.mutate(!!copy || data?.status === 'failed')}
          >
            {generating ? '生成中…' : copy ? '重新生成' : data?.status === 'failed' ? '重试生成' : '生成文案'}
          </button>
        </div>
      </header>

      {error && <div className="error-box" role="alert">{error}</div>}
      {copyError && <p className="social-copy-notice" role="alert">{copyError}</p>}
      {stale && <p className="social-copy-notice" role="status">
        {scriptDirty ? '先保存脚本，再重新生成与当前内容一致的发布文案。' : '文案对应旧版脚本。请重新生成后使用。'}
      </p>}
      {task.status !== 'complete' && <p className="social-copy-notice">视频尚未完成。文案基于已保存的脚本，供发布前准备。</p>}

      <div className="social-copy-fields">
        {[
          { key: 'title', label: 'YouTube 标题', hint: '用于视频上传标题', value: copy?.youtube_title || '', count: copy ? `${Array.from(copy.youtube_title).length} / 100` : '最多 100 字符', rows: 3 },
          { key: 'notes', label: 'YouTube Show Notes', hint: '粘贴到视频下方的说明栏', value: copy?.youtube_show_notes || '', count: copy ? `${Array.from(copy.youtube_show_notes).length.toLocaleString()} / 5,000` : '最多 5,000 字符', rows: 13 },
          { key: 'x', label: 'X 推文', hint: '首句新闻事实 · 按意思分段 · 最多 280 加权字符', value: copy?.x_post || '', count: copy ? `${copy.x_weighted_length} / 280` : '标准单条推文', rows: 6 },
        ].map((field, index) => (
          <div className={`social-copy-field social-copy-${field.key}`} key={field.key}>
            <div className="social-copy-field-heading">
              <label htmlFor={`social-copy-${field.key}`}><span>{String(index + 1).padStart(2, '0')}</span>{field.label}</label>
              <button type="button" className="btn-ghost" disabled={!field.value || stale}
                aria-label={`复制 ${field.label}`} onClick={() => copyText(field.key, field.value)}>
                {copied === field.key ? '已复制' : '复制'}
              </button>
            </div>
            <p className="social-copy-hint">{field.hint}</p>
            <textarea id={`social-copy-${field.key}`} readOnly value={field.value} rows={field.rows}
              placeholder={generating ? '正在根据本期脚本编写…' : eligible ? '生成后可在此复制文案' : '脚本就绪后可生成文案'} />
            <span className="social-copy-count">{field.count}</span>
          </div>
        ))}
      </div>
      <footer className="social-copy-footer">
        <span>Show Notes 与推文使用 <a href="https://github.com/blader/humanizer" target="_blank" rel="noreferrer">Humanizer</a> 编写</span>
        {data?.generated_at && <time dateTime={data.generated_at}>更新于 {new Date(data.generated_at).toLocaleString()}</time>}
      </footer>
      <span className="sr-only" role="status">{copied ? '文案已复制到剪贴板' : ''}</span>
    </section>
  )
}

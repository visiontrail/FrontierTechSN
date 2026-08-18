import { useQuery } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import { accountRunImageUrl, fetchAccountRun } from '../api'
import { IconChevronLeft } from './Icons'

const ACTIVE = new Set(['queued', 'planning', 'generating_image', 'publishing'])
const HISTORY_STAGES = ['planning', 'generating_image', 'publishing', 'published']
const ENGAGEMENT_STAGES = ['planning', 'publishing', 'published']

function stageClass(status: string, stage: string, stages: string[]): string {
  if (status === 'failed') return ''
  const current = status === 'queued' ? 0 : stages.indexOf(status)
  const index = stages.indexOf(stage)
  if (index < current || status === 'published') return 'done'
  if (index === current) return 'active'
  return ''
}

function textValue(value: unknown): string {
  return typeof value === 'string' || typeof value === 'number' ? String(value) : '—'
}

function objectRows(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value)
    ? value.filter((item): item is Record<string, unknown> => typeof item === 'object' && item !== null)
    : []
}

function optionalText(value: unknown): string | null {
  return typeof value === 'string' && value ? value : null
}

export default function AccountRunDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const { data: run, isLoading } = useQuery({
    queryKey: ['account-run', id],
    queryFn: () => fetchAccountRun(id!),
    enabled: !!id,
    refetchInterval: (query) => ACTIVE.has(query.state.data?.status ?? '') ? 2000 : false,
  })
  if (isLoading || !run) return <div className="empty-state">Loading edition…</div>
  const engagement = run.feature_type === 'x_engagement'
  const sources = Array.isArray(run.content.source_notes) ? run.content.source_notes : []
  const replies = objectRows(run.content.replies)
  const quotes = objectRows(run.content.quote_reposts)
  const mediaActions = [...replies, ...quotes].filter((item) => optionalText(item.media_explanation))
  const stages = engagement ? ENGAGEMENT_STAGES : HISTORY_STAGES

  return (
    <section className="ops-detail-workspace">
      <header className="ops-detail-command">
        <button type="button" className="icon-btn" aria-label="Back to account operations" onClick={() => navigate('/account-operations')}><IconChevronLeft /></button>
        <div>
          <span className="eyebrow">{run.event_date} · @{run.account_handle}</span>
          <h1>{run.title || (engagement ? 'Replies & Reposts run' : 'Today in History edition')}</h1>
          <p>{run.id}</p>
        </div>
        <div className="ops-detail-actions">
          <span className={`badge ${run.status}`}>{run.status.replace('_', ' ')}</span>
          {run.post_url && <a className="btn-primary" href={run.post_url} target="_blank" rel="noreferrer">View on X ↗</a>}
        </div>
      </header>

      <div className="ops-stage-rail">
        {stages.map((stage, index) => <div key={stage} className={`stage ${stageClass(run.status, stage, stages)}`}><span>0{index + 1}</span>{stage.replace('_', ' ')}</div>)}
      </div>
      {run.error_message && <div className="error-box">{run.error_message}</div>}

      <div className="ops-detail-grid">
        <main>
          {engagement ? (
            <section className="ops-engagement-actions">
              <header><span className="eyebrow">Published actions</span><strong>{replies.length} replies · {quotes.length} quote-reposts</strong></header>
              {!replies.length && !quotes.length && <div className="ops-ledger-empty"><strong>No post cleared the bar.</strong><span>The agent completed the scan without forcing an interaction.</span></div>}
              {replies.map((reply, index) => <article key={`reply-${index}`} className="ops-action-card"><div><span>Reply · @{textValue(reply.target_author)}</span>{optionalText(reply.result_url) && <a href={optionalText(reply.result_url)!} target="_blank" rel="noreferrer">View on X ↗</a>}</div><blockquote>{textValue(reply.reply_text)}</blockquote><a href={optionalText(reply.target_url) ?? '#'} target="_blank" rel="noreferrer">Original post ↗</a></article>)}
              {quotes.map((quote, index) => <article key={`quote-${index}`} className="ops-action-card ops-action-card--quote"><div><span>Quote-repost · @{textValue(quote.target_author)}</span>{optionalText(quote.result_url) && <a href={optionalText(quote.result_url)!} target="_blank" rel="noreferrer">View on X ↗</a>}</div><blockquote>{textValue(quote.quote_text)}</blockquote><p>{textValue(quote.selection_reason)}</p><a href={optionalText(quote.target_url) ?? '#'} target="_blank" rel="noreferrer">Original post ↗</a></article>)}
            </section>
          ) : (
            <>
              {run.image_path ? <figure className="ops-edition-image"><img src={accountRunImageUrl(run.id)} alt={run.title || 'Today in History illustration'} /><figcaption>Generated through ChatGPT Web · preserved with this run</figcaption></figure> : <div className="ops-image-pending">Image plate pending</div>}
              <section className="ops-post-card"><span className="eyebrow">Published copy</span><blockquote>{run.post_text || 'Copy is being curated…'}</blockquote><footer><span>{run.post_text?.length ?? 0} characters</span><span>{run.external_post_id ? `X · ${run.external_post_id}` : 'Not yet published'}</span></footer></section>
            </>
          )}
        </main>
        <aside>
          <section className="ops-context-card">
            <span className="eyebrow">{engagement ? 'Run controls' : 'Historical note'}</span>
            <dl>
              {engagement ? <><div><dt>Feed</dt><dd>{run.content.following_feed_used === true ? 'Following' : 'Unverified'}</dd></div><div><dt>Scanned</dt><dd>{textValue(run.content.scanned_posts)}</dd></div></> : <><div><dt>Year</dt><dd>{textValue(run.content.year)}</dd></div><div><dt>Place</dt><dd>{textValue(run.content.location)}</dd></div></>}
              <div><dt>Executor</dt><dd>{run.executor === 'opencode' ? 'OpenCode agent' : run.executor === 'claude_sdk' ? 'Claude Agent SDK' : 'Pipeline'}</dd></div>
              <div><dt>Trigger</dt><dd>{run.trigger}</dd></div>
            </dl>
            {engagement ? (
              <>{mediaActions.length ? <><h3>Grok media explanations</h3><ol>{mediaActions.map((item, index) => <li key={index}><a href={optionalText(item.explained_original_url) ?? optionalText(item.target_url) ?? '#'} target="_blank" rel="noreferrer">@{textValue(item.target_author)}</a>: {textValue(item.media_explanation)}</li>)}</ol></> : <><h3>Media checks</h3><p>No published action depended on image or video interpretation.</p></>}</>
            ) : (
              <><h3>What happened</h3><p>{textValue(run.content.event_summary)}</p><h3>Why it matters</h3><p>{textValue(run.content.historical_reflection)}</p>{!!sources.length && <><h3>Research trail</h3><ol>{sources.map((source, index) => <li key={index}>{textValue(source)}</li>)}</ol></>}{run.chatgpt_conversation_url && <a href={run.chatgpt_conversation_url} target="_blank" rel="noreferrer">Open ChatGPT provenance ↗</a>}</>
            )}
          </section>
          <section className="ops-run-log"><div><span className="eyebrow">Run log</span><strong>{ACTIVE.has(run.status) ? 'Live' : 'Closed'}</strong></div><pre>{run.log_text || 'Waiting for the worker…'}</pre></section>
        </aside>
      </div>
    </section>
  )
}

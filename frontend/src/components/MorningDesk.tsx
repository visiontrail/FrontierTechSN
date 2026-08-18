import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  fetchDailyAutomation,
  fetchDailySources,
  runDailyNow,
  updateDailyAutomation,
  type DailyAutomationSettings,
} from '../api'

const SOURCE_LABELS: Record<string, string> = {
  zh: '中文',
  en: 'English',
}

function formatDateTime(value: string | null) {
  if (!value) return 'Not scheduled'
  return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value))
}

export default function MorningDesk() {
  const queryClient = useQueryClient()
  const { data } = useQuery({ queryKey: ['daily-automation'], queryFn: fetchDailyAutomation })
  const { data: sources = [] } = useQuery({ queryKey: ['daily-sources'], queryFn: fetchDailySources })
  const [draftOverride, setDraftOverride] = useState<DailyAutomationSettings | null>(null)
  const draft = draftOverride ?? data?.settings ?? null

  const save = useMutation({
    mutationFn: (value: DailyAutomationSettings) => updateDailyAutomation(value),
    onSuccess: (next) => {
      setDraftOverride(next.settings)
      queryClient.setQueryData(['daily-automation'], next)
    },
  })
  const run = useMutation({
    mutationFn: () => runDailyNow(true, 1),
    onSuccess: (task) => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
      queryClient.invalidateQueries({ queryKey: ['daily-automation'] })
      window.location.assign(`/tasks/${task.id}`)
    },
  })

  const patch = <K extends keyof DailyAutomationSettings>(key: K, value: DailyAutomationSettings[K]) => {
    setDraftOverride((current) => ({ ...(current ?? data!.settings), [key]: value }))
  }

  if (!draft || !data) return <div className="empty-state">Opening the morning desk…</div>

  const priority = sources.filter((source) => source.priority <= 6)
  const secondary = sources.filter((source) => source.priority > 6)

  return (
    <div className="morning-desk">
      <header className="morning-masthead">
        <div>
          <span className="morning-kicker">Autonomous edition · Asia/Singapore</span>
          <h1>Frontier Tech<br />Morning Desk</h1>
        </div>
        <div className="morning-status">
          <span className={draft.enabled ? 'is-live' : ''}><i />{draft.enabled ? 'Daily run enabled' : 'Schedule paused'}</span>
          <strong>{draft.generation_time}</strong>
          <small>Next edition · {formatDateTime(data.next_run_at)}</small>
        </div>
      </header>

      <section className="morning-lead-grid">
        <article className="morning-lead">
          <span className="morning-section-label">Edition contract</span>
          <h2>Evidence first. Broadcast ready.</h2>
          <p>Each run researches a bilingual source set, writes a date-stamped morning-news script, checks every factual claim in ChatGPT, generates a Gemini music bed, then mixes it below the narration.</p>
          <div className="morning-contracts">
            <span><b>01</b> 10-attempt yhroot policy</span>
            <span><b>02</b> Paper-collage on by default</span>
            <span><b>03</b> Script ↔ speech hard gate</span>
            <span><b>04</b> Account discovered at publish time</span>
          </div>
        </article>

        <aside className="morning-control-card">
          <span className="morning-section-label">Run control</span>
          <label className="morning-switch">
            <span><strong>Daily automation</strong><small>Catch up once after a restart</small></span>
            <input type="checkbox" checked={draft.enabled} onChange={(e) => patch('enabled', e.target.checked)} />
          </label>
          <div className="morning-fields">
            <label><span>Desk time</span><input type="time" value={draft.generation_time} onChange={(e) => patch('generation_time', e.target.value)} /></label>
            <label><span>Run length</span><select value={draft.target_duration_minutes} onChange={(e) => patch('target_duration_minutes', Number(e.target.value))}><option value={6}>6 min</option><option value={8}>8 min</option><option value={10}>10 min</option><option value={12}>12 min</option></select></label>
            <label><span>Language</span><select value={draft.language} onChange={(e) => patch('language', e.target.value as 'en' | 'zh')}><option value="en">English</option><option value="zh">中文</option></select></label>
            <label><span>Stories</span><select value={draft.max_stories} onChange={(e) => patch('max_stories', Number(e.target.value))}><option value={4}>4</option><option value={6}>6</option><option value={8}>8</option></select></label>
          </div>
          <label className="morning-switch">
            <span><strong>Automatic distribution</strong><small>YouTube · X · Apple feed</small></span>
            <input type="checkbox" checked={draft.auto_publish} onChange={(e) => patch('auto_publish', e.target.checked)} />
          </label>
          <p className="morning-safety">The default YouTube visibility is private. The system records the current account identity and exact publication URL; no account is hard-coded.</p>
          <div className="morning-actions">
            <button type="button" className="btn-primary" disabled={save.isPending} onClick={() => save.mutate(draft)}>{save.isPending ? 'Saving…' : 'Save desk'}</button>
            <button type="button" className="btn-ghost" disabled={run.isPending} onClick={() => run.mutate()}>{run.isPending ? 'Queuing…' : 'Run 1-min test'}</button>
          </div>
          {(save.isError || run.isError) && <div className="error-box">{String((save.error || run.error) as Error)}</div>}
        </aside>
      </section>

      <section className="morning-sources">
        <div className="morning-section-head"><div><span className="morning-section-label">Signal board</span><h2>12 monitored sources</h2></div><p>Six primary desks set the agenda; six secondary desks widen or verify the frame. Per-source failure is isolated, while source quorum remains a hard gate.</p></div>
        <div className="source-table-wrap">
          <table className="source-table">
            <thead><tr><th>Rank</th><th>Desk</th><th>Language</th><th>Coverage</th><th>Cadence</th><th>Signal</th></tr></thead>
            <tbody>
              {[...priority, ...secondary].map((source) => (
                <tr key={source.id} className={source.priority <= 6 ? 'is-priority' : ''}>
                  <td>{String(source.priority).padStart(2, '0')}</td>
                  <td><a href={source.homepage} target="_blank" rel="noreferrer">{source.name}</a>{source.priority <= 6 && <small>PRIMARY</small>}</td>
                  <td>{SOURCE_LABELS[source.language] || source.language}</td>
                  <td>{source.coverage}</td>
                  <td>{source.frequency}</td>
                  <td><span className="source-stars">{'★'.repeat(source.rating)}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {data.last_task_id && <footer className="morning-last"><span>Last edition</span><strong>{data.last_run_date}</strong><Link to={`/tasks/${data.last_task_id}`}>Open run →</Link></footer>}
    </div>
  )
}

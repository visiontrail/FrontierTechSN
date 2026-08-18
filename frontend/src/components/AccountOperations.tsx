import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import {
  fetchAccountAutomations,
  fetchAccountOpsStatus,
  fetchAccountRuns,
  pauseAccountAutomation,
  resumeAccountAutomation,
  runAccountAutomation,
  updateAccountAutomation,
} from '../api'
import type {
  AccountAutomation,
  AccountAutomationFeature,
  AccountAutomationUpdate,
  AccountOpsStatus,
  AccountRun,
} from '../api'

const ACTIVE = new Set(['queued', 'planning', 'generating_image', 'publishing'])

function formatDateTime(value: string | null): string {
  if (!value) return '—'
  return new Intl.DateTimeFormat(undefined, {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  }).format(new Date(value))
}

function countdown(target: string | null): string {
  if (!target) return ''
  const ms = new Date(target).getTime() - Date.now()
  if (ms <= 0) return 'imminent'
  const hours = Math.floor(ms / 3_600_000)
  const minutes = Math.floor((ms % 3_600_000) / 60_000)
  if (hours >= 1) return `in ${hours}h ${minutes}m`
  if (minutes >= 1) return `in ${minutes}m`
  return `in ${Math.floor(ms / 1_000)}s`
}

function runTitle(run: AccountRun): string {
  if (run.title) return run.title
  return run.feature_type === 'x_engagement'
    ? `Replies & reposts · ${run.event_date}`
    : `Today in History · ${run.event_date}`
}

function automationDraft(automation: AccountAutomation): AccountAutomationUpdate {
  return {
    name: automation.name,
    account_handle: automation.account_handle,
    enabled: automation.enabled,
    schedule_time: automation.schedule_time,
    schedule_times: automation.schedule_times,
    timezone: automation.timezone,
    prompt_template: automation.prompt_template,
    reply_style_prompt: automation.reply_style_prompt,
    max_replies: automation.max_replies,
    max_quote_reposts: automation.max_quote_reposts,
    scan_limit: automation.scan_limit,
    executor: automation.executor,
    opencode_model: automation.opencode_model,
  }
}

function AutomationStatusBar({
  automation,
  status,
}: {
  automation: AccountAutomation
  status: AccountOpsStatus | undefined
}) {
  const queryClient = useQueryClient()
  const pause = useMutation({
    mutationFn: () => pauseAccountAutomation(automation.id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['account-automations'] }),
  })
  const resume = useMutation({
    mutationFn: () => resumeAccountAutomation(automation.id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['account-automations'] }),
  })

  const workerAlive = status?.worker_alive ?? false
  const isPaused = !automation.enabled
  let dotClass = 'ops-status-dot--idle'
  let label = 'Paused'
  let detail = 'No scheduled runs'
  if (!workerAlive) {
    dotClass = 'ops-status-dot--offline'
    label = 'Scheduler offline'
    detail = 'Worker process is not running — restart the server'
  } else if (!isPaused) {
    dotClass = 'ops-status-dot--live'
    label = 'Running'
    detail = automation.next_run_at
      ? `Next run ${formatDateTime(automation.next_run_at)} (${countdown(automation.next_run_at)})`
      : 'Calculating next run…'
  }

  return (
    <div className="ops-status-bar">
      <div className="ops-status-info">
        <span className={`ops-status-dot ${dotClass}`} />
        <div><strong>{label}</strong><span>{detail}</span></div>
      </div>
      <button
        type="button"
        className={isPaused ? 'btn-primary' : 'btn-ghost'}
        disabled={pause.isPending || resume.isPending || !workerAlive}
        onClick={() => (isPaused ? resume.mutate() : pause.mutate())}
      >
        {isPaused
          ? (resume.isPending ? 'Resuming…' : 'Resume')
          : (pause.isPending ? 'Pausing…' : 'Pause')}
      </button>
      {(pause.isError || resume.isError) && (
        <span className="ops-status-error">{((pause.error || resume.error) as Error).message}</span>
      )}
    </div>
  )
}

function ScheduleTimes({
  values,
  onChange,
}: {
  values: string[]
  onChange: (values: string[]) => void
}) {
  const update = (index: number, value: string) => {
    onChange(values.map((item, itemIndex) => itemIndex === index ? value : item).sort())
  }
  return (
    <div className="form-group ops-times-field">
      <div className="ops-field-label"><label>Daily run times</label><small>Local to the timezone below</small></div>
      <div className="ops-time-list">
        {values.map((value, index) => (
          <div className="ops-time-row" key={`${index}-${value}`}>
            <input aria-label={`Daily run time ${index + 1}`} type="time" value={value} onChange={(event) => update(index, event.target.value)} />
            <button type="button" className="btn-ghost" aria-label={`Remove ${value}`} disabled={values.length === 1} onClick={() => onChange(values.filter((_, itemIndex) => itemIndex !== index))}>×</button>
          </div>
        ))}
        <button type="button" className="ops-add-time" disabled={values.length >= 12} onClick={() => onChange([...values, '12:00'].sort())}>+ Add time</button>
      </div>
    </div>
  )
}

function AutomationEditor({
  automation,
  status,
}: {
  automation: AccountAutomation
  status: AccountOpsStatus | undefined
}) {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [draft, setDraft] = useState<AccountAutomationUpdate>(() => automationDraft(automation))
  const engagement = automation.feature_type === 'x_engagement'

  const save = useMutation({
    mutationFn: () => updateAccountAutomation(automation.id, draft),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['account-automations'] }),
  })
  const runNow = useMutation({
    mutationFn: () => runAccountAutomation(automation.id),
    onSuccess: (run) => {
      queryClient.invalidateQueries({ queryKey: ['account-runs'] })
      navigate(`/account-operations/runs/${run.id}`)
    },
  })
  const set = <K extends keyof AccountAutomationUpdate>(key: K, value: AccountAutomationUpdate[K]) => {
    setDraft((current) => ({ ...current, [key]: value }))
  }
  const setTimes = (values: string[]) => {
    const times = [...new Set(values)].sort()
    setDraft((current) => ({ ...current, schedule_times: times, schedule_time: times[0] }))
  }

  const executorLabel = draft.executor === 'opencode'
    ? 'OpenCode'
    : draft.executor === 'claude_sdk' ? 'Claude Agent SDK' : 'Pipeline'

  return (
    <section className="ops-config-panel">
      <div className="ops-config-head">
        <div>
          <span className="eyebrow">{engagement ? 'Conversation agent' : 'Daily commission'}</span>
          <h2>{engagement ? 'Replies & Reposts' : 'Today in History'}</h2>
          <p>{engagement
            ? 'A Following-only field desk for natural replies, Grok-assisted media reading, and rare quote-reposts.'
            : 'A quiet historical record, researched and illustrated through ChatGPT Web.'}</p>
        </div>
        <label className="ops-switch">
          <input type="checkbox" checked={draft.enabled} onChange={(event) => set('enabled', event.target.checked)} />
          <span>{draft.enabled ? 'Scheduled' : 'Paused'}</span>
        </label>
      </div>

      <AutomationStatusBar automation={automation} status={status} />

      <div className="ops-route-line" aria-label="Operation route">
        <span>{executorLabel}</span><i>→</i><span>OpenCLI</span><i>→</i><span>{engagement ? 'Following + Grok' : 'ChatGPT'}</span><i>→</i><span>@{draft.account_handle}</span>
      </div>

      <div className="ops-form-grid">
        <div className="form-group">
          <label htmlFor="ops-account">Required X account</label>
          <input id="ops-account" value={draft.account_handle} onChange={(event) => set('account_handle', event.target.value)} />
          <small>Agent switches accounts and verifies this handle before every write.</small>
        </div>
        <div className="form-group">
          <label htmlFor="ops-timezone">Timezone</label>
          <input id="ops-timezone" value={draft.timezone} onChange={(event) => set('timezone', event.target.value)} />
        </div>
        <div className="form-group">
          <label htmlFor="ops-executor">Orchestrator</label>
          <select id="ops-executor" value={draft.executor} onChange={(event) => set('executor', event.target.value as AccountAutomationUpdate['executor'])}>
            <option value="opencode">OpenCode agent</option>
            <option value="claude_sdk">Claude Agent SDK</option>
            {!engagement && <option value="pipeline">Deterministic pipeline</option>}
          </select>
        </div>
        {draft.executor !== 'pipeline' && (
          <div className="form-group">
            <label htmlFor="ops-model">Agent model</label>
            <input id="ops-model" value={draft.opencode_model} onChange={(event) => set('opencode_model', event.target.value)} />
          </div>
        )}

        <ScheduleTimes values={draft.schedule_times} onChange={setTimes} />

        {engagement && (
          <>
            <div className="form-group">
              <label htmlFor="ops-max-replies">Replies per run</label>
              <input id="ops-max-replies" type="number" min="1" max="10" value={draft.max_replies} onChange={(event) => set('max_replies', Number(event.target.value))} />
              <small>A maximum, never a quota.</small>
            </div>
            <div className="form-group">
              <label htmlFor="ops-max-quotes">Quote-reposts per run</label>
              <input id="ops-max-quotes" type="number" min="0" max="2" value={draft.max_quote_reposts} onChange={(event) => set('max_quote_reposts', Number(event.target.value))} />
              <small>Zero is normal; two is the hard ceiling.</small>
            </div>
            <div className="form-group">
              <label htmlFor="ops-scan-limit">Following posts to scan</label>
              <input id="ops-scan-limit" type="number" min="5" max="100" value={draft.scan_limit} onChange={(event) => set('scan_limit', Number(event.target.value))} />
            </div>
            <div className="form-group ops-prompt-field">
              <label htmlFor="ops-style-prompt">Human reply style</label>
              <textarea id="ops-style-prompt" value={draft.reply_style_prompt} onChange={(event) => set('reply_style_prompt', event.target.value)} spellCheck={false} />
              <small>Editable voice layer for both replies and quote commentary.</small>
            </div>
          </>
        )}

        <div className="form-group ops-prompt-field">
          <label htmlFor="ops-prompt">{engagement ? 'Selection & safety prompt' : 'Curatorial prompt'}</label>
          <textarea id="ops-prompt" value={draft.prompt_template} onChange={(event) => set('prompt_template', event.target.value)} spellCheck={false} />
          <small>{engagement
            ? 'Following-only selection, Grok media handling, and quote-repost threshold.'
            : <>Available variables: {'{date}'}, {'{month_name}'}, {'{day}'}, {'{year}'}</>}</small>
        </div>
      </div>

      <div className="ops-config-foot">
        <p>Next run <strong>{draft.enabled ? formatDateTime(automation.next_run_at) : 'paused'}</strong></p>
        <div>
          <button type="button" className="btn-ghost" disabled={save.isPending} onClick={() => save.mutate()}>{save.isPending ? 'Saving…' : 'Save configuration'}</button>
          <button type="button" className="btn-primary" disabled={runNow.isPending} onClick={() => runNow.mutate()}>{runNow.isPending ? 'Queuing…' : 'Run now'}</button>
        </div>
      </div>
      {(save.isError || runNow.isError) && <div className="error-box">{((save.error || runNow.error) as Error).message}</div>}
    </section>
  )
}

type Section = 'today-in-history' | 'replies-reposts'

const SECTIONS: { id: Section; feature: AccountAutomationFeature; label: string; hint: string; index: string }[] = [
  { id: 'today-in-history', feature: 'today_in_history', label: 'Today in History', hint: 'Daily editorial desk', index: '01' },
  { id: 'replies-reposts', feature: 'x_engagement', label: 'Replies & Reposts', hint: 'Following conversation agent', index: '02' },
]

type SubTab = 'commission' | 'ledger'
const SUBTABS: { id: SubTab; label: string }[] = [
  { id: 'commission', label: 'Configuration' },
  { id: 'ledger', label: 'Run ledger' },
]

export default function AccountOperations() {
  const [section, setSection] = useState<Section>('today-in-history')
  const [subtab, setSubtab] = useState<SubTab>('commission')
  const navigate = useNavigate()
  const selected = SECTIONS.find((item) => item.id === section) ?? SECTIONS[0]
  const { data: automations, isLoading: loadingAutomations } = useQuery({ queryKey: ['account-automations'], queryFn: fetchAccountAutomations })
  const { data: status } = useQuery({ queryKey: ['account-ops-status'], queryFn: fetchAccountOpsStatus, refetchInterval: 5_000 })
  const { data: runs, isLoading: loadingRuns } = useQuery({
    queryKey: ['account-runs'],
    queryFn: fetchAccountRuns,
    refetchInterval: (query) => query.state.data?.some((run) => ACTIVE.has(run.status)) ? 2500 : false,
  })
  const automation = automations?.find((item) => item.feature_type === selected.feature)
  const selectedRuns = runs?.filter((run) => run.feature_type === selected.feature) ?? []
  const published = runs?.filter((run) => run.status === 'published').length ?? 0
  const active = runs?.filter((run) => ACTIVE.has(run.status)).length ?? 0
  const failed = runs?.filter((run) => run.status === 'failed').length ?? 0
  const liveDesks = automations?.filter((item) => item.enabled).length ?? 0
  const workerAlive = status?.worker_alive ?? false

  let footDotClass = 'ops-status-dot--offline'
  let footLabel = 'Scheduler offline'
  if (workerAlive && liveDesks > 0) {
    footDotClass = 'ops-status-dot--live'
    footLabel = `${active} in flight · ${published} complete`
  } else if (workerAlive) {
    footDotClass = 'ops-status-dot--idle'
    footLabel = 'Paused'
  }

  return (
    <div className="ops-console">
      <aside className="ops-nav">
        <header className="ops-nav-heading"><span className="eyebrow">Quiet Atlas desk</span><h1>Account Ops</h1><p>Autonomous publishing and conversation with a complete operating record.</p></header>
        <nav className="ops-tabs" role="tablist" aria-label="Account operations sections">
          {SECTIONS.map((item) => (
            <button key={item.id} role="tab" aria-selected={section === item.id} aria-controls={`ops-panel-${item.id}`} className={`ops-tab ${section === item.id ? 'is-active' : ''}`} onClick={() => { setSection(item.id); setSubtab('commission') }}>
              <span className="ops-tab-index">{item.index}</span><span className="ops-tab-copy"><span className="ops-tab-label">{item.label}</span><span className="ops-tab-hint">{item.hint}</span></span><span className="ops-tab-arrow" aria-hidden="true">→</span>
            </button>
          ))}
        </nav>
        <footer className="ops-nav-foot"><span className={`ops-status-dot ${footDotClass}`} /><span><strong>{footLabel}</strong>{failed > 0 && <small>{failed} failed</small>}</span></footer>
      </aside>

      <section className="ops-panel" id={`ops-panel-${section}`} role="tabpanel" aria-label={selected.label}>
        <div className="ops-panel-scroll"><div>
          <nav className="ops-subtabs" role="tablist" aria-label={`${selected.label} sections`}>
            {SUBTABS.map((item) => <button key={item.id} role="tab" aria-selected={subtab === item.id} className={`ops-subtab ${subtab === item.id ? 'is-active' : ''}`} onClick={() => setSubtab(item.id)}>{item.label}</button>)}
          </nav>

          {subtab === 'commission' && (
            loadingAutomations ? <div className="table-message">Loading configuration…</div> : automation ? (
              <AutomationEditor key={`${automation.id}-${automation.updated_at}`} automation={automation} status={status} />
            ) : <div className="empty-state">No {selected.label} automation configured.</div>
          )}
          {subtab === 'ledger' && (
            <section className="ops-ledger">
              <div className="ops-ledger-head"><div><span className="eyebrow">Operating ledger</span><h2>{selected.label}</h2></div><span>{selectedRuns.length} records</span></div>
              {loadingRuns ? <div className="table-message">Loading run ledger…</div> : !selectedRuns.length ? (
                <div className="ops-ledger-empty"><strong>No runs yet.</strong><span>Run this automation to create its first audited record.</span></div>
              ) : (
                <div className="ops-table-scroll"><table className="ops-table">
                  <thead><tr><th>Run</th><th>Status</th><th>Trigger</th><th>Account</th><th>Created</th><th /></tr></thead>
                  <tbody>{selectedRuns.map((run) => (
                    <tr key={run.id} tabIndex={0} onClick={() => navigate(`/account-operations/runs/${run.id}`)} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') navigate(`/account-operations/runs/${run.id}`) }}>
                      <td><strong>{runTitle(run)}</strong><small>{run.event_date} · {run.id.slice(-6)}</small></td><td><span className={`badge ${run.status}`}>{run.status.replace('_', ' ')}</span></td><td>{run.trigger}</td><td>@{run.account_handle}</td><td>{formatDateTime(run.created_at)}</td><td><span className="row-arrow">→</span></td>
                    </tr>
                  ))}</tbody>
                </table></div>
              )}
            </section>
          )}
        </div></div>
      </section>
    </div>
  )
}

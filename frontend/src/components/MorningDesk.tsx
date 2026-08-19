import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import {
  fetchDailyAutomation,
  fetchDailySources,
  fetchSettingsSchema,
  fetchTtsModels,
  fetchVoices,
  runDailyNow,
  updateDailyAutomation,
  type DailyAutomationSettings,
} from '../api'

const SOURCE_LABELS: Record<string, string> = {
  zh: '中文',
  en: 'English',
}

const TIMEZONE_SUGGESTIONS = [
  'Asia/Singapore',
  'Asia/Shanghai',
  'Asia/Tokyo',
  'Europe/London',
  'America/New_York',
  'America/Los_Angeles',
]

function formatDateTime(value: string | null) {
  if (!value) return 'Not scheduled'
  return new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value))
}

function inRange(value: number, minimum: number, maximum: number) {
  return Number.isFinite(value) && value >= minimum && value <= maximum
}

export default function MorningDesk() {
  const queryClient = useQueryClient()
  const {
    data,
    isError: automationIsError,
    error: automationError,
  } = useQuery({ queryKey: ['daily-automation'], queryFn: fetchDailyAutomation })
  const { data: sources = [] } = useQuery({ queryKey: ['daily-sources'], queryFn: fetchDailySources })
  const { data: settingsSchema } = useQuery({ queryKey: ['settings-schema'], queryFn: fetchSettingsSchema })
  const {
    data: ttsModels = [],
    isError: ttsModelsIsError,
  } = useQuery({ queryKey: ['tts-models'], queryFn: fetchTtsModels })
  const [draftOverride, setDraftOverride] = useState<DailyAutomationSettings | null>(null)
  const persisted = data?.settings
  const draft = draftOverride ?? (persisted ? {
    ...persisted,
    footage_clip_count: persisted.footage_clip_count ?? 8,
    news_image_count: persisted.news_image_count ?? 4,
  } : null)
  const {
    data: voices = [],
    isFetching: voicesAreLoading,
    isError: voicesAreError,
  } = useQuery({
    queryKey: ['voices', draft?.tts_model],
    queryFn: () => fetchVoices(draft!.tts_model),
    enabled: Boolean(draft?.tts_model),
  })

  const applySaved = (next: Awaited<ReturnType<typeof updateDailyAutomation>>) => {
    setDraftOverride(next.settings)
    queryClient.setQueryData(['daily-automation'], next)
  }

  const save = useMutation({
    mutationFn: (value: DailyAutomationSettings) => updateDailyAutomation(value),
    onSuccess: applySaved,
  })
  const openQueuedTask = (task: Awaited<ReturnType<typeof runDailyNow>>) => {
    queryClient.invalidateQueries({ queryKey: ['tasks'] })
    queryClient.invalidateQueries({ queryKey: ['daily-automation'] })
    window.location.assign(`/tasks/${task.id}`)
  }

  const runTest = useMutation({
    mutationFn: async (value: DailyAutomationSettings) => {
      const saved = await updateDailyAutomation(value)
      // Saving and queuing are two separate requests. Reflect the committed
      // recipe immediately so a queueing failure never looks like the recipe
      // was lost or remains unsaved.
      applySaved(saved)
      return runDailyNow(true, 1)
    },
    onSuccess: openQueuedTask,
  })
  const runNow = useMutation({
    mutationFn: async (value: DailyAutomationSettings) => {
      const saved = await updateDailyAutomation(value)
      applySaved(saved)
      // No duration override: the backend snapshots the saved full-length
      // recipe and follows the same render and distribution path as a
      // scheduled edition.
      return runDailyNow(false, null)
    },
    onSuccess: openQueuedTask,
  })

  const patch = <K extends keyof DailyAutomationSettings>(key: K, value: DailyAutomationSettings[K]) => {
    if (!draft) return
    setDraftOverride((current) => ({ ...(current ?? draft), [key]: value }))
  }

  const toggleTarget = (target: DailyAutomationSettings['publish_targets'][number]) => {
    const selected = draft?.publish_targets ?? []
    patch(
      'publish_targets',
      selected.includes(target) ? selected.filter((item) => item !== target) : [...selected, target],
    )
  }

  if (!draft || !data) {
    if (automationIsError) return <div className="error-box">{String(automationError)}</div>
    return <div className="empty-state">Opening the morning desk…</div>
  }

  const priority = sources.filter((source) => source.priority <= 6)
  const secondary = sources.filter((source) => source.priority > 6)
  const publicationFields = Object.fromEntries(
    (settingsSchema?.groups.find((group) => group.id === 'publication')?.fields ?? [])
      .map((field) => [field.key, field.value]),
  )
  const globalPublishingEnabled = Boolean(publicationFields.VIDEO_AUTO_PUBLISH_ENABLED)
  const youtubeIdentity = String(publicationFields.VIDEO_PUBLISH_YOUTUBE_CHANNEL_NAME || 'Channel not configured')
  const youtubeReady = Boolean(
    publicationFields.VIDEO_PUBLISH_YOUTUBE_ENABLED
    && publicationFields.VIDEO_PUBLISH_YOUTUBE_CHANNEL_NAME
    && publicationFields.VIDEO_PUBLISH_YOUTUBE_CHANNEL_ID,
  )
  const xIdentity = publicationFields.VIDEO_PUBLISH_X_HANDLE
    ? `@${String(publicationFields.VIDEO_PUBLISH_X_HANDLE).replace(/^@/, '')}`
    : 'Handle not configured'
  const xReady = Boolean(publicationFields.VIDEO_PUBLISH_X_ENABLED && publicationFields.VIDEO_PUBLISH_X_HANDLE)
  const podcastReady = Boolean(publicationFields.VIDEO_PUBLISH_APPLE_PODCAST_ENABLED)
  const distributionEnabled = globalPublishingEnabled && draft.auto_publish
  const distributionState = !globalPublishingEnabled
    ? 'Blocked by global kill switch'
    : draft.auto_publish ? 'Automatic dispatch armed' : 'Paused · selections retained'
  const destinations = [
    {
      target: 'youtube' as const,
      index: '01',
      mark: 'YT',
      name: 'YouTube',
      capability: 'Native video upload',
      identity: youtubeIdentity,
      ready: youtubeReady,
    },
    {
      target: 'x' as const,
      index: '02',
      mark: 'X',
      name: 'x.com',
      capability: 'Native video post',
      identity: xIdentity,
      ready: xReady,
    },
    {
      target: 'apple_podcast' as const,
      index: '03',
      mark: 'RSS',
      name: 'Podcast RSS',
      capability: 'Feed generation only',
      identity: 'Local feed · no Apple account submission',
      ready: podcastReady,
    },
  ]
  const selectedModel = ttsModels.find((model) => model.id === draft.tts_model)
  const selectedVoiceIsValid = voices.some((voice) => voice.name === draft.voice)
  const effectiveVoice = selectedVoiceIsValid ? draft.voice : (voices[0]?.name ?? draft.voice)
  const effectiveVoiceIsValid = voices.some((voice) => voice.name === effectiveVoice)
  const recipe = effectiveVoice === draft.voice ? draft : { ...draft, voice: effectiveVoice }
  const catalogReady = Boolean(draft.tts_model && effectiveVoice)
    && !voicesAreLoading
    && (voicesAreError || effectiveVoiceIsValid)
    && (ttsModelsIsError || ttsModels.some((model) => model.id === draft.tts_model))
  const configReady = catalogReady
    && Boolean(draft.generation_time && draft.timezone.trim())
    && inRange(draft.target_duration_minutes, 1, 30)
    && inRange(draft.max_stories, 3, 12)
    && inRange(draft.source_window_hours, 12, 96)
    && inRange(draft.collage_broll_count, 2, 10)
    && inRange(draft.news_image_count, 2, 12)
    && inRange(draft.footage_clip_count, 1, 30)
  const dirty = JSON.stringify(recipe) !== JSON.stringify(data.settings)
  const actionPending = save.isPending || runTest.isPending || runNow.isPending
  const actionError = save.error || runTest.error || runNow.error

  return (
    <div className="morning-desk">
      <header className="morning-masthead">
        <div>
          <span className="morning-kicker">Autonomous edition · {draft.timezone}</span>
          <h1>Frontier Tech<br />Morning Desk</h1>
        </div>
        <div className="morning-status">
          <span className={draft.enabled ? 'is-live' : ''}><i />{draft.enabled ? 'Daily run enabled' : 'Schedule paused'}</span>
          <strong>{draft.generation_time}</strong>
          <small>Next edition · {formatDateTime(data.next_run_at)}</small>
        </div>
      </header>

      <section className="morning-config" aria-labelledby="morning-config-title">
        <header className="morning-config-head">
          <div>
            <span className="morning-section-label">Automation specification</span>
            <h2 id="morning-config-title">One saved recipe.<br />Every morning.</h2>
            <p>This contract is copied into every scheduled edition. Change it here once; the next research, voice, visual, and distribution run follows it.</p>
          </div>
          <label className={`morning-master-switch ${draft.enabled ? 'is-live' : ''}`}>
            <span><small>Desk state</small><strong>{draft.enabled ? 'Automation armed' : 'Automation paused'}</strong></span>
            <input type="checkbox" checked={draft.enabled} onChange={(event) => patch('enabled', event.target.checked)} />
          </label>
        </header>

        <div className="morning-config-grid">
          <article className="morning-config-panel">
            <header><span>01</span><div><strong>Schedule</strong><small>When the newsroom wakes</small></div></header>
            <div className="morning-field-grid">
              <label><span>Desk time</span><input type="time" value={draft.generation_time} onChange={(event) => patch('generation_time', event.target.value)} /></label>
              <label><span>Timezone</span><input type="text" list="morning-timezones" value={draft.timezone} onChange={(event) => patch('timezone', event.target.value)} /></label>
              <datalist id="morning-timezones">{TIMEZONE_SUGGESTIONS.map((timezone) => <option value={timezone} key={timezone} />)}</datalist>
            </div>
            <label className="morning-inline-switch">
              <input type="checkbox" checked={draft.catch_up_after_restart} onChange={(event) => patch('catch_up_after_restart', event.target.checked)} />
              <span><strong>Catch up after restart</strong><small>Queue today’s edition if the desk comes back after its scheduled time.</small></span>
            </label>
          </article>

          <article className="morning-config-panel">
            <header><span>02</span><div><strong>Editorial brief</strong><small>Length, language, and evidence window</small></div></header>
            <div className="morning-field-grid morning-field-grid--four">
              <label><span>Run length</span><div className="morning-unit-input"><input type="number" min="1" max="30" value={draft.target_duration_minutes} onChange={(event) => patch('target_duration_minutes', Number(event.target.value))} /><i>min</i></div></label>
              <label><span>Language</span><select value={draft.language} onChange={(event) => patch('language', event.target.value as 'en' | 'zh')}><option value="en">English</option><option value="zh">中文</option></select></label>
              <label><span>Stories</span><input type="number" min="3" max="12" value={draft.max_stories} onChange={(event) => patch('max_stories', Number(event.target.value))} /></label>
              <label><span>Source window</span><div className="morning-unit-input"><input type="number" min="12" max="96" step="12" value={draft.source_window_hours} onChange={(event) => patch('source_window_hours', Number(event.target.value))} /><i>hr</i></div></label>
            </div>
            <p className="morning-panel-note">The source quorum remains a hard gate; widening the window never lowers evidence requirements.</p>
          </article>

          <article className="morning-config-panel">
            <header><span>03</span><div><strong>Voice desk</strong><small>Model-dependent single-host delivery</small></div></header>
            <div className="morning-field-grid">
              <label>
                <span>TTS model</span>
                <select value={draft.tts_model} onChange={(event) => patch('tts_model', event.target.value)}>
                  {!ttsModels.some((model) => model.id === draft.tts_model) && <option value={draft.tts_model}>{draft.tts_model}</option>}
                  {ttsModels.map((model) => <option value={model.id} key={model.id}>{model.provider} — {model.label}</option>)}
                </select>
              </label>
              <label>
                <span>Host voice</span>
                <select value={effectiveVoice} disabled={voicesAreLoading} onChange={(event) => patch('voice', event.target.value)}>
                  {(voicesAreLoading || voicesAreError) && !selectedVoiceIsValid && <option value={draft.voice}>{voicesAreLoading ? 'Loading compatible voices…' : draft.voice}</option>}
                  {voices.map((voice) => <option value={voice.name} key={voice.name}>{voice.name} · {voice.gender}</option>)}
                </select>
              </label>
            </div>
            <div className="morning-contract-line"><span>Delivery</span><strong>Monologue · {selectedModel?.single_speaker ? 'single-speaker engine' : 'single host selected'}</strong></div>
            {(ttsModelsIsError || voicesAreError) && <p className="morning-panel-warning">The live voice catalog could not be loaded. Existing saved values remain available.</p>}
          </article>

          <article className="morning-config-panel morning-config-panel--visual">
            <header><span>04</span><div><strong>Visual recipe</strong><small>Collage, news imagery, and footage are separate</small></div></header>
            <div className="morning-visual-row">
              <div className="morning-visual-copy"><b>Paper-Collage</b><small>Generated visual metaphors · always on</small></div>
              <label><span>Clips / edition</span><input type="number" min="2" max="10" value={draft.collage_broll_count} onChange={(event) => patch('collage_broll_count', Number(event.target.value))} /></label>
            </div>
            <div className="morning-visual-row is-enabled">
              <div className="morning-visual-copy"><b>News imagery</b><small>OpenCLI-grounded logos and event stills · inline + full screen</small></div>
              <label><span>Images / edition</span><input type="number" min="2" max="12" value={draft.news_image_count} onChange={(event) => patch('news_image_count', Number(event.target.value))} /></label>
            </div>
            <div className={`morning-visual-row ${draft.public_footage_enabled ? 'is-enabled' : ''}`}>
              <label className="morning-inline-switch morning-inline-switch--compact">
                <input type="checkbox" checked={draft.public_footage_enabled} onChange={(event) => patch('public_footage_enabled', event.target.checked)} />
                <span><strong>Public footage</strong><small>Scout eligible external B-roll</small></span>
              </label>
              <label><span>Clips / edition</span><input type="number" min="1" max="30" disabled={!draft.public_footage_enabled} value={draft.footage_clip_count} onChange={(event) => patch('footage_clip_count', Number(event.target.value))} /></label>
            </div>
            <label className="morning-music-field"><span>Program music</span><select value={draft.background_music_provider} onChange={(event) => patch('background_music_provider', event.target.value as DailyAutomationSettings['background_music_provider'])}><option value="gemini_create_music">Gemini Create Music · local fallback</option><option value="local">Deterministic local bed</option></select></label>
          </article>
        </div>

        <footer className="morning-config-actions">
          <div>
            <span className={`morning-save-state ${dirty ? 'is-dirty' : ''}`}><i />{dirty ? 'Unsaved changes' : 'Saved recipe'}</span>
            <small>Tests are one minute and never publish. Full runs use the saved duration and distribution settings.</small>
          </div>
          <div className="morning-action-buttons">
            <button type="button" className="btn-ghost" disabled={actionPending || !configReady} onClick={() => save.mutate(recipe)}>{save.isPending ? 'Saving…' : 'Save recipe'}</button>
            <button type="button" className="btn-primary" disabled={actionPending || !configReady} onClick={() => runTest.mutate(recipe)}>{runTest.isPending ? 'Saving & queuing test…' : 'Save & run 1-min test'}</button>
            <button
              type="button"
              className="btn-primary morning-start-now"
              disabled={actionPending || !configReady}
              title="Save this recipe and start a full end-to-end edition immediately"
              onClick={() => runNow.mutate(recipe)}
            >
              {runNow.isPending ? 'Saving & starting full run…' : 'Start full run now'}
            </button>
          </div>
          {actionError && <div className="error-box">{String(actionError)}</div>}
        </footer>
      </section>

      <section className={`morning-distribution ${distributionEnabled ? 'is-active' : 'is-paused'}`}>
        <header className="morning-distribution-head">
          <div className="morning-distribution-copy">
            <span className="morning-section-label">Dispatch matrix</span>
            <h2>Automatic distribution</h2>
            <p>Send every completed edition to the selected destinations. Platform choices are retained while dispatch is paused.</p>
          </div>
          <div className="morning-distribution-controls">
            <label className={`morning-visibility ${distributionEnabled && draft.publish_targets.includes('youtube') ? '' : 'is-disabled'}`}>
              <span>YouTube visibility</span>
              <select
                value={draft.publish_visibility}
                disabled={!distributionEnabled || !draft.publish_targets.includes('youtube')}
                onChange={(event) => patch('publish_visibility', event.target.value as 'private' | 'unlisted' | 'public')}
              >
                <option value="public">Public</option>
                <option value="unlisted">Unlisted</option>
                <option value="private">Private</option>
              </select>
            </label>
            <label className={`morning-distribution-master ${distributionEnabled ? 'is-live' : ''}`}>
              <span><small>Edition dispatch</small><strong>{distributionState}</strong></span>
              <input type="checkbox" checked={draft.auto_publish} onChange={(event) => patch('auto_publish', event.target.checked)} />
            </label>
          </div>
        </header>

        <div className="morning-destinations" aria-label="Automatic publication targets" aria-disabled={!distributionEnabled}>
          {destinations.map((destination) => {
            const selected = draft.publish_targets.includes(destination.target)
            const ready = distributionEnabled && selected && destination.ready
            const state = !distributionEnabled
              ? 'Paused'
              : !selected ? 'Not selected' : destination.ready ? 'Ready' : 'Setup required'
            return (
              <label
                className={`morning-destination ${selected ? 'is-selected' : ''} ${ready ? 'is-ready' : ''}`}
                key={destination.target}
              >
                <span className="morning-destination-index">{destination.index}</span>
                <span className="morning-destination-mark">{destination.mark}</span>
                <span className="morning-destination-switch">
                  <input
                    type="checkbox"
                    checked={selected}
                    disabled={!distributionEnabled}
                    onChange={() => toggleTarget(destination.target)}
                  />
                  <i aria-hidden="true" />
                </span>
                <span className="morning-destination-copy">
                  <b>{destination.name}</b>
                  <small>{destination.capability}</small>
                </span>
                <span className="morning-destination-identity">{destination.identity}</span>
                <span className="morning-destination-state"><i />{state}</span>
              </label>
            )
          })}
        </div>

        <footer className="morning-distribution-foot">
          <p>Automatic writes require the configured account to match the active browser identity. Test runs remain private and require exact-receipt cleanup.</p>
          <Link to="/settings?tab=publishing">Configure platform accounts →</Link>
        </footer>
      </section>

      <section className="morning-sources">
        <div className="morning-section-head"><div><span className="morning-section-label">Signal board</span><h2>{sources.length || 12} monitored sources</h2></div><p>Six primary desks set the agenda; six secondary desks widen or verify the frame. Per-source failure is isolated, while source quorum remains a hard gate.</p></div>
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

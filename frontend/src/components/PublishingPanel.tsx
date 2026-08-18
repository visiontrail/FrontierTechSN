import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  fetchSettingsSchema,
  updateSettingsValues,
  type SettingField,
  type SettingValue,
} from '../api'

const PLATFORM_FIELDS = {
  youtube: [
    'VIDEO_PUBLISH_YOUTUBE_ENABLED',
    'VIDEO_PUBLISH_YOUTUBE_CHANNEL_NAME',
    'VIDEO_PUBLISH_YOUTUBE_CHANNEL_ID',
  ],
  x: ['VIDEO_PUBLISH_X_ENABLED', 'VIDEO_PUBLISH_X_HANDLE'],
  podcast: [
    'VIDEO_PUBLISH_APPLE_PODCAST_ENABLED',
    'VIDEO_PUBLISH_APPLE_FEED_TITLE',
    'VIDEO_PUBLISH_APPLE_FEED_AUTHOR',
    'VIDEO_PUBLISH_APPLE_FEED_PUBLIC_BASE_URL',
  ],
} as const

type Platform = keyof typeof PLATFORM_FIELDS

const PLATFORM_COPY: Record<Platform, { mark: string; title: string; capability: string; note: string }> = {
  youtube: {
    mark: 'YT',
    title: 'YouTube',
    capability: 'Native video upload',
    note: 'Channel name and Channel ID must both match YouTube Studio before upload begins.',
  },
  x: {
    mark: 'X',
    title: 'x.com',
    capability: 'Native video post',
    note: 'The active session must match the configured @handle before the composer opens.',
  },
  podcast: {
    mark: 'RSS',
    title: 'Podcast feed',
    capability: 'Feed generation only',
    note: 'Creates a standards-compliant RSS feed. Apple Podcasts Connect submission is not automated.',
  },
}

export default function PublishingPanel() {
  const queryClient = useQueryClient()
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['settings-schema'],
    queryFn: fetchSettingsSchema,
  })
  const publication = data?.groups.find((group) => group.id === 'publication')
  const fields = useMemo(
    () => Object.fromEntries((publication?.fields ?? []).map((field) => [field.key, field])),
    [publication?.fields],
  ) as Record<string, SettingField>
  const [drafts, setDrafts] = useState<Record<string, SettingValue>>({})

  const value = (key: string) => key in drafts ? drafts[key] : fields[key]?.value
  const setValue = (key: string, next: SettingValue) => setDrafts((current) => ({ ...current, [key]: next }))
  const dirty = Object.keys(drafts).filter((key) => fields[key] && drafts[key] !== fields[key].value)

  const save = useMutation({
    mutationFn: () => updateSettingsValues(Object.fromEntries(dirty.map((key) => [key, drafts[key]]))),
    onSuccess: (schema) => {
      queryClient.setQueryData(['settings-schema'], schema)
      setDrafts({})
    },
  })

  if (isLoading) return <div className="empty-state">Opening publishing control…</div>
  if (isError) return <div className="error-box">{(error as Error).message}</div>
  if (!publication) return <div className="error-box">Publishing settings are unavailable.</div>

  const masterEnabled = Boolean(value('VIDEO_AUTO_PUBLISH_ENABLED'))

  function renderField(key: string) {
    const field = fields[key]
    if (!field) return null
    const current = value(key)
    if (field.type === 'bool') {
      return (
        <label className="publish-platform-switch" key={key}>
          <input type="checkbox" checked={Boolean(current)} onChange={(event) => setValue(key, event.target.checked)} />
          <span aria-hidden="true" />
          <b>{current ? 'On' : 'Off'}</b>
        </label>
      )
    }
    return (
      <label className="publish-identity-field" key={key}>
        <span>{field.label}</span>
        <input
          type="text"
          value={String(current ?? '')}
          placeholder={field.placeholder}
          spellCheck={false}
          onChange={(event) => setValue(key, event.target.value)}
        />
        <small>{field.description}</small>
      </label>
    )
  }

  function readiness(platform: Platform) {
    const keys = PLATFORM_FIELDS[platform]
    const enabled = Boolean(value(keys[0]))
    if (!enabled) return { ready: false, label: 'Disabled' }
    if (platform === 'youtube') {
      const ready = Boolean(
        String(value(PLATFORM_FIELDS.youtube[1]) ?? '').trim()
        && String(value(PLATFORM_FIELDS.youtube[2]) ?? '').trim(),
      )
      return { ready, label: ready ? 'Identity configured' : 'Identity required' }
    }
    if (platform === 'x') {
      const ready = Boolean(String(value(PLATFORM_FIELDS.x[1]) ?? '').trim())
      return { ready, label: ready ? 'Identity configured' : 'Handle required' }
    }
    const ready = Boolean(
      String(value(PLATFORM_FIELDS.podcast[1]) ?? '').trim()
      && String(value(PLATFORM_FIELDS.podcast[2]) ?? '').trim(),
    )
    return { ready, label: ready ? 'Feed metadata ready' : 'Metadata required' }
  }

  return (
    <div className="publishing-console">
      <header className="publishing-head">
        <div>
          <span className="eyebrow">Distribution safety</span>
          <h2>Publishing &amp; Accounts</h2>
          <p>Choose explicit destinations and pin each browser-backed adapter to one expected identity. A mismatch stops the run before any social write.</p>
        </div>
        <label className={`publish-master ${masterEnabled ? 'is-live' : ''}`}>
          <span><small>Global kill switch</small><strong>{masterEnabled ? 'Automatic publishing armed' : 'Human review only'}</strong></span>
          <input type="checkbox" checked={masterEnabled} onChange={(event) => setValue('VIDEO_AUTO_PUBLISH_ENABLED', event.target.checked)} />
        </label>
      </header>

      <div className="publishing-rule">
        <span>Dispatch contract</span>
        <p>Global switch <b>and</b> platform switch <b>and</b> task opt-in <b>and</b> exact account match</p>
      </div>

      <div className="publishing-platforms">
        {(Object.keys(PLATFORM_FIELDS) as Platform[]).map((platform, index) => {
          const copy = PLATFORM_COPY[platform]
          const state = readiness(platform)
          const keys = PLATFORM_FIELDS[platform]
          return (
            <section className={`publish-platform-card ${state.ready ? 'is-ready' : ''}`} key={platform}>
              <header>
                <span className="publish-platform-index">0{index + 1}</span>
                <span className="publish-platform-mark">{copy.mark}</span>
                <div><h3>{copy.title}</h3><p>{copy.capability}</p></div>
                {renderField(keys[0])}
              </header>
              <div className="publish-platform-state"><i />{state.label}</div>
              <p className="publish-platform-note">{copy.note}</p>
              <div className="publish-identity-grid">
                {keys.slice(1).map((key) => renderField(key))}
              </div>
            </section>
          )
        })}
      </div>

      {save.isError && <div className="error-box">{(save.error as Error).message}</div>}
      <footer className="settings-savebar publishing-savebar">
        <span>{dirty.length ? `${dirty.length} unsaved change${dirty.length === 1 ? '' : 's'}` : 'Configuration matches the running service'}</span>
        <button type="button" disabled={!dirty.length} onClick={() => setDrafts({})}>Discard</button>
        <button type="button" className="btn-primary" disabled={!dirty.length || save.isPending} onClick={() => save.mutate()}>
          {save.isPending ? 'Saving…' : 'Save publishing configuration'}
        </button>
      </footer>
    </div>
  )
}

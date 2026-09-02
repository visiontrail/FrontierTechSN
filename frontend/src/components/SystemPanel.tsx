import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { fetchSettingsSchema, fetchTtsModels, resetSettingsValues, updateSettingsValues } from '../api'
import type { SettingField, SettingValue, SettingsSchema } from '../api'

/**
 * Every setting that used to live in `.env`, edited in place.
 *
 * The form is generic on purpose: the backend describes each field (type,
 * options, bounds, default) in `backend/settings_store.py`, so a new setting
 * shows up here without a frontend change.
 */
export default function SystemPanel() {
  const queryClient = useQueryClient()
  const { data, isLoading, isError, error } = useQuery<SettingsSchema>({
    queryKey: ['settings-schema'],
    queryFn: fetchSettingsSchema,
  })
  const { data: ttsModels = [] } = useQuery({
    queryKey: ['tts-models'],
    queryFn: fetchTtsModels,
  })

  const groups = useMemo(() => data?.groups ?? [], [data?.groups])
  const [groupId, setGroupId] = useState<string | null>(null)
  // Only edited keys live here, so a background refetch never clobbers typing.
  const [drafts, setDrafts] = useState<Record<string, SettingValue>>({})
  // A blanked secret is indistinguishable from an untouched one, so clearing is
  // an explicit act we have to remember.
  const [cleared, setCleared] = useState<string[]>([])
  const [restartKeys, setRestartKeys] = useState<string[]>([])

  const activeGroup = groups.find((g) => g.id === groupId) ?? groups[0] ?? null
  const fields = useMemo(() => groups.flatMap((g) => g.fields), [groups])

  const defaultTtsModelField = fields.find((field) => field.key === 'TTS_DEFAULT_MODEL')
  const selectedDefaultTtsModel = String(
    drafts.TTS_DEFAULT_MODEL ?? defaultTtsModelField?.value ?? '',
  )
  const selectedTtsModel = ttsModels.find((model) => model.id === selectedDefaultTtsModel)

  function isVisible(field: SettingField): boolean {
    if (field.key !== 'ORPHEUS_TTS_SPEED_PERCENT') return true
    return selectedTtsModel?.speed_adjustable === true
  }

  function isDirty(field: SettingField): boolean {
    if (!(field.key in drafts)) return false
    if (field.type === 'secret') return drafts[field.key] !== '' || cleared.includes(field.key)
    return drafts[field.key] !== field.value
  }

  const dirtyFields = fields.filter((field) => isVisible(field) && isDirty(field))
  const dirtyByGroup = (id: string) =>
    (groups.find((g) => g.id === id)?.fields ?? [])
      .filter((field) => isVisible(field) && isDirty(field)).length

  function setDraft(key: string, value: SettingValue) {
    setDrafts((d) => ({ ...d, [key]: value }))
    setCleared((c) => c.filter((k) => k !== key))
  }

  function clearAll() {
    setDrafts({})
    setCleared([])
  }

  function onSaved(schema: SettingsSchema) {
    queryClient.setQueryData(['settings-schema'], schema)
    queryClient.invalidateQueries({ queryKey: ['settings-schema'] })
    // Voices, models and endpoints shown elsewhere may have moved with it.
    queryClient.invalidateQueries({ queryKey: ['voices'] })
    queryClient.invalidateQueries({ queryKey: ['tts-models'] })
    setRestartKeys(schema.restart_required)
    clearAll()
  }

  const saveMutation = useMutation({
    mutationFn: () => {
      const payload: Record<string, SettingValue> = {}
      for (const field of dirtyFields) payload[field.key] = drafts[field.key]
      return updateSettingsValues(payload)
    },
    onSuccess: onSaved,
  })

  const resetMutation = useMutation({
    mutationFn: (keys: string[]) => resetSettingsValues(keys),
    onSuccess: onSaved,
  })

  if (isLoading) return <p>Loading…</p>
  if (isError) return <div className="error-box">{(error as Error).message}</div>

  function renderControl(field: SettingField) {
    const draft = field.key in drafts ? drafts[field.key] : field.value

    if (field.type === 'bool') {
      return (
        <label className="setting-toggle">
          <input
            type="checkbox"
            checked={Boolean(draft)}
            onChange={(e) => setDraft(field.key, e.target.checked)}
          />
          <span>{draft ? 'Enabled' : 'Disabled'}</span>
        </label>
      )
    }

    if (field.type === 'choice') {
      return (
        <select value={String(draft)} onChange={(e) => setDraft(field.key, e.target.value)}>
          {field.options.map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      )
    }

    if (field.type === 'int') {
      return (
        <div className="setting-inline">
          <input
            type="number"
            min={field.minimum ?? undefined}
            max={field.maximum ?? undefined}
            value={String(draft)}
            onChange={(e) => {
              // Keep half-typed input ("-", "1e") as text rather than NaN; the
              // backend rejects it with a readable message if it is submitted.
              const raw = e.target.value
              const parsed = Number(raw)
              setDraft(field.key, raw === '' || Number.isNaN(parsed) ? raw : parsed)
            }}
          />
          {field.unit && <span className="setting-unit">{field.unit}</span>}
        </div>
      )
    }

    if (field.type === 'secret') {
      const typed = String(draft ?? '')
      return (
        <div className="setting-inline">
          <input
            type="password"
            value={typed}
            placeholder={field.is_set ? `saved: ${field.masked}` : 'not set'}
            onChange={(e) => setDraft(field.key, e.target.value)}
          />
          {field.is_set && (
            <button
              type="button"
              onClick={() => {
                setDrafts((d) => ({ ...d, [field.key]: '' }))
                setCleared((c) => (c.includes(field.key) ? c : [...c, field.key]))
              }}
            >
              Clear
            </button>
          )}
        </div>
      )
    }

    return (
      <input
        type="text"
        value={String(draft ?? '')}
        placeholder={field.placeholder}
        spellCheck={false}
        onChange={(e) => setDraft(field.key, e.target.value)}
      />
    )
  }

  function renderDefault(field: SettingField) {
    if (field.type === 'secret') {
      return field.default_masked ? field.default_masked : '(none)'
    }
    if (field.type === 'bool') return field.default ? 'enabled' : 'disabled'
    const text = String(field.default ?? '')
    return text === '' ? '(blank)' : text
  }

  return (
    <div className="split-panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">System &amp; Environment</h2>
          <p className="panel-sub">
            Everything the app used to read from <code>.env</code>, editable here. Saved values go
            to <code>data/settings.json</code> and apply to the next pipeline stage immediately —
            <code>.env</code> now only seeds the defaults you can reset back to.
          </p>
        </div>
      </div>

      {restartKeys.length > 0 && (
        <div className="warn-box" style={{ marginBottom: 12 }}>
          ⚠ Saved. {restartKeys.map((k) => <code key={k} className="var-chip">{k}</code>)} needs a
          restart (<code>./scripts/start.sh</code>) before it fully takes effect.
        </div>
      )}

      <div className="split-2">
        <div className="split-list">
          {groups.map((group) => {
            const pending = dirtyByGroup(group.id)
            return (
              <button
                key={group.id}
                className={`list-item ${group.id === activeGroup?.id ? 'is-active' : ''}`}
                onClick={() => setGroupId(group.id)}
              >
                <div className="list-item-top">
                  <span className="list-item-title">{group.label}</span>
                  {pending > 0 && <span className="badge badge-warn">{pending}</span>}
                </div>
                <span className="badge badge-muted">{group.fields.filter(isVisible).length} settings</span>
              </button>
            )
          })}
        </div>

        <div className="split-detail">
          {activeGroup && (
            <>
              <p className="panel-sub" style={{ marginTop: 0 }}>{activeGroup.description}</p>

              <div className="setting-list">
                {activeGroup.fields.filter(isVisible).map((field) => (
                  <div key={field.key} className={`setting-row ${isDirty(field) ? 'is-dirty' : ''}`}>
                    <div className="setting-head">
                      <span className="setting-label">{field.label}</span>
                      {field.is_overridden && <span className="badge badge-accent">custom</span>}
                      {field.restart_required && (
                        <span className="badge badge-warn" title="Applies on the next app start">
                          restart
                        </span>
                      )}
                      <code className="file-chip">{field.key}</code>
                    </div>

                    {field.description && <p className="setting-desc">{field.description}</p>}

                    <div className="setting-control">{renderControl(field)}</div>

                    <div className="setting-foot">
                      <span>default: <code>{renderDefault(field)}</code></span>
                      {field.is_overridden && (
                        <button
                          type="button"
                          className="link-button"
                          disabled={resetMutation.isPending}
                          onClick={() => resetMutation.mutate([field.key])}
                        >
                          Reset to default
                        </button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      {(saveMutation.isError || resetMutation.isError) && (
        <div className="error-box" style={{ marginTop: 12 }}>
          {((saveMutation.error ?? resetMutation.error) as Error).message}
        </div>
      )}

      <div className="settings-savebar">
        <button
          className="btn-primary"
          disabled={dirtyFields.length === 0 || saveMutation.isPending}
          onClick={() => saveMutation.mutate()}
        >
          {saveMutation.isPending ? 'Saving…' : `Save changes${dirtyFields.length ? ` (${dirtyFields.length})` : ''}`}
        </button>
        <button disabled={dirtyFields.length === 0} onClick={clearAll}>
          Discard
        </button>
        <span style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--text-dim)' }}>
          {dirtyFields.length > 0
            ? `${dirtyFields.length} unsaved change${dirtyFields.length > 1 ? 's' : ''}`
            : 'All changes saved'}
        </span>
      </div>
    </div>
  )
}

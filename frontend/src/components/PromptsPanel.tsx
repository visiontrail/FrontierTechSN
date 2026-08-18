import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { fetchPrompts, updatePrompt, resetPrompt } from '../api'

export default function PromptsPanel() {
  const queryClient = useQueryClient()
  const { data: prompts = [], isLoading } = useQuery({ queryKey: ['prompts'], queryFn: fetchPrompts })

  const [selectedKey, setSelectedKey] = useState<string | null>(null)
  const [drafts, setDrafts] = useState<Record<string, string>>({})

  const activeKey = selectedKey ?? prompts[0]?.key ?? null
  const selected = prompts.find((p) => p.key === activeKey) ?? null
  const draft = selected ? drafts[selected.key] ?? selected.content : ''
  const setDraft = (value: string) => {
    if (!selected) return
    setDrafts((current) => ({ ...current, [selected.key]: value }))
  }
  const clearDraft = (key: string) => {
    setDrafts((current) => {
      const next = { ...current }
      delete next[key]
      return next
    })
  }

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ['prompts'] })

  const saveMutation = useMutation({
    mutationFn: () => updatePrompt(selected!.key, draft),
    onSuccess: (updated) => {
      invalidate()
      clearDraft(updated.key)
    },
  })

  const resetMutation = useMutation({
    mutationFn: () => resetPrompt(selected!.key),
    onSuccess: (updated) => {
      invalidate()
      clearDraft(updated.key)
    },
  })

  const dirty = selected ? draft !== selected.content : false
  // Warn if a required {placeholder} the pipeline substitutes is missing.
  const missing = selected
    ? selected.variables.filter((v) => !new RegExp(`\\{${v}\\}`).test(draft))
    : []

  if (isLoading) return <p>Loading…</p>

  return (
    <div className="split-panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Pipeline Prompts</h2>
          <p className="panel-sub">
            System prompts the pipeline sends at each stage. Edits take effect on the next task —
            no restart. Use <code>{'{placeholder}'}</code> tokens for values the pipeline fills in.
          </p>
        </div>
      </div>

      <div className="split-2">
        <div className="split-list">
          {prompts.map((p) => (
            <button
              key={p.key}
              className={`list-item ${p.key === activeKey ? 'is-active' : ''}`}
              onClick={() => setSelectedKey(p.key)}
            >
              <div className="list-item-top">
                <span className="list-item-title">{p.label}</span>
                {p.is_modified && <span className="badge badge-warn">edited</span>}
              </div>
              <span className="badge badge-muted">{p.stage}</span>
            </button>
          ))}
        </div>

        <div className="split-detail">
          {selected && (
            <>
              <p className="panel-sub" style={{ marginTop: 0 }}>{selected.description}</p>
              <div className="meta-row">
                <span className="badge badge-muted">{selected.stage}</span>
                <code className="file-chip">{selected.file}</code>
                {selected.variables.length > 0 && (
                  <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>
                    variables:{' '}
                    {selected.variables.map((v) => (
                      <code key={v} className="var-chip">{`{${v}}`}</code>
                    ))}
                  </span>
                )}
              </div>

              <textarea
                className="code-area"
                value={draft}
                spellCheck={false}
                onChange={(e) => setDraft(e.target.value)}
              />

              {missing.length > 0 && (
                <div className="warn-box">
                  ⚠ Missing variables the pipeline expects:{' '}
                  {missing.map((v) => (
                    <code key={v} className="var-chip">{`{${v}}`}</code>
                  ))}
                  . The stage may misbehave without them.
                </div>
              )}

              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 12 }}>
                <button
                  className="btn-primary"
                  disabled={!dirty || saveMutation.isPending}
                  onClick={() => saveMutation.mutate()}
                >
                  {saveMutation.isPending ? 'Saving…' : 'Save'}
                </button>
                <button disabled={!dirty} onClick={() => clearDraft(selected.key)}>
                  Revert edits
                </button>
                <button
                  disabled={!selected.has_default || !selected.is_modified || resetMutation.isPending}
                  title={selected.has_default ? 'Restore the shipped default' : 'No default snapshot'}
                  onClick={() => resetMutation.mutate()}
                >
                  Reset to default
                </button>
                <span style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--text-dim)' }}>
                  {dirty ? 'Unsaved changes' : 'Saved'}
                </span>
              </div>
              {saveMutation.isError && (
                <div className="error-box" style={{ marginTop: 12 }}>{(saveMutation.error as Error).message}</div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

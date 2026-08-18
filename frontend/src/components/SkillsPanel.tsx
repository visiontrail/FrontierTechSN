import { useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { fetchSkills, fetchSkill, updateSkill, uploadSkill } from '../api'
import type { Skill } from '../api'

export default function SkillsPanel() {
  const queryClient = useQueryClient()
  const { data: skills = [], isLoading } = useQuery({ queryKey: ['skills'], queryFn: fetchSkills })

  const [selectedName, setSelectedName] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [editing, setEditing] = useState(false)
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const uploadInputRef = useRef<HTMLInputElement>(null)

  const listItem = skills.find((s) => s.name === selectedName) ?? null

  const { data: detail } = useQuery({
    queryKey: ['skill', selectedName],
    queryFn: () => fetchSkill(selectedName!),
    enabled: selectedName !== null,
  })

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ['skills'] })
    queryClient.invalidateQueries({ queryKey: ['skill', selectedName] })
  }

  const toggleMutation = useMutation({
    mutationFn: (input: { name: string; enabled: boolean }) =>
      updateSkill(input.name, { enabled: input.enabled }),
    onSuccess: invalidate,
  })

  const saveMutation = useMutation({
    mutationFn: () => updateSkill(selectedName!, { body: draft }),
    onSuccess: () => {
      invalidate()
      setEditing(false)
    },
  })

  const uploadMutation = useMutation({
    mutationFn: (file: File) => uploadSkill(file),
    onSuccess: (skill) => {
      queryClient.setQueryData(['skill', skill.name], skill)
      queryClient.invalidateQueries({ queryKey: ['skills'] })
      setSelectedName(skill.name)
      setEditing(false)
      setUploadFile(null)
      if (uploadInputRef.current) uploadInputRef.current.value = ''
    },
  })

  const enabledCount = skills.filter((s) => s.enabled).length

  if (isLoading) return <p>Loading…</p>

  return (
    <div className="split-panel">
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Skills</h2>
          <p className="panel-sub">
            The HyperFrames <code>SKILL.md</code> bundles under <code>.claude/skills</code> that guide
            Claude Agent SDK calls. Disabled skills are blocked at runtime; editing rewrites the skill body.
          </p>
        </div>
        <span className="badge badge-muted">{enabledCount}/{skills.length} enabled</span>
      </div>

      <form
        className="skill-import"
        onSubmit={(event) => {
          event.preventDefault()
          if (uploadFile) uploadMutation.mutate(uploadFile)
        }}
      >
        <div>
          <strong>Load a skill package</strong>
          <p>Upload a <code>.skill</code>, <code>.zip</code>, or standalone <code>SKILL.md</code>. ZIP packages keep their scripts and reference assets.</p>
        </div>
        <div className="skill-import-actions">
          <label className="sr-only" htmlFor="skill-package">Skill package</label>
          <input
            ref={uploadInputRef}
            id="skill-package"
            type="file"
            accept=".skill,.zip,.md,application/zip,text/markdown"
            disabled={uploadMutation.isPending}
            onChange={(event) => {
              setUploadFile(event.target.files?.[0] ?? null)
              uploadMutation.reset()
            }}
          />
          <button className="btn-primary" type="submit" disabled={!uploadFile || uploadMutation.isPending}>
            {uploadMutation.isPending ? 'Loading…' : 'Load skill'}
          </button>
        </div>
        {uploadMutation.isError && (
          <div className="error-box">{(uploadMutation.error as Error).message}</div>
        )}
        {uploadMutation.isSuccess && (
          <div className="success-box">Loaded <strong>{uploadMutation.data.name}</strong> and enabled it.</div>
        )}
      </form>

      <div className="split-2">
        <div className="split-list">
          {skills.map((s: Skill) => (
            <button
              key={s.name}
              className={`list-item ${s.name === selectedName ? 'is-active' : ''}`}
              onClick={() => {
                setSelectedName(s.name)
                setEditing(false)
              }}
            >
              <div className="list-item-top">
                <span className="list-item-title" style={{ opacity: s.enabled ? 1 : 0.5 }}>{s.name}</span>
                {!s.enabled && <span className="badge badge-muted">off</span>}
              </div>
              <span className="list-item-desc">{s.description}</span>
            </button>
          ))}
        </div>

        <div className="split-detail">
          {listItem && (
            <>
              <div className="panel-head" style={{ marginBottom: 8 }}>
                <h3 style={{ fontSize: 16, margin: 0 }}>{listItem.name}</h3>
                <label className="switch">
                  <input
                    type="checkbox"
                    checked={listItem.enabled}
                    disabled={toggleMutation.isPending}
                    onChange={(e) => toggleMutation.mutate({ name: listItem.name, enabled: e.target.checked })}
                    style={{ width: 'auto' }}
                  />
                  <span>{listItem.enabled ? 'Enabled' : 'Disabled'}</span>
                </label>
              </div>
              <p className="panel-sub" style={{ marginTop: 0 }}>{listItem.description}</p>
              <div className="meta-row">
                <code className="file-chip">{listItem.path}</code>
                {listItem.is_symlink && <span className="badge badge-muted">symlink</span>}
              </div>

              {editing ? (
                <>
                  <textarea
                    className="code-area"
                    value={draft}
                    spellCheck={false}
                    onChange={(e) => setDraft(e.target.value)}
                  />
                  <p style={{ fontSize: 12, color: 'var(--text-faint)', marginTop: 6 }}>
                    Only the markdown body is editable; the <code>name</code>/<code>description</code>{' '}
                    frontmatter is preserved.
                    {listItem.is_symlink && ' This skill is a symlink — edits change the shared source.'}
                  </p>
                  <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
                    <button className="btn-primary" disabled={saveMutation.isPending} onClick={() => saveMutation.mutate()}>
                      {saveMutation.isPending ? 'Saving…' : 'Save'}
                    </button>
                    <button onClick={() => { setEditing(false); setDraft(detail?.body ?? '') }}>Cancel</button>
                  </div>
                  {saveMutation.isError && (
                    <div className="error-box" style={{ marginTop: 12 }}>{(saveMutation.error as Error).message}</div>
                  )}
                </>
              ) : (
                <>
                  <pre className="skill-body">{detail?.body ?? 'Loading…'}</pre>
                  <div style={{ marginTop: 12 }}>
                    <button
                      onClick={() => {
                        setDraft(detail?.body ?? '')
                        setEditing(true)
                      }}
                      disabled={!detail}
                    >
                      Edit body
                    </button>
                  </div>
                </>
              )}
            </>
          )}
          {!listItem && <p style={{ color: 'var(--text-dim)' }}>Select a skill to view its instructions.</p>}
        </div>
      </div>
    </div>
  )
}

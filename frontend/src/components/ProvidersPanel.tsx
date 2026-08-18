import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  fetchProviders,
  createProvider,
  updateProvider,
  deleteProvider,
  testProvider,
} from '../api'
import type { Provider, ProviderInput, ProviderTestRequest, ProviderTestResult } from '../api'

const EMPTY_FORM: ProviderInput = {
  name: '',
  endpoint: '',
  api_key: '',
  model: '',
  is_default: false,
}

export default function ProvidersPanel() {
  const queryClient = useQueryClient()
  const { data: providers = [], isLoading } = useQuery({
    queryKey: ['providers'],
    queryFn: fetchProviders,
  })

  const [editingId, setEditingId] = useState<number | null>(null)
  const [form, setForm] = useState<ProviderInput>(EMPTY_FORM)
  const [showForm, setShowForm] = useState(false)

  const [testing, setTesting] = useState<string | null>(null)
  const [testResults, setTestResults] = useState<Record<string, ProviderTestResult>>({})

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ['providers'] })

  async function runTest(key: string, input: ProviderTestRequest) {
    setTesting(key)
    setTestResults((r) => {
      const next = { ...r }
      delete next[key]
      return next
    })
    try {
      const result = await testProvider(input)
      setTestResults((r) => ({ ...r, [key]: result }))
    } catch (e) {
      setTestResults((r) => ({ ...r, [key]: { ok: false, message: (e as Error).message } }))
    } finally {
      setTesting(null)
    }
  }

  const saveMutation = useMutation({
    mutationFn: () =>
      editingId === null ? createProvider(form) : updateProvider(editingId, form),
    onSuccess: () => {
      invalidate()
      resetForm()
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (id: number) => deleteProvider(id),
    onSuccess: invalidate,
  })

  const defaultMutation = useMutation({
    mutationFn: (id: number) => updateProvider(id, { is_default: true }),
    onSuccess: invalidate,
  })

  function resetForm() {
    setForm(EMPTY_FORM)
    setEditingId(null)
    setShowForm(false)
  }

  function startEdit(p: Provider) {
    setEditingId(p.id)
    setForm({ name: p.name, endpoint: p.endpoint, api_key: '', model: p.model, is_default: p.is_default })
    setShowForm(true)
  }

  const canSave = form.name.trim() && form.endpoint.trim() && form.model.trim()
  const canTestForm = form.endpoint.trim() && form.model.trim()

  function renderTestResult(key: string) {
    if (testing === key) {
      return <span style={{ fontSize: 12, color: 'var(--text-dim)' }}>Testing…</span>
    }
    const r = testResults[key]
    if (!r) return null
    return (
      <span style={{ fontSize: 12, color: r.ok ? 'var(--accent-green)' : 'var(--accent-red)' }}>
        {r.ok
          ? `✓ ${r.message}${r.latency_ms != null ? ` (${r.latency_ms} ms)` : ''}`
          : `✗ ${r.message}`}
      </span>
    )
  }

  return (
    <div>
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Models &amp; Providers</h2>
          <p className="panel-sub">
            The AI gateways (endpoint + model + key) that drive summarization and scriptwriting.
            The <strong>default</strong> provider is used when a task doesn&apos;t pick one.
          </p>
        </div>
      </div>

      {isLoading ? (
        <p>Loading…</p>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 16 }}>
          {providers.length === 0 && (
            <p style={{ color: 'var(--text-dim)' }}>No providers configured.</p>
          )}
          {providers.map((p) => (
            <div key={p.id} className="card provider-row">
              <div className="provider-info">
                <div style={{ fontWeight: 600 }}>
                  {p.name}{' '}
                  {p.is_default && (
                    <span className="badge badge-accent">default</span>
                  )}
                </div>
                <div style={{ fontSize: 13, color: 'var(--text-dim)', wordBreak: 'break-all' }}>
                  {p.model} — {p.endpoint}
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-faint)' }}>
                  key: {p.api_key_masked || '(none)'}
                </div>
              </div>
              <div className="provider-actions">
                <div className="provider-actions-row">
                  <button disabled={testing === `card-${p.id}`} onClick={() => runTest(`card-${p.id}`, { provider_id: p.id })}>
                    Test
                  </button>
                  {!p.is_default && <button onClick={() => defaultMutation.mutate(p.id)}>Set default</button>}
                  <button onClick={() => startEdit(p)}>Edit</button>
                  <button onClick={() => deleteMutation.mutate(p.id)}>Delete</button>
                </div>
                {renderTestResult(`card-${p.id}`)}
              </div>
            </div>
          ))}
        </div>
      )}

      {showForm ? (
        <div className="card" style={{ padding: 16 }}>
          <h3 style={{ marginBottom: 12, fontSize: 15 }}>
            {editingId === null ? 'Add Provider' : 'Edit Provider'}
          </h3>
          <div className="form-group">
            <label>Name</label>
            <input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="Local Ollama" />
          </div>
          <div className="form-group">
            <label>Endpoint</label>
            <input
              value={form.endpoint}
              onChange={(e) => setForm({ ...form, endpoint: e.target.value })}
              placeholder="http://localhost:11434/v1/chat/completions"
            />
          </div>
          <div className="grid-2">
            <div className="form-group">
              <label>Model</label>
              <input value={form.model} onChange={(e) => setForm({ ...form, model: e.target.value })} placeholder="llama3" />
            </div>
            <div className="form-group">
              <label>API Key {editingId !== null && '(leave blank to keep)'}</label>
              <input
                type="password"
                value={form.api_key ?? ''}
                onChange={(e) => setForm({ ...form, api_key: e.target.value })}
                placeholder="sk-…"
              />
            </div>
          </div>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: 14, marginBottom: 12 }}>
            <input
              type="checkbox"
              checked={form.is_default ?? false}
              onChange={(e) => setForm({ ...form, is_default: e.target.checked })}
              style={{ width: 'auto' }}
            />
            Set as default provider
          </label>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <button className="btn-primary" disabled={!canSave || saveMutation.isPending} onClick={() => saveMutation.mutate()}>
              {saveMutation.isPending ? 'Saving…' : 'Save'}
            </button>
            <button
              disabled={!canTestForm || testing === 'form'}
              onClick={() =>
                runTest('form', {
                  provider_id: editingId,
                  endpoint: form.endpoint.trim(),
                  model: form.model.trim(),
                  api_key: form.api_key?.trim() || undefined,
                })
              }
            >
              Test
            </button>
            <button onClick={resetForm}>Cancel</button>
            <span style={{ marginLeft: 'auto' }}>{renderTestResult('form')}</span>
          </div>
          {saveMutation.isError && (
            <div className="error-box" style={{ marginTop: 12 }}>{(saveMutation.error as Error).message}</div>
          )}
        </div>
      ) : (
        <button className="btn-primary" onClick={() => { setForm(EMPTY_FORM); setEditingId(null); setShowForm(true) }}>
          + Add Provider
        </button>
      )}
    </div>
  )
}

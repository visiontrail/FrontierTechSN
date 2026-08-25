import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  fetchProviders,
  fetchProviderCatalog,
  createProvider,
  updateProvider,
  deleteProvider,
  testProvider,
} from '../api'
import type {
  Provider,
  ProviderCatalogEntry,
  ProviderInput,
  ProviderTestRequest,
  ProviderTestResult,
} from '../api'

const CUSTOM_MODEL = '__custom__'
const ENDPOINT_PLACEHOLDER = /\{[^{}]+\}/
const CUSTOM_PROFILE: ProviderCatalogEntry = {
  id: 'custom',
  label: 'Custom Anthropic-compatible endpoint',
  default_endpoint: '',
  default_model: '',
  models: [],
  notes: 'Enter the endpoint and model ID manually.',
  endpoint_needs_input: false,
}

const EMPTY_FORM: ProviderInput = {
  provider_type: 'custom',
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
  const {
    data: loadedCatalog = [],
    isError: catalogIsError,
  } = useQuery({
    queryKey: ['provider-catalog'],
    queryFn: fetchProviderCatalog,
    staleTime: Infinity,
  })
  const catalog = loadedCatalog.length ? loadedCatalog : [CUSTOM_PROFILE]

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
    setForm({
      provider_type: p.provider_type,
      name: p.name,
      endpoint: p.endpoint,
      api_key: '',
      model: p.model,
      is_default: p.is_default,
    })
    setShowForm(true)
  }

  const selectedProfile = catalog.find((profile) => profile.id === form.provider_type)
    ?? CUSTOM_PROFILE
  const selectedModelPreset = selectedProfile.models.includes(form.model)
    ? form.model
    : CUSTOM_MODEL

  function selectProvider(providerType: string) {
    const nextProfile = catalog.find((profile) => profile.id === providerType) ?? CUSTOM_PROFILE
    const nameWasGenerated = !form.name.trim() || form.name === selectedProfile.label
    setForm({
      ...form,
      provider_type: nextProfile.id,
      name: nameWasGenerated ? nextProfile.label : form.name,
      endpoint: nextProfile.default_endpoint,
      model: nextProfile.default_model,
    })
  }

  function startAdd() {
    setForm(EMPTY_FORM)
    setEditingId(null)
    setShowForm(true)
  }

  const endpointReady = Boolean(form.endpoint.trim()) && !ENDPOINT_PLACEHOLDER.test(form.endpoint)
  const canSave = Boolean(form.name.trim() && endpointReady && form.model.trim())
  const canTestForm = Boolean(endpointReady && form.model.trim())

  function renderTestResult(key: string) {
    if (testing === key) {
      return <span className="provider-test-result is-testing">Testing…</span>
    }
    const r = testResults[key]
    if (!r) return null
    return (
      <span className={`provider-test-result ${r.ok ? 'is-ok' : 'is-fail'}`}>
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
            Add independent AI gateways from the RavenAIService provider catalog. Choosing a
            provider fills its endpoint and links the matching model presets; custom values remain
            editable. The <strong>default</strong> row is used when a task doesn&apos;t pick one.
          </p>
          <p className="provider-routing-note">Independent provider rows · no primary/backup routing</p>
        </div>
      </div>

      {catalogIsError && (
        <div className="provider-catalog-warning" role="status">
          The provider catalog could not be loaded. Manual provider configuration remains available.
        </div>
      )}

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
                  <span className="badge badge-muted">
                    {catalog.find((profile) => profile.id === p.provider_type)?.label ?? p.provider_type}
                  </span>{' '}
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
        <div className="card provider-form-card">
          <div className="provider-form-head">
            <div>
              <span className="provider-form-kicker">Catalog-linked configuration</span>
              <h3>{editingId === null ? 'Add Provider' : 'Edit Provider'}</h3>
            </div>
            <span className="provider-form-index">{editingId === null ? 'NEW' : `#${editingId}`}</span>
          </div>

          <div className="grid-2 provider-form-grid">
            <div className="form-group">
              <label htmlFor="provider-type">Provider</label>
              <select
                id="provider-type"
                value={form.provider_type}
                onChange={(event) => selectProvider(event.target.value)}
              >
                {catalog.map((profile) => (
                  <option value={profile.id} key={profile.id}>{profile.label} · {profile.id}</option>
                ))}
              </select>
              <small>{selectedProfile.notes}</small>
            </div>
            <div className="form-group">
              <label htmlFor="provider-name">Display name</label>
              <input
                id="provider-name"
                value={form.name}
                onChange={(event) => setForm({ ...form, name: event.target.value })}
                placeholder="DeepSeek — production"
              />
              <small>Name this credential or gateway instance.</small>
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="provider-endpoint">Endpoint / Base URL</label>
            <input
              id="provider-endpoint"
              value={form.endpoint}
              onChange={(e) => setForm({ ...form, endpoint: e.target.value })}
              placeholder={selectedProfile.default_endpoint || 'http://localhost:11434'}
            />
            {selectedProfile.endpoint_needs_input ? (
              <small className="provider-field-warning">
                Replace the endpoint placeholder with your workspace-specific value before testing.
              </small>
            ) : (
              <small>The preset remains editable for proxies and private gateways.</small>
            )}
          </div>

          <div className="grid-2 provider-form-grid">
            <div className="form-group">
              <label htmlFor="provider-model-preset">Model preset</label>
              {selectedProfile.models.length ? (
                <select
                  id="provider-model-preset"
                  value={selectedModelPreset}
                  onChange={(event) => {
                    if (event.target.value !== CUSTOM_MODEL) {
                      setForm({ ...form, model: event.target.value })
                    }
                  }}
                >
                  {selectedProfile.models.map((model) => (
                    <option value={model} key={model}>{model}</option>
                  ))}
                  <option value={CUSTOM_MODEL}>Custom model ID…</option>
                </select>
              ) : (
                <div className="provider-manual-field">Manual model ID</div>
              )}
              <small>Known models are shortcuts, not a whitelist.</small>
            </div>
            <div className="form-group">
              <label htmlFor="provider-model">Model ID</label>
              <input
                id="provider-model"
                value={form.model}
                onChange={(event) => setForm({ ...form, model: event.target.value })}
                placeholder={selectedProfile.default_model || 'model-id'}
              />
              <small>Edit directly when the provider ships a newer model.</small>
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="provider-key">API Key {editingId !== null && '(leave blank to keep)'}</label>
            <input
              id="provider-key"
              type="password"
              value={form.api_key ?? ''}
              onChange={(e) => setForm({ ...form, api_key: e.target.value })}
              placeholder="sk-…"
              autoComplete="new-password"
            />
            <small>The key is stored server-side and only returned in masked form.</small>
          </div>

          <label className="provider-default-toggle">
            <input
              type="checkbox"
              checked={form.is_default ?? false}
              onChange={(e) => setForm({ ...form, is_default: e.target.checked })}
            />
            Set as default provider
          </label>

          <div className="provider-form-actions">
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
            <span className="provider-form-test-status">{renderTestResult('form')}</span>
          </div>
          {saveMutation.isError && (
            <div className="error-box" style={{ marginTop: 12 }}>{(saveMutation.error as Error).message}</div>
          )}
        </div>
      ) : (
        <button className="btn-primary" onClick={startAdd}>
          + Add Provider
        </button>
      )}
    </div>
  )
}

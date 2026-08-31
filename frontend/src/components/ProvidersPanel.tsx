import { useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  fetchProviders,
  fetchProviderCatalog,
  createProvider,
  updateProvider,
  deleteProvider,
  testProvider,
  testProviderRoute,
} from '../api'
import type {
  Provider,
  ProviderCatalogEntry,
  ProviderInput,
  ProviderTestRequest,
  ProviderTestResult,
  ProviderRouteTestResult,
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
  route_role: 'standalone',
}

const ROLE_LABELS: Record<NonNullable<ProviderInput['route_role']>, string> = {
  primary: 'PRIMARY',
  backup: 'BACKUP',
  standalone: 'STANDALONE',
}

function parseKeyPool(value: string): string[] {
  return value
    .split(/[\n,]+/)
    .map((key) => key.trim())
    .filter(Boolean)
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
  const [apiKeysText, setApiKeysText] = useState('')
  const [showForm, setShowForm] = useState(false)

  const [testing, setTesting] = useState<string | null>(null)
  const [testResults, setTestResults] = useState<Record<string, ProviderTestResult>>({})
  const [routeTestResult, setRouteTestResult] = useState<ProviderRouteTestResult | null>(null)

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
    mutationFn: () => {
      const isPrimary = form.route_role === 'primary'
      const parsedKeys = parseKeyPool(apiKeysText)
      const payload: ProviderInput = {
        ...form,
        api_key: isPrimary ? undefined : form.api_key?.trim() || undefined,
        api_keys: isPrimary
          ? (apiKeysText.trim() ? parsedKeys : editingId === null ? [] : undefined)
          : undefined,
      }
      return editingId === null ? createProvider(payload) : updateProvider(editingId, payload)
    },
    onSuccess: () => {
      invalidate()
      setRouteTestResult(null)
      resetForm()
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (id: number) => deleteProvider(id),
    onSuccess: () => {
      invalidate()
      setRouteTestResult(null)
    },
  })

  const routeTestMutation = useMutation({
    mutationFn: testProviderRoute,
    onMutate: () => setRouteTestResult(null),
    onSuccess: setRouteTestResult,
    onError: (error) => setRouteTestResult({
      ok: false,
      message: (error as Error).message,
      fallback_used: false,
      events: [],
    }),
  })

  function resetForm() {
    setForm(EMPTY_FORM)
    setApiKeysText('')
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
      route_role: p.route_role,
    })
    setApiKeysText('')
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
    setApiKeysText('')
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
      <div className={`provider-test-result ${r.ok ? 'is-ok' : 'is-fail'}`}>
        <span>
          {r.ok
            ? `✓ ${r.message}${r.latency_ms != null ? ` (${r.latency_ms} ms)` : ''}`
            : `✗ ${r.message}`}
        </span>
        {r.key_results && r.key_results.length > 1 && (
          <div className="provider-key-test-list">
            {r.key_results.map((result) => (
              <span key={result.key_id} className={result.ok ? 'is-ok' : 'is-fail'}>
                {result.ok ? '✓' : '✗'} {result.key_id}
                {result.latency_ms != null ? ` · ${result.latency_ms} ms` : ''}
              </span>
            ))}
          </div>
        )}
      </div>
    )
  }

  return (
    <div>
      <div className="panel-head">
        <div>
          <h2 className="panel-title">Models &amp; Providers</h2>
          <p className="panel-sub">
            Configure the persisted primary and backup routes used by every task. The primary route
            accepts a pool of API keys and distributes calls across them; the backup route is used
            only when the primary route is unavailable.
          </p>
          <p className="provider-routing-note">Admin-owned routing · no .env role switches · secrets stay server-side</p>
        </div>
        <button
          className="provider-route-test-button"
          disabled={routeTestMutation.isPending || !providers.some((provider) => provider.route_role === 'primary')}
          onClick={() => routeTestMutation.mutate()}
        >
          {routeTestMutation.isPending ? 'Testing route…' : 'Test primary → backup route'}
        </button>
      </div>

      {routeTestResult && (
        <div className={`provider-route-result ${routeTestResult.ok ? 'is-ok' : 'is-fail'}`} role="status">
          <div className="provider-route-result-head">
            <strong>{routeTestResult.ok ? 'Route test passed' : 'Route test failed'}</strong>
            {routeTestResult.route_role && (
              <span className={`provider-role-badge is-${routeTestResult.route_role}`}>
                {ROLE_LABELS[routeTestResult.route_role]}
              </span>
            )}
          </div>
          <div>{routeTestResult.message}</div>
          {routeTestResult.provider_name && (
            <div className="provider-route-meta">
              {routeTestResult.provider_name} · {routeTestResult.model}
              {routeTestResult.key_id ? ` · key ${routeTestResult.key_id}` : ''}
              {routeTestResult.fallback_used ? ' · fallback used' : ''}
              {routeTestResult.latency_ms != null ? ` · ${routeTestResult.latency_ms} ms` : ''}
            </div>
          )}
          {routeTestResult.events.length > 0 && (
            <ol className="provider-route-events">
              {routeTestResult.events.map((event, index) => <li key={`${index}-${event}`}>{event}</li>)}
            </ol>
          )}
        </div>
      )}

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
                  <span className={`provider-role-badge is-${p.route_role}`}>
                    {ROLE_LABELS[p.route_role]}
                  </span>
                </div>
                <div style={{ fontSize: 13, color: 'var(--text-dim)', wordBreak: 'break-all' }}>
                  {p.model} — {p.endpoint}
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-faint)' }}>
                  {p.route_role === 'primary'
                    ? `${p.api_key_count} key${p.api_key_count === 1 ? '' : 's'} in rotation · first ${p.api_key_masked || '(none)'}`
                    : `key: ${p.api_key_masked || '(none)'}`}
                </div>
              </div>
              <div className="provider-actions">
                <div className="provider-actions-row">
                  <button disabled={testing === `card-${p.id}`} onClick={() => runTest(`card-${p.id}`, { provider_id: p.id })}>
                    {p.route_role === 'primary' && p.api_key_count > 1 ? 'Test all keys' : 'Test'}
                  </button>
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

          <div className="form-group provider-route-role-field">
            <label htmlFor="provider-route-role">Routing role</label>
            <select
              id="provider-route-role"
              value={form.route_role ?? 'standalone'}
              onChange={(event) => setForm({
                ...form,
                route_role: event.target.value as NonNullable<ProviderInput['route_role']>,
              })}
            >
              <option value="primary">Primary — normal traffic, multi-key rotation</option>
              <option value="backup">Backup — automatic failover, single key</option>
              <option value="standalone">Standalone — selected explicitly by a task</option>
            </select>
            <small>Saving a primary or backup route replaces the previous provider in that role.</small>
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

          {form.route_role === 'primary' ? (
            <div className="form-group">
              <label htmlFor="provider-key-pool">
                Primary API key pool {editingId !== null && '(leave blank to keep existing keys)'}
              </label>
              <textarea
                id="provider-key-pool"
                className="provider-key-pool-input"
                rows={5}
                value={apiKeysText}
                onChange={(event) => setApiKeysText(event.target.value)}
                placeholder={'sk-primary-01\nsk-primary-02\nsk-primary-03'}
                autoComplete="new-password"
              />
              <small>
                One key per line. Calls reserve keys in a persisted round-robin sequence; duplicate
                keys are rejected. Existing values are never returned to this page.
              </small>
            </div>
          ) : (
            <div className="form-group">
              <label htmlFor="provider-key">API Key {editingId !== null && '(leave blank to keep)'}</label>
              <input
                id="provider-key"
                type="password"
                value={form.api_key ?? ''}
                onChange={(event) => setForm({ ...form, api_key: event.target.value })}
                placeholder="sk-…"
                autoComplete="new-password"
              />
              <small>The key is stored server-side and only returned in masked form.</small>
            </div>
          )}

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
                  api_key: form.route_role === 'primary' ? undefined : form.api_key?.trim() || undefined,
                  api_keys: form.route_role === 'primary'
                    ? (apiKeysText.trim() ? parseKeyPool(apiKeysText) : undefined)
                    : undefined,
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

import { useState } from 'react'
import ProvidersPanel from './ProvidersPanel'
import PromptsPanel from './PromptsPanel'
import SkillsPanel from './SkillsPanel'
import SystemPanel from './SystemPanel'

type Tab = 'models' | 'prompts' | 'skills' | 'system'

const TABS: { id: Tab; label: string; hint: string; index: string }[] = [
  { id: 'models', label: 'Models', hint: 'Providers & endpoints', index: '01' },
  { id: 'prompts', label: 'Prompts', hint: 'Pipeline instructions', index: '02' },
  { id: 'skills', label: 'Skills', hint: 'Composition bundles', index: '03' },
  { id: 'system', label: 'System', hint: 'Runtime & environment', index: '04' },
]

export default function SettingsPage() {
  const [tab, setTab] = useState<Tab>('models')

  return (
    <div className="admin-console">
      <aside className="admin-nav">
        <header className="admin-heading">
          <span className="eyebrow">System control</span>
          <h1>Admin</h1>
          <p>Configure the intelligence behind every production run.</p>
        </header>

        <nav className="admin-tabs" role="tablist" aria-label="Admin sections">
          {TABS.map((t) => (
            <button
              key={t.id}
              role="tab"
              aria-selected={tab === t.id}
              aria-controls={`admin-panel-${t.id}`}
              className={`admin-tab ${tab === t.id ? 'is-active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              <span className="admin-tab-index">{t.index}</span>
              <span className="admin-tab-copy">
                <span className="admin-tab-label">{t.label}</span>
                <span className="admin-tab-hint">{t.hint}</span>
              </span>
              <span className="admin-tab-arrow" aria-hidden="true">→</span>
            </button>
          ))}
        </nav>

        <footer className="admin-nav-foot">
          <span className="system-pulse" aria-hidden="true" />
          <span><strong>Pipeline online</strong><small>Changes apply to new tasks</small></span>
        </footer>
      </aside>

      <section
        className="admin-panel"
        id={`admin-panel-${tab}`}
        role="tabpanel"
        aria-label={TABS.find((item) => item.id === tab)?.label}
      >
        <div className="admin-panel-scroll">
          {tab === 'models' && <ProvidersPanel />}
          {tab === 'prompts' && <PromptsPanel />}
          {tab === 'skills' && <SkillsPanel />}
          {tab === 'system' && <SystemPanel />}
        </div>
      </section>
    </div>
  )
}

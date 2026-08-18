import { useEffect, useState } from 'react'
import { Routes, Route, NavLink, Link, useLocation, useNavigate } from 'react-router-dom'
import TaskList from './components/TaskList'
import TaskForm from './components/TaskForm'
import TaskDetail from './components/TaskDetail'
import SettingsPage from './components/SettingsPage'
import AccountOperations from './components/AccountOperations'
import AccountRunDetail from './components/AccountRunDetail'
import ContentPlanning from './components/ContentPlanning'
import MorningDesk from './components/MorningDesk'
import {
  IconTasks,
  IconNew,
  IconAdmin,
  IconAtlas,
  IconCalendar,
  IconSun,
  IconMoon,
  IconMenu,
  IconChevronLeft,
  IconChevronRight,
} from './components/Icons'

type Theme = 'light' | 'dark'

const NAV = [
  { to: '/morning-desk', label: 'Morning Desk', hint: 'Daily tech briefing', Icon: IconSun, end: false },
  { to: '/', label: 'Tasks', hint: 'Pipeline queue', Icon: IconTasks, end: true },
  { to: '/new', label: 'New Task', hint: 'Start a render', Icon: IconNew, end: false },
  { to: '/planning', label: 'Content Plan', hint: 'Series & release calendar', Icon: IconCalendar, end: false },
  { to: '/account-operations', label: 'Account Ops', hint: 'Autonomous publishing', Icon: IconAtlas, end: false },
  { to: '/settings', label: 'Admin', hint: 'Models & prompts', Icon: IconAdmin, end: false },
]

/* `full` drops the centred measure; `chrome: false` lets workspace routes
   provide their own hierarchy instead of repeating it in a global top bar. */
type PageMeta = {
  title: string
  sub: string
  back?: boolean
  action?: 'new'
  full?: boolean
  chrome?: boolean
}

function pageMeta(pathname: string): PageMeta {
  if (pathname === '/') return { title: 'Tasks', sub: 'Every podcast render in the pipeline', full: true, chrome: false }
  if (pathname === '/new') return { title: 'New Task', sub: 'Configure a podcast video run', full: true, chrome: false }
  if (pathname === '/planning') return { title: 'Content Plan', sub: 'Editorial series and release calendar', full: true, chrome: false }
  if (pathname === '/morning-desk') return { title: 'Morning Desk', sub: 'Daily frontier-tech briefing automation', full: true, chrome: false }
  if (pathname.startsWith('/tasks/')) return {
    title: 'Task Detail',
    sub: 'Stages, script review, and output',
    back: true,
    full: true,
    chrome: false,
  }
  if (pathname === '/settings') return {
    title: 'Admin Console',
    sub: 'Models, prompts, and skills that drive the pipeline',
    full: true,
    chrome: false,
  }
  if (pathname.startsWith('/account-operations')) return {
    title: 'Account Operations',
    sub: 'Scheduled, auditable autonomous publishing',
    full: true,
    chrome: false,
  }
  return { title: 'FrontierTechSN', sub: '' }
}

export default function App() {
  const location = useLocation()
  const navigate = useNavigate()

  const [theme, setTheme] = useState<Theme>(() => {
    const stored = localStorage.getItem('theme')
    if (stored === 'light' || stored === 'dark') return stored
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  })
  const [rail, setRail] = useState(() => localStorage.getItem('nav-rail') === '1')
  const [drawer, setDrawer] = useState(false)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem('theme', theme)
  }, [theme])

  useEffect(() => {
    localStorage.setItem('nav-rail', rail ? '1' : '0')
  }, [rail])

  useEffect(() => {
    if (!drawer) return
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setDrawer(false)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [drawer])

  const nextTheme = theme === 'dark' ? 'light' : 'dark'
  const meta = pageMeta(location.pathname)

  return (
    <div className={`app-shell${rail ? ' is-rail' : ''}${drawer ? ' is-drawer-open' : ''}`}>
      <div className="ambient-field" aria-hidden="true" />

      <aside className="sidebar" aria-label="Primary">
        <Link
          to="/"
          className="brand-lockup"
          title="FrontierTechSN"
          onClick={() => setDrawer(false)}
        >
          <span className="brand-mark" aria-hidden="true">FT</span>
          <span className="brand-text">
            <strong>FrontierTechSN</strong>
            <small>Autonomous news desk</small>
          </span>
        </Link>

        {/* The drawer is modal on small screens — navigating always dismisses it. */}
        <nav className="nav" onClick={() => setDrawer(false)}>
          <p className="nav-section">Workspace</p>
          {NAV.map(({ to, label, hint, Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              title={label}
              className={({ isActive }) => `nav-item${isActive ? ' is-active' : ''}`}
            >
              <span className="nav-icon"><Icon /></span>
              <span className="nav-text">
                <strong>{label}</strong>
                <small>{hint}</small>
              </span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-foot">
          <button
            type="button"
            className="rail-btn"
            aria-label={`Switch to ${nextTheme} theme`}
            title={`Switch to ${nextTheme} theme`}
            onClick={() => setTheme(nextTheme)}
          >
            <span className="nav-icon">{theme === 'dark' ? <IconSun /> : <IconMoon />}</span>
            <span className="nav-text"><strong>{theme === 'dark' ? 'Light mode' : 'Dark mode'}</strong></span>
          </button>
          <button
            type="button"
            className="rail-btn collapse-btn"
            aria-label={rail ? 'Expand navigation' : 'Collapse navigation'}
            title={rail ? 'Expand navigation' : 'Collapse navigation'}
            onClick={() => setRail((v) => !v)}
          >
            <span className="nav-icon">{rail ? <IconChevronRight /> : <IconChevronLeft />}</span>
            <span className="nav-text"><strong>Collapse</strong></span>
          </button>
        </div>
      </aside>

      <button
        type="button"
        className="scrim"
        aria-label="Close navigation"
        tabIndex={drawer ? 0 : -1}
        onClick={() => setDrawer(false)}
      />

      <div className={`app-main${meta.full ? ' is-full' : ''}${meta.chrome === false ? ' no-topbar' : ''}`}>
        {meta.chrome !== false && (
          <header className="topbar">
            <div className="topbar-inner">
              <button
                type="button"
                className="icon-btn drawer-btn"
                aria-label="Open navigation"
                onClick={() => setDrawer(true)}
              >
                <IconMenu />
              </button>
              {meta.back && (
                <button
                  type="button"
                  className="icon-btn"
                  aria-label="Back to tasks"
                  onClick={() => navigate('/')}
                >
                  <IconChevronLeft />
                </button>
              )}
              <div className="topbar-heading">
                <h1>{meta.title}</h1>
                {meta.sub && <p>{meta.sub}</p>}
              </div>
              {meta.action === 'new' && (
                <Link to="/new" className="topbar-cta">
                  <button className="btn-primary" type="button">New Task</button>
                </Link>
              )}
            </div>
          </header>
        )}

        {meta.chrome === false && (
          <button
            type="button"
            className="icon-btn drawer-btn canvas-drawer-btn"
            aria-label="Open navigation"
            onClick={() => setDrawer(true)}
          >
            <IconMenu />
          </button>
        )}

        <main className="app-canvas">
          <div className="page" key={location.pathname}>
            <Routes>
              <Route path="/" element={<TaskList />} />
              <Route path="/new" element={<TaskForm />} />
              <Route path="/planning" element={<ContentPlanning />} />
              <Route path="/morning-desk" element={<MorningDesk />} />
              <Route path="/tasks/:id" element={<TaskDetail />} />
              <Route path="/account-operations" element={<AccountOperations />} />
              <Route path="/account-operations/runs/:id" element={<AccountRunDetail />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Routes>
          </div>
        </main>
      </div>
    </div>
  )
}

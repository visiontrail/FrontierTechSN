import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent } from 'react'
import { Link, useLocation, useNavigate, useParams } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  createExplainer, ExplainerApiError, fetchExplainer, fetchExplainerDefaults, fetchExplainers,
  isExplainerRunning, runExplainer, saveExplainer, saveExplainerDefaults,
} from '../explainerApi'
import type { ExplainerForm, ExplainerProject, ExplainerScene, ExplainerStatus, ExplainerStyle } from '../explainerApi'
import './ExplainerStudio.css'

const STATUS: Record<ExplainerStatus, string> = {
  draft: 'Draft', writing_script: 'Writing script', script_ready: 'Script ready', generating_audio: 'Generating audio',
  generating_images: 'Generating images', generating_motion: 'Generating motion', rendering: 'Rendering video', completed: 'Completed',
  failed: 'Failed', cancelled: 'Cancelled', interrupted: 'Interrupted',
}

type Draft = { form: ExplainerForm; revision?: number }
const storageKey = (id?: string) => `explainer-studio:draft:${id ?? 'new'}`

function readDraft(id?: string): Draft | null {
  try {
    const draft = JSON.parse(localStorage.getItem(storageKey(id)) ?? 'null') as Draft | null
    if (!draft?.form || typeof draft.form.title !== 'string' || typeof draft.form.research !== 'string'
      || !draft.form.style || !Array.isArray(draft.form.scenes)) return null
    return draft
  } catch { return null }
}

function formOf(project: ExplainerForm): ExplainerForm {
  const { title, research, language, target_minutes, tts_model, voice, captions_enabled, style, scenes } = project
  return { title, research, language, target_minutes, tts_model, voice, captions_enabled, style, scenes }
}

function narrationCount(scenes: ExplainerScene[], language: string) {
  const text = scenes.map((scene) => scene.narration).join(' ')
  return language === 'zh' ? text.replace(/[\s\p{P}\p{S}]/gu, '').length : text.trim().split(/\s+/u).filter(Boolean).length
}

function durationLabel(seconds: number) {
  const rounded = Math.max(0, Math.round(seconds))
  return `${Math.floor(rounded / 60)}:${String(rounded % 60).padStart(2, '0')}`
}

function StylePreview({ style }: { style: ExplainerStyle }) {
  return <div className="explainer-style-preview" style={{ backgroundColor: style.background_color, color: style.ink_color }}>
    <span className="explainer-preview-caption">STYLE STUDY</span>
    <svg viewBox="0 0 320 170" role="img" aria-label="Style preview with a line character and simple concept shapes">
      <g fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="104" cy="53" r="20" /><path d="M102 74l-2 48m0-26l-25 15m26-18l32-15m-33 44l-19 31m19-31l23 30" />
        <path d="M97 51h1m13-1h1m-14 10q6 4 11-1" />
        <rect x="186" y="90" width="42" height="42" rx="3" /><path d="M207 78V65m-8 7l8-8 8 8M163 111h-17m9-7l-8 7 8 7" />
        <circle cx="206" cy="37" r="21" stroke={style.accent_color} /><path d="M197 37l7 7 12-14" stroke={style.accent_color} />
        <path d="M56 156h192" strokeWidth="1" opacity="0.4" />
      </g>
    </svg>
    <span className="explainer-preview-footer">One scene. One clear idea.</span>
  </div>
}

export default function ExplainerStudio() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const location = useLocation()
  const queryClient = useQueryClient()
  const fileRef = useRef<HTMLInputElement>(null)
  const [override, setOverride] = useState<Draft | null>(() => readDraft(id))
  const [pending, setPending] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(() => typeof location.state?.explainerError === 'string' ? location.state.explainerError : null)
  const [conflict, setConflict] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const [localSaveError, setLocalSaveError] = useState(false)
  const { data: defaults, error: defaultsError } = useQuery({ queryKey: ['explainer-defaults'], queryFn: fetchExplainerDefaults })
  const { data: library, error: libraryError } = useQuery({ queryKey: ['explainers'], queryFn: fetchExplainers, refetchInterval: 7000 })
  const { data: project, error: projectError, isLoading: projectLoading } = useQuery({
    queryKey: ['explainer', id], queryFn: () => fetchExplainer(id!), enabled: !!id,
    refetchInterval: (query) => isExplainerRunning(query.state.data?.status) ? 2000 : 7000,
  })
  const form = override?.form ?? (project ? formOf(project) : !id && defaults ? {
    title: '', research: '', scenes: [], language: defaults.language, target_minutes: defaults.target_minutes,
    tts_model: defaults.tts_model, voice: defaults.voice, captions_enabled: defaults.captions_enabled, style: defaults.style,
  } : null)
  const running = isExplainerRunning(project?.status)
  const locked = running || !!pending || (!!id && !project)
  const dirty = override !== null

  useEffect(() => {
    if (!dirty || !localSaveError) return
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = '' }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [dirty, localSaveError])

  function edit(next: ExplainerForm) {
    const draft: Draft = { form: next, revision: override?.revision ?? project?.revision }
    setOverride(draft)
    setNotice(null)
    try { localStorage.setItem(storageKey(id), JSON.stringify(draft)); setLocalSaveError(false) }
    catch { setLocalSaveError(true) }
  }

  function patch<K extends keyof ExplainerForm>(key: K, value: ExplainerForm[K]) {
    if (form) edit({ ...form, [key]: value })
  }

  function patchStyle<K extends keyof ExplainerStyle>(key: K, value: ExplainerStyle[K]) {
    if (form) patch('style', { ...form.style, [key]: value })
  }

  function patchScene(index: number, values: Partial<ExplainerScene>) {
    if (form) patch('scenes', form.scenes.map((scene, i) => i === index ? { ...scene, ...values } : scene))
  }

  function rememberSaved(saved: ExplainerProject) {
    queryClient.setQueryData(['explainer', saved.id], saved)
    void queryClient.invalidateQueries({ queryKey: ['explainers'] })
    setOverride(null)
    try { localStorage.removeItem(storageKey(id)); localStorage.removeItem(storageKey(saved.id)) } catch { /* server copy is saved */ }
    setConflict(false)
  }

  async function saveCurrent(): Promise<ExplainerProject> {
    if (!form) throw new Error('The studio has not loaded yet.')
    if (!form.title.trim()) throw new Error('Enter a video title first.')
    if (form.scenes.some((scene) => !scene.narration.trim() || scene.visual_prompt.trim().length < 3)) throw new Error('Complete the narration and visual description for every scene, or remove empty scenes before saving.')
    if (!Number.isInteger(form.target_minutes) || form.target_minutes < 1 || form.target_minutes > 60) throw new Error('Target duration must be a whole number from 1 to 60 minutes.')
    if (id && !project) throw new Error('Wait for the project to load before saving.')
    if (project && !dirty) return project
    const saved = id
      ? await saveExplainer(id, form, override?.revision ?? project!.revision)
      : await createExplainer(form)
    rememberSaved(saved)
    return saved
  }

  async function act(action: 'save' | 'script' | 'render' | 'cancel' | 'defaults') {
    setError(null)
    setNotice(null)
    setPending(action)
    let savedId = id
    let actionError: string | null = null
    try {
      if (action === 'defaults') {
        if (!form) return
        const { language, target_minutes, tts_model, voice, captions_enabled, style } = form
        await saveExplainerDefaults({ language, target_minutes, tts_model, voice, captions_enabled, style })
        await queryClient.invalidateQueries({ queryKey: ['explainer-defaults'] })
        setNotice('Style, language and voice settings saved as defaults for new projects.')
      } else if (action === 'cancel' && id) {
        const cancelled = await runExplainer(id, 'cancel')
        queryClient.setQueryData(['explainer', id], cancelled)
        void queryClient.invalidateQueries({ queryKey: ['explainers'] })
        setNotice('Cancellation requested. Finishing the current step.')
      } else {
        const saved = await saveCurrent()
        savedId = saved.id
        if (action === 'script' || action === 'render') {
          rememberSaved(await runExplainer(saved.id, action))
        } else setNotice('Project saved.')
      }
    } catch (cause) {
      actionError = cause instanceof Error ? cause.message : String(cause)
      setError(actionError)
      setConflict(cause instanceof ExplainerApiError && cause.status === 409)
    } finally {
      setPending(null)
      if (!id && savedId) navigate(`/explainer-studio/${savedId}`, { replace: true, state: { explainerError: actionError } })
    }
  }

  async function importResearch(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file || !form) return
    if (file.size > 2_000_000) { setError('Choose a TXT or Markdown file smaller than 2 MB.'); return }
    if (!/\.(txt|md|markdown)$/i.test(file.name)) { setError('Import a TXT or Markdown file.'); return }
    try {
      const text = await file.text()
      patch('research', `${form.research.trim()}${form.research.trim() ? '\n\n' : ''}${text}`)
      setNotice(`Content from ${file.name} appended.`)
    } catch { setError('Could not read the file. Try again or paste the text directly.') }
  }

  function downloadScript() {
    if (!form?.scenes.length) return
    let chapter = ''
    const paragraphs = [form.title, ...form.scenes.flatMap((scene) => {
      const heading = scene.chapter && scene.chapter !== chapter ? [scene.chapter] : []
      chapter = scene.chapter
      return [...heading, scene.narration]
    })]
    const url = URL.createObjectURL(new Blob([paragraphs.join('\n\n') + '\n'], { type: 'text/plain;charset=utf-8' }))
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `${form.title.replace(/[<>:"/\\|?*\p{Cc}]/gu, '-').slice(0, 80) || 'explainer'}-narration.txt`
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    window.setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  const count = form ? narrationCount(form.scenes, form.language) : 0
  const estimate = form ? count / (form.language === 'zh' ? 240 : 145) : 0
  const selectedModel = defaults?.tts_models.find((model) => model.id === form?.tts_model)
  const knownVoices = selectedModel?.voices ?? []
  const voices = form?.voice && !knownVoices.includes(form.voice) ? [form.voice, ...knownVoices] : knownVoices
  const invalidScenes = form?.scenes.some((scene) => !scene.narration.trim() || scene.visual_prompt.trim().length < 3) ?? false
  const progress = project?.progress
  const percentage = progress?.total ? Math.min(100, Math.max(0, (progress.completed / progress.total) * 100)) : 0
  const imageInterval = defaults?.provider_limits?.chatgpt_interval_seconds
  const imageWaitHours = form && imageInterval ? Math.max(0, form.scenes.length - 1) * imageInterval / 3600 : 0
  const loadError = projectError ?? defaultsError

  return <div className="explainer-studio">
    <header className="explainer-header">
      <div><p className="explainer-eyebrow">EXPLAINER STUDIO / 01</p><h1>Explainer Studio<span>Turn research into a story.</span></h1>
        <p className="explainer-intro">Turn your research into a long-form video with line illustrations, narration and occasional animation, all in your own style.</p></div>
      <Link to="/explainer-studio" className="explainer-button explainer-new">+ New project</Link>
    </header>

    <div className="explainer-workspace">
      <aside className="explainer-library" aria-label="Explainer projects">
        <div className="explainer-library-heading"><h2>My explainers</h2><span>{library?.projects.length ?? '—'}</span></div>
        {libraryError && <p className="explainer-small-error" role="alert">Could not load projects.</p>}
        {!library && !libraryError && <p className="explainer-muted">Loading projects…</p>}
        {library?.projects.length === 0 && <p className="explainer-library-empty">Your first explainer starts<br /> with a piece of research.</p>}
        <nav aria-label="Select an explainer project">
          {library?.projects.map((item) => <Link key={item.id} to={`/explainer-studio/${item.id}`} className={`explainer-project-link${id === item.id ? ' is-selected' : ''}`} aria-current={id === item.id ? 'page' : undefined}>
            <span className={`explainer-status-dot${isExplainerRunning(item.status) ? ' is-running' : ''}`} />
            <span><strong>{item.title}</strong><small>{STATUS[item.status] ?? item.status} · {new Date(item.updated_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}</small></span>
          </Link>)}
        </nav>
        <div className="explainer-workflow-note"><span>WORKFLOW</span><p>Research → Script & storyboard<br />→ Images & audio → Video</p></div>
      </aside>

      <div className="explainer-editor">
        {(error || loadError) && <div className="explainer-alert is-error" role="alert"><p>{error ?? (loadError instanceof Error ? loadError.message : String(loadError))}</p>
          {conflict && <><p>The server version has changed. Your local edits are preserved. Copy anything you want to keep before reloading the server version.</p><button type="button" onClick={() => { setOverride(null); try { localStorage.removeItem(storageKey(id)) } catch { /* no-op */ } setError(null); setConflict(false); void queryClient.invalidateQueries({ queryKey: ['explainer', id] }) }}>Discard local edits and reload</button></>}
        </div>}
        {notice && <div className="explainer-alert" role="status">{notice}</div>}
        {localSaveError && <div className="explainer-alert is-error" role="alert">Browser draft storage is full. Save the project to avoid losing your edits.</div>}
        {id && projectLoading && <div className="explainer-loading">Loading explainer project…</div>}
        {!form && !projectLoading && !loadError && <div className="explainer-loading">Preparing the studio…</div>}

        {form && <>
          <div className="explainer-project-bar"><span className="explainer-badge">{project ? STATUS[project.status] : 'New project'}</span>
            <span className="explainer-save-state">{dirty ? 'Draft kept locally · Not yet saved to project' : project ? 'Saved to project' : 'Start with your research'}</span>
            <button type="button" onClick={() => void act('save')} disabled={locked || (!!project && !dirty) || !form.title.trim()}>{pending === 'save' ? 'Saving…' : 'Save project'}</button>
          </div>

          {project && (running || project.error || ['cancelled', 'interrupted'].includes(project.status)) && <section className="explainer-progress-panel" aria-label="Production progress">
            <div className="explainer-section-heading"><div><p className="explainer-eyebrow">PRODUCTION</p><h2>{STATUS[project.status]}</h2></div>
              {running && <button type="button" disabled={!!pending} onClick={() => void act('cancel')}>{pending === 'cancel' ? 'Cancelling…' : 'Cancel run'}</button>}</div>
            <p role="status" aria-live="polite">{progress?.message || 'Preparing…'}</p>
            {running && <progress max={100} value={percentage} aria-label="Video generation progress" />}
            {progress && progress.total > 0 && <small>{progress.completed} / {progress.total}</small>}
            {project.error && <p className="explainer-small-error">{project.error}</p>}
            {!running && <p className="explainer-muted">Edit the research or settings, then regenerate the script or video.</p>}
          </section>}

          {project?.artifacts.video_url && <section className="explainer-result">
            <div className="explainer-section-heading"><div><p className="explainer-eyebrow">THE FINISHED STORY</p><h2>Your explainer video</h2></div>
              {project.artifacts.duration_seconds != null && <span>{durationLabel(project.artifacts.duration_seconds)}</span>}</div>
            <video key={project.artifacts.video_url} src={project.artifacts.video_url} controls preload="metadata" aria-label="Explainer video preview" />
            <div className="explainer-output-links"><a className="explainer-button is-primary" href={project.artifacts.video_url} download>Download video ↓</a>
              {project.artifacts.audio_url && <a className="explainer-button" href={project.artifacts.audio_url} download>Download narration</a>}
              {project.artifacts.timeline_url && <a className="explainer-button" href={project.artifacts.timeline_url} target="_blank" rel="noreferrer">View timeline ↗</a>}</div>
          </section>}

          <fieldset disabled={locked} className="explainer-fieldset">
            <section className="explainer-section">
              <div className="explainer-section-heading"><div><p className="explainer-eyebrow">01 / THE SOURCE</p><h2>Research & direction</h2></div><span className="explainer-section-aside">Give your story a source</span></div>
              <label htmlFor="explainer-title">Video title / working title</label><input id="explainer-title" value={form.title} onChange={(e) => patch('title', e.target.value)} placeholder="Why do we mistake being busy for making progress?" maxLength={200} />
              <div className="explainer-research-heading"><label htmlFor="explainer-research">Research, arguments & sources</label><button type="button" onClick={() => fileRef.current?.click()}>Import TXT / Markdown</button></div>
              <input ref={fileRef} type="file" accept=".txt,.md,.markdown,text/plain,text/markdown" hidden onChange={(event) => void importResearch(event)} />
              <textarea id="explainer-research" className="explainer-research" value={form.research} maxLength={200000} onChange={(e) => patch('research', e.target.value)} placeholder={'Paste Deep Research, reading notes or your own research.\n\nInclude the central question, evidence and links, your perspective, examples to keep and wording to avoid.'} />
              <p className="explainer-field-help">The script uses this material as its source (at least 40 characters). You can edit the narration and visual description for each scene after generation.</p>
              <div className="explainer-settings-row">
                <div><label htmlFor="explainer-language">Narration language</label><select id="explainer-language" value={form.language} onChange={(e) => patch('language', e.target.value as 'en' | 'zh')}><option value="en">English</option><option value="zh">Chinese</option></select></div>
                <div><label htmlFor="explainer-minutes">Target duration (minutes)</label><input id="explainer-minutes" type="number" min={1} max={60} step={1} value={form.target_minutes} onChange={(e) => patch('target_minutes', Number(e.target.value))} /></div>
                <div><label htmlFor="explainer-tts">Voice model</label><select id="explainer-tts" value={form.tts_model} onChange={(e) => { const model = defaults?.tts_models.find((item) => item.id === e.target.value); edit({ ...form, tts_model: e.target.value, voice: model?.voices[0] ?? form.voice }) }}>
                  {form.tts_model && !defaults?.tts_models.some((model) => model.id === form.tts_model) && <option value={form.tts_model}>{form.tts_model}</option>}
                  {defaults?.tts_models.map((model) => <option key={model.id} value={model.id}>{model.label}{model.language === 'en' ? ' · English only' : ''}</option>)}</select></div>
                <div><label htmlFor="explainer-voice">Voice</label><select id="explainer-voice" value={form.voice} onChange={(e) => patch('voice', e.target.value)}>{voices.map((voice) => <option key={voice} value={voice}>{voice}</option>)}</select></div>
              </div>
              {form.language === 'zh' && selectedModel?.language === 'en' && <p className="explainer-small-error">This voice model only supports English. Choose a model that supports Chinese before generating a Chinese video.</p>}
              <label className="explainer-check"><input type="checkbox" checked={form.captions_enabled} onChange={(e) => patch('captions_enabled', e.target.checked)} /><span>Show narration captions in the finished video</span></label>
            </section>

            <section className="explainer-section">
              <div className="explainer-section-heading"><div><p className="explainer-eyebrow">02 / VISUAL LANGUAGE</p><h2>Make it your own</h2></div><button type="button" onClick={() => void act('defaults')}>{pending === 'defaults' ? 'Saving…' : 'Save as defaults'}</button></div>
              <div className="explainer-style-grid">
                <div><div className="explainer-presets" aria-label="Style presets">{defaults?.presets.map((preset) => <button key={preset.id} type="button" onClick={() => patch('style', { ...preset.style })}>{preset.label}</button>)}</div>
                  <label htmlFor="explainer-character">Character description</label><textarea id="explainer-character" rows={2} value={form.style.character_description} onChange={(e) => patchStyle('character_description', e.target.value)} />
                  <label htmlFor="explainer-visual">Art direction</label><textarea id="explainer-visual" rows={3} value={form.style.visual_direction} onChange={(e) => patchStyle('visual_direction', e.target.value)} />
                  <label htmlFor="explainer-tone">Writing tone</label><textarea id="explainer-tone" rows={2} value={form.style.writing_tone} onChange={(e) => patchStyle('writing_tone', e.target.value)} />
                </div>
                <div><StylePreview style={form.style} /><div className="explainer-colors">
                  {([['background_color', 'Background'], ['ink_color', 'Ink'], ['accent_color', 'Accent']] as const).map(([key, label]) => <label key={key}><input type="color" value={form.style[key]} onChange={(e) => patchStyle(key, e.target.value)} /><span>{label}</span></label>)}
                </div><p className="explainer-field-help">This preview shows your palette. Generated images use the character description and each scene’s visual prompt.</p></div>
              </div>
              <details className="explainer-advanced"><summary>Pacing & motion</summary><div className="explainer-settings-row is-three">
                <div><label htmlFor="explainer-motion">Motion style</label><select id="explainer-motion" value={form.style.motion_mode} onChange={(e) => patchStyle('motion_mode', e.target.value as ExplainerStyle['motion_mode'])}><option value="none">Still illustrations</option><option value="subtle">Gentle pan & zoom</option><option value="gemini">Gemini animation</option></select></div>
                <div><label htmlFor="explainer-spacing">Animate every N scenes (0 = manual)</label><input id="explainer-spacing" type="number" min={0} max={100} value={form.style.motion_every_n_scenes} onChange={(e) => patchStyle('motion_every_n_scenes', Number(e.target.value))} /></div>
                <div><label htmlFor="explainer-seconds">Target seconds per scene</label><input id="explainer-seconds" type="number" min={2} max={45} value={form.style.seconds_per_scene} onChange={(e) => patchStyle('seconds_per_scene', Number(e.target.value))} /></div>
              </div><p className="explainer-field-help">Target seconds guide script segmentation. Each scene lasts until its narration ends. You can select individual scenes for Gemini animation in the storyboard.</p></details>
              <div className="explainer-generate-row"><p>Research → Full narration + Storyboard</p><button type="button" className="explainer-button is-primary" disabled={!form.title.trim() || form.research.trim().length < 40 || invalidScenes} onClick={() => void act('script')}>{form.scenes.length ? 'Regenerate script ↗' : 'Generate script ↗'}</button></div>
            </section>

            <section className="explainer-section explainer-script">
              <div className="explainer-section-heading"><div><p className="explainer-eyebrow">03 / STORYBOARD</p><h2>Narration & visuals</h2></div><button type="button" disabled={form.scenes.length === 0} onClick={downloadScript}>Export narration ↓</button></div>
              <div className="explainer-script-summary"><span>{form.scenes.length} scenes</span><span>{count.toLocaleString('en-US')} {form.language === 'zh' ? 'characters' : 'words'}</span><span>Estimated narration: {estimate.toFixed(1)} min</span><span>Target: {form.target_minutes} min</span></div>
              <p className="explainer-render-estimate">{form.scenes.length > 0 && imageInterval ? `${form.scenes.length} images. With the current ${Math.round(imageInterval / 60)}-minute request interval, generating all images from scratch adds at least about ${imageWaitHours.toFixed(1)} hours of waiting, plus generation and rendering time. Reusing existing assets can shorten this.` : 'The current image request interval applies. Increase scene duration to reduce the number of images.'}</p>
              {form.scenes.length === 0 && <div className="explainer-empty-story"><span>“</span><h3>Start with a question worth explaining.</h3><p>Add your research and generate a script, or add your first scene manually.</p></div>}
              <div className="explainer-scenes">
                {form.scenes.map((scene, index) => <article className="explainer-scene" key={scene.id}>
                  <div className="explainer-scene-top"><span className="explainer-scene-number">{String(index + 1).padStart(2, '0')}</span><label className="explainer-sr-only" htmlFor={`chapter-${scene.id}`}>Scene {index + 1} chapter</label><input id={`chapter-${scene.id}`} value={scene.chapter} placeholder="Chapter title" onChange={(e) => patchScene(index, { chapter: e.target.value })} />
                    <button type="button" aria-label={`Remove scene ${index + 1}`} className="explainer-remove" onClick={() => patch('scenes', form.scenes.filter((_, i) => i !== index))}>Remove</button></div>
                  <div className="explainer-scene-columns"><div><label htmlFor={`narration-${scene.id}`}>Narration <span>VOICE</span></label><textarea id={`narration-${scene.id}`} rows={5} value={scene.narration} onChange={(e) => patchScene(index, { narration: e.target.value })} placeholder="What the narrator says in this scene…" /></div>
                    <div><label htmlFor={`visual-${scene.id}`}>Visual description <span>FRAME</span></label><textarea id={`visual-${scene.id}`} rows={5} value={scene.visual_prompt} onChange={(e) => patchScene(index, { visual_prompt: e.target.value })} placeholder="What is the character doing? What visual metaphor explains the idea?" /></div></div>
                  <div className="explainer-scene-bottom"><div><label htmlFor={`text-${scene.id}`}>On-screen text (optional)</label><input id={`text-${scene.id}`} value={scene.on_screen_text} onChange={(e) => patchScene(index, { on_screen_text: e.target.value })} placeholder="Keep it brief" /></div>
                    {form.style.motion_mode === 'gemini' && form.style.motion_every_n_scenes > 0 && (index + 1) % form.style.motion_every_n_scenes === 0
                      ? <label className="explainer-check"><input type="checkbox" checked disabled /><span>Animation scheduled automatically</span></label>
                      : <label className="explainer-check"><input type="checkbox" checked={scene.motion} onChange={(e) => patchScene(index, { motion: e.target.checked })} /><span>Animate scene{form.style.motion_mode === 'none' ? ' (still mode selected)' : ''}</span></label>}</div>
                </article>)}
              </div>
              <button type="button" className="explainer-add-scene" onClick={() => { const used = new Set(form.scenes.map((scene) => scene.id)); let number = form.scenes.length + 1; while (used.has(`scene-${String(number).padStart(3, '0')}`)) number++; patch('scenes', [...form.scenes, { id: `scene-${String(number).padStart(3, '0')}`, chapter: '', narration: '', visual_prompt: '', on_screen_text: '', motion: false }]) }}>+ Add scene</button>
              {invalidScenes && <p className="explainer-small-error">Every scene needs narration and a visual description before you can generate a video.</p>}
              <div className="explainer-render-row"><div><strong>Bring your story to life.</strong><p>Edits are saved automatically, then each scene gets narration and an illustration. The video follows the actual audio duration.</p></div><button type="button" className="explainer-button is-primary" disabled={!form.title.trim() || form.scenes.length === 0 || invalidScenes || (form.language === 'zh' && selectedModel?.language === 'en')} onClick={() => void act('render')}>{project?.status === 'failed' || project?.status === 'interrupted' ? 'Regenerate video →' : 'Generate video →'}</button></div>
            </section>
          </fieldset>
          {project && project.logs.length > 0 && <details className="explainer-logs"><summary>Run log · {project.logs.length}</summary><pre>{project.logs.join('\n')}</pre></details>}
          {pending && <div className="explainer-pending" role="status" aria-live="polite">{pending === 'script' ? 'Saving and starting script generation…' : pending === 'render' ? 'Saving and starting video generation…' : 'Processing…'}</div>}
        </>}
      </div>
    </div>
  </div>
}

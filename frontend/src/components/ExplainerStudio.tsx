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
  draft: '草稿', writing_script: '编写脚本', script_ready: '脚本已就绪', generating_audio: '生成配音',
  generating_images: '绘制场景', generating_motion: '生成动态镜头', rendering: '合成视频', completed: '已完成',
  failed: '生成失败', cancelled: '已取消', interrupted: '运行中断',
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
    <span className="explainer-preview-caption">STYLE STUDY / 风格示意</span>
    <svg viewBox="0 0 320 170" role="img" aria-label="简笔人物与概念图形的风格示意">
      <g fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="104" cy="53" r="20" /><path d="M102 74l-2 48m0-26l-25 15m26-18l32-15m-33 44l-19 31m19-31l23 30" />
        <path d="M97 51h1m13-1h1m-14 10q6 4 11-1" />
        <rect x="186" y="90" width="42" height="42" rx="3" /><path d="M207 78V65m-8 7l8-8 8 8M163 111h-17m9-7l-8 7 8 7" />
        <circle cx="206" cy="37" r="21" stroke={style.accent_color} /><path d="M197 37l7 7 12-14" stroke={style.accent_color} />
        <path d="M56 156h192" strokeWidth="1" opacity="0.4" />
      </g>
    </svg>
    <span className="explainer-preview-footer">一个画面，讲清一个意思。</span>
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
    if (!form) throw new Error('工作台尚未载入')
    if (!form.title.trim()) throw new Error('请先填写视频标题。')
    if (form.scenes.some((scene) => !scene.narration.trim() || scene.visual_prompt.trim().length < 3)) throw new Error('请补全每个镜头的旁白和画面描述，或删除空镜头后再保存。')
    if (!Number.isInteger(form.target_minutes) || form.target_minutes < 1 || form.target_minutes > 60) throw new Error('目标时长需要是 1 到 60 之间的整数。')
    if (id && !project) throw new Error('请等待项目载入后再保存。')
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
        setNotice('风格、语言和配音设置已保存为新项目默认值。')
      } else if (action === 'cancel' && id) {
        const cancelled = await runExplainer(id, 'cancel')
        queryClient.setQueryData(['explainer', id], cancelled)
        void queryClient.invalidateQueries({ queryKey: ['explainers'] })
        setNotice('已发送取消请求，正在结束当前步骤。')
      } else {
        const saved = await saveCurrent()
        savedId = saved.id
        if (action === 'script' || action === 'render') {
          rememberSaved(await runExplainer(saved.id, action))
        } else setNotice('项目已保存。')
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
    if (file.size > 2_000_000) { setError('请选择小于 2 MB 的 TXT 或 Markdown 文件。'); return }
    if (!/\.(txt|md|markdown)$/i.test(file.name)) { setError('支持导入 TXT 或 Markdown 文件。'); return }
    try {
      const text = await file.text()
      patch('research', `${form.research.trim()}${form.research.trim() ? '\n\n' : ''}${text}`)
      setNotice(`已追加 ${file.name} 的内容。`)
    } catch { setError('文件读取失败，请重试或直接粘贴文字。') }
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
    anchor.download = `${form.title.replace(/[<>:"/\\|?*\p{Cc}]/gu, '-').slice(0, 80) || 'explainer'}-旁白脚本.txt`
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
      <div><p className="explainer-eyebrow">EXPLAINER STUDIO / 01</p><h1>长篇图解<span>把研究，讲成故事。</span></h1>
        <p className="explainer-intro">从你提供的调研出发，用简笔画、旁白和少量动态镜头，完成一支有自己风格的长视频。</p></div>
      <Link to="/explainer-studio" className="explainer-button explainer-new">＋ 新建项目</Link>
    </header>

    <div className="explainer-workspace">
      <aside className="explainer-library" aria-label="图解项目">
        <div className="explainer-library-heading"><h2>我的图解</h2><span>{library?.projects.length ?? '—'}</span></div>
        {libraryError && <p className="explainer-small-error" role="alert">项目列表读取失败。</p>}
        {!library && !libraryError && <p className="explainer-muted">载入项目…</p>}
        {library?.projects.length === 0 && <p className="explainer-library-empty">第一支图解，<br />从一份研究开始。</p>}
        <nav aria-label="选择图解项目">
          {library?.projects.map((item) => <Link key={item.id} to={`/explainer-studio/${item.id}`} className={`explainer-project-link${id === item.id ? ' is-selected' : ''}`} aria-current={id === item.id ? 'page' : undefined}>
            <span className={`explainer-status-dot${isExplainerRunning(item.status) ? ' is-running' : ''}`} />
            <span><strong>{item.title}</strong><small>{STATUS[item.status] ?? item.status} · {new Date(item.updated_at).toLocaleDateString('zh-CN', { month: 'numeric', day: 'numeric' })}</small></span>
          </Link>)}
        </nav>
        <div className="explainer-workflow-note"><span>工作路径</span><p>研究材料 → 旁白与分镜<br />→ 画面与配音 → 视频</p></div>
      </aside>

      <div className="explainer-editor">
        {(error || loadError) && <div className="explainer-alert is-error" role="alert"><p>{error ?? (loadError instanceof Error ? loadError.message : String(loadError))}</p>
          {conflict && <><p>服务器状态已变化，本地编辑已保留。可以复制内容后重新载入服务器版本。</p><button type="button" onClick={() => { setOverride(null); try { localStorage.removeItem(storageKey(id)) } catch { /* no-op */ } setError(null); setConflict(false); void queryClient.invalidateQueries({ queryKey: ['explainer', id] }) }}>放弃本地编辑并重新载入</button></>}
        </div>}
        {notice && <div className="explainer-alert" role="status">{notice}</div>}
        {localSaveError && <div className="explainer-alert is-error" role="alert">浏览器草稿空间不足，请保存项目，避免丢失修改。</div>}
        {id && projectLoading && <div className="explainer-loading">正在载入图解项目…</div>}
        {!form && !projectLoading && !loadError && <div className="explainer-loading">正在准备工作台…</div>}

        {form && <>
          <div className="explainer-project-bar"><span className="explainer-badge">{project ? STATUS[project.status] : '新项目'}</span>
            <span className="explainer-save-state">{dirty ? '编辑已保留在本机 · 尚未保存到项目' : project ? '已保存到项目' : '先放入你的研究材料'}</span>
            <button type="button" onClick={() => void act('save')} disabled={locked || (!!project && !dirty) || !form.title.trim()}>{pending === 'save' ? '保存中…' : '保存项目'}</button>
          </div>

          {project && (running || project.error || ['cancelled', 'interrupted'].includes(project.status)) && <section className="explainer-progress-panel" aria-label="生成进度">
            <div className="explainer-section-heading"><div><p className="explainer-eyebrow">PRODUCTION</p><h2>{STATUS[project.status]}</h2></div>
              {running && <button type="button" disabled={!!pending} onClick={() => void act('cancel')}>{pending === 'cancel' ? '取消中…' : '取消运行'}</button>}</div>
            <p role="status" aria-live="polite">{progress?.message || '正在准备…'}</p>
            {running && <progress max={100} value={percentage} aria-label="视频生成进度" />}
            {progress && progress.total > 0 && <small>{progress.completed} / {progress.total}</small>}
            {project.error && <p className="explainer-small-error">{project.error}</p>}
            {!running && <p className="explainer-muted">可以修改材料或设置，然后重新生成脚本或视频。</p>}
          </section>}

          {project?.artifacts.video_url && <section className="explainer-result">
            <div className="explainer-section-heading"><div><p className="explainer-eyebrow">THE FINISHED STORY</p><h2>你的图解视频</h2></div>
              {project.artifacts.duration_seconds != null && <span>{durationLabel(project.artifacts.duration_seconds)}</span>}</div>
            <video key={project.artifacts.video_url} src={project.artifacts.video_url} controls preload="metadata" aria-label="图解视频预览" />
            <div className="explainer-output-links"><a className="explainer-button is-primary" href={project.artifacts.video_url} download>下载视频 ↓</a>
              {project.artifacts.audio_url && <a className="explainer-button" href={project.artifacts.audio_url} download>下载旁白</a>}
              {project.artifacts.timeline_url && <a className="explainer-button" href={project.artifacts.timeline_url} target="_blank" rel="noreferrer">查看时间轴 ↗</a>}</div>
          </section>}

          <fieldset disabled={locked} className="explainer-fieldset">
            <section className="explainer-section">
              <div className="explainer-section-heading"><div><p className="explainer-eyebrow">01 / THE SOURCE</p><h2>研究与方向</h2></div><span className="explainer-section-aside">把材料交给故事</span></div>
              <label htmlFor="explainer-title">视频标题 / 工作标题</label><input id="explainer-title" value={form.title} onChange={(e) => patch('title', e.target.value)} placeholder="例如：为什么我们总把忙碌当成进步？" maxLength={200} />
              <div className="explainer-research-heading"><label htmlFor="explainer-research">调研、论点与参考来源</label><button type="button" onClick={() => fileRef.current?.click()}>导入 TXT / Markdown</button></div>
              <input ref={fileRef} type="file" accept=".txt,.md,.markdown,text/plain,text/markdown" hidden onChange={(event) => void importResearch(event)} />
              <textarea id="explainer-research" className="explainer-research" value={form.research} maxLength={200000} onChange={(e) => patch('research', e.target.value)} placeholder={'粘贴 Deep Research、读书笔记或你自己的研究。\n\n可以包含：核心问题、证据与链接、你的判断、想保留的例子，以及不希望使用的表述。'} />
              <p className="explainer-field-help">旁白脚本以这些材料为基础（至少 40 个字符）。生成后可直接编辑每个镜头的旁白和画面描述。</p>
              <div className="explainer-settings-row">
                <div><label htmlFor="explainer-language">旁白语言</label><select id="explainer-language" value={form.language} onChange={(e) => patch('language', e.target.value as 'en' | 'zh')}><option value="en">English</option><option value="zh">中文</option></select></div>
                <div><label htmlFor="explainer-minutes">目标时长（分钟）</label><input id="explainer-minutes" type="number" min={1} max={60} step={1} value={form.target_minutes} onChange={(e) => patch('target_minutes', Number(e.target.value))} /></div>
                <div><label htmlFor="explainer-tts">配音模型</label><select id="explainer-tts" value={form.tts_model} onChange={(e) => { const model = defaults?.tts_models.find((item) => item.id === e.target.value); edit({ ...form, tts_model: e.target.value, voice: model?.voices[0] ?? form.voice }) }}>
                  {form.tts_model && !defaults?.tts_models.some((model) => model.id === form.tts_model) && <option value={form.tts_model}>{form.tts_model}</option>}
                  {defaults?.tts_models.map((model) => <option key={model.id} value={model.id}>{model.label}{model.language === 'en' ? ' · 仅英文' : ''}</option>)}</select></div>
                <div><label htmlFor="explainer-voice">声音</label><select id="explainer-voice" value={form.voice} onChange={(e) => patch('voice', e.target.value)}>{voices.map((voice) => <option key={voice} value={voice}>{voice}</option>)}</select></div>
              </div>
              {form.language === 'zh' && selectedModel?.language === 'en' && <p className="explainer-small-error">当前配音模型只支持英文，生成中文视频前请选择支持中文的模型。</p>}
              <label className="explainer-check"><input type="checkbox" checked={form.captions_enabled} onChange={(e) => patch('captions_enabled', e.target.checked)} /><span>在成片中显示旁白字幕</span></label>
            </section>

            <section className="explainer-section">
              <div className="explainer-section-heading"><div><p className="explainer-eyebrow">02 / VISUAL LANGUAGE</p><h2>让它像你</h2></div><button type="button" onClick={() => void act('defaults')}>{pending === 'defaults' ? '保存中…' : '设为新项目默认'}</button></div>
              <div className="explainer-style-grid">
                <div><div className="explainer-presets" aria-label="风格预设">{defaults?.presets.map((preset) => <button key={preset.id} type="button" onClick={() => patch('style', { ...preset.style })}>{preset.label}</button>)}</div>
                  <label htmlFor="explainer-character">人物设定</label><textarea id="explainer-character" rows={2} value={form.style.character_description} onChange={(e) => patchStyle('character_description', e.target.value)} />
                  <label htmlFor="explainer-visual">画面方向</label><textarea id="explainer-visual" rows={3} value={form.style.visual_direction} onChange={(e) => patchStyle('visual_direction', e.target.value)} />
                  <label htmlFor="explainer-tone">叙述语气</label><textarea id="explainer-tone" rows={2} value={form.style.writing_tone} onChange={(e) => patchStyle('writing_tone', e.target.value)} />
                </div>
                <div><StylePreview style={form.style} /><div className="explainer-colors">
                  {([['background_color', '背景'], ['ink_color', '线条'], ['accent_color', '强调色']] as const).map(([key, label]) => <label key={key}><input type="color" value={form.style[key]} onChange={(e) => patchStyle(key, e.target.value)} /><span>{label}</span></label>)}
                </div><p className="explainer-field-help">示意图用于预览配色。实际画面会结合人物设定和逐镜头描述生成。</p></div>
              </div>
              <details className="explainer-advanced"><summary>镜头节奏与动态设置</summary><div className="explainer-settings-row is-three">
                <div><label htmlFor="explainer-motion">动态方式</label><select id="explainer-motion" value={form.style.motion_mode} onChange={(e) => patchStyle('motion_mode', e.target.value as ExplainerStyle['motion_mode'])}><option value="none">静态图解</option><option value="subtle">轻微推拉镜头</option><option value="gemini">Gemini 短动画</option></select></div>
                <div><label htmlFor="explainer-spacing">自动动态间隔（0 = 仅手动）</label><input id="explainer-spacing" type="number" min={0} max={100} value={form.style.motion_every_n_scenes} onChange={(e) => patchStyle('motion_every_n_scenes', Number(e.target.value))} /></div>
                <div><label htmlFor="explainer-seconds">单镜头目标秒数</label><input id="explainer-seconds" type="number" min={2} max={45} value={form.style.seconds_per_scene} onChange={(e) => patchStyle('seconds_per_scene', Number(e.target.value))} /></div>
              </div><p className="explainer-field-help">目标秒数用于拆分脚本；每个画面最终持续到对应旁白结束。Gemini 动态镜头可在分镜中逐一选择。</p></details>
              <div className="explainer-generate-row"><p>研究材料 → 完整旁白 + 对应分镜</p><button type="button" className="explainer-button is-primary" disabled={!form.title.trim() || form.research.trim().length < 40 || invalidScenes} onClick={() => void act('script')}>{form.scenes.length ? '重新生成脚本 ↗' : '生成长篇脚本 ↗'}</button></div>
            </section>

            <section className="explainer-section explainer-script">
              <div className="explainer-section-heading"><div><p className="explainer-eyebrow">03 / STORYBOARD</p><h2>旁白与画面，一起读</h2></div><button type="button" disabled={form.scenes.length === 0} onClick={downloadScript}>导出完整旁白 ↓</button></div>
              <div className="explainer-script-summary"><span>{form.scenes.length} 个镜头</span><span>{count.toLocaleString()} {form.language === 'zh' ? '字' : 'words'}</span><span>估算旁白 {estimate.toFixed(1)} 分钟</span><span>目标 {form.target_minutes} 分钟</span></div>
              <p className="explainer-render-estimate">{form.scenes.length > 0 && imageInterval ? `共 ${form.scenes.length} 张图。沿用每次 ${Math.round(imageInterval / 60)} 分钟的图片请求间隔，全部新生成仅间隔等待就至少约 ${imageWaitHours.toFixed(1)} 小时，另需实际生成与合成时间；重用已有素材可缩短。` : '沿用现有图片生成间隔，可调大每镜时长以减少图片数。'}</p>
              {form.scenes.length === 0 && <div className="explainer-empty-story"><span>“</span><h3>先有一个值得讲清楚的问题。</h3><p>填入研究材料后生成脚本，或手动添加你的第一个镜头。</p></div>}
              <div className="explainer-scenes">
                {form.scenes.map((scene, index) => <article className="explainer-scene" key={scene.id}>
                  <div className="explainer-scene-top"><span className="explainer-scene-number">{String(index + 1).padStart(2, '0')}</span><label className="explainer-sr-only" htmlFor={`chapter-${scene.id}`}>镜头 {index + 1} 章节</label><input id={`chapter-${scene.id}`} value={scene.chapter} placeholder="章节标题" onChange={(e) => patchScene(index, { chapter: e.target.value })} />
                    <button type="button" aria-label={`删除镜头 ${index + 1}`} className="explainer-remove" onClick={() => patch('scenes', form.scenes.filter((_, i) => i !== index))}>删除</button></div>
                  <div className="explainer-scene-columns"><div><label htmlFor={`narration-${scene.id}`}>旁白 <span>VOICE</span></label><textarea id={`narration-${scene.id}`} rows={5} value={scene.narration} onChange={(e) => patchScene(index, { narration: e.target.value })} placeholder="这个镜头中要说的话…" /></div>
                    <div><label htmlFor={`visual-${scene.id}`}>画面描述 <span>FRAME</span></label><textarea id={`visual-${scene.id}`} rows={5} value={scene.visual_prompt} onChange={(e) => patchScene(index, { visual_prompt: e.target.value })} placeholder="人物在做什么？用什么视觉比喻？" /></div></div>
                  <div className="explainer-scene-bottom"><div><label htmlFor={`text-${scene.id}`}>画面关键词（可选）</label><input id={`text-${scene.id}`} value={scene.on_screen_text} onChange={(e) => patchScene(index, { on_screen_text: e.target.value })} placeholder="尽量简短" /></div>
                    {form.style.motion_mode === 'gemini' && form.style.motion_every_n_scenes > 0 && (index + 1) % form.style.motion_every_n_scenes === 0
                      ? <label className="explainer-check"><input type="checkbox" checked disabled /><span>按间隔自动生成动画</span></label>
                      : <label className="explainer-check"><input type="checkbox" checked={scene.motion} onChange={(e) => patchScene(index, { motion: e.target.checked })} /><span>动态镜头{form.style.motion_mode === 'none' ? '（当前为静态模式）' : ''}</span></label>}</div>
                </article>)}
              </div>
              <button type="button" className="explainer-add-scene" onClick={() => { const used = new Set(form.scenes.map((scene) => scene.id)); let number = form.scenes.length + 1; while (used.has(`scene-${String(number).padStart(3, '0')}`)) number++; patch('scenes', [...form.scenes, { id: `scene-${String(number).padStart(3, '0')}`, chapter: '', narration: '', visual_prompt: '', on_screen_text: '', motion: false }]) }}>＋ 添加镜头</button>
              {invalidScenes && <p className="explainer-small-error">每个镜头都需要旁白和画面描述，填写完整后即可生成视频。</p>}
              <div className="explainer-render-row"><div><strong>把故事做出来。</strong><p>自动保存修改，逐镜头配音、绘图，再按实际音频时长合成。</p></div><button type="button" className="explainer-button is-primary" disabled={!form.title.trim() || form.scenes.length === 0 || invalidScenes || (form.language === 'zh' && selectedModel?.language === 'en')} onClick={() => void act('render')}>{project?.status === 'failed' || project?.status === 'interrupted' ? '重新生成视频 →' : '生成视频 →'}</button></div>
            </section>
          </fieldset>
          {project && project.logs.length > 0 && <details className="explainer-logs"><summary>运行记录 · {project.logs.length}</summary><pre>{project.logs.join('\n')}</pre></details>}
          {pending && <div className="explainer-pending" role="status" aria-live="polite">{pending === 'script' ? '正在保存并启动脚本生成…' : pending === 'render' ? '正在保存并启动视频生成…' : '正在处理…'}</div>}
        </>}
      </div>
    </div>
  </div>
}

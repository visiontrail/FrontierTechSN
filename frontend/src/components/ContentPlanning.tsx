import { useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { Link } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  approveContentPlanItem,
  createContentPlanItem,
  createContentSeries,
  deleteContentPlanItem,
  fetchContentPlanItems,
  fetchContentPlanningStatus,
  fetchContentSeries,
  fetchProviders,
  fetchTtsModels,
  fetchVoices,
  recordContentPlanPublication,
  updateContentPlanItem,
  updateContentSeries,
  DEFAULT_CLOSING_REMARKS,
} from '../api'
import type { ContentPlanInput, ContentPlanItem, TaskConfig } from '../api'
import { localInputToIso, toLocalInputValue } from '../schedule'

const PLAN_LABELS: Record<string, string> = {
  draft: 'Draft',
  scheduled: 'Scheduled',
  generating: 'Generating',
  review: 'Needs review',
  ready: 'Approved',
  published: 'Published',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

function nextDate(hours: number): string {
  const date = new Date(Date.now() + hours * 60 * 60 * 1000)
  date.setMinutes(0, 0, 0)
  return toLocalInputValue(date)
}

function formatDate(value: string | null): string {
  if (!value) return 'Not scheduled'
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(new Date(value))
}

function shortDate(value: string | null): { day: string; month: string } {
  if (!value) return { day: '—', month: 'Draft' }
  const date = new Date(value)
  return {
    day: new Intl.DateTimeFormat(undefined, { day: '2-digit' }).format(date),
    month: new Intl.DateTimeFormat(undefined, { month: 'short' }).format(date).toUpperCase(),
  }
}

type PlanDraft = {
  seriesId: string
  title: string
  brief: string
  sourceType: 'topic' | 'youtube'
  sourceUrl: string
  episode: string
  generationAt: string
  publishAt: string
  platform: string
  taskConfig: TaskConfig
}

const DEFAULT_TASK_CONFIG = (): TaskConfig => ({
  target_duration_minutes: 10,
  script_format: 'monologue',
  speaker_count: 1,
  voice_1: 'Carter',
  voice_2: 'Alice',
  closing_remarks: DEFAULT_CLOSING_REMARKS,
  include_character: false,
  captions_enabled: false,
  tts_model: 'vibevoice-0.5b',
  video_template: 'podcast',
  video_orientation: 'landscape',
  opening_style: 'editorial_motion',
  processing_mode: 'full_text',
  provider_id: null,
  footage_enabled: true,
  footage_provider: 'hybrid',
  footage_license_policy: 'review_required',
  footage_clip_count: 8,
  footage_orientation: 'landscape',
  footage_multimodal_analyzer: 'gemini_web',
  collage_broll_enabled: true,
  collage_broll_count: 4,
  thumbnail_enabled: true,
  auto_render: true,
})

const EMPTY_PLAN = (): PlanDraft => ({
  seriesId: '',
  title: '',
  brief: '',
  sourceType: 'youtube',
  sourceUrl: '',
  episode: '',
  generationAt: nextDate(24),
  publishAt: nextDate(48),
  platform: 'YouTube',
  taskConfig: DEFAULT_TASK_CONFIG(),
})

export default function ContentPlanning() {
  const queryClient = useQueryClient()
  const seriesDialogRef = useRef<HTMLDialogElement>(null)
  const planDialogRef = useRef<HTMLDialogElement>(null)
  const [seriesFormOpen, setSeriesFormOpen] = useState(false)
  const [planFormOpen, setPlanFormOpen] = useState(false)
  const [activeSeries, setActiveSeries] = useState<string>('all')
  const [seriesDraft, setSeriesDraft] = useState({ name: '', theme: '', description: '' })
  const [planDraft, setPlanDraft] = useState<PlanDraft>(EMPTY_PLAN)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [publicationUrls, setPublicationUrls] = useState<Record<string, string>>({})

  useEffect(() => {
    const dialog = seriesDialogRef.current
    if (!dialog) return
    if (seriesFormOpen && !dialog.open) {
      dialog.showModal()
      dialog.querySelector<HTMLElement>('[data-modal-autofocus]')?.focus()
    }
    if (!seriesFormOpen && dialog.open) dialog.close()
  }, [seriesFormOpen])

  useEffect(() => {
    const dialog = planDialogRef.current
    if (!dialog) return
    if (planFormOpen && !dialog.open) {
      dialog.showModal()
      dialog.querySelector<HTMLElement>('[data-modal-autofocus]')?.focus()
    }
    if (!planFormOpen && dialog.open) dialog.close()
  }, [planFormOpen])

  const { data: planningStatus } = useQuery({
    queryKey: ['content-planning-status'],
    queryFn: fetchContentPlanningStatus,
  })
  const { data: series = [], isLoading: seriesLoading } = useQuery({
    queryKey: ['content-series'],
    queryFn: fetchContentSeries,
  })
  const { data: items = [], isLoading: itemsLoading } = useQuery({
    queryKey: ['content-plan-items'],
    queryFn: fetchContentPlanItems,
    refetchInterval: 5000,
  })
  const { data: providers = [] } = useQuery({ queryKey: ['providers'], queryFn: fetchProviders })
  const { data: ttsModels = [] } = useQuery({ queryKey: ['tts-models'], queryFn: fetchTtsModels })
  const selectedTtsModel = planDraft.taskConfig.tts_model || 'vibevoice-0.5b'
  const { data: voices = [] } = useQuery({
    queryKey: ['voices', selectedTtsModel],
    queryFn: () => fetchVoices(selectedTtsModel),
    enabled: planFormOpen,
  })
  const voiceNames = voices.map((voice) => voice.name)
  const selectedVoice1 = voiceNames.includes(planDraft.taskConfig.voice_1)
    ? planDraft.taskConfig.voice_1
    : (voiceNames[0] || planDraft.taskConfig.voice_1)
  const selectedVoice2 = voiceNames.includes(planDraft.taskConfig.voice_2)
    ? planDraft.taskConfig.voice_2
    : (voiceNames[1] || voiceNames[0] || planDraft.taskConfig.voice_2)

  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ['content-series'] })
    queryClient.invalidateQueries({ queryKey: ['content-plan-items'] })
    queryClient.invalidateQueries({ queryKey: ['tasks'] })
  }

  const seriesMutation = useMutation({
    mutationFn: createContentSeries,
    onSuccess: (created) => {
      setSeriesDraft({ name: '', theme: '', description: '' })
      setSeriesFormOpen(false)
      setActiveSeries(created.id)
      setPlanDraft((draft) => ({ ...draft, seriesId: created.id }))
      refresh()
    },
  })

  const planMutation = useMutation({
    mutationFn: ({ id, input }: { id: string | null; input: ContentPlanInput }) => (
      id ? updateContentPlanItem(id, input) : createContentPlanItem(input)
    ),
    onSuccess: () => {
      setPlanDraft(EMPTY_PLAN())
      setEditingId(null)
      setPlanFormOpen(false)
      refresh()
    },
  })

  const deleteMutation = useMutation({ mutationFn: deleteContentPlanItem, onSuccess: refresh })
  const approveMutation = useMutation({ mutationFn: approveContentPlanItem, onSuccess: refresh })
  const publishMutation = useMutation({
    mutationFn: ({ id, url }: { id: string; url: string | null }) => recordContentPlanPublication(id, url),
    onSuccess: (_, variables) => {
      setPublicationUrls((current) => ({ ...current, [variables.id]: '' }))
      refresh()
    },
  })
  const archiveSeriesMutation = useMutation({
    mutationFn: (id: string) => updateContentSeries(id, { archived: true }),
    onSuccess: () => {
      setActiveSeries('all')
      refresh()
    },
  })

  const visibleItems = useMemo(
    () => activeSeries === 'all' ? items : items.filter((item) => item.series_id === activeSeries),
    [activeSeries, items],
  )

  const counts = useMemo(() => ({
    planned: items.filter((item) => ['draft', 'scheduled'].includes(item.status)).length,
    production: items.filter((item) => item.status === 'generating').length,
    review: items.filter((item) => item.status === 'review').length,
    published: items.filter((item) => item.status === 'published').length,
  }), [items])

  const submitSeries = (event: FormEvent) => {
    event.preventDefault()
    seriesMutation.mutate(seriesDraft)
  }

  const submitPlan = (event: FormEvent) => {
    event.preventDefault()
    const input: ContentPlanInput = {
      series_id: planDraft.seriesId || null,
      title: planDraft.title,
      brief: planDraft.brief,
      source_type: planDraft.sourceType,
      source_url: planDraft.sourceType === 'youtube' ? planDraft.sourceUrl : null,
      episode_number: planDraft.episode ? Number(planDraft.episode) : null,
      generation_at: localInputToIso(planDraft.generationAt),
      publish_at: localInputToIso(planDraft.publishAt),
      platform: planDraft.platform || 'manual',
      auto_publish_requested: false,
      task_config: {
        ...planDraft.taskConfig,
        voice_1: selectedVoice1,
        voice_2: selectedVoice2,
      },
    }
    planMutation.mutate({ id: editingId, input })
  }

  const openSeriesForm = () => {
    setPlanFormOpen(false)
    setEditingId(null)
    setSeriesFormOpen(true)
  }

  const openNewPlanForm = () => {
    setSeriesFormOpen(false)
    setEditingId(null)
    setPlanDraft({ ...EMPTY_PLAN(), seriesId: activeSeries === 'all' ? '' : activeSeries })
    setPlanFormOpen(true)
  }

  const editPlan = (item: ContentPlanItem) => {
    setSeriesFormOpen(false)
    setEditingId(item.id)
    setPlanDraft({
      seriesId: item.series_id || '',
      title: item.title,
      brief: item.brief,
      sourceType: item.source_type,
      sourceUrl: item.source_url || '',
      episode: item.episode_number ? String(item.episode_number) : '',
      generationAt: item.generation_at ? toLocalInputValue(new Date(item.generation_at)) : '',
      publishAt: item.publish_at ? toLocalInputValue(new Date(item.publish_at)) : '',
      platform: item.platform,
      taskConfig: { ...DEFAULT_TASK_CONFIG(), ...item.task_config },
    })
    setPlanFormOpen(true)
  }

  const setTaskConfig = (patch: Partial<TaskConfig>) => {
    setPlanDraft((draft) => ({
      ...draft,
      taskConfig: { ...draft.taskConfig, ...patch },
    }))
  }

  return (
    <section className="planning-workspace">
      <header className="planning-command">
        <div className="planning-title-block">
          <span className="eyebrow">Editorial runway · {new Date().getFullYear()}</span>
          <h1>Content Plan</h1>
          <p>Shape the series. Set the clock. Keep publication human.</p>
        </div>
        <div className="planning-actions">
          <button className="btn-ghost" type="button" onClick={openSeriesForm}>New Series</button>
          <button className="btn-primary" type="button" onClick={openNewPlanForm}>Plan a video</button>
        </div>
      </header>

      <div className="planning-ledger" aria-label="Plan summary">
        <div><span>Planned</span><strong>{counts.planned}</strong><small>ideas on the runway</small></div>
        <div><span>In production</span><strong>{counts.production}</strong><small>tasks currently running</small></div>
        <div><span>Needs review</span><strong>{counts.review}</strong><small>finished, never auto-posted</small></div>
        <div><span>Published</span><strong>{counts.published}</strong><small>manually confirmed</small></div>
      </div>

      <dialog
        ref={seriesDialogRef}
        className="planning-modal planning-modal--series"
        aria-labelledby="series-form-title"
        aria-describedby="series-form-description"
        onClose={() => setSeriesFormOpen(false)}
        onClick={(event) => {
          if (event.target === event.currentTarget) event.currentTarget.close()
        }}
      >
        {seriesFormOpen && (
          <form className="planning-form series-form" onSubmit={submitSeries}>
            <div className="planning-form-heading">
              <span className="form-index">01</span>
              <div>
                <h2 id="series-form-title">Define a series</h2>
                <p id="series-form-description">A durable editorial lens, not a one-off title.</p>
              </div>
              <button className="planning-modal-close" type="button" aria-label="Close new series form" onClick={() => seriesDialogRef.current?.close()}>×</button>
            </div>
            <div className="planning-field-grid">
              <div className="form-group">
                <label htmlFor="series-name">Series name</label>
                <input id="series-name" data-modal-autofocus required maxLength={120} placeholder="Borderlands" value={seriesDraft.name} onChange={(e) => setSeriesDraft({ ...seriesDraft, name: e.target.value })} />
              </div>
              <div className="form-group">
                <label htmlFor="series-theme">Theme / territory</label>
                <input id="series-theme" maxLength={120} placeholder="Countries, borders, identity" value={seriesDraft.theme} onChange={(e) => setSeriesDraft({ ...seriesDraft, theme: e.target.value })} />
              </div>
              <div className="form-group planning-span-2">
                <label htmlFor="series-description">Editorial promise</label>
                <textarea id="series-description" rows={3} placeholder="What question will every episode pursue?" value={seriesDraft.description} onChange={(e) => setSeriesDraft({ ...seriesDraft, description: e.target.value })} />
              </div>
            </div>
            <div className="planning-form-footer">
              {seriesMutation.isError && <span className="planning-form-error">{(seriesMutation.error as Error).message}</span>}
              <button className="btn-ghost" type="button" onClick={() => seriesDialogRef.current?.close()}>Cancel</button>
              <button className="btn-primary" type="submit" disabled={seriesMutation.isPending}>{seriesMutation.isPending ? 'Creating…' : 'Create series'}</button>
            </div>
          </form>
        )}
      </dialog>

      <dialog
        ref={planDialogRef}
        className="planning-modal planning-modal--plan"
        aria-labelledby="plan-form-title"
        aria-describedby="plan-form-description"
        onClose={() => {
          setPlanFormOpen(false)
          setEditingId(null)
        }}
        onClick={(event) => {
          if (event.target === event.currentTarget) event.currentTarget.close()
        }}
      >
        {planFormOpen && (
          <form className="planning-form plan-form" onSubmit={submitPlan}>
            <div className="planning-form-heading">
              <span className="form-index">{editingId ? 'REV' : '02'}</span>
              <div>
                <h2 id="plan-form-title">{editingId ? 'Revise the plan' : 'Schedule a video'}</h2>
                <p id="plan-form-description">The generation time creates and controls a real pipeline task.</p>
              </div>
              <button className="planning-modal-close" type="button" aria-label="Close video plan form" onClick={() => planDialogRef.current?.close()}>×</button>
            </div>
            <div className="planning-field-grid planning-field-grid--three">
              <div className="form-group">
                <label htmlFor="plan-series">Series</label>
                <select id="plan-series" value={planDraft.seriesId} onChange={(e) => setPlanDraft({ ...planDraft, seriesId: e.target.value })}>
                  <option value="">Standalone video</option>
                  {series.filter((entry) => !entry.archived).map((entry) => <option key={entry.id} value={entry.id}>{entry.name}</option>)}
                </select>
              </div>
              <div className="form-group planning-span-2">
                <label htmlFor="plan-title">Working title</label>
                <input id="plan-title" data-modal-autofocus required maxLength={180} placeholder="The country that moved its capital overnight" value={planDraft.title} onChange={(e) => setPlanDraft({ ...planDraft, title: e.target.value })} />
              </div>
              <div className="form-group planning-span-3">
                <label htmlFor="plan-brief">Editorial brief</label>
                <textarea id="plan-brief" required minLength={10} rows={4} placeholder="State the angle, essential facts, audience promise, and questions the script must answer." value={planDraft.brief} onChange={(e) => setPlanDraft({ ...planDraft, brief: e.target.value })} />
              </div>
              <div className="form-group">
                <label htmlFor="plan-source-type">Video source</label>
                <select id="plan-source-type" value={planDraft.sourceType} onChange={(e) => setPlanDraft({ ...planDraft, sourceType: e.target.value as PlanDraft['sourceType'] })}>
                  <option value="youtube">YouTube URL</option>
                  <option value="topic">Topic / research brief</option>
                </select>
              </div>
              <div className="form-group planning-span-2">
                <label htmlFor="plan-source-url">YouTube URL</label>
                <input
                  id="plan-source-url"
                  type="url"
                  required={planDraft.sourceType === 'youtube'}
                  disabled={planDraft.sourceType !== 'youtube'}
                  placeholder="https://www.youtube.com/watch?v=..."
                  value={planDraft.sourceUrl}
                  onChange={(e) => setPlanDraft({ ...planDraft, sourceUrl: e.target.value })}
                />
                <small>{planDraft.sourceType === 'youtube' ? 'This exact URL is saved on the scheduled task.' : 'The editorial brief becomes the research source.'}</small>
              </div>
              <div className="form-group">
                <label htmlFor="plan-episode">Episode number</label>
                <input id="plan-episode" type="number" min="1" placeholder="1" value={planDraft.episode} onChange={(e) => setPlanDraft({ ...planDraft, episode: e.target.value })} />
              </div>
              <div className="form-group">
                <label htmlFor="plan-generation">Begin generation</label>
                <input id="plan-generation" type="datetime-local" value={planDraft.generationAt} onInput={(e) => setPlanDraft({ ...planDraft, generationAt: e.currentTarget.value })} />
                <small>Creates a linked task and releases it at this exact time.</small>
              </div>
              <div className="form-group">
                <label htmlFor="plan-publish">Target publication</label>
                <input id="plan-publish" type="datetime-local" value={planDraft.publishAt} min={planDraft.generationAt || undefined} onInput={(e) => setPlanDraft({ ...planDraft, publishAt: e.currentTarget.value })} />
                <small>A deadline only; it does not post automatically.</small>
              </div>
              <div className="form-group planning-span-3">
                <label htmlFor="plan-platform">Publication destination</label>
                <input id="plan-platform" maxLength={80} placeholder="YouTube" value={planDraft.platform} onChange={(e) => setPlanDraft({ ...planDraft, platform: e.target.value })} />
              </div>
            </div>
            <section className="plan-config-section" aria-labelledby="plan-config-title">
              <div className="plan-config-heading">
                <div><span className="eyebrow">Generation configuration</span><h3 id="plan-config-title">Build settings</h3></div>
                <small>Saved now and applied when the scheduled task starts.</small>
              </div>
              <div className="planning-field-grid planning-field-grid--three">
                <div className="form-group">
                  <label htmlFor="plan-duration">Target duration</label>
                  <select id="plan-duration" value={planDraft.taskConfig.target_duration_minutes} onChange={(e) => setTaskConfig({ target_duration_minutes: Number(e.target.value) })}>
                    {[5, 10, 15, 20].map((value) => <option key={value} value={value}>{value} min</option>)}
                  </select>
                </div>
                <div className="form-group">
                  <label htmlFor="plan-script-format">Script format</label>
                  <select id="plan-script-format" value={planDraft.taskConfig.script_format} onChange={(e) => {
                    const format = e.target.value as 'monologue' | 'dialogue'
                    const dialogueModel = ttsModels.find((model) => !model.single_speaker)?.id
                    setTaskConfig({
                      script_format: format,
                      speaker_count: format === 'monologue' ? 1 : 2,
                      ...(format === 'dialogue' && ttsModels.find((model) => model.id === selectedTtsModel)?.single_speaker && dialogueModel ? { tts_model: dialogueModel } : {}),
                    })
                  }}>
                    <option value="monologue">Solo talk-show</option>
                    <option value="dialogue">Two-host dialogue</option>
                  </select>
                </div>
                <div className="form-group">
                  <label htmlFor="plan-provider">AI provider</label>
                  <select id="plan-provider" value={planDraft.taskConfig.provider_id ?? ''} onChange={(e) => setTaskConfig({ provider_id: e.target.value ? Number(e.target.value) : null })}>
                    <option value="">Default provider</option>
                    {providers.map((provider) => <option key={provider.id} value={provider.id}>{provider.name} — {provider.model}</option>)}
                  </select>
                </div>
                <div className="form-group">
                  <label htmlFor="plan-tts-model">TTS model</label>
                  <select id="plan-tts-model" value={selectedTtsModel} onChange={(e) => {
                    const model = ttsModels.find((entry) => entry.id === e.target.value)
                    setTaskConfig({
                      tts_model: e.target.value,
                      ...(model?.single_speaker ? { script_format: 'monologue', speaker_count: 1 } : {}),
                    })
                  }}>
                    {ttsModels.length === 0 && <option value={selectedTtsModel}>{selectedTtsModel}</option>}
                    {ttsModels.map((model) => <option key={model.id} value={model.id} disabled={planDraft.taskConfig.script_format === 'dialogue' && model.single_speaker}>{model.provider} — {model.label}</option>)}
                  </select>
                </div>
                <div className="form-group">
                  <label htmlFor="plan-voice-1">Host voice</label>
                  <select id="plan-voice-1" value={selectedVoice1} onChange={(e) => setTaskConfig({ voice_1: e.target.value })}>
                    {voices.length === 0 && <option value={selectedVoice1}>{selectedVoice1}</option>}
                    {voices.map((voice) => <option key={voice.name} value={voice.name}>{voice.name}</option>)}
                  </select>
                </div>
                <div className="form-group">
                  <label htmlFor="plan-voice-2">Co-host voice</label>
                  <select id="plan-voice-2" disabled={planDraft.taskConfig.script_format !== 'dialogue'} value={selectedVoice2} onChange={(e) => setTaskConfig({ voice_2: e.target.value })}>
                    {voices.length === 0 && <option value={selectedVoice2}>{selectedVoice2}</option>}
                    {voices.map((voice) => <option key={voice.name} value={voice.name}>{voice.name}</option>)}
                  </select>
                </div>
                <div className="form-group">
                  <label htmlFor="plan-template">Video template</label>
                  <select id="plan-template" value={planDraft.taskConfig.video_template} onChange={(e) => setTaskConfig({ video_template: e.target.value })}>
                    <option value="podcast">Documentary</option><option value="kinetic">Kinetic</option><option value="swiss">Swiss Grid</option><option value="minimal">Minimal</option><option value="shanshui">Shan Shui</option>
                  </select>
                </div>
                <div className="form-group">
                  <label htmlFor="plan-orientation">Final frame</label>
                  <select id="plan-orientation" value={planDraft.taskConfig.video_orientation} onChange={(e) => {
                    const orientation = e.target.value as 'landscape' | 'portrait'
                    setTaskConfig({ video_orientation: orientation, footage_orientation: orientation })
                  }}>
                    <option value="landscape">Landscape 16:9</option><option value="portrait">Portrait 9:16</option>
                  </select>
                </div>
                <div className="form-group">
                  <label htmlFor="plan-opening">Opening style</label>
                  <select id="plan-opening" value={planDraft.taskConfig.opening_style} onChange={(e) => setTaskConfig({ opening_style: e.target.value as 'editorial_motion' | 'paper_collage' })}>
                    <option value="editorial_motion">Editorial motion</option><option value="paper_collage">Paper collage</option>
                  </select>
                </div>
                <div className="form-group planning-span-3">
                  <label htmlFor="plan-closing-remarks">Closing remarks</label>
                  <textarea
                    id="plan-closing-remarks"
                    rows={3}
                    maxLength={500}
                    required
                    value={planDraft.taskConfig.closing_remarks ?? DEFAULT_CLOSING_REMARKS}
                    onChange={(e) => setTaskConfig({ closing_remarks: e.target.value })}
                  />
                  <small>Spoken verbatim and included in the final timed video scenes.</small>
                </div>
              </div>
              <div className="plan-switch-grid">
                {([
                  ['captions_enabled', 'Captions', 'Burn concise captions into the video'],
                  ['include_character', 'Animated character', 'Add the Lottie host overlay'],
                  ['thumbnail_enabled', 'Viral thumbnail', 'Generate cover art before TTS'],
                  ['auto_render', 'Auto render', 'Continue after TTS without audio approval'],
                ] as const).map(([key, label, note]) => (
                  <label className="plan-switch" key={key}>
                    <input type="checkbox" checked={Boolean(planDraft.taskConfig[key])} onChange={(e) => setTaskConfig({ [key]: e.target.checked })} />
                    <span><strong>{label}</strong><small>{note}</small></span>
                  </label>
                ))}
              </div>
              <div className="planning-field-grid planning-field-grid--three plan-media-config">
                <label className="plan-switch">
                  <input type="checkbox" checked={Boolean(planDraft.taskConfig.footage_enabled)} onChange={(e) => setTaskConfig({ footage_enabled: e.target.checked })} />
                  <span><strong>Public footage</strong><small>Scout and download eligible B-roll</small></span>
                </label>
                <div className="form-group">
                  <label htmlFor="plan-footage-provider">Footage source</label>
                  <select id="plan-footage-provider" disabled={!planDraft.taskConfig.footage_enabled} value={planDraft.taskConfig.footage_provider} onChange={(e) => {
                    const provider = e.target.value as 'wikimedia' | 'hybrid' | 'opencli_web'
                    setTaskConfig({ footage_provider: provider, footage_license_policy: provider === 'wikimedia' ? 'open_only' : 'review_required' })
                  }}>
                    <option value="hybrid">Commons + YouTube</option><option value="wikimedia">Wikimedia only</option><option value="opencli_web">YouTube only</option>
                  </select>
                </div>
                <div className="form-group">
                  <label htmlFor="plan-footage-count">Footage clips</label>
                  <input id="plan-footage-count" type="number" min="1" max="30" disabled={!planDraft.taskConfig.footage_enabled} value={planDraft.taskConfig.footage_clip_count} onChange={(e) => setTaskConfig({ footage_clip_count: Number(e.target.value) })} />
                </div>
                <label className="plan-switch">
                  <input type="checkbox" checked={Boolean(planDraft.taskConfig.collage_broll_enabled)} onChange={(e) => setTaskConfig({ collage_broll_enabled: e.target.checked })} />
                  <span><strong>Paper-collage B-roll</strong><small>Generate recurring visual metaphors</small></span>
                </label>
                <div className="form-group">
                  <label htmlFor="plan-collage-count">Collage clips</label>
                  <input id="plan-collage-count" type="number" min="2" max="10" disabled={!planDraft.taskConfig.collage_broll_enabled} value={planDraft.taskConfig.collage_broll_count} onChange={(e) => setTaskConfig({ collage_broll_count: Number(e.target.value) })} />
                </div>
              </div>
            </section>
            <div className="planning-safety-note">
              <span className="safety-lock" aria-hidden="true">×</span>
              <div><strong>Automatic publication is locked</strong><p>Every finished video must be reviewed, approved, and manually published. The global pipeline switch is {planningStatus?.auto_publish_enabled ? 'enabled, but this plan remains opted out' : 'off'}.</p></div>
            </div>
            <div className="planning-form-footer">
              {planMutation.isError && <span className="planning-form-error">{(planMutation.error as Error).message}</span>}
              <button className="btn-ghost" type="button" onClick={() => planDialogRef.current?.close()}>Cancel</button>
              <button className="btn-primary" type="submit" disabled={planMutation.isPending}>{planMutation.isPending ? 'Saving…' : editingId ? 'Save changes' : 'Commit to calendar'}</button>
            </div>
          </form>
        )}
      </dialog>

      <div className="planning-layout">
        <aside className="series-rail">
          <div className="series-rail-head"><span>Series index</span><small>{series.length} collections</small></div>
          <button type="button" className={`series-card series-card--all${activeSeries === 'all' ? ' is-active' : ''}`} onClick={() => setActiveSeries('all')}>
            <span className="series-monogram">ALL</span><span><strong>Master calendar</strong><small>{items.length} total videos</small></span>
          </button>
          {seriesLoading ? <div className="series-loading">Loading series…</div> : series.filter((entry) => !entry.archived).map((entry, index) => (
            <button key={entry.id} type="button" className={`series-card${activeSeries === entry.id ? ' is-active' : ''}`} onClick={() => setActiveSeries(entry.id)}>
              <span className="series-monogram">{String(index + 1).padStart(2, '0')}</span>
              <span className="series-card-copy"><strong>{entry.name}</strong><small>{entry.theme || 'Open theme'} · {entry.item_count} videos</small></span>
            </button>
          ))}
          {activeSeries !== 'all' && (
            <button className="series-archive" type="button" onClick={() => {
              if (confirm('Archive this series? Its video plans will remain available.')) archiveSeriesMutation.mutate(activeSeries)
            }}>Archive selected series</button>
          )}
        </aside>

        <main className="rundown-board">
          <div className="rundown-head">
            <div><span className="eyebrow">Production rundown</span><h2>{activeSeries === 'all' ? 'All planned videos' : series.find((entry) => entry.id === activeSeries)?.name}</h2></div>
            <span className={`publication-mode ${planningStatus?.auto_publish_enabled ? 'is-enabled' : ''}`}><i /> Human review</span>
          </div>

          {itemsLoading ? (
            <div className="rundown-empty"><span className="loading-mark" />Loading editorial calendar…</div>
          ) : visibleItems.length === 0 ? (
            <div className="rundown-empty"><strong>No videos on this runway</strong><p>Plan the first topic and its generation task will appear here and in Tasks.</p><button className="btn-primary" onClick={openNewPlanForm}>Plan a video</button></div>
          ) : (
            <div className="rundown-list">
              {visibleItems.map((item) => {
                const date = shortDate(item.generation_at)
                const canEdit = ['draft', 'scheduled'].includes(item.status)
                const canDelete = ['draft', 'scheduled'].includes(item.status)
                return (
                  <article className={`rundown-item rundown-item--${item.status}`} key={item.id}>
                    <div className="rundown-date"><strong>{date.day}</strong><span>{date.month}</span></div>
                    <div className="rundown-content">
                      <div className="rundown-meta">
                        <span>{item.series_name || 'Standalone'}</span>
                        {item.episode_number && <span>EP {String(item.episode_number).padStart(2, '0')}</span>}
                        <span>{item.source_type === 'youtube' ? 'YouTube source' : 'Topic source'}</span>
                        <span>{item.task_config.target_duration_minutes} min</span>
                        <span>{item.platform}</span>
                      </div>
                      <h3>{item.title}</h3>
                      <p>{item.brief}</p>
                      <div className="rundown-timeline">
                        <div className="timeline-stop timeline-stop--generation"><i /><span><small>Generation</small><strong>{formatDate(item.generation_at)}</strong></span></div>
                        <div className="timeline-line" />
                        <div className="timeline-stop timeline-stop--publish"><i /><span><small>Target release</small><strong>{formatDate(item.publish_at)}</strong></span></div>
                      </div>
                      {item.error_message && <div className="planning-inline-error">{item.error_message}</div>}
                      {item.status === 'published' && (
                        <div className="publication-proof">
                          <span>Published {formatDate(item.published_at)}</span>
                          {item.publication_url && <a href={item.publication_url} target="_blank" rel="noopener">Open published video ↗</a>}
                        </div>
                      )}
                      {item.status === 'ready' && (
                        <div className="publication-recorder">
                          <input aria-label={`Publication URL for ${item.title}`} placeholder="Paste the live publication URL (optional)" value={publicationUrls[item.id] || ''} onChange={(e) => setPublicationUrls((current) => ({ ...current, [item.id]: e.target.value }))} />
                          <button className="btn-primary" disabled={publishMutation.isPending} onClick={() => publishMutation.mutate({ id: item.id, url: publicationUrls[item.id] || null })}>Mark published</button>
                        </div>
                      )}
                    </div>
                    <div className="rundown-controls">
                      <span className={`plan-status plan-status--${item.status}`}><i />{PLAN_LABELS[item.status]}</span>
                      <div className="rundown-buttons">
                        {item.task_id && <Link className="rundown-link" to={`/tasks/${item.task_id}`}>Open task ↗</Link>}
                        {item.status === 'review' && <button className="btn-primary" disabled={approveMutation.isPending} onClick={() => approveMutation.mutate(item.id)}>Approve for manual publish</button>}
                        {canEdit && <button className="btn-ghost" onClick={() => editPlan(item)}>Edit plan</button>}
                        {canDelete && <button className="rundown-delete" onClick={() => { if (confirm('Delete this plan and its queued task?')) deleteMutation.mutate(item.id) }}>Delete</button>}
                      </div>
                    </div>
                  </article>
                )
              })}
            </div>
          )}
        </main>
      </div>
    </section>
  )
}

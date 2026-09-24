export interface ExplainerStyle {
  background_color: string
  ink_color: string
  accent_color: string
  character_description: string
  visual_direction: string
  writing_tone: string
  motion_mode: 'none' | 'subtle' | 'gemini'
  motion_every_n_scenes: number
  seconds_per_scene: number
}

export interface ExplainerScene {
  id: string
  chapter: string
  narration: string
  visual_prompt: string
  on_screen_text: string
  motion: boolean
}

export interface ExplainerForm {
  title: string
  research: string
  language: 'en' | 'zh'
  target_minutes: number
  tts_model: string
  voice: string
  captions_enabled: boolean
  style: ExplainerStyle
  scenes: ExplainerScene[]
}

export type ExplainerStatus = 'draft' | 'writing_script' | 'script_ready' | 'generating_audio'
  | 'generating_images' | 'generating_motion' | 'rendering' | 'completed' | 'failed' | 'cancelled' | 'interrupted'

export interface ExplainerProject extends ExplainerForm {
  id: string
  status: ExplainerStatus
  revision: number
  created_at: string
  updated_at: string
  error: string | null
  logs: string[]
  progress: { completed: number; total: number; message: string }
  artifacts: {
    video_url?: string
    audio_url?: string
    timeline_url?: string
    duration_seconds?: number
    scene_artifacts?: unknown[]
  }
}

export type ExplainerDefaultValues = Omit<ExplainerForm, 'title' | 'research' | 'scenes'>

export interface ExplainerDefaults extends ExplainerDefaultValues {
  tts_models: { id: string; label: string; language: string; voices: string[] }[]
  presets: { id: string; label: string; style: ExplainerStyle }[]
  provider_limits?: {
    chatgpt_interval_seconds: number
    gemini_interval_min_seconds: number
    gemini_interval_max_seconds: number
  }
}

export class ExplainerApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, method = 'GET', body?: unknown): Promise<T> {
  const response = await fetch(`/api/explainers${path}`, {
    method,
    ...(body === undefined ? {} : { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    const detail = payload?.detail
    const message = typeof detail === 'string' ? detail
      : Array.isArray(detail) ? detail.map((item: { msg?: string }) => item.msg ?? 'Invalid input').join('; ')
      : `Request failed (${response.status})`
    throw new ExplainerApiError(message, response.status)
  }
  return response.json() as Promise<T>
}

export const fetchExplainers = () => request<{ projects: ExplainerProject[] }>('')
export const fetchExplainerDefaults = () => request<ExplainerDefaults>('/defaults')
export const saveExplainerDefaults = (values: ExplainerDefaultValues) => request<ExplainerDefaults>('/defaults', 'PUT', values)
export const fetchExplainer = (id: string) => request<ExplainerProject>(`/${encodeURIComponent(id)}`)
export const createExplainer = (form: ExplainerForm) => request<ExplainerProject>('', 'POST', form)
export const saveExplainer = (id: string, form: ExplainerForm, revision: number) => request<ExplainerProject>(`/${encodeURIComponent(id)}`, 'PUT', { ...form, revision })
export const runExplainer = (id: string, action: 'script' | 'render' | 'cancel') => request<ExplainerProject>(`/${encodeURIComponent(id)}/${action}`, 'POST')

export const isExplainerRunning = (status?: ExplainerStatus) => !!status && [
  'writing_script', 'generating_audio', 'generating_images', 'generating_motion', 'rendering',
].includes(status)

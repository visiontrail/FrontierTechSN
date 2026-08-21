const BASE = '';

export interface TaskConfig {
  target_duration_minutes: number;
  script_format?: 'monologue' | 'dialogue';
  speaker_count: number;
  voice_1: string;
  voice_2: string;
  closing_remarks: string;
  include_character: boolean;
  captions_enabled?: boolean;
  tts_model?: string;
  video_template?: string;
  video_orientation?: 'landscape' | 'portrait';
  opening_style?: 'editorial_motion' | 'paper_collage';
  processing_mode?: string;
  ai_endpoint?: string;
  ai_model?: string;
  provider_id?: number | null;
  footage_enabled?: boolean;
  footage_provider?: 'wikimedia' | 'hybrid' | 'opencli_web';
  footage_license_policy?: 'open_only' | 'review_required';
  footage_clip_count?: number;
  footage_orientation?: 'landscape' | 'portrait';
  footage_multimodal_analyzer?: 'gemini_web';
  collage_broll_enabled?: boolean;
  collage_broll_count?: number;
  news_images_enabled?: boolean;
  news_image_count?: number;
  thumbnail_enabled?: boolean;
  auto_render?: boolean;
  news_language?: 'en' | 'zh';
  news_max_stories?: number;
  news_window_hours?: number;
  news_timezone?: string;
  background_music_enabled?: boolean;
  background_music_provider?: 'gemini_create_music' | 'local';
  background_music_bed_db?: number;
  background_music_duck_db?: number;
  auto_publish?: boolean;
  publish_targets?: Array<'youtube' | 'x' | 'apple_podcast'>;
  publish_visibility?: 'private' | 'unlisted' | 'public';
  publish_test_mode?: boolean;
}

export interface Provider {
  id: number;
  name: string;
  endpoint: string;
  api_key_masked: string;
  model: string;
  is_default: boolean;
  created_at: string;
}

export interface ProviderInput {
  name: string;
  endpoint: string;
  api_key?: string;
  model: string;
  is_default?: boolean;
}

export interface ProviderTestRequest {
  provider_id?: number | null;
  endpoint?: string;
  model?: string;
  api_key?: string;
}

export interface ProviderTestResult {
  ok: boolean;
  message: string;
  latency_ms?: number | null;
}

export interface Task {
  id: string;
  created_at: string;
  updated_at: string;
  source_type: 'topic' | 'youtube' | 'epub' | 'pdf' | 'news_daily';
  source_url: string | null;
  source_title: string | null;
  generated_title: string | null;
  status: 'queued' | 'researching' | 'extracting' | 'digesting' | 'titling' | 'reviewing' | 'sourcing' | 'tts' | 'awaiting_review' | 'music' | 'composing' | 'publishing' | 'complete' | 'failed';
  error_message: string | null;
  config: TaskConfig;
  /** UTC ISO-8601 start time; the worker holds the task until it passes. */
  scheduled_at: string | null;
  output_dir: string | null;
  script_path: string | null;
  audio_path: string | null;
  video_path: string | null;
  video_artifact_state: VideoArtifactState | null;
  thumbnail_path: string | null;
  duration_seconds: number | null;
  suppress_next_auto_publish: boolean;
  publication_safety_hold: boolean;
  origin_type: 'manual' | 'content_plan' | 'daily_news';
  origin_id: string | null;
  origin_label: string | null;
  planned_publish_at: string | null;
}

export interface FootageQuery {
  query: string;
  purpose: string;
}

export interface FootageClip {
  id: string;
  query: string;
  purpose: string;
  provider: string;
  provider_id: string;
  title: string;
  source_page_url: string;
  creator: string;
  license: string;
  license_url?: string;
  attribution_required: boolean;
  duration_seconds: number;
  width: number;
  height: number;
  bytes: number;
  mime_type: string;
  description?: string;
  sha256: string;
  local_path: string;
  status: string;
  platform?: 'bilibili' | 'youtube';
  review_required?: boolean;
  rights_status?: string;
  source_duration_seconds?: number;
  script_excerpt?: string;
  evidence_frames?: string[];
  analysis?: {
    start_seconds: number;
    end_seconds: number;
    confidence: number;
    reason: string;
    analyzer: string;
    status: string;
  };
}

export interface FootageManifest {
  task_id: string;
  status: 'not_started' | 'planning' | 'searching' | 'ready' | 'partial' | 'no_results';
  provider: string;
  provider_id: string;
  license_policy: string;
  license_allowlist: string[];
  requested_clip_count: number;
  planner: string;
  queries: FootageQuery[];
  clips: FootageClip[];
  errors: Array<{ query?: string; stage?: string; message: string }>;
  rights_review_required?: boolean;
  publication_blockers?: string[];
}

export interface Settings {
  ai_endpoint: string;
  ai_model: string;
  tts_device: string;
  default_voice_1: string;
  default_voice_2: string;
  available_voices: Record<string, { gender: string; lang: string }>;
}

export interface DailyAutomationSettings {
  enabled: boolean;
  generation_time: string;
  timezone: string;
  catch_up_after_restart: boolean;
  target_duration_minutes: number;
  language: 'en' | 'zh';
  max_stories: number;
  source_window_hours: number;
  tts_model: string;
  voice: string;
  collage_broll_count: number;
  news_image_count: number;
  public_footage_enabled: boolean;
  footage_clip_count: number;
  background_music_provider: 'gemini_create_music' | 'local';
  auto_publish: boolean;
  publish_targets: Array<'youtube' | 'x' | 'apple_podcast'>;
  publish_visibility: 'private' | 'unlisted' | 'public';
}

export interface DailyAutomationResponse {
  settings: DailyAutomationSettings;
  next_run_at: string | null;
  last_run_at: string | null;
  last_run_date: string | null;
  last_task_id: string | null;
}

export interface DailyNewsSource {
  id: string;
  language: 'en' | 'zh';
  name: string;
  type: string;
  coverage: string;
  frequency: string;
  rating: number;
  priority: number;
  homepage: string;
  feed_url: string | null;
  fetch_mode: string;
  enabled: boolean;
}

export interface PublicationEntry {
  status: 'published' | 'deleted' | 'feed_ready';
  url: string;
  test_mode?: boolean;
  visibility?: 'private' | 'unlisted' | 'public';
  identity?: Record<string, string> | null;
  published_at?: string;
  deleted_at?: string;
  sha256?: string;
  matches_current_media?: boolean;
}

export interface PublicationManifest {
  task_id: string;
  updated_at?: string;
  current_video_sha256?: string | null;
  current_audio_sha256?: string | null;
  platforms: Partial<Record<'youtube' | 'x' | 'apple_podcast', PublicationEntry>>;
}

/** One runtime setting (formerly a .env variable), as described by the backend. */
export type SettingValue = string | number | boolean;

export interface SettingField {
  key: string;
  label: string;
  type: 'string' | 'secret' | 'int' | 'bool' | 'choice' | 'path';
  description: string;
  placeholder: string;
  unit: string;
  options: string[];
  value: SettingValue;
  default: SettingValue;
  is_overridden: boolean;
  restart_required: boolean;
  allow_blank: boolean;
  /** Secrets only — the value itself is never sent to the client. */
  masked?: string | null;
  default_masked?: string | null;
  is_set?: boolean | null;
}

export interface SettingGroup {
  id: string;
  label: string;
  description: string;
  fields: SettingField[];
}

export interface SettingsSchema {
  groups: SettingGroup[];
  /** Keys whose last change needs a restart to fully take effect. */
  restart_required: string[];
}

export interface Prompt {
  key: string;
  file: string;
  label: string;
  stage: string;
  description: string;
  variables: string[];
  has_default: boolean;
  content: string;
  is_modified: boolean;
  missing_variables: string[];
}

export interface Skill {
  name: string;
  slug: string;
  description: string;
  path: string;
  is_symlink: boolean;
  enabled: boolean;
  body?: string;
}

export type AccountAutomationExecutor = 'opencode' | 'claude_sdk' | 'pipeline';
export type AccountAutomationFeature = 'today_in_history' | 'x_engagement';
export type AccountRunStatus = 'queued' | 'planning' | 'generating_image' | 'publishing' | 'published' | 'failed';

export interface AccountOpsStatus {
  worker_alive: boolean;
  last_tick_at: string | null;
  poll_interval: number;
}

export interface AccountAutomation {
  id: string;
  created_at: string;
  updated_at: string;
  name: string;
  feature_type: AccountAutomationFeature;
  platform: 'x';
  account_handle: string;
  enabled: boolean;
  schedule_time: string;
  schedule_times: string[];
  timezone: string;
  prompt_template: string;
  reply_style_prompt: string;
  max_replies: number;
  max_quote_reposts: number;
  scan_limit: number;
  executor: AccountAutomationExecutor;
  opencode_model: string;
  next_run_at: string | null;
  last_run_at: string | null;
}

export type AccountAutomationUpdate = Pick<
  AccountAutomation,
  'name' | 'account_handle' | 'enabled' | 'schedule_time' | 'schedule_times' | 'timezone' | 'prompt_template' | 'reply_style_prompt' | 'max_replies' | 'max_quote_reposts' | 'scan_limit' | 'executor' | 'opencode_model'
>;

export interface AccountRun {
  id: string;
  automation_id: string;
  automation_name: string;
  feature_type: AccountAutomationFeature;
  account_handle: string;
  platform: 'x';
  trigger: 'manual' | 'scheduled';
  status: AccountRunStatus;
  scheduled_for: string | null;
  event_date: string;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  title: string | null;
  post_text: string | null;
  image_path: string | null;
  chatgpt_conversation_url: string | null;
  post_url: string | null;
  external_post_id: string | null;
  executor: AccountAutomationExecutor;
  content: Record<string, unknown>;
  error_message: string | null;
  log_text: string;
}

export async function fetchTasks(): Promise<Task[]> {
  const res = await fetch(`${BASE}/api/tasks`);
  const data = await res.json();
  return data.tasks;
}

export async function fetchTask(id: string): Promise<Task> {
  const res = await fetch(`${BASE}/api/tasks/${id}`);
  if (!res.ok) throw new Error('Task not found');
  return res.json();
}

/** Move a queued task's start time; pass null to release it immediately. */
export async function scheduleTask(taskId: string, scheduledAt: string | null): Promise<Task> {
  const res = await fetch(`${BASE}/api/tasks/${taskId}/schedule`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scheduled_at: scheduledAt }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteTask(id: string): Promise<void> {
  const res = await fetch(`${BASE}/api/tasks/${id}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(await res.text());
}

export async function fetchSettings(): Promise<Settings> {
  const res = await fetch(`${BASE}/api/settings`);
  return res.json();
}

// ── Runtime settings ─────────────────────────────────────────────────
async function settingsRequest(path: string, init: RequestInit): Promise<SettingsSchema> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    const payload = (await res.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? `Request failed (${res.status})`);
  }
  return res.json();
}

export async function fetchSettingsSchema(): Promise<SettingsSchema> {
  return settingsRequest('/api/settings/schema', { method: 'GET' });
}

/** Persist and apply the given settings. Only the submitted keys change. */
export async function updateSettingsValues(
  values: Record<string, SettingValue>,
): Promise<SettingsSchema> {
  return settingsRequest('/api/settings/values', {
    method: 'PUT',
    body: JSON.stringify({ values }),
  });
}

/** Drop saved overrides, restoring the values the process started with. */
export async function resetSettingsValues(keys: string[]): Promise<SettingsSchema> {
  return settingsRequest('/api/settings/values/reset', {
    method: 'POST',
    body: JSON.stringify({ keys }),
  });
}

export interface VoiceOption {
  name: string;
  gender: string;
  lang: string;
  resolved_name: string;
  preview_available: boolean;
}

export interface TtsModelOption {
  id: string;
  label: string;
  provider: string;
  single_speaker: boolean;
  is_default: boolean;
}

export async function fetchTtsModels(): Promise<TtsModelOption[]> {
  const res = await fetch(`${BASE}/api/voices/models`);
  if (!res.ok) throw new Error('Failed to load TTS models');
  return res.json();
}

export async function fetchVoices(ttsModel?: string): Promise<VoiceOption[]> {
  const qs = ttsModel ? `?tts_model=${encodeURIComponent(ttsModel)}` : '';
  const res = await fetch(`${BASE}/api/voices${qs}`);
  if (!res.ok) throw new Error('Failed to load voices');
  return res.json();
}

export async function fetchProviders(): Promise<Provider[]> {
  const res = await fetch(`${BASE}/api/providers`);
  const data = await res.json();
  return data.providers;
}

export async function createProvider(input: ProviderInput): Promise<Provider> {
  const res = await fetch(`${BASE}/api/providers`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function updateProvider(id: number, input: Partial<ProviderInput>): Promise<Provider> {
  const res = await fetch(`${BASE}/api/providers/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteProvider(id: number): Promise<void> {
  await fetch(`${BASE}/api/providers/${id}`, { method: 'DELETE' });
}

export async function testProvider(input: ProviderTestRequest): Promise<ProviderTestResult> {
  const res = await fetch(`${BASE}/api/providers/test`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ── Prompts ──────────────────────────────────────────────────────────
export async function fetchPrompts(): Promise<Prompt[]> {
  const res = await fetch(`${BASE}/api/prompts`);
  const data = await res.json();
  return data.prompts;
}

export async function updatePrompt(key: string, content: string): Promise<Prompt> {
  const res = await fetch(`${BASE}/api/prompts/${key}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function resetPrompt(key: string): Promise<Prompt> {
  const res = await fetch(`${BASE}/api/prompts/${key}/reset`, { method: 'POST' });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// ── Skills ───────────────────────────────────────────────────────────
export async function fetchSkills(): Promise<Skill[]> {
  const res = await fetch(`${BASE}/api/skills`);
  const data = await res.json();
  return data.skills;
}

export async function fetchSkill(name: string): Promise<Skill> {
  const res = await fetch(`${BASE}/api/skills/${encodeURIComponent(name)}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function updateSkill(
  name: string,
  input: { enabled?: boolean; body?: string },
): Promise<Skill> {
  const res = await fetch(`${BASE}/api/skills/${encodeURIComponent(name)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function uploadSkill(file: File): Promise<Skill> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/api/skills/upload`, {
    method: 'POST',
    body: form,
  });
  if (!res.ok) {
    const payload = await res.json().catch(() => null) as { detail?: string } | null;
    throw new Error(payload?.detail ?? `Skill upload failed (${res.status})`);
  }
  return res.json();
}

// ── Account operations ─────────────────────────────────────────────
export async function fetchAccountOpsStatus(): Promise<AccountOpsStatus> {
  const res = await fetch(`${BASE}/api/account-operations/status`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchAccountAutomations(): Promise<AccountAutomation[]> {
  const res = await fetch(`${BASE}/api/account-operations/automations`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json() as { automations: AccountAutomation[] };
  return data.automations;
}

export async function updateAccountAutomation(
  id: string,
  input: Partial<AccountAutomationUpdate>,
): Promise<AccountAutomation> {
  const res = await fetch(`${BASE}/api/account-operations/automations/${encodeURIComponent(id)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function runAccountAutomation(id: string): Promise<AccountRun> {
  const res = await fetch(`${BASE}/api/account-operations/automations/${encodeURIComponent(id)}/run`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function pauseAccountAutomation(id: string): Promise<AccountAutomation> {
  const res = await fetch(`${BASE}/api/account-operations/automations/${encodeURIComponent(id)}/pause`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function resumeAccountAutomation(id: string): Promise<AccountAutomation> {
  const res = await fetch(`${BASE}/api/account-operations/automations/${encodeURIComponent(id)}/resume`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchAccountRuns(): Promise<AccountRun[]> {
  const res = await fetch(`${BASE}/api/account-operations/runs`);
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json() as { runs: AccountRun[] };
  return data.runs;
}

export async function fetchAccountRun(id: string): Promise<AccountRun> {
  const res = await fetch(`${BASE}/api/account-operations/runs/${encodeURIComponent(id)}`);
  if (!res.ok) throw new Error('Account-operation run not found');
  return res.json();
}

export function accountRunImageUrl(id: string): string {
  return `${BASE}/api/account-operations/runs/${encodeURIComponent(id)}/image`;
}

export async function fetchScript(taskId: string): Promise<string> {
  const res = await fetch(scriptUrl(taskId));
  if (!res.ok) throw new Error(await res.text());
  return res.text();
}

export async function updateScript(taskId: string, content: string): Promise<void> {
  const res = await fetch(scriptUrl(taskId), {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content }),
  });
  if (!res.ok) throw new Error(await res.text());
}

export async function regenerateTask(taskId: string): Promise<Task> {
  const res = await fetch(`${BASE}/api/tasks/${taskId}/regenerate`, { method: 'POST' });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function renderTask(taskId: string): Promise<Task> {
  const res = await fetch(`${BASE}/api/tasks/${taskId}/render`, { method: 'POST' });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function fetchFootage(taskId: string): Promise<FootageManifest> {
  const res = await fetch(`${BASE}/api/tasks/${taskId}/footage`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function acquireFootage(taskId: string, queries: string[] = []): Promise<Task> {
  const res = await fetch(`${BASE}/api/tasks/${taskId}/footage/acquire`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ queries }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function footageFileUrl(taskId: string, clipId: string): string {
  return `${BASE}/api/tasks/${taskId}/footage/${clipId}/file`;
}

export type VideoArtifactState = 'final' | 'retained';

export function videoUrl(taskId: string, artifact: VideoArtifactState): string {
  return `${BASE}/api/tasks/${taskId}/video?artifact=${artifact}`;
}

export function audioUrl(taskId: string): string {
  return `${BASE}/api/tasks/${taskId}/audio`;
}

export function scriptUrl(taskId: string): string {
  return `${BASE}/api/tasks/${taskId}/script`;
}

export function thumbnailUrl(taskId: string): string {
  return `${BASE}/api/tasks/${taskId}/thumbnail`;
}

export function thumbnailPromptUrl(taskId: string): string {
  return `${BASE}/api/tasks/${taskId}/thumbnail/prompt`;
}

export function logsStreamUrl(taskId: string): string {
  return `${BASE}/api/tasks/${taskId}/logs/stream`;
}

async function dailyRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}/api/daily-news${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) {
    const payload = await res.json().catch(() => null) as { detail?: string } | null;
    throw new Error(payload?.detail ?? `Daily news request failed (${res.status})`);
  }
  return res.json();
}

export function fetchDailyAutomation(): Promise<DailyAutomationResponse> {
  return dailyRequest('');
}

export function updateDailyAutomation(settings: DailyAutomationSettings): Promise<DailyAutomationResponse> {
  return dailyRequest('', { method: 'PUT', body: JSON.stringify(settings) });
}

export async function fetchDailySources(): Promise<DailyNewsSource[]> {
  const payload = await dailyRequest<{ sources: DailyNewsSource[] }>('/sources');
  return payload.sources;
}

export function runDailyNow(testMode = true, durationMinutes: number | null = 1): Promise<Task> {
  const params = new URLSearchParams({
    test_mode: String(testMode),
  });
  if (durationMinutes !== null) params.set('duration_minutes', String(durationMinutes));
  return dailyRequest(`/run-now?${params}`, { method: 'POST' });
}

export async function fetchPublications(taskId: string): Promise<PublicationManifest> {
  const res = await fetch(`${BASE}/api/tasks/${taskId}/publications`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function publishTestEpisode(
  taskId: string,
  options: {
    targets?: Array<'youtube' | 'x' | 'apple_podcast'>
    testMode?: boolean
    visibility?: 'private' | 'unlisted' | 'public'
  } = {},
): Promise<PublicationManifest> {
  const res = await fetch(`${BASE}/api/tasks/${taskId}/publications`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      targets: options.targets || ['youtube', 'x'],
      test_mode: options.testMode ?? true,
      visibility: options.visibility || 'private',
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function reconcilePublicationSafetyHold(
  taskId: string,
  confirmedDeletedTestTargets: Array<'youtube' | 'x'> = [],
): Promise<Task> {
  const res = await fetch(`${BASE}/api/tasks/${taskId}/publications/reconcile`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      acknowledge_no_unrecorded_publication: true,
      confirmed_deleted_test_targets: confirmedDeletedTestTargets,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteTestEpisode(taskId: string): Promise<PublicationManifest> {
  const res = await fetch(`${BASE}/api/tasks/${taskId}/publications/delete-test`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ targets: ['youtube', 'x'] }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

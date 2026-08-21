import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  fetchTask,
  deleteTask,
  fetchScript,
  updateScript,
  regenerateTask,
  renderTask,
  scheduleTask,
  videoUrl,
  audioUrl,
  scriptUrl,
  thumbnailUrl,
  thumbnailPromptUrl,
} from '../api'
import LogPanel from './LogPanel'
import FootagePanel from './FootagePanel'
import PublicationPanel from './PublicationPanel'
import { IconChevronLeft } from './Icons'
import { countdown, formatStart, isPendingStart, localInputToIso, toLocalInputValue } from '../schedule'

const STAGES = ['researching', 'digesting', 'reviewing', 'sourcing', 'tts', 'music', 'composing', 'publishing', 'complete'] as const
const STAGE_LABELS: Record<string, string> = {
  researching: 'Research',
  extracting: 'Extract',
  digesting: 'Digest',
  titling: 'Title',
  reviewing: 'Fact check',
  sourcing: 'Footage',
  tts: 'TTS',
  awaiting_review: 'Review',
  music: 'Music',
  composing: 'Compose',
  publishing: 'Publish',
  complete: 'Done',
}

function stageState(current: string, stage: string): 'done' | 'active' | '' {
  const ci = STAGES.indexOf(current as typeof STAGES[number])
  const si = STAGES.indexOf(stage as typeof STAGES[number])
  if (ci < 0) return ''
  if (si < ci) return 'done'
  if (si === ci) return current === 'complete' ? 'done' : 'active'
  return ''
}

function retainedVideoWarning(status: string, publicationSafetyHold: boolean): string {
  if (publicationSafetyHold) {
    return 'Publication stopped with an indeterminate result after this cut was validated. The cut remains retained while the task is failed; verify every destination before taking further action.'
  }
  if (status === 'failed') {
    return 'This task did not complete. The download is the last validated cut retained before failure; it may belong to an earlier render and is not proof that the failed attempt produced a final result.'
  }
  if (status === 'queued') {
    return 'A new attempt is queued. The download remains the last validated cut until the task completes.'
  }
  if (status === 'awaiting_review') {
    return 'This task is awaiting review. The download is the last validated cut, not the current draft.'
  }
  if (status === 'publishing') {
    return 'Publishing is still in progress. This validated cut remains retained until the task reaches Complete.'
  }
  return 'This task is still processing. The download remains the last validated cut until the task completes.'
}

export default function TaskDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: task, isLoading } = useQuery({
    queryKey: ['task', id],
    queryFn: () => fetchTask(id!),
    enabled: !!id,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status && !['complete', 'failed', 'awaiting_review'].includes(status) ? 2000 : false
    },
  })

  const hasScript = !!task?.script_path
  const { data: scriptText } = useQuery({
    queryKey: ['script', id],
    queryFn: () => fetchScript(id!),
    enabled: !!id && hasScript,
  })

  const [draftOverride, setDraftOverride] = useState<string | null>(null)
  const draft = draftOverride ?? scriptText ?? ''
  const dirty = draftOverride !== null && draftOverride !== scriptText

  const [startOverride, setStartOverride] = useState<string | null>(null)
  const [titleCopied, setTitleCopied] = useState(false)

  const deleteMutation = useMutation({
    mutationFn: () => deleteTask(id!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
      navigate('/')
    },
  })

  const saveMutation = useMutation({
    mutationFn: () => updateScript(id!, draft),
    onSuccess: () => {
      setDraftOverride(null)
      queryClient.invalidateQueries({ queryKey: ['script', id] })
      queryClient.invalidateQueries({ queryKey: ['task', id] })
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
    },
  })

  const regenMutation = useMutation({
    mutationFn: () => regenerateTask(id!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['task', id] })
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
    },
  })

  const renderMutation = useMutation({
    mutationFn: () => renderTask(id!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['task', id] })
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
    },
  })

  const scheduleMutation = useMutation({
    mutationFn: (scheduledAt: string | null) => scheduleTask(id!, scheduledAt),
    onSuccess: () => {
      setStartOverride(null)
      queryClient.invalidateQueries({ queryKey: ['task', id] })
      queryClient.invalidateQueries({ queryKey: ['tasks'] })
    },
  })

  if (isLoading || !task) return <div className="empty-state">Loading...</div>

  const isRunning = !['complete', 'failed', 'queued', 'awaiting_review'].includes(task.status)
  const awaitingReview = task.status === 'awaiting_review'
  const publicationRecovery = task.status === 'failed' && task.publication_safety_hold
  const retryingFailedRender = task.status === 'failed' && !publicationRecovery && !!task.audio_path && hasScript
  const scriptLocked = publicationRecovery || isRunning || task.status === 'queued' || saveMutation.isPending || renderMutation.isPending || regenMutation.isPending
  // Parked in the queue behind a future start time — still cancellable.
  const parked = task.status === 'queued' && isPendingStart(task.scheduled_at)
  const startDraft = startOverride ?? (task.scheduled_at ? toLocalInputValue(new Date(task.scheduled_at)) : '')
  const startMoved = !!startOverride && localInputToIso(startDraft) !== task.scheduled_at

  return (
    <div className="detail-workspace">
      <header className="detail-command">
        <button
          type="button"
          className="icon-btn detail-back"
          aria-label="Back to tasks"
          title="Back to tasks"
          onClick={() => navigate('/')}
        >
          <IconChevronLeft />
        </button>

        <div className="detail-identity">
          <span className="eyebrow">
            {task.source_type.toUpperCase()} &middot; {task.id.slice(0, 8)}
          </span>
          <h1>{task.generated_title || task.source_title || task.id}</h1>
          {task.source_url && task.source_type !== 'topic' && <p className="detail-source">{task.source_url}</p>}
        </div>

        <div className="detail-command-side">
          <span className={`badge ${parked ? 'scheduled' : task.status}`}>
            {parked
              ? 'Scheduled'
              : task.status === 'tts'
                ? 'Generating Audio'
                : task.status === 'titling'
                  ? 'Generating Title'
                : task.status === 'awaiting_review'
                  ? 'Awaiting Review'
                  : task.status}
            {isRunning && ' ...'}
          </span>
          <div className="detail-quick-actions">
            {task.script_path && (
              <a href={scriptUrl(task.id)} target="_blank" rel="noopener">
                <button className="btn-ghost" type="button">Script</button>
              </a>
            )}
            {task.audio_path && (
              <a href={audioUrl(task.id)} target="_blank" rel="noopener">
                <button className="btn-ghost" type="button">Audio</button>
              </a>
            )}
            {task.video_path && task.video_artifact_state && (
              <a href={videoUrl(task.id, task.video_artifact_state)} download>
                <button className="btn-primary" type="button">
                  {task.video_artifact_state === 'final' ? 'Download Video' : 'Download Last Validated Cut'}
                </button>
              </a>
            )}
            {task.thumbnail_path && (
              <a href={thumbnailUrl(task.id)} download>
                <button className="btn-primary" type="button">Download Thumbnail</button>
              </a>
            )}
            <button
              className="btn-danger"
              type="button"
              disabled={publicationRecovery || deleteMutation.isPending || isRunning || renderMutation.isPending || regenMutation.isPending}
              title={publicationRecovery ? 'Resolve the publication safety hold before deleting this task' : isRunning ? 'Task cannot be deleted while it is processing' : ''}
              onClick={() => { if (confirm('Delete this task?')) deleteMutation.mutate() }}
            >
              Delete
            </button>
          </div>
        </div>
      </header>

      {deleteMutation.isError && (
        <div className="error-box">{(deleteMutation.error as Error).message}</div>
      )}

      {task.video_artifact_state === 'retained' && (
        <div className="retained-video-warning" role="status">
          <strong>Last validated cut retained</strong>
          <span>{retainedVideoWarning(task.status, task.publication_safety_hold)}</span>
        </div>
      )}

      <div className="detail-progress">
        <div className="pipeline-stages">
          {STAGES.map((s) => (
            <div key={s} className={`stage ${stageState(task.status, s)}`}>
              {STAGE_LABELS[s]}
            </div>
          ))}
        </div>
        <dl className="detail-facts">
          <div>
            <dt>Duration</dt>
            <dd>{task.config.target_duration_minutes} min</dd>
          </div>
          <div>
            <dt>Voices</dt>
            <dd>{task.config.voice_1} + {task.config.voice_2}</dd>
          </div>
          <div>
            <dt>Character</dt>
            <dd>{task.config.include_character ? 'On' : 'Off'}</dd>
          </div>
          <div>
            <dt>Captions</dt>
            <dd>{task.config.captions_enabled !== false ? 'On · single line' : 'Off'}</dd>
          </div>
          <div>
            <dt>Origin</dt>
            <dd>{task.origin_type === 'content_plan' ? 'Content plan' : task.origin_type === 'daily_news' ? 'Morning desk' : 'Manual task'}</dd>
          </div>
          <div>
            <dt>Target release</dt>
            <dd>{task.planned_publish_at ? formatStart(task.planned_publish_at) : 'Not planned'}</dd>
          </div>
          <div>
            <dt>Delivery</dt>
            <dd>{task.origin_type === 'daily_news' && task.config.auto_publish ? 'Automatic · identity gated' : 'Manual review'}</dd>
          </div>
          <div>
            <dt>Spoken ending</dt>
            <dd title={task.config.closing_remarks}>{task.config.closing_remarks || 'Default close'}</dd>
          </div>
        </dl>
      </div>

      <div className="detail-body">
        <section className="detail-column detail-main">
          {task.origin_type === 'content_plan' && (
            <div className="task-origin-banner">
              <div>
                <span className="eyebrow">Task source · retired editorial plan</span>
                <strong>{task.origin_label || task.origin_id}</strong>
                {task.source_type === 'topic' && task.source_url && <p>{task.source_url}</p>}
              </div>
              <span className="task-origin-retired">Legacy provenance retained</span>
            </div>
          )}
          {parked && (
            <div className="schedule-bar">
              <div className="schedule-bar-copy">
                <strong>Starts {formatStart(task.scheduled_at!)}</strong>
                <small>Held in the queue · {countdown(task.scheduled_at!)}</small>
              </div>
              <div className="schedule-bar-controls">
                <input
                  type="datetime-local"
                  aria-label="Start time"
                  value={startDraft}
                  min={toLocalInputValue(new Date())}
                  onInput={(e) => setStartOverride(e.currentTarget.value)}
                />
                <button
                  className="btn-ghost"
                  disabled={!startMoved || scheduleMutation.isPending}
                  onClick={() => scheduleMutation.mutate(localInputToIso(startDraft))}
                >
                  Move
                </button>
                <button
                  className="btn-primary"
                  disabled={scheduleMutation.isPending}
                  onClick={() => scheduleMutation.mutate(null)}
                >
                  {scheduleMutation.isPending ? 'Working…' : 'Start now'}
                </button>
              </div>
            </div>
          )}
          {scheduleMutation.isError && (
            <div className="error-box">{(scheduleMutation.error as Error).message}</div>
          )}

          {task.status === 'failed' && task.error_message && (
            <div className="error-box">{task.error_message}</div>
          )}

          {task.generated_title && (
            <section className="detail-panel title-result">
              <div className="title-result-head">
                <div>
                  <span className="eyebrow">Independent Agent output</span>
                  <h3>Publication title</h3>
                </div>
                <span className="title-result-mark">TITLE / READY</span>
              </div>
              <p className="title-result-copy">{task.generated_title}</p>
              <div className="title-result-footer">
                <div>
                  <span>Source title</span>
                  <strong>{task.source_title || 'Untitled source'}</strong>
                </div>
                <button
                  className="btn-ghost"
                  type="button"
                  onClick={async () => {
                    try {
                      await navigator.clipboard.writeText(task.generated_title!)
                      setTitleCopied(true)
                      window.setTimeout(() => setTitleCopied(false), 1600)
                    } catch {
                      setTitleCopied(false)
                    }
                  }}
                >
                  {titleCopied ? 'Copied' : 'Copy title'}
                </button>
              </div>
            </section>
          )}

          {task.thumbnail_path && (
            <section className="detail-panel thumbnail-result">
              <h3>Viral Thumbnail</h3>
              <img src={thumbnailUrl(task.id)} alt={`Thumbnail for ${task.source_title || task.id}`} />
              <div className="actions">
                <a href={thumbnailUrl(task.id)} download>
                  <button className="btn-primary" type="button">Download Thumbnail</button>
                </a>
                <a href={thumbnailPromptUrl(task.id)} target="_blank" rel="noopener">
                  <button className="btn-ghost" type="button">View Image Prompt</button>
                </a>
              </div>
            </section>
          )}

          {task.status === 'complete' && task.video_path && (
            <section className="detail-panel">
              <h3>Final Cut</h3>
              <video
                key={task.updated_at}
                controls
                src={`${videoUrl(task.id, 'final')}&v=${encodeURIComponent(task.updated_at)}`}
              />
              <div className="actions">
                <button
                  className="btn-ghost"
                  disabled={renderMutation.isPending || regenMutation.isPending || dirty}
                  title={dirty ? 'Save your changes first' : ''}
                  onClick={() => renderMutation.mutate()}
                >
                  {renderMutation.isPending ? 'Starting...' : 'Re-render Video'}
                </button>
              </div>
              {renderMutation.isError && (
                <div className="error-box" style={{ marginTop: 8 }}>
                  {(renderMutation.error as Error).message}
                </div>
              )}
            </section>
          )}

          {(task.status === 'complete' || publicationRecovery) && task.video_path && task.origin_type === 'daily_news' && (
            <PublicationPanel
              taskId={task.id}
              recoveryRequired={publicationRecovery}
            />
          )}

          {task.audio_path && task.status !== 'complete' && (
            <section className="detail-panel">
              <h3>Audio Preview</h3>
              <audio controls src={audioUrl(task.id)} />
              {(awaitingReview || retryingFailedRender) && (
                <>
                  <p className="detail-hint">
                    {retryingFailedRender
                      ? 'Retry only the video render with the saved script and audio.'
                      : 'Listen to the generated audio. Render the video to continue, or edit the script below and re-generate the audio.'}
                  </p>
                  <div className="actions">
                    <button
                      className="btn-primary"
                      disabled={renderMutation.isPending || regenMutation.isPending || dirty}
                      title={dirty ? 'Save your changes first' : ''}
                      onClick={() => renderMutation.mutate()}
                    >
                      {renderMutation.isPending
                        ? 'Starting...'
                        : retryingFailedRender
                          ? 'Retry Video Render'
                          : 'Render Video'}
                    </button>
                  </div>
                  {renderMutation.isError && (
                    <div className="error-box" style={{ marginTop: 8 }}>
                      {(renderMutation.error as Error).message}
                    </div>
                  )}
                </>
              )}
            </section>
          )}

          {hasScript && (
            <section className="detail-panel">
              <h3>Script</h3>
              <textarea
                className="script-editor"
                value={draft}
                onChange={(e) => setDraftOverride(e.target.value)}
                disabled={scriptLocked}
                spellCheck={false}
              />
              <div className="actions">
                <button
                  className="btn-ghost"
                  disabled={!dirty || saveMutation.isPending || scriptLocked}
                  title={scriptLocked ? 'Script is locked while the task is processing' : ''}
                  onClick={() => saveMutation.mutate()}
                >
                  {saveMutation.isPending ? 'Saving...' : 'Save Script'}
                </button>
                <button
                  className="btn-primary"
                  disabled={publicationRecovery || isRunning || task.status === 'queued' || dirty || regenMutation.isPending || renderMutation.isPending}
                  title={publicationRecovery ? 'Resolve the publication safety hold first' : dirty ? 'Save your changes first' : isRunning ? 'Task is processing' : ''}
                  onClick={() => regenMutation.mutate()}
                >
                  {regenMutation.isPending ? 'Starting...' : 'Re-generate Audio'}
                </button>
              </div>
              {regenMutation.isError && (
                <div className="error-box" style={{ marginTop: 8 }}>
                  {(regenMutation.error as Error).message}
                </div>
              )}
            </section>
          )}

          {task.config.footage_enabled && (
            <FootagePanel task={task} />
          )}
        </section>

        <aside className="detail-column detail-rail">
          <LogPanel taskId={task.id} taskStatus={task.status} fill />
        </aside>
      </div>
    </div>
  )
}

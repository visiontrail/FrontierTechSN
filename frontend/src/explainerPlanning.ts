import type { ExplainerForm, ExplainerProject } from './explainerApi'
import { isExplainerTask } from './explainerTasks.ts'

export function canQueueExplainer(form: Pick<ExplainerForm, 'title' | 'research' | 'scenes' | 'target_minutes'>) {
  return !!form.title.trim()
    && Number.isInteger(form.target_minutes) && form.target_minutes >= 1 && form.target_minutes <= 60
    && (form.research.trim().length >= 40 || form.scenes.length > 0)
    && form.scenes.every((scene) => !!scene.narration.trim() && scene.visual_prompt.trim().length >= 3)
}

export function selectTopicPlans(projects: ExplainerProject[] = []) {
  return projects.filter((project) => !isExplainerTask(project))
    .sort((a, b) => b.updated_at.localeCompare(a.updated_at))
}

export function selectQueuedExplainers(projects: ExplainerProject[] = []) {
  return projects.filter((project) => project.status === 'queued')
    .sort((a, b) => (a.queue?.position ?? Number.MAX_SAFE_INTEGER) - (b.queue?.position ?? Number.MAX_SAFE_INTEGER)
      || (a.queue?.enqueued_at ?? a.updated_at).localeCompare(b.queue?.enqueued_at ?? b.updated_at)
      || a.id.localeCompare(b.id))
}

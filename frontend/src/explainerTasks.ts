import type { ExplainerProject } from './explainerApi'

/** Saved topics become Tasks when submitted or when generation has left evidence. */
export function isExplainerTask(project: ExplainerProject): boolean {
  if (project.status !== 'draft' && project.status !== 'script_ready') return true
  return project.logs.length > 0
    || Object.values(project.artifacts).some((value) => Array.isArray(value) ? value.length > 0 : value != null && value !== '')
    || project.progress.completed > 0
    || project.progress.total > 0
    || !!project.progress.message.trim()
}

export function selectExplainerTasks(projects: ExplainerProject[] = []): ExplainerProject[] {
  return projects.filter(isExplainerTask)
}

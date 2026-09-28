import assert from 'node:assert/strict'
import test from 'node:test'
import type { ExplainerProject, ExplainerStatus } from '../src/explainerApi.ts'
import { isExplainerTask, selectExplainerTasks } from '../src/explainerTasks.ts'

function project(id: string, changes: Partial<ExplainerProject> = {}): ExplainerProject {
  return {
    id, status: 'draft', logs: [], artifacts: {}, progress: { completed: 0, total: 0, message: '' },
    ...changes,
  } as ExplainerProject
}

test('saved topics and manually supplied storyboards stay out of the production list', () => {
  assert.equal(isExplainerTask(project('research-only')), false)
  assert.equal(isExplainerTask(project('manual-scenes', { status: 'script_ready' })), false)
  assert.deepEqual(selectExplainerTasks(), [])
})

test('queued, running, delivered, and stopped productions all remain Tasks', () => {
  const statuses: ExplainerStatus[] = [
    'queued', 'writing_script', 'generating_audio', 'generating_images', 'generating_motion',
    'rendering', 'completed', 'failed', 'cancelled', 'interrupted',
  ]
  const projects = statuses.map((status) => project(status, { status }))
  assert.deepEqual(selectExplainerTasks(projects).map((item) => item.status), statuses)
  assert.equal(projects.length, statuses.length)
})

test('a generated script and an edited production remain reachable from Tasks', () => {
  assert.equal(isExplainerTask(project('script-done', {
    status: 'script_ready', progress: { completed: 4, total: 4, message: 'Script ready' },
  })), true)
  // Editing a previous production resets status/artifacts/progress, but preserves its logs.
  assert.equal(isExplainerTask(project('edited', { logs: ['Writing script'] })), true)
  assert.equal(isExplainerTask(project('audio-produced', {
    status: 'script_ready', artifacts: { audio_url: '/audio.wav' },
  })), true)
  assert.equal(isExplainerTask(project('empty-artifacts', { artifacts: { scene_artifacts: [] } })), false)
})

test('dequeue before production returns a research topic to the saved topic list', () => {
  const draft = project('topic')
  assert.equal(isExplainerTask({ ...draft, status: 'queued' }), true)
  assert.equal(isExplainerTask(draft), false)
})

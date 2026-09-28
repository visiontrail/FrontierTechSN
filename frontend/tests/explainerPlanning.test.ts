import assert from 'node:assert/strict'
import test from 'node:test'
import type { ExplainerForm, ExplainerProject } from '../src/explainerApi.ts'
import { canQueueExplainer, selectQueuedExplainers, selectTopicPlans } from '../src/explainerPlanning.ts'

const research = { title: 'A saved topic', research: 'A research question, source evidence, and a direction for the story.', scenes: [], target_minutes: 5 }
const scene = { id: 'scene-001', narration: 'An explanation.', visual_prompt: 'A visual explanation.', chapter: '', on_screen_text: '', motion: false }

test('research-only topics can queue a full video without generating scenes first', () => {
  assert.equal(canQueueExplainer(research), true)
  assert.equal(canQueueExplainer({ ...research, research: 'x'.repeat(40) }), true)
  assert.equal(canQueueExplainer({ ...research, research: ` ${'x'.repeat(39)} ` }), false)
  assert.equal(canQueueExplainer({ ...research, research: '', scenes: [scene] }), true)
})

test('queue eligibility protects incomplete stories while leaving them available as topic plans', () => {
  const invalid: Partial<ExplainerForm>[] = [
    { title: '  ' }, { research: 'short' }, { target_minutes: 0 }, { target_minutes: 61 }, { target_minutes: 1.5 },
    { scenes: [{ ...scene, narration: ' ' }] }, { scenes: [{ ...scene, visual_prompt: 'ab' }] },
  ]
  for (const change of invalid) assert.equal(canQueueExplainer({ ...research, ...change }), false, JSON.stringify(change))
})

function project(id: string, changes: Partial<ExplainerProject> = {}): ExplainerProject {
  return { ...research, id, status: 'draft', logs: [], artifacts: {}, progress: { completed: 0, total: 0, message: '' }, updated_at: '2026-09-28T01:00:00Z', ...changes } as ExplainerProject
}

test('topic plans contain only unsubmitted drafts and exclude production tasks', () => {
  const projects = [
    project('old', { updated_at: '2026-09-27T01:00:00Z' }), project('new'),
    project('script', { status: 'script_ready' }), project('retry', { status: 'failed' }),
    project('cancel', { status: 'cancelled' }), project('interrupt', { status: 'interrupted' }),
    project('waiting', { status: 'queued' }), project('active', { status: 'generating_audio' }),
    project('complete', { status: 'completed' }),
  ]
  const originalOrder = projects.map((item) => item.id)
  assert.deepEqual(selectTopicPlans(projects).map((item) => item.id), ['new', 'script', 'old'])
  assert.deepEqual(projects.map((item) => item.id), originalOrder)
  assert.deepEqual(selectTopicPlans(), [])
})

test('queue display follows server positions even when library update order differs', () => {
  const queue = (position: number | null, enqueued_at: string) => ({ operation: 'full' as const, position, enqueued_at })
  const items = [
    project('second', { status: 'queued', queue: queue(2, '2026-09-28T02:00:00Z') }),
    project('draft'), project('first', { status: 'queued', queue: queue(1, '2026-09-28T01:00:00Z') }),
    project('unranked-later', { status: 'queued', queue: queue(null, '2026-09-28T04:00:00Z') }),
    project('unranked-earlier', { status: 'queued', queue: queue(null, '2026-09-28T03:00:00Z') }),
  ]
  assert.deepEqual(selectQueuedExplainers(items).map((item) => item.id), ['first', 'second', 'unranked-earlier', 'unranked-later'])
  assert.equal(items[0].id, 'second')
  assert.deepEqual(selectQueuedExplainers(), [])
})

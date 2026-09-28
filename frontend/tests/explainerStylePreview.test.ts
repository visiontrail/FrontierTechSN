import assert from 'node:assert/strict'
import test from 'node:test'
import type { ExplainerProject, ExplainerStyle } from '../src/explainerApi.ts'
import { selectStylePreviews, sameExplainerVisualStyle } from '../src/explainerStylePreview.ts'

const style: ExplainerStyle = {
  background_color: '#F4EBDD', ink_color: '#302A25', accent_color: '#C56545',
  character_description: 'A round-headed character in a terracotta shirt.',
  visual_direction: 'Warm textured paper and charcoal outlines.', writing_tone: 'Clear and practical.',
  motion_mode: 'none', gemini_watermark_removal: true,
  gemini_watermark_region: { x: 87.5, y: 77.78, width: 6.25, height: 11.11 },
  motion_every_n_scenes: 0, seconds_per_scene: 4,
}

function project(id: string, changes: Partial<ExplainerProject> = {}): ExplainerProject {
  return {
    id, title: `Project ${id}`, status: 'completed', revision: 1,
    created_at: '2026-09-28T00:00:00Z', updated_at: '2026-09-28T00:00:00Z',
    research: '', language: 'en', target_minutes: 1, tts_model: 'pocket-tts-en', voice: 'alba',
    captions_enabled: false, style: { ...style }, scenes: [], error: null, logs: [],
    progress: { completed: 1, total: 1, message: 'Complete' },
    artifacts: { scene_artifacts: [{ id: 'scene-001', image_path: 'artifacts/scene-001/image/hash/image.png' }] },
    ...changes,
  }
}

test('the current project artwork is preferred over matching library artwork', () => {
  const current = project('current')
  const result = selectStylePreviews(style, current, [project('other'), current])
  assert.deepEqual(result.map((item) => [item.projectId, item.isCurrentProject]), [['current', true], ['other', false]])
  assert.equal(result[0].projectTitle, current.title)
  assert.equal(result[0].sceneId, 'scene-001')
  assert.equal(result[0].imageUrl, '/api/explainers/current/assets/artifacts/scene-001/image/hash/image.png')
})

test('new or unfinished projects can use artwork from a completed project with the same visual style', () => {
  const library = [project('other')]
  assert.equal(selectStylePreviews(style, undefined, library)[0].projectId, 'other')
  assert.equal(selectStylePreviews(style, project('draft', { status: 'draft', artifacts: {} }), library)[0].projectId, 'other')
})

test('each visual field invalidates previously generated artwork', () => {
  for (const field of ['background_color', 'ink_color', 'accent_color', 'character_description', 'visual_direction'] as const) {
    const changed = { ...style, [field]: `${style[field]} changed` }
    assert.equal(sameExplainerVisualStyle(style, changed), false, field)
    assert.deepEqual(selectStylePreviews(changed, project('current'), [project('other')]), [], field)
  }
})

test('restoring a color through the color input matches uppercase saved colors', () => {
  const saved = project('current')
  for (const field of ['background_color', 'ink_color', 'accent_color'] as const) {
    const changed = { ...style, [field]: '#112233' }
    assert.deepEqual(selectStylePreviews(changed, saved), [], field)
    const restored = { ...changed, [field]: style[field].toLowerCase() }
    assert.equal(sameExplainerVisualStyle(restored, style), true, field)
    assert.equal(selectStylePreviews(restored, saved)[0].projectId, 'current', field)
  }
})

test('character descriptions and art directions remain case-sensitive exact matches', () => {
  for (const field of ['character_description', 'visual_direction'] as const) {
    for (const value of [style[field].toLowerCase(), `${style[field]} `]) {
      const changed = { ...style, [field]: value }
      assert.equal(sameExplainerVisualStyle(style, changed), false, field)
      assert.deepEqual(selectStylePreviews(changed, project('current')), [], field)
    }
  }
})

test('writing, motion, voice, captions and pacing changes do not change still illustration style', () => {
  const changed: ExplainerStyle = {
    ...style, writing_tone: 'Conversational.', motion_mode: 'gemini', seconds_per_scene: 12,
    motion_every_n_scenes: 4, gemini_watermark_removal: false,
    gemini_watermark_region: { x: 80, y: 80, width: 5, height: 5 },
  }
  const saved = project('other', { style: changed, voice: 'different', captions_enabled: true })
  assert.equal(sameExplainerVisualStyle(style, changed), true)
  assert.equal(selectStylePreviews(style, undefined, [saved]).length, 1)
})

test('empty, unfinished, failed, cleared and top-level stale artwork is never selected', () => {
  const stale = Object.assign(project('stale', { artifacts: {} }), {
    scene_artifacts: [{ id: 'scene-001', image_path: 'old.png' }],
  })
  assert.deepEqual(selectStylePreviews(style), [])
  assert.deepEqual(selectStylePreviews(style, undefined, [
    project('empty', { artifacts: {} }), project('draft', { status: 'draft' }),
    project('failed', { status: 'failed' }), project('running', { status: 'generating_images' }), stale,
  ]), [])
})

test('the current project never falls back to a stale same-id library entry', () => {
  const old = project('current')
  for (const changes of [
    { artifacts: {} }, { status: 'script_ready' as const },
    { style: { ...style, visual_direction: 'A different visual style.' } },
  ]) {
    assert.deepEqual(selectStylePreviews(style, project('current', changes), [old]), [])
  }
  assert.deepEqual(selectStylePreviews(style, undefined, [old], 'current'), [])
})

test('asset URLs encode each path segment without losing directory separators', () => {
  const saved = project('project id', { artifacts: { scene_artifacts: [
    { id: 'scene-001', image_path: 'artifacts/scene 01/手绘 #1 100%.png' },
  ] } })
  assert.equal(selectStylePreviews(style, saved)[0].imageUrl,
    '/api/explainers/project%20id/assets/artifacts/scene%2001/%E6%89%8B%E7%BB%98%20%231%20100%25.png')
})

test('unsafe paths and unsupported file types are rejected', () => {
  for (const path of [
    '', '/image.png', '../image.png', 'artifacts/../../image.png', 'artifacts/./image.png',
    'artifacts//image.png', 'artifacts\\image.png', 'https://example.com/image.png',
    'data:image/png;base64,image.png', 'artifacts/\u0000image.png', 'artifacts/image.mp4',
    'artifacts/image.png?download=1',
  ]) {
    const saved = project('current', { artifacts: { scene_artifacts: [{ id: 'scene-001', image_path: path }] } })
    assert.deepEqual(selectStylePreviews(style, saved), [], path)
  }
})

test('missing and duplicate image paths do not produce candidates', () => {
  const saved = project('current', { artifacts: { scene_artifacts: [
    { id: 'scene-001' }, { id: 'scene-002', image_path: 'image.png' },
    { id: 'scene-003', image_path: 'image.png' },
  ] } })
  assert.deepEqual(selectStylePreviews(style, saved).map((item) => item.sceneId), ['scene-002'])
})

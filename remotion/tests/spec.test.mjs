import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import {validateSpec} from '../src/spec.ts'
import {sceneState} from '../src/state.ts'
const sample = () => JSON.parse(fs.readFileSync(new URL('../examples/compute-flow.json', import.meta.url), 'utf8'))
test('validated spec is detached and preserves semantic metadata', () => {
  const source = sample(); source.copy = ['Source-bound supplied label']
  const spec = validateSpec(source); source.actors[0].position[0] = 12
  assert.equal(spec.actors[0].position[0], -2.5); assert.deepEqual(spec.copy, ['Source-bound supplied label'])
})
test('malformed and executable model fields fail closed', () => {
  for (const change of [s => s.url = 'https://bad', s => s.actors[0].code = 'alert(1)', s => s.actors[0].position[0] = NaN,
    s => s.actors[0].kind = 'html', s => s.actors[1].id = 'chip', s => s.links[0].to = 'missing',
    s => s.actors[0].motion_end = .1, s => s.width = 1279, s => s.duration_frames = 1801,
    s => s.camera.end_position = s.camera.position.map(v => -v), s => s.actors[1].explode = .5,
    s => s.actors[0].scale = [0, 1, 1], s => s.actors[0].color = 'url(http://bad)',
    s => s.camera.position = [0, .2, 0], s => s.links[0].particles = 13]) {
    const s = sample(); change(s); assert.throws(() => validateSpec(s))
  }
})
test('every supported semantic actor validates without external assets', () => {
  for (const kind of ['chip', 'rack', 'satellite', 'robot', 'data', 'object']) {
    const s = sample(); s.actors = [{id: 'subject', kind, position: [0, 0, 0]}]; s.links = []
    assert.equal(validateSpec(s).actors[0].kind, kind)
  }
})
test('out of order seeking produces the exact same scene state', () => {
  const spec = validateSpec(sample())
  const expected = sceneState(spec, 45)
  for (const frame of [89, 0, 13, 88, 1]) sceneState(spec, frame)
  assert.deepEqual(sceneState(spec, 45), expected)
  assert.notDeepEqual(sceneState(spec, 0).camera.position, expected.camera.position)
  assert.equal(sceneState(spec, 89).actors[0].explode, .6)
})
test('all motion including camera, robots, and particles settles with supplied cue', () => {
  const s = sample(); s.actors.forEach(a => {a.motion_end = .6})
  const spec = validateSpec(s)
  const early = sceneState(spec, 20); const settled = sceneState(spec, 55); const end = sceneState(spec, 89)
  assert.notDeepEqual(early.actors, settled.actors)
  assert.deepEqual(settled.actors, end.actors)
  assert.deepEqual(settled.camera, end.camera)
  assert.equal(settled.flowPhase, end.flowPhase)
})
test('frame-derived translation and rotation have finite exact endpoints', () => {
  const s = sample(); Object.assign(s.actors[0], {end_position: [1, 2, 3], end_rotation: [0, Math.PI, 0]})
  const spec = validateSpec(s); const a = sceneState(spec, 89).actors[0]
  assert.deepEqual(a.position, [1, 2, 3]); assert.deepEqual(a.rotation, [0, Math.PI, 0])
  assert.deepEqual(sceneState(spec, -99), sceneState(spec, 0))
  assert.deepEqual(sceneState(spec, 999), sceneState(spec, 89))
})

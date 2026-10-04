import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { patchSnapshotMediaTiming } from './patch-snapshot-media-timing.mjs'

const cli = new URL('../node_modules/hyperframes/dist/cli.js', import.meta.url)
const installed = readFileSync(cli, 'utf8')
const patched = patchSnapshotMediaTiming(installed)
const marker = '// FrontierTechSN: snapshot media uses composition mount offsets'

function selector(code, videos) {
  // Run the exact installed snapshot selector after an in-memory patch; never
  // execute the CLI entrypoint, open a browser, or modify its installed file.
  const snapshotStart = code.indexOf('async function captureSnapshots(')
  const start = code.indexOf('const active = await page.evaluate((t2) => {', snapshotStart)
  const end = code.indexOf('\n          }, time);', start)
  assert.ok(snapshotStart > 0 && start > snapshotStart && end > start)
  const body = code.slice(start + 'const active = await page.evaluate((t2) => {'.length, end)
  return new Function('document', `return (t2) => {${body}\n}`)({
    querySelectorAll: (query) => {
      assert.equal(query, 'video[data-start]')
      return videos
    },
  })
}

function parent(start, outer = null, attributes = ['data-composition-id']) {
  return { dataset: { start: String(start) }, parentElement: outer, hasAttribute: (name) => attributes.includes(name) }
}

function video({ start = 0, duration = 8.2, parentElement = null, ...other } = {}) {
  return {
    id: 'scene-02-shot-5-media', dataset: { start: String(start), duration: String(duration) },
    defaultPlaybackRate: 1, duration, src: 'http://localhost/three_d/clip.mp4', currentSrc: '',
    loop: false, parentElement, ...other,
  }
}

test('real scene-02 sample times select the intended 3D clip after adding its mount offset', () => {
  const clip = video({ start: 25.4, parentElement: parent(13.4, parent(0)) })
  const original = selector(installed.replace(marker, ''), [clip])
  const fixed = selector(patched, [clip])
  // The original installed code is unpatched in the regression fixture; keep
  // this test useful after postinstall has also applied the prepared patch.
  const originalStart = 25.4
  for (const time of [39.067, 42.4, 42.867, 45.4]) {
    if (!installed.includes(marker)) assert.equal(original(time).length, 0)
    assert.equal(time < originalStart + clip.duration, false)
    const [active] = fixed(time)
    assert.equal(active.id, clip.id)
    assert.ok(Math.abs(active.relTime - (time - 38.8)) < 1e-9)
  }
})

test('nested composition starts accumulate once; ordinary timed wrappers do not shift media', () => {
  const root = parent(0)
  const outer = parent(100, root, ['data-composition-file'])
  const inner = parent(7, outer, ['data-composition-id', 'data-composition-file'])
  const flattened = parent(999, inner, ['data-hf-inner-root'])
  const ordinary = parent(888, flattened, ['data-start'])
  const select = selector(patched, [video({ start: 3, duration: 5, parentElement: ordinary })])
  assert.equal(select(109.999).length, 0)
  assert.equal(select(110)[0].relTime, 0)
  assert.equal(select(114.999).length, 1)
  assert.equal(select(115).length, 0)
})

test('root-level media preserves half-open timing and media-local seek with playback rate', () => {
  const clip = video({
    duration: 20, defaultPlaybackRate: 2,
    dataset: { start: '4', duration: '5', playbackStart: '3', mediaStart: '1' },
  })
  const select = selector(patched, [clip])
  assert.equal(select(3.999).length, 0)
  assert.equal(select(4)[0].relTime, 3)
  assert.equal(select(6)[0].relTime, 7)
  assert.equal(select(9).length, 0)
})

test('nested loop playback keeps source offsets and wraps in source time', () => {
  const clip = video({
    parentElement: parent(20), duration: 6, loop: true, defaultPlaybackRate: 2,
    dataset: { start: '4', duration: '10', mediaStart: '2' },
  })
  const select = selector(patched, [clip])
  assert.equal(select(24)[0].relTime, 2)
  assert.equal(select(26)[0].relTime, 2)
  assert.equal(select(27)[0].relTime, 4)
  assert.equal(select(34).length, 0)
})

test('separate mount instances resolve independently with a nonzero root offset', () => {
  const root = parent(10)
  const first = video({ id: 'first', start: 2, duration: 5, parentElement: parent(20, root) })
  const second = video({ id: 'second', start: 2, duration: 5, parentElement: parent(22, root) })
  const select = selector(patched, [first, second])
  assert.deepEqual(select(32).map((item) => item.id), ['first'])
  assert.deepEqual(select(34).map((item) => item.id), ['first', 'second'])
  assert.deepEqual(select(37).map((item) => item.id), ['second'])
  assert.equal(select(39).length, 0)
})

test('unresolved timeline references fail closed instead of guessing a zero offset', () => {
  for (const start of ['after(scene-01)', '12 seconds', '', 'Infinity']) {
    const nested = selector(patched, [video({ parentElement: parent(start) })])
    assert.throws(() => nested(1), /requires resolved numeric timing/)
    const local = selector(patched, [video({ dataset: { start, duration: '5' } })])
    assert.throws(() => local(1), /requires resolved numeric timing/)
  }
})

test('nested mount duration and end clip activity without shifting the media source time', () => {
  const outer = parent(100)
  outer.dataset.duration = '5'
  const inner = parent(3, outer)
  inner.dataset.duration = '9'
  const select = selector(patched, [video({ start: 0, duration: 9, parentElement: inner })])
  assert.equal(select(103)[0].relTime, 0)
  assert.equal(select(104)[0].relTime, 1)
  assert.equal(select(105).length, 0)
  assert.equal(select(106).length, 0)
  outer.dataset.end = '104'
  assert.equal(select(103.5)[0].relTime, 0.5)
  assert.equal(select(104).length, 0)
  outer.dataset.end = 'after(other)'
  assert.throws(() => select(103), /requires resolved numeric timing/)
})

test('a clip starting before its mount preserves elapsed source time at the visible boundary', () => {
  const select = selector(patched, [video({ start: -2, duration: 5, parentElement: parent(10) })])
  assert.equal(select(9.999).length, 0)
  assert.equal(select(10)[0].relTime, 2)
  assert.equal(select(13).length, 0)
})

test('only the snapshot function and patch marker change; renderer and compiler stay identical', () => {
  const stripMarker = (source) => source.replace(`${marker}\n`, '')
  const source = stripMarker(installed)
  const result = stripMarker(patched)
  const start = 'async function captureSnapshots('
  const from = source.indexOf(start)
  const nextFunction = source.indexOf('\nasync function ', from + start.length)
  assert.ok(from > 0 && nextFunction > from)
  const tail = source.slice(nextFunction)
  assert.equal(result.slice(0, result.indexOf(start)), source.slice(0, from))
  assert.ok(result.endsWith(tail))
})

test('snapshot patch is idempotent and rejects unknown, partial or duplicated bundles', () => {
  assert.equal(patchSnapshotMediaTiming(patched), patched)
  assert.throws(() => patchSnapshotMediaTiming('unknown bundle'), /review the installed version/)
  assert.throws(() => patchSnapshotMediaTiming(patched.replace('let start = numericTiming(v2, "start", "0")', 'let start = 0')), /review the installed version/)
  assert.throws(() => patchSnapshotMediaTiming(patched.replace('t2 >= visibleStart', 't2 >= start')), /review the installed version/)
  assert.throws(() => patchSnapshotMediaTiming(patched.replace(marker, '')), /review the installed version/)
  assert.throws(() => patchSnapshotMediaTiming(patched + patched), /review the installed version/)
  assert.throws(() => patchSnapshotMediaTiming(`${marker}\n${patched}`), /review the installed version/)
  assert.equal(readFileSync(cli, 'utf8'), installed)
})

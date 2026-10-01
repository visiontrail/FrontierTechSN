import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { setTimeout as delay } from 'node:timers/promises'
import { test } from 'node:test'
import { captureWithCancellation, launchBrowserOnceMore } from './render-recovery.mjs'
import { patchRenderRecovery } from './patch-render-recovery.mjs'

const silent = () => {}
const launchError = new Error('Failed to launch the browser process:  Code: null\n\nstderr:\n')

test('retry a terminated launch once and return the real browser', async () => {
  let attempts = 0
  const waits = []
  const browser = { connected: true }
  const result = await launchBrowserOnceMore(async () => {
    if (++attempts === 1) throw launchError
    return browser
  }, { wait: async (ms) => waits.push(ms), log: silent })
  assert.equal(result, browser)
  assert.equal(attempts, 2)
  assert.deepEqual(waits, [1000])
})

test('a repeated launch failure remains a failure after exactly two attempts', async () => {
  let attempts = 0
  await assert.rejects(launchBrowserOnceMore(async () => {
    attempts++
    throw launchError
  }, { wait: async () => {}, log: silent }), (error) => error === launchError)
  assert.equal(attempts, 2)
})

for (const message of ['spawn ENOENT', 'Timed out after 120000 ms', 'Navigation failed', 'Frame capture failed']) {
  test(`do not retry ${message}`, async () => {
    let attempts = 0
    await assert.rejects(launchBrowserOnceMore(async () => {
      attempts++
      throw new Error(message)
    }, { wait: async () => assert.fail('unexpected retry'), log: silent }), { message })
    assert.equal(attempts, 1)
  })
}

test('successful workers retain their complete results', async () => {
  const result = await captureWithCancellation([0, 1], async (id) => ({ workerId: id, framesCaptured: 20 }), undefined, silent)
  assert.deepEqual(result, [{ workerId: 0, framesCaptured: 20 }, { workerId: 1, framesCaptured: 20 }])
})

test('worker failure cancels sibling capture and waits for its cleanup', async () => {
  let cleaned = false
  let captured = 0
  const results = await captureWithCancellation([0, 1], async (id, signal) => {
    if (id === 0) {
      await delay(5)
      return { workerId: id, error: launchError.message }
    }
    while (!signal.aborted && captured < 100) {
      captured++
      await delay(2)
    }
    await delay(10)
    cleaned = true
    return { workerId: id, framesCaptured: captured, error: signal.aborted ? 'Parallel worker cancelled' : undefined }
  }, undefined, silent)
  assert.ok(captured < 100)
  assert.equal(cleaned, true)
  assert.equal(results[0].error, launchError.message)
  assert.equal(results[1].error, 'Parallel worker cancelled')
})

test('unexpected rejection drains siblings before propagating the cause', async () => {
  let cleaned = false
  await assert.rejects(captureWithCancellation([0, 1], async (id, signal) => {
    if (id === 0) throw new Error('cannot create output directory')
    if (!signal.aborted) {
      await new Promise((resolve) => signal.addEventListener('abort', resolve, { once: true }))
    }
    await delay(10)
    cleaned = true
    return { workerId: id, error: 'Parallel worker cancelled' }
  }, undefined, silent), /cannot create output directory/)
  assert.equal(cleaned, true)
})

test('caller cancellation reaches all workers and an already aborted call starts none', async () => {
  const controller = new AbortController()
  const pending = captureWithCancellation([0, 1], async (id, signal) => {
    await new Promise((resolve) => signal.addEventListener('abort', resolve, { once: true }))
    return { workerId: id, error: 'Parallel worker cancelled' }
  }, controller.signal, silent)
  controller.abort()
  assert.equal((await pending).length, 2)
  await assert.rejects(captureWithCancellation([0], () => assert.fail('started after cancellation'), controller.signal, silent), { name: 'AbortError' })
})

test('patch the installed worker dispatcher, preserve error reporting and reject unknown bundles', async () => {
  const installed = readFileSync(new URL('../node_modules/hyperframes/dist/cli.js', import.meta.url), 'utf8')
  const patched = patchRenderRecovery(installed)
  assert.equal(patchRenderRecovery(patched), patched)
  assert.equal((patched.match(/frontierLaunchBrowser\(\(\) => ppt.launch/g) || []).length, 1)
  assert.throws(() => patchRenderRecovery('unknown bundle'), /expected 1 anchor/)
  const start = patched.indexOf('async function executeParallelCapture(')
  const end = patched.indexOf('\nasync function mergeWorkerFrames(', start)
  const dispatcher = new Function('frontierCaptureWorkers', 'executeWorkerTask', 'formatWorkerFailure', `${patched.slice(start, end)}; return executeParallelCapture`)(
    (tasks, execute, signal) => captureWithCancellation(tasks, execute, signal, silent),
    async (task, _url, _options, _hook, signal) => {
      if (task.workerId === 0) return { workerId: 0, error: launchError.message }
      await new Promise((resolve) => signal.addEventListener('abort', resolve, { once: true }))
      return { workerId: 1, error: 'Parallel worker cancelled' }
    },
    (result) => `Worker ${result.workerId}: ${result.error}`,
  )
  await assert.rejects(dispatcher('', '', [{ workerId: 0, startFrame: 0, endFrame: 10 }, { workerId: 1, startFrame: 10, endFrame: 20 }], {}, () => {}, undefined), /\[Parallel\] Capture failed: Worker 0: Failed to launch/)
})

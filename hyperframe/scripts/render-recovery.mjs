import { setTimeout as delay } from 'node:timers/promises'

// Retry only the observed process-exit error, before a capture session exists.
// Missing executables, timeouts, navigation and frame errors remain failures.
export async function launchBrowserOnceMore(launch, { wait = delay, log = console.warn } = {}) {
  try {
    return await launch()
  } catch (error) {
    if (!/^Failed to launch the browser process:\s*Code:\s*null\b/.test(error?.message ?? '')) {
      throw error
    }
    log('[Render:recovery] Browser exited during launch; retrying that launch once in 1000ms')
    await wait(1000)
    return await launch()
  }
}

// Workers return error records rather than rejecting. Cancel sibling frame
// loops as soon as such a record arrives, but drain every worker's cleanup.
export async function captureWithCancellation(tasks, execute, signal, log = console.warn) {
  signal?.throwIfAborted()
  const controller = new AbortController()
  const abort = () => controller.abort(signal.reason)
  signal?.addEventListener('abort', abort, { once: true })
  const started = performance.now()
  const cancel = (error) => {
    if (!controller.signal.aborted) {
      log(`[Render:recovery] Worker failed after ${Math.round(performance.now() - started)}ms; cancelling sibling capture: ${error}`)
      controller.abort()
    }
  }
  try {
    const settled = await Promise.allSettled(tasks.map(async (task) => {
      try {
        const result = await execute(task, controller.signal)
        if (result.error) cancel(result.error)
        return result
      } catch (error) {
        cancel(error)
        throw error
      }
    }))
    const rejection = settled.find((result) => result.status === 'rejected')
    if (rejection) throw rejection.reason
    return settled.map((result) => result.value)
  } finally {
    signal?.removeEventListener('abort', abort)
  }
}

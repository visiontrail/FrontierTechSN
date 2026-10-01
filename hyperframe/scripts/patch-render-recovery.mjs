const marker = '// FrontierTechSN: bounded browser launch recovery and sibling cancellation'
const source = 'import { launchBrowserOnceMore as frontierLaunchBrowser, captureWithCancellation as frontierCaptureWorkers } from "../../../scripts/render-recovery.mjs";'

function replaceOnce(code, before, after) {
  const count = code.split(before).length - 1
  if (count !== 1) {
    throw new Error(`HyperFrames render-recovery patch expected 1 anchor, found ${count}; review the installed version before rendering`)
  }
  return code.replace(before, after)
}

export function patchRenderRecovery(code) {
  if (code.includes(marker)) return code
  let patched = replaceOnce(code, `  let browser = await ppt.launch({
    headless: true,
    args: chromeArgs,
    defaultViewport: null,
    executablePath,
    timeout: browserTimeout,
    protocolTimeout
  });`, `  let browser = await frontierLaunchBrowser(() => ppt.launch({
    headless: true,
    args: chromeArgs,
    defaultViewport: null,
    executablePath,
    timeout: browserTimeout,
    protocolTimeout
  }));`)
  patched = replaceOnce(patched, `  const results = await Promise.all(
    tasks.map(
      (task) => executeWorkerTask(
        task,
        serverUrl,
        captureOptions,
        createBeforeCaptureHook,
        signal,
        onFrameCaptured,
        onFrameBuffer,
        config,
        parallel
      )
    )
  );`, `  const results = await frontierCaptureWorkers(
    tasks,
    (task, workerSignal) => executeWorkerTask(
      task,
      serverUrl,
      captureOptions,
      createBeforeCaptureHook,
      workerSignal,
      onFrameCaptured,
      onFrameBuffer,
      config,
      parallel
    ),
    signal
  );`)
  return patched.replace(/^(#![^\n]*\n)?/, (prefix) => `${prefix}${marker}\n${source}\n`)
}

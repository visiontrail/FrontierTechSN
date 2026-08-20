import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import vm from 'node:vm'

import { selectChatGPTModel } from './node_modules/@jackwener/opencli/clis/chatgpt/utils.js'
import {
  sendGeminiMessage,
  waitForGeminiResponse,
} from './node_modules/@jackwener/opencli/clis/gemini/utils.js'
import {
  uploadFrame,
  uploadFrames,
  submittedVideoPrompt,
} from './node_modules/@jackwener/opencli/clis/gemini/video.js'

function videoFrameFixture(t, names = ['first-frame.png']) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'opencli-video-upload-'))
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }))
  return names.map((name, index) => {
    const file = path.join(directory, name)
    fs.writeFileSync(file, Buffer.from(`frame-${index + 1}`))
    return file
  })
}

function videoUploadPage({
  native = 'success',
  cdp = false,
  clear = false,
  busyReads = 0,
  inputReadyAfterReads = 0,
  documentHasFocus = true,
  uploadBusy = false,
} = {}) {
  const actions = []
  let attachments = 0
  let remainingBusyReads = busyReads
  const discoveryReads = new Map()
  let selector = ''
  const page = {
    actions,
    async click(value) {
      actions.push(['click', value])
    },
    async wait(value) {
      actions.push(['wait', value])
    },
    async evaluate(script) {
      // Parse the exact inner script handed to Browser Bridge. `node --check`
      // only validates this test/module and misses template-literal unescaping
      // that can corrupt a regex before Runtime.evaluate sees it.
      new vm.Script(script)
      if (script.includes("input.setAttribute('data-opencli-video-upload-target'")) {
        const match = script.match(/const marker = ("(?:[^"\\]|\\.)*")/)
        const marker = match ? JSON.parse(match[1]) : 'missing-marker'
        const expectedMatch = marker.match(/^opencli-video-upload-(\d+)-/)
        const expectedCount = Number(expectedMatch?.[1] || 1)
        const reads = (discoveryReads.get(expectedCount) || 0) + 1
        discoveryReads.set(expectedCount, reads)
        actions.push(['discoverInput', reads, expectedCount])
        const requiredReads = Array.isArray(inputReadyAfterReads)
          ? Number(inputReadyAfterReads[expectedCount - 1] || 0)
          : inputReadyAfterReads
        if (reads <= requiredReads) {
          return {
            ok: false,
            inputs: [],
            documentHasFocus,
            busy: uploadBusy,
            busyCount: uploadBusy ? 1 : 0,
            button: {
              connected: true,
              disabled: false,
              visible: true,
            },
          }
        }
        selector = `[data-opencli-video-upload-target="${marker}"]`
        return {
          ok: true,
          selector,
          selected: {
            index: 0,
            name: 'Filedata',
            type: 'file',
            accept: 'image/*',
            disabled: false,
            connected: true,
            files: [],
          },
          inputCount: 1,
        }
      }
      if (script.includes("document.querySelectorAll('gem-media-attachment').length")) {
        const busy = remainingBusyReads > 0
        if (remainingBusyReads > 0) remainingBusyReads -= 1
        return {
          attachments,
          busy,
          busyCount: busy ? 1 : 0,
          inputFiles: clear || attachments === 0 ? [[]] : [[{ name: 'frame.png' }]],
          textTail: 'Videos Flash Landscape (16:9)',
        }
      }
      if (script.includes('const transfer = new DataTransfer()')) {
        actions.push(['DataTransfer', selector])
        if (!clear) attachments += 1
        return { ok: true, bytes: 7, fileCount: 1 }
      }
      if (script.includes("removeAttribute('data-opencli-video-upload-target')")) {
        actions.push(['cleanup', selector])
        return true
      }
      throw new Error(`Unexpected Gemini video upload script: ${String(script).slice(0, 120)}`)
    },
  }
  if (native) {
    page.setFileInput = async (files, target) => {
      actions.push(['setFileInput', files, target])
      if (native === 'error') throw new Error('Not allowed')
      if (!clear) attachments += 1
    }
  }
  if (cdp) {
    page.cdp = async (method, params = {}) => {
      actions.push(['cdp', method, params])
      if (cdp === 'error' && method === 'Runtime.evaluate') {
        throw new Error('CDP method not permitted: Runtime.evaluate')
      }
      if (method === 'Runtime.evaluate') return { result: { objectId: 'live-file-input' } }
      if (method === 'DOM.setFileInputFiles' && !clear) attachments += 1
      return {}
    }
  }
  return page
}

test('Gemini video uploads a live keyframe through the native file-input path and waits for idle', async (t) => {
  const [frame] = videoFrameFixture(t)
  const page = videoUploadPage({ native: 'success', busyReads: 1 })

  const result = await uploadFrame(page, frame, 1, {
    attachmentTimeoutMs: 1000,
    clearedIdleLimit: 2,
  })

  assert.equal(result.method, 'page.setFileInput')
  assert.equal(result.state.attachments, 1)
  assert.equal(result.state.busy, false)
  const native = page.actions.find(([action]) => action === 'setFileInput')
  assert.deepEqual(native[1], [frame])
  assert.match(native[2], /^\[data-opencli-video-upload-target=/)
  assert.equal(page.actions.some(([action]) => action === 'DataTransfer'), false)
  assert.ok(page.actions.filter(([action, value]) => action === 'wait' && value === 1).length >= 2)
})

test('Gemini video waits for a delayed live file input without opening a second chooser', async (t) => {
  const [frame] = videoFrameFixture(t)
  const page = videoUploadPage({ native: 'success', inputReadyAfterReads: 3 })

  const result = await uploadFrame(page, frame, 1, {
    attachmentTimeoutMs: 1000,
    inputReadyTimeoutMs: 5000,
    inputPollIntervalMs: 1000,
    inputReopenAfterMs: 4000,
  })

  assert.equal(result.method, 'page.setFileInput')
  assert.equal(page.actions.filter(([action]) => action === 'discoverInput').length, 4)
  assert.equal(page.actions.filter(([action]) => action === 'click').length, 1)
})

test('Gemini video retries the upload control at most once when hydration never creates an input', async (t) => {
  const [frame] = videoFrameFixture(t)
  const page = videoUploadPage({
    native: 'success',
    inputReadyAfterReads: Number.POSITIVE_INFINITY,
  })

  await assert.rejects(
    uploadFrame(page, frame, 1, {
      inputReadyTimeoutMs: 4000,
      inputPollIntervalMs: 1000,
      inputReopenAfterMs: 2000,
    }),
    (error) => {
      assert.match(error.message, /keyframe 1 upload failed at discover_live_input/)
      assert.match(error.message, /"pollAttempts":4/)
      assert.match(error.message, /"reopened":true/)
      assert.match(error.message, /"reason":"hydration_retry","ok":true/)
      assert.match(error.message, /"inputs":\[\]/)
      return true
    },
  )

  assert.equal(page.actions.filter(([action]) => action === 'discoverInput').length, 4)
  assert.equal(page.actions.filter(([action]) => action === 'click').length, 2)
})

test('Gemini video never re-clicks while a native chooser may hold page focus', async (t) => {
  const [frame] = videoFrameFixture(t)
  const page = videoUploadPage({
    native: 'success',
    inputReadyAfterReads: Number.POSITIVE_INFINITY,
    documentHasFocus: false,
  })

  await assert.rejects(
    uploadFrame(page, frame, 1, {
      inputReadyTimeoutMs: 3000,
      inputPollIntervalMs: 1000,
      inputReopenAfterMs: 1000,
    }),
    /keyframe 1 upload failed at discover_live_input/,
  )

  assert.equal(page.actions.filter(([action]) => action === 'click').length, 1)
})

test('Gemini video uses direct CDP file injection when the native page helper is unavailable', async (t) => {
  const [frame] = videoFrameFixture(t)
  const page = videoUploadPage({ native: null, cdp: true })

  const result = await uploadFrame(page, frame, 1, { attachmentTimeoutMs: 1000 })

  assert.equal(result.method, 'DOM.setFileInputFiles')
  const injection = page.actions.find(
    ([action, method]) => action === 'cdp' && method === 'DOM.setFileInputFiles',
  )
  assert.deepEqual(injection[2].files, [frame])
  assert.equal(injection[2].objectId, 'live-file-input')
  assert.equal(page.actions.some(([action]) => action === 'DataTransfer'), false)
})

test('Gemini video continues from a rejected native action to direct CDP injection', async (t) => {
  const [frame] = videoFrameFixture(t)
  const page = videoUploadPage({ native: 'error', cdp: true })

  const result = await uploadFrame(page, frame, 1, { attachmentTimeoutMs: 1000 })

  assert.equal(result.method, 'DOM.setFileInputFiles')
  assert.equal(page.actions.filter(([action]) => action === 'setFileInput').length, 1)
  assert.equal(
    page.actions.filter(
      ([action, method]) => action === 'cdp' && method === 'DOM.setFileInputFiles',
    ).length,
    1,
  )
  assert.equal(page.actions.some(([action]) => action === 'DataTransfer'), false)
})

test('Gemini video falls back to DataTransfer only when native injection fails', async (t) => {
  const [frame] = videoFrameFixture(t)
  const page = videoUploadPage({ native: 'error' })

  const result = await uploadFrame(page, frame, 1, { attachmentTimeoutMs: 1000 })

  assert.equal(result.method, 'DataTransfer')
  assert.equal(page.actions.filter(([action]) => action === 'setFileInput').length, 1)
  assert.equal(page.actions.filter(([action]) => action === 'DataTransfer').length, 1)
})

test('Gemini video fails closed with substep and input state when the UI clears a native upload', async (t) => {
  const [frame] = videoFrameFixture(t)
  const page = videoUploadPage({ native: 'success', clear: true })

  await assert.rejects(
    uploadFrame(page, frame, 1, { attachmentTimeoutMs: 1000, clearedIdleLimit: 2 }),
    (error) => {
      assert.match(error.message, /keyframe 1 upload failed at wait_for_attachment/)
      assert.match(error.message, /"method":"page\.setFileInput"/)
      assert.match(error.message, /"selector":"\[data-opencli-video-upload-target=/)
      assert.match(error.message, /"inputFiles":\[\[\]\]/)
      assert.match(error.message, /"clearedIdleSamples":2/)
      assert.doesNotMatch(error.message, /OPENCLI_CAPABILITY_UNAVAILABLE/)
      return true
    },
  )
  assert.equal(page.actions.some(([action]) => action === 'DataTransfer'), false)
})

test('Gemini video reports a stable upload capability code only after every path is blocked', async (t) => {
  const [frame] = videoFrameFixture(t)
  const page = videoUploadPage({ native: 'error', cdp: 'error', clear: true })

  await assert.rejects(
    uploadFrame(page, frame, 1, { attachmentTimeoutMs: 1000, clearedIdleLimit: 2 }),
    (error) => {
      assert.match(
        error.message,
        /OPENCLI_CAPABILITY_UNAVAILABLE:GEMINI_VIDEO_LOCAL_FILE_UPLOAD/,
      )
      assert.match(error.message, /"method":"DataTransfer"/)
      assert.match(error.message, /"method":"page\.setFileInput"/)
      assert.match(error.message, /-32000|Not allowed/i)
      assert.match(error.message, /CDP method not permitted: Runtime\.evaluate/)
      assert.match(error.message, /"attachments":0/)
      assert.match(error.message, /"inputFiles":\[\[\]\]/)
      return true
    },
  )
  assert.equal(page.actions.filter(([action]) => action === 'DataTransfer').length, 1)
})

test('Gemini video independently waits for first and delayed last keyframes in order', async (t) => {
  const frames = videoFrameFixture(t, ['first-frame.png', 'last-frame.jpg'])
  const page = videoUploadPage({ native: 'success', inputReadyAfterReads: [0, 2] })

  await uploadFrames(page, frames, {
    attachmentTimeoutMs: 1000,
    inputReadyTimeoutMs: 4000,
    inputPollIntervalMs: 1000,
    inputReopenAfterMs: 3000,
  })

  const uploads = page.actions.filter(([action]) => action === 'setFileInput')
  assert.deepEqual(uploads.map(([, files]) => files[0]), frames)
  assert.notEqual(uploads[0][2], uploads[1][2])
  assert.equal(page.actions.filter(([action]) => action === 'click').length, 2)
  assert.equal(page.actions.filter(([action]) => action === 'cleanup').length, 2)
  assert.deepEqual(
    page.actions
      .filter(([action, , expectedCount]) => action === 'discoverInput' && expectedCount === 2)
      .map(([, reads]) => reads),
    [1, 2, 3],
  )
})

test('Gemini video submitted-state page script compiles and executes after template expansion', async () => {
  let pathname = '/videos'
  let draft = ''
  let userQuery = false
  const page = {
    async evaluate(script) {
      return vm.runInNewContext(script, {
        document: {
          querySelector(selector) {
            if (selector === 'user-query') return userQuery ? {} : null
            if (selector.includes('[contenteditable="true"]')) {
              return { textContent: draft }
            }
            return null
          },
        },
        location: { pathname },
      })
    },
  }

  assert.equal(await submittedVideoPrompt(page), false)
  pathname = '/app/abc123'
  assert.equal(await submittedVideoPrompt(page), true)
  draft = 'prompt still present'
  assert.equal(await submittedVideoPrompt(page), false)
  pathname = '/videos'
  draft = ''
  userQuery = true
  assert.equal(await submittedVideoPrompt(page), true)
})

test('Gemini clicks a discovered semantic send button instead of pressing Enter', async () => {
  const actions = []
  let composerHasText = false
  const page = {
    async evaluate(script) {
      if (script === 'window.location.href') return 'https://gemini.google.com/app/test'
      if (script.includes('bestButton instanceof HTMLElement')) {
        return {
          action: 'button',
          label: 'Send message',
          x: 123,
          y: 456,
        }
      }
      if (script.includes('hasText: !!(composer')) {
        return { hasText: composerHasText }
      }
      if (script.includes('inputText') && script.includes('InputEvent')) {
        return { hasText: composerHasText }
      }
      if (script.includes('Could not find Gemini composer')) return { ok: true }
      throw new Error(`Unexpected Gemini evaluate script: ${String(script).slice(0, 100)}`)
    },
    async nativeType(text) {
      actions.push(['nativeType', text])
      composerHasText = true
    },
    async fillText(selector, text) {
      actions.push(['fillText', selector, text])
      composerHasText = true
      return { verified: true, actual: text }
    },
    async nativeClick(x, y) {
      actions.push(['nativeClick', x, y])
    },
    async nativeKeyPress(key) {
      actions.push(['nativeKeyPress', key])
    },
    async click(selector) {
      actions.push(['click', selector])
      composerHasText = false
    },
    async wait() {},
  }

  const result = await sendGeminiMessage(page, 'hello')

  assert.equal(result, 'button')
  assert.deepEqual(actions, [
    ['fillText', '[contenteditable="true"][aria-label*="Gemini"]', 'hello'],
    [
      'click',
      'button[aria-label="Send message"], button[aria-label="发送消息"], button[aria-label="提交"]',
    ],
  ])
})

test('Gemini accepts an owned short reply already present at submission confirmation', async () => {
  const prompt = 'REVIEW_REQUEST_ID:0123456789abcdef0123456789abcdef\nReply exactly 1P2P'
  const current = {
    url: 'https://gemini.google.com/app/owned-turn',
    turns: [],
    transcriptLines: [prompt, '1P2P'],
    composerHasText: false,
    isGenerating: false,
    structuredTurnsTrusted: false,
  }
  const page = {
    async evaluate(script) {
      if (script === 'window.location.href') return current.url
      if (script.includes('structuredTurnsTrusted')) return current
      throw new Error(`Unexpected Gemini response script: ${String(script).slice(0, 100)}`)
    },
    async wait() {},
  }

  const result = await waitForGeminiResponse(page, {
    snapshot: current,
    preSendAssistantCount: 0,
    userAnchorTurn: null,
    reason: 'composer_transcript',
  }, prompt, 6)

  assert.equal(result, '1P2P')
})

test('Gemini does not click twice when the first click submits but reports an error', async () => {
  let composerHasText = false
  let clickCount = 0
  const page = {
    async evaluate(script) {
      if (script === 'window.location.href') return 'https://gemini.google.com/app/test'
      if (script.includes('bestButton instanceof HTMLElement')) {
        return { action: 'button', label: 'Send message', x: 123, y: 456 }
      }
      if (script.includes('hasText: !!(composer')) {
        return { hasText: composerHasText }
      }
      if (script.includes('Could not find Gemini composer')) return { ok: true }
      throw new Error(`Unexpected Gemini evaluate script: ${String(script).slice(0, 100)}`)
    },
    async fillText(_selector, text) {
      composerHasText = true
      return { verified: true, actual: text }
    },
    async click() {
      clickCount += 1
      composerHasText = false
      throw new Error('transport response was lost after dispatch')
    },
    async wait() {},
  }

  const result = await sendGeminiMessage(page, 'hello')

  assert.equal(result, 'button')
  assert.equal(clickCount, 1)
})

test('ChatGPT waits for a delayed slider after one model-trigger click', async () => {
  let currentModel = 'advanced'
  let nativeClicks = 0
  let pickerReads = 0
  const pressed = []
  const page = {
    async evaluate(script) {
      if (script === 'window.location.href') return 'https://chatgpt.com/'
      if (script.includes('hasComposer') && script.includes('hasLoginGate')) {
        return {
          url: 'https://chatgpt.com/',
          title: 'ChatGPT',
          hasComposer: true,
          isLoggedIn: true,
          hasLoginGate: false,
        }
      }
      if (script.includes('findEntryForText')) {
        return {
          model: currentModel,
          label: currentModel === 'balanced' ? 'Medium' : 'Advanced',
        }
      }
      if (script.includes('menuButtonSelectors')) {
        return { found: true, x: 100, y: 50 }
      }
      if (script.includes('contentFound')) {
        pickerReads += 1
        const ready = nativeClicks === 1 && pickerReads >= 3
        return { ready, expanded: nativeClicks === 1, contentFound: ready }
      }
      if (script.includes('keyboardTarget')) {
        return { found: true, current: 2, minimum: 0, maximum: 4 }
      }
      throw new Error(`Unexpected ChatGPT evaluate script: ${String(script).slice(0, 80)}`)
    },
    async nativeClick() {
      nativeClicks += 1
    },
    async nativeKeyPress(key) {
      pressed.push(`native:${key}`)
    },
    async pressKey(key) {
      pressed.push(key)
      if (key === 'ArrowLeft') currentModel = 'balanced'
    },
    async wait() {},
  }

  const result = await selectChatGPTModel(page, 'medium')

  assert.deepEqual(result, { Status: 'Success', Model: 'Medium' })
  assert.equal(nativeClicks, 1)
  assert.ok(pickerReads >= 3)
  assert.deepEqual(pressed, ['ArrowLeft', 'Escape'])
})

test('ChatGPT does not toggle an already expanded model picker closed', async () => {
  let currentModel = 'advanced'
  let nativeClicks = 0
  const pressed = []
  const page = {
    async evaluate(script) {
      if (script === 'window.location.href') return 'https://chatgpt.com/'
      if (script.includes('hasComposer') && script.includes('hasLoginGate')) {
        return {
          url: 'https://chatgpt.com/',
          title: 'ChatGPT',
          hasComposer: true,
          isLoggedIn: true,
          hasLoginGate: false,
        }
      }
      if (script.includes('findEntryForText')) {
        return {
          model: currentModel,
          label: currentModel === 'balanced' ? 'Medium' : 'Advanced',
        }
      }
      if (script.includes('menuButtonSelectors')) return { found: true, x: 100, y: 50 }
      if (script.includes('contentFound')) {
        return { ready: false, expanded: true, contentFound: true }
      }
      if (script.includes('keyboardTarget')) {
        return { found: true, current: 2, minimum: 0, maximum: 4 }
      }
      throw new Error(`Unexpected ChatGPT evaluate script: ${String(script).slice(0, 80)}`)
    },
    async nativeClick() {
      nativeClicks += 1
    },
    async nativeKeyPress(key) {
      pressed.push(`native:${key}`)
    },
    async pressKey(key) {
      pressed.push(key)
      if (key === 'ArrowLeft') currentModel = 'balanced'
    },
    async wait() {},
  }

  const result = await selectChatGPTModel(page, 'medium')

  assert.deepEqual(result, { Status: 'Success', Model: 'Medium' })
  assert.equal(nativeClicks, 0)
  assert.deepEqual(pressed, ['ArrowLeft', 'Escape'])
})

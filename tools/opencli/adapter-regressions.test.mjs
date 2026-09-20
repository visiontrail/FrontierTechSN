import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'
import vm from 'node:vm'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

import { isGenerating, getCurrentChatGPTModel, selectChatGPTModel, uploadChatGPTImages, startNewChat, getVisibleMessages } from './node_modules/@jackwener/opencli/clis/chatgpt/utils.js'
import { detailCommand as chatgptDetailCommand } from './node_modules/@jackwener/opencli/clis/chatgpt/detail.js'
import { modelCommand as chatgptModelCommand } from './node_modules/@jackwener/opencli/clis/chatgpt/model.js'
import { askCommand as geminiAskCommand } from './node_modules/@jackwener/opencli/clis/gemini/ask.js'
import { askCommand as chatgptAskCommand } from './node_modules/@jackwener/opencli/clis/chatgpt/ask.js'
import {
  attachGeminiFile,
  sendGeminiMessage,
  startNewGeminiChat,
  waitForGeminiResponse,
  requireGeminiGeneratedReply,
  getGeminiVisibleTurns,
} from './node_modules/@jackwener/opencli/clis/gemini/utils.js'
import {
  uploadFrame,
  uploadFrames,
  submittedVideoPrompt,
  waitForVideo,
  downloadVideo,
  videoCommand,
} from './node_modules/@jackwener/opencli/clis/gemini/video.js'

function videoTransferPage(payload, { failChunk = 0, httpStatus = 200 } = {}) {
  let chunks = 0
  const context = vm.createContext({
    document: { querySelector(selector) {
      assert.equal(selector, 'generated-video video')
      return { currentSrc: 'https://example.test/generated.mp4' }
    } },
    async fetch(url, options) {
      assert.equal(url, 'https://example.test/generated.mp4')
      assert.equal(options.credentials, 'include')
      assert.ok(options.signal)
      return { ok: httpStatus === 200, status: httpStatus, async blob() { return new Blob([payload]) } }
    },
    AbortController, setTimeout, clearTimeout, btoa,
  })
  return {
    context,
    async wait() { await new Promise(resolve => setImmediate(resolve)) },
    async evaluate(script) {
      if (script.includes('blob.slice(') && ++chunks === failChunk) throw new Error('bridge disconnected')
      return await vm.runInContext(script, context)
    },
  }
}

test('Gemini video downloads bytes directly into the task directory across multiple bridge chunks', async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'gemini-task-download-'))
  try {
    const output = path.join(directory, 'collage_broll', 'scene-02', 'video', 'gemini-web-original.mp4')
    const payload = Buffer.alloc(420123, 37)
    payload.write('ftyp', 4)
    const page = videoTransferPage(payload)
    const result = await downloadVideo(page, output, 30)
    assert.deepEqual(result, { downloaded: true, filename: output, size: payload.length })
    assert.deepEqual(fs.readFileSync(output), payload)
    assert.deepEqual(fs.readdirSync(path.dirname(output)), ['gemini-web-original.mp4'])
    assert.equal(Object.keys(page.context).some(key => key.startsWith('__opencliGeminiVideo_')), false)
  } finally { fs.rmSync(directory, { recursive: true, force: true }) }
})

test('Gemini interrupted downloads remove task partials and preserve the previous complete video', async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'gemini-task-interrupted-'))
  try {
    const output = path.join(directory, 'video.mp4')
    fs.writeFileSync(output, 'previous complete video')
    const payload = Buffer.alloc(420123, 37)
    payload.write('ftyp', 4)
    const page = videoTransferPage(payload, { failChunk: 2 })
    await assert.rejects(downloadVideo(page, output, 30), /bridge disconnected/)
    assert.equal(fs.readFileSync(output, 'utf8'), 'previous complete video')
    assert.deepEqual(fs.readdirSync(directory), ['video.mp4'])
    assert.equal(Object.keys(page.context).some(key => key.startsWith('__opencliGeminiVideo_')), false)
  } finally { fs.rmSync(directory, { recursive: true, force: true }) }
})

test('Gemini video rejects HTTP failures and non-video bodies without publishing files', async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'gemini-task-invalid-'))
  try {
    const output = path.join(directory, 'video.mp4')
    await assert.rejects(downloadVideo(videoTransferPage(Buffer.alloc(2048), { httpStatus: 403 }), output, 30), /HTTP 403/)
    await assert.rejects(downloadVideo(videoTransferPage(Buffer.alloc(2048, 65)), output, 30), /not return an MP4/)
    assert.deepEqual(fs.readdirSync(directory), [])
  } finally { fs.rmSync(directory, { recursive: true, force: true }) }
})

test('Gemini stalled fetch is polled briefly and aborted without leaving a transfer or partial file', async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'gemini-task-deadline-'))
  try {
    const page = videoTransferPage(Buffer.alloc(2048))
    let signal
    let reads = 0
    page.context.fetch = (_url, options) => {
      signal = options.signal
      return new Promise((_, reject) => {
        signal.addEventListener('abort', () => reject(new Error('fetch aborted')), { once: true })
      })
    }
    const evaluate = page.evaluate
    page.evaluate = async script => { reads++; return evaluate(script) }
    await assert.rejects(downloadVideo(page, path.join(directory, 'video.mp4'), 0.02), /deadline|aborted/)
    assert.equal(signal.aborted, true)
    assert.ok(reads > 2)
    assert.deepEqual(fs.readdirSync(directory), [])
    assert.equal(Object.keys(page.context).some(key => key.startsWith('__opencliGeminiVideo_')), false)
  } finally { fs.rmSync(directory, { recursive: true, force: true }) }
})

test('Gemini video spinner without changing response stops early and records progress', async () => {
  let clock = 1000
  const checkpoint = {}
  const page = {
    async wait(seconds) { clock += seconds * 1000 },
    async evaluate(script) {
      new vm.Script(script)
      return { videos: [], text: 'Defining the Parameters', url: 'https://gemini.google.com/app/abc123' }
    },
  }
  await assert.rejects(waitForVideo(page, [], 1800, {
    now: () => clock, stallTimeoutSeconds: 60, checkpoint,
  }), /GEMINI_VIDEO_GENERATION_STALLED/)
  assert.equal(clock, 66000)
  assert.equal(checkpoint.url, 'https://gemini.google.com/app/abc123')
})

test('Gemini response progress resets the stall timer and a completed video wins', async () => {
  let clock = 1000
  let reads = 0
  const page = {
    async wait(seconds) { clock += seconds * 1000 },
    async evaluate() {
      reads++
      return { text: reads < 10 ? 'Defining' : 'Rendering', videos: reads === 20 ? [{ src: 'new', readyState: 4 }] : [] }
    },
  }
  assert.equal((await waitForVideo(page, [], 1800, { now: () => clock, stallTimeoutSeconds: 60 })).src, 'new')
})

test('Gemini resume preserves the original total and no-progress deadlines', async () => {
  let clock = 60000
  const checkpoint = { deadline: 70000, lastProgressAt: 1000,
    progressSignature: JSON.stringify({ text: 'Defining', videos: [] }) }
  const page = {
    async wait(seconds) { clock += seconds * 1000 },
    async evaluate() { return { text: 'Defining', videos: [] } },
  }
  await assert.rejects(waitForVideo(page, [], 1800, {
    now: () => clock, stallTimeoutSeconds: 60, checkpoint,
  }), /GENERATION_STALLED/)
  clock = 70000
  await assert.rejects(waitForVideo(page, [], 1800, {
    now: () => clock, checkpoint,
  }), /GENERATION_TIMEOUT/)
})

test('Gemini checkpoint rejects wrong resume ownership and expired generation before browser operations', async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'gemini-resume-'))
  try {
    const stateFile = path.join(directory, 'state.json')
    const requestId = 'a'.repeat(64)
    const kwargs = { resume: 'https://gemini.google.com/app/abc123', output: path.join(directory, 'video.mp4'),
      'state-file': stateFile, 'request-id': requestId }
    fs.writeFileSync(stateFile, JSON.stringify({ request_id: requestId, url: 'https://gemini.google.com/app/other' }))
    await assert.rejects(videoCommand.func({}, kwargs), /does not match/)
    fs.writeFileSync(stateFile, JSON.stringify({ request_id: requestId, url: kwargs.resume, deadline: Date.now() - 1 }))
    await assert.rejects(videoCommand.func({}, kwargs), /GENERATION_TIMEOUT/)
    assert.equal(JSON.parse(fs.readFileSync(stateFile)).status, 'timeout')
  } finally { fs.rmSync(directory, { recursive: true, force: true }) }
})

test('Gemini download recovery uses the remaining original budget without waiting for generation again', async () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'gemini-download-resume-'))
  try {
    const stateFile = path.join(directory, 'state.json')
    const checkpoint = { request_id: 'b'.repeat(64), url: 'https://gemini.google.com/app/abc123',
      status: 'downloading', deadline: Date.now() - 1000, totalDeadline: Date.now() + 120000 }
    fs.writeFileSync(stateFile, JSON.stringify(checkpoint))
    let downloadReached = false
    const page = {
      async goto(url) { assert.equal(url, checkpoint.url) },
      async evaluate(script) {
        assert.match(script, /fetch\(source/)
        downloadReached = true
        return { ok: false, reason: 'simulated network interruption' }
      },
    }
    const kwargs = { resume: checkpoint.url, output: path.join(directory, 'video.mp4'),
      'state-file': stateFile, 'request-id': checkpoint.request_id }
    await assert.rejects(videoCommand.func(page, kwargs), /in-page download failed/)
    assert.equal(downloadReached, true)
    assert.deepEqual(JSON.parse(fs.readFileSync(stateFile)), checkpoint)
    checkpoint.totalDeadline = Date.now() - 1
    fs.writeFileSync(stateFile, JSON.stringify(checkpoint))
    await assert.rejects(videoCommand.func({}, kwargs), /GENERATION_TIMEOUT/)
  } finally { fs.rmSync(directory, { recursive: true, force: true }) }
})

test('Gemini video stops waiting when its submitted conversation disappears into the home composer', async () => {
  let reads = 0
  const page = {
    async wait() {},
    async evaluate(script) {
      new vm.Script(script)
      reads++
      return { videos: [], text: '', emptyHome: true }
    },
  }
  await assert.rejects(waitForVideo(page, [], 60), /submitted conversation is no longer available/)
  assert.equal(reads, 3)
})

test('Gemini video tolerates transient empty navigation and resets the lost-conversation counter', async () => {
  const states = [true, true, false, true, true]
  let reads = 0
  const page = {
    async wait() {},
    async evaluate() {
      const emptyHome = states[reads++]
      return { text: '', emptyHome, videos: reads > states.length ? [{ src: 'owned-video', readyState: 4 }] : [] }
    },
  }
  assert.equal((await waitForVideo(page, [], 60)).src, 'owned-video')
  assert.equal(reads, 6)
})

test('Gemini generation errors fail explicitly while ordinary review content is preserved', () => {
  for (const text of ['', 'I seem to be encountering an error. Can I try something else for you?',
    'I encountered an error doing what you asked. Could you try again?',
    'Sorry, something went wrong. Please try your request again.',
    "I'm having a hard time fulfilling your request. Can I help you with something else instead?"]) {
    assert.throws(() => requireGeminiGeneratedReply(text), /Gemini generation/)
  }
  const review = '{"image_received":true,"reviews":[{"issues":["An error message is visible on screen."]}]}'
  assert.equal(requireGeminiGeneratedReply(review), review)
  assert.equal(requireGeminiGeneratedReply('I can explain an error in your code.'), 'I can explain an error in your code.')
})

for (const failure of ['I encountered an error doing what you asked. Could you try again?',
  'I seem to be encountering an error. Can I try something else for you?',
  'Sorry, something went wrong. Please try your request again.',
  "I'm having a hard time fulfilling your request. Can I help you with something else instead?"]) {
test('Gemini reports an owned stable generation error while the stop button remains visible: ' + failure, async () => {
  const user = { Role: 'User', Text: 'Review the supplied image.' }
  const baseline = { url: 'https://gemini.google.com/app/current', turns: [user],
    transcriptLines: [user.Text], composerHasText: false, isGenerating: true,
    structuredTurnsTrusted: true }
  let reads = 0
  const page = {
    async wait() {},
    async evaluate(script) {
      if (script === 'window.location.href') return baseline.url
      if (script.includes('structuredTurnsTrusted')) {
        reads++
        return { ...baseline, turns: [user, { Role: 'Assistant', Text: failure }] }
      }
      throw new Error('Unexpected Gemini snapshot script')
    },
  }
  await assert.rejects(waitForGeminiResponse(page, {
    snapshot: baseline, userAnchorTurn: user,
  }, user.Text, 10), /Gemini generation failed/)
  assert.equal(reads, 2)
})
}

test('Gemini ignores old generation errors and waits for the owned answer to finish', async () => {
  const user = { Role: 'User', Text: 'Review the supplied image.' }
  const old = { Role: 'Assistant', Text: 'I encountered an error doing what you asked. Could you try again?' }
  const answer = '{"image_received":true,"reviews":[]}'
  const baseline = { url: 'https://gemini.google.com/app/current', turns: [old, user],
    transcriptLines: [], composerHasText: false, isGenerating: true,
    structuredTurnsTrusted: true }
  let reads = 0
  const page = {
    async wait() {},
    async evaluate(script) {
      if (script === 'window.location.href') return baseline.url
      if (script.includes('structuredTurnsTrusted')) {
        reads++
        return { ...baseline, isGenerating: reads < 4,
          turns: [old, user, { Role: 'Assistant', Text: answer }] }
      }
      throw new Error('Unexpected Gemini snapshot script')
    },
  }
  assert.equal(await waitForGeminiResponse(page, {
    snapshot: baseline, userAnchorTurn: user,
  }, user.Text, 10), answer)
  assert.equal(reads, 4)
})

test('Gemini reads provider errors from message-content when no Markdown answer exists', async () => {
  const failure = 'I encountered an error doing what you asked. Could you try again?'
  const body = { innerText: failure }
  class Element {
    constructor(tagName, text) { this.tagName = tagName; this.innerText = text }
    getBoundingClientRect() { return { width: 500, height: 100 } }
    getAttribute() { return null }
    querySelectorAll() { return [] }
    querySelector(selector) { return this.tagName === 'MODEL-RESPONSE' && selector === 'message-content' ? body : null }
    compareDocumentPosition() { return this.tagName === 'USER-QUERY' ? 4 : 2 }
  }
  const roots = [new Element('USER-QUERY', 'Review the image.'), new Element('MODEL-RESPONSE', 'Gemini said ' + failure)]
  const page = { async evaluate(script) {
    if (script === 'window.location.href') return 'https://gemini.google.com/app/current'
    return vm.runInNewContext(script, { HTMLElement: Element,
      Node: { DOCUMENT_POSITION_FOLLOWING: 4, DOCUMENT_POSITION_PRECEDING: 2 },
      window: { getComputedStyle: () => ({ display: 'block', visibility: 'visible' }) },
      document: { querySelectorAll: selector => selector === 'user-query, model-response' ? roots : [] },
    })
  } }
  const turns = await getGeminiVisibleTurns(page)
  assert.deepEqual(turns.map(turn => [turn.Role, turn.Text]), [
    ['User', 'Review the image.'], ['Assistant', failure],
  ])
})

test('Gemini canonical turns exclude changing prompt summaries and nested speaker labels', async () => {
  class Element {
    constructor(tagName, text) { this.tagName = tagName; this.innerText = text; this.textContent = text }
    getBoundingClientRect() { return { width: 600, height: 200 } }
    getAttribute() { return null }
    querySelector() { return null }
    querySelectorAll() { return [] }
    compareDocumentPosition() { return this.tagName === 'USER-QUERY' ? 4 : 2 }
  }
  const prompt = 'Review the image.\nReturn exactly one JSON object.'
  const answer = '{"image_received":true,"reviews":[{"id":"scene-09","score":72}]}'
  const user = new Element('USER-QUERY', 'You said Review the image… Show more')
  const model = new Element('MODEL-RESPONSE', 'Gemini said JSON' + answer)
  user.querySelectorAll = selector => selector === '.query-text-line'
    ? prompt.split('\n').map(line => new Element('P', line)) : []
  model.querySelector = selector => selector === '.markdown, .model-response-text'
    ? new Element('DIV', answer) : null
  let broadReads = 0
  const page = { async evaluate(script) {
    if (script === 'window.location.href') return 'https://gemini.google.com/app/current'
    return vm.runInNewContext(script, { HTMLElement: Element, Node: {
      DOCUMENT_POSITION_FOLLOWING: 4, DOCUMENT_POSITION_PRECEDING: 2,
    }, window: { getComputedStyle: () => ({ display: 'block', visibility: 'visible' }) },
    document: { querySelectorAll: selector => {
      if (selector === 'user-query, model-response') return [user, model]
      broadReads++
      return [new Element('DIV', 'Gemini said'), user, model]
    } } })
  } }
  const before = await getGeminiVisibleTurns(page)
  user.innerText = 'You said Review the image. Return exactly one JSON object. Show less'
  const after = await getGeminiVisibleTurns(page)
  assert.equal(broadReads, 0)
  assert.equal(JSON.stringify(before), JSON.stringify(after))
  assert.equal(before.length, 2)
  assert.equal(before[0].Role, 'User'); assert.equal(before[0].Text, prompt)
  assert.equal(before[1].Role, 'Assistant'); assert.equal(before[1].Text, answer)
  model.querySelector = () => null
  assert.equal((await getGeminiVisibleTurns(page)).length, 1)
})

test('Gemini rejects whole-page transcript fallback even when the prompt is escaped or collapsed', async () => {
  const prompt = 'Review this image. Return {"image_received":true,"reviews":[]}.'
  const baseline = { url: 'https://gemini.google.com/app/current', turns: [],
    transcriptLines: [], composerHasText: false, isGenerating: false, structuredTurnsTrusted: false }
  const pageText = 'GeminiNew chatSearch chatsConversation with Gemini You said '
    + JSON.stringify(prompt) + ' Gemini said JSON{"image_received":true,"reviews":[]}'
  const page = { async wait() {}, async evaluate(script) {
    if (script === 'window.location.href') return baseline.url
    if (script.includes('structuredTurnsTrusted')) return { ...baseline, transcriptLines: [pageText] }
    throw new Error('Unexpected snapshot script')
  } }
  assert.equal(await waitForGeminiResponse(page, { snapshot: baseline, userAnchorTurn: null }, prompt, 10), '')
})

test('ChatGPT reads section turns and full collapsed prompts without UI labels', async () => {
  class Element {
    constructor(text) { this.textContent = text; this.innerText = text; this.innerHTML = text }
    getBoundingClientRect() { return { width: 600, height: 200 } }
    getAttribute() { return null }
    querySelector() { return null }
  }
  const prompt = new Element('The complete multi-line prompt.\nIncluding its hidden final line.')
  const answer = new Element('{"image_received":true,"reviews":[]}')
  const user = new Element('You said: Show more'); const assistant = new Element('ChatGPT said: Copy response')
  user.querySelector = selector => selector === 'h4' ? new Element('You said:')
    : selector.includes('collapsible-user-message-content') ? prompt : null
  assistant.querySelector = selector => selector === 'h4' ? new Element('ChatGPT said:')
    : selector === '.markdown' ? answer : null
  const page = { async evaluate(script) {
    return vm.runInNewContext(script, { HTMLElement: Element,
      window: { getComputedStyle: () => ({ display: 'block', visibility: 'visible' }) },
      document: { querySelectorAll: selector => selector === '[data-message-author-role], [data-testid^="conversation-turn-"]' ? [user, assistant] : [] },
    })
  } }
  const rows = await getVisibleMessages(page, { textOnly: true })
  assert.equal(rows.length, 2)
  assert.equal(rows[0].Role, 'User'); assert.equal(rows[0].Text, prompt.textContent)
  assert.equal(rows[1].Role, 'Assistant'); assert.equal(rows[1].Text, answer.textContent)
})

for (const outcome of ['loaded', 'blank', 'login', 'conversation']) {
test(`ChatGPT new chat distinguishes stalled navigation from login: ${outcome}`, async () => {
  let navigations = 0
  const page = {
    async goto(url) { assert.equal(url, 'https://chatgpt.com'); navigations++ },
    async wait() { throw new Error('Composer has not mounted yet') },
    async evaluate(script) {
      const loaded = outcome === 'loaded' && navigations === 2
      if (script === 'window.location.href') return outcome === 'conversation' ? 'https://chatgpt.com/c/existing' : 'about:blank'
      if (script.includes('hasLoginGate')) return { hasComposer: loaded, hasLoginGate: outcome === 'login', isLoggedIn: outcome !== 'login' }
      throw new Error('Unexpected navigation script')
    },
  }
  if (['loaded', 'login'].includes(outcome)) await startNewChat(page)
  else await assert.rejects(startNewChat(page), /did not finish loading/)
  assert.equal(navigations, ['loaded', 'blank'].includes(outcome) ? 2 : 1)
})
}

test('browser patches reject an unsupported upstream before changing any adapters', (t) => {
  const source = path.dirname(fileURLToPath(import.meta.url))
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'opencli-patch-version-'))
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))
  const packageDir = path.join(root, 'node_modules', '@jackwener', 'opencli')
  fs.mkdirSync(packageDir, { recursive: true })
  fs.writeFileSync(path.join(packageDir, 'package.json'), JSON.stringify({ version: '0.0.0' }))
  fs.copyFileSync(path.join(source, 'patch-opencli.mjs'), path.join(root, 'patch-opencli.mjs'))
  assert.throws(
    () => execFileSync(process.execPath, [path.join(root, 'patch-opencli.mjs')], { stdio: 'pipe' }),
    error => /OpenCLI patches require 1\.8\.8; found 0\.0\.0/.test(String(error.stderr)),
  )
  assert.deepEqual(fs.readdirSync(packageDir), ['package.json'])
})

test('reapplying browser patches keeps every adapter byte-identical and recovery arguments unique', (t) => {
  const source = path.dirname(fileURLToPath(import.meta.url))
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'opencli-patch-idempotence-'))
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))
  const packagePath = 'node_modules/@jackwener/opencli'
  fs.cpSync(path.join(source, packagePath), path.join(root, packagePath), { recursive: true })
  fs.cpSync(path.join(source, 'patches'), path.join(root, 'patches'), { recursive: true })
  fs.copyFileSync(path.join(source, 'patch-opencli.mjs'), path.join(root, 'patch-opencli.mjs'))
  const apply = () => execFileSync(process.execPath, [path.join(root, 'patch-opencli.mjs')])
  const adapters = ['clis/gemini/ask.js', 'clis/gemini/utils.js', 'clis/gemini/models.js',
    'clis/gemini/video.js', 'clis/chatgpt/ask.js', 'clis/chatgpt/utils.js',
    'clis/chatgpt/model.js', 'clis/chatgpt/detail.js', 'dist/src/execution.js', 'cli-manifest.json']
  // Simulate the installed adapter before these two provider errors were
  // recognized. Upgrading must extend its helper rather than insert it twice.
  const utilsPath = path.join(root, packagePath, 'clis/gemini/utils.js')
  const currentUtils = fs.readFileSync(utilsPath, 'utf8')
  const legacyUtils = currentUtils
    .replace("\n        || normalized === 'sorry, something went wrong. please try your request again.'", '')
    .replace(`\n        || normalized === "i'm having a hard time fulfilling your request. can i help you with something else instead?"`, '')
  assert.notEqual(legacyUtils, currentUtils)
  fs.writeFileSync(utilsPath, legacyUtils)
  apply()
  assert.equal(fs.readFileSync(utilsPath, 'utf8'), currentUtils)
  const contents = adapters.map(name => fs.readFileSync(path.join(root, packagePath, name), 'utf8'))
  apply()
  adapters.forEach((name, index) => assert.equal(
    fs.readFileSync(path.join(root, packagePath, name), 'utf8'), contents[index], name,
  ))
  const detail = contents[adapters.indexOf('clis/chatgpt/detail.js')]
  for (const name of ['refresh', 'cooldown']) assert.equal(
    [...detail.matchAll(new RegExp(`name: '${name}'`, 'g'))].length, 1,
  )
  assert.equal([...detail.matchAll(/let currentId = '';/g)].length, 1)
})

test('ChatGPT generation includes status outside the message in a section turn', async () => {
  const thinking = {
    children: [], textContent: 'Thinking',
    closest() { return null },
  }
  const answer = {
    children: [], textContent: 'W',
    closest() { return {} },
  }
  const section = {
    children: [thinking, answer],
    querySelectorAll() { return [thinking, answer] },
  }
  const page = {
    async evaluate(script) {
      return vm.runInNewContext(script, {
        document: {
          querySelector() { return null },
          querySelectorAll(selector) {
            if (selector === '[data-testid^="conversation-turn-"]') return [section]
            if (selector === '[data-message-author-role]') return [answer]
            return []
          },
        },
      })
    },
  }
  assert.equal(await isGenerating(page), true)
  thinking.textContent = ''
  answer.textContent = 'An answer mentioning Thinking'
  assert.equal(await isGenerating(page), false)
})

test('ChatGPT ask exposes image attachments and refuses to send a missing image', async () => {
  assert.ok(chatgptAskCommand.args.some(arg => arg.name === 'file'))
  const page = {
    async evaluate(script) {
      if (script === 'window.location.href') return 'https://chatgpt.com/'
      if (script.includes('isLoggedIn')) return { isLoggedIn: true, hasComposer: true, hasLoginGate: false }
      if (script.includes('stop-button')) return false
      throw new Error('Unexpected action after failed attachment')
    },
  }
  await assert.rejects(chatgptAskCommand.func(page, { prompt: 'Review this image', file: '/missing-review-image.jpg' }), /not found|does not exist/i)
})

for (const nativeError of ['Page.fileChooserOpened not received within 5s', 'Browser target crashed']) {
test(`ChatGPT image upload recovers only supported native-picker failures: ${nativeError}`, async (t) => {
  const [image] = videoFrameFixture(t, ['contact-sheet.jpg'])
  let transfers = 0
  const page = {
    async setFileInput() { throw new Error(nativeError) },
    async sleep() {},
    async evaluate(script) {
      if (script.includes('const reactiveInputs')) return true
      if (script.includes('const dt = new DataTransfer()')) { transfers++; return { ok: true } }
      if (script.includes('matchedNames')) return true
      throw new Error('Unexpected upload operation')
    },
  }
  if (nativeError.includes('fileChooserOpened')) {
    assert.equal((await uploadChatGPTImages(page, [image])).ok, true)
    assert.equal(transfers, 1)
  } else {
    await assert.rejects(uploadChatGPTImages(page, [image]), /Browser target crashed/)
    assert.equal(transfers, 0)
  }
})
}

for (const hydrated of [false, true]) {
test(`ChatGPT selects the active image callback instead of the first generic file input: ${hydrated}`, async (t) => {
  const [image] = videoFrameFixture(t, ['contact-sheet.jpg'])
  let selected = -1, nativeCalls = 0
  const inputs = [
    { accept: '', __reactProps$x: { onChange: null } },
    { accept: 'image/*', __reactProps$x: { onChange: hydrated ? () => {} : null } },
  ].map((input, index) => ({ ...input, disabled: false, setAttribute() { selected = index } }))
  const scope = { querySelectorAll: () => inputs }
  const composer = { closest: () => ({ parentElement: scope }) }
  const page = {
    async sleep() {},
    async setFileInput(_files, selector) { nativeCalls++; assert.equal(selected, 1); assert.match(selector, /data-opencli-chatgpt-image-input/); throw new Error('fileChooserOpened not received') },
    async evaluate(script) {
      if (script.includes('const reactiveInputs')) return vm.runInNewContext(script, { document: { querySelector: () => composer, querySelectorAll: () => [] } })
      if (script.includes('const dt = new DataTransfer()')) return { ok: true }
      if (script.includes('matchedNames')) return true
      throw new Error('Unexpected upload operation')
    },
  }
  const result = await uploadChatGPTImages(page, [image])
  assert.equal(result.ok, hydrated)
  assert.equal(nativeCalls, hydrated ? 1 : 0)
})
}

for (const current of ['https://chatgpt.com/c/abcdefgh1234', 'https://chatgpt.com/c/otherchat1234', 'about:blank']) {
  test(`ChatGPT detail navigates only when the target differs from ${current}`, async () => {
    const navigations = []
    const page = {
      async goto(url) { navigations.push(url) },
      async wait() {},
      async evaluate(script) {
        if (script === 'window.location.href') return current
        if (script.includes('isLoggedIn')) return { isLoggedIn: true, hasLoginGate: false }
        if (script.includes('const roleOf')) return [{ role: 'assistant', text: 'W1P;2P', html: '' }]
        if (script.includes('stop-button')) return false
        throw new Error(`Unexpected detail evaluation: ${script.slice(0, 100)}`)
      },
    }
    const rows = await chatgptDetailCommand.func(page, { id: 'abcdefgh1234', wait: false })
    assert.equal(rows[0].Text, 'W1P;2P')
    assert.deepEqual(navigations, current.endsWith('/abcdefgh1234') ? [] : ['https://chatgpt.com/c/abcdefgh1234'])
  })
}

function videoFrameFixture(t, names = ['first-frame.png']) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'opencli-video-upload-'))
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }))
  return names.map((name, index) => {
    const file = path.join(directory, name)
    fs.writeFileSync(file, Buffer.from(`frame-${index + 1}`))
    return file
  })
}

function geminiAttachmentPage(nativeError, inputSelector = 'input[name="Filedata"]') {
  const actions = []
  return {
    actions,
    async goto(url, options) {
      actions.push(['goto', url, options])
    },
    async wait(seconds) {
      actions.push(['wait', seconds])
    },
    async click(selector) {
      actions.push(['click', selector])
    },
    async setFileInput(files, selector) {
      actions.push(['setFileInput', files, selector])
      throw new Error(nativeError)
    },
    async cdp(method, params) {
      actions.push(['cdp', method, params])
    },
    async evaluate(script) {
      if (script === 'window.location.href') return 'https://gemini.google.com/app'
      if (script.includes('button.click();')) return false
      if (script.includes('const inputSelector = selectors.find')) {
        return { inputSelector, buttonSelector: '', expanded: true }
      }
      if (
        script.includes("input.dispatchEvent(new Event('change'")
        && !script.includes('new DataTransfer()')
      ) {
        actions.push(['change', inputSelector])
        return true
      }
      if (script.includes('const transfer = new DataTransfer()')) {
        actions.push(['DataTransfer'])
        return { ok: true }
      }
      if (script.includes('__opencliGeminiUpload_')) return true
      if (script.includes('const candidates = Array.from')) return { ready: true }
      throw new Error(`Unexpected Gemini attachment script: ${String(script).slice(0, 120)}`)
    },
  }
}

test('Gemini ask falls back when Browser Bridge misses fileChooserOpened', async (t) => {
  const [image] = videoFrameFixture(t, ['contact-sheet.jpg'])
  const page = geminiAttachmentPage(
    'Page.fileChooserOpened not received within 5s — the input may not have opened a file chooser',
  )

  await assert.doesNotReject(attachGeminiFile(page, image))

  assert.equal(page.actions.filter(([action]) => action === 'setFileInput').length, 1)
  assert.equal(page.actions.filter(([action]) => action === 'DataTransfer').length, 1)
})

test('Gemini attachment waits for the hydrated upload control and recovers its replacement', async (t) => {
  const [image] = videoFrameFixture(t, ['contact-sheet.jpg'])
  const page = geminiAttachmentPage('fileChooserOpened not received')
  let readyWaits = 0
  let clicks = 0
  const wait = page.wait
  page.wait = async value => {
    if (value?.selector === 'button[aria-label="Upload & tools"]') readyWaits++
    return wait(value)
  }
  page.click = async () => {
    clicks++
    assert.equal(readyWaits, clicks)
    if (clicks < 3) throw new Error('CSS selector matched 0 elements')
  }
  await attachGeminiFile(page, image)
  assert.equal(clicks, 3)
  assert.equal(page.actions.filter(([action]) => action === 'DataTransfer').length, 1)
})

test('Gemini attachment never uploads when its skeleton page does not hydrate', async (t) => {
  const [image] = videoFrameFixture(t, ['contact-sheet.jpg'])
  const page = geminiAttachmentPage('fileChooserOpened not received')
  const wait = page.wait
  page.wait = async value => {
    if (value?.selector) throw new Error('Upload control readiness timed out')
    return wait(value)
  }
  await assert.rejects(attachGeminiFile(page, image), /readiness timed out/)
  assert.equal(page.actions.some(([action]) => ['click', 'setFileInput', 'DataTransfer'].includes(action)), false)
})

for (const scenario of ['fresh', 'draft', 'attachment', 'conversation']) {
test(`Gemini new chat preserves only an already empty fresh page: ${scenario}`, async () => {
  let navigations = 0
  const page = {
    async evaluate(script) {
      if (script === 'window.location.href') return 'https://gemini.google.com/app'
      return vm.runInNewContext(script, {
        location: { pathname: scenario === 'conversation' ? '/app/current' : '/app' },
        document: { querySelector(selector) {
          if (selector.includes('contenteditable')) return { innerText: scenario === 'draft' ? 'unsent draft' : '' }
          return scenario === 'attachment' ? {} : null
        } },
      })
    },
    async goto() { navigations++ },
    async wait() {},
  }
  await startNewGeminiChat(page)
  assert.equal(navigations, scenario === 'fresh' ? 0 : 1)
})
}

test('Gemini ask does not hide unrelated native upload failures', async (t) => {
  const [image] = videoFrameFixture(t, ['contact-sheet.jpg'])
  const page = geminiAttachmentPage('Browser target crashed')

  await assert.rejects(attachGeminiFile(page, image), /Browser target crashed/)
  assert.equal(page.actions.some(([action]) => action === 'DataTransfer'), false)
})

test('Gemini verifies a native upload acknowledgement before skipping the file transfer', async (t) => {
  const [image] = videoFrameFixture(t, ['contact-sheet.jpg'])
  const page = geminiAttachmentPage(null)
  page.setFileInput = async () => undefined
  const originalEvaluate = page.evaluate
  page.evaluate = async script => {
    if (script.includes('const file = input.files?.[0]')) {
      return vm.runInNewContext(script, { document: { querySelector: () => ({ files: [] }) } })
    }
    return originalEvaluate(script)
  }
  await attachGeminiFile(page, image)
  assert.equal(page.actions.filter(([action]) => action === 'DataTransfer').length, 1)
})

for (const interrupted of [false, true]) {
  test(`Gemini large-file fallback stays below the daemon limit and cleans transfer state: ${interrupted}`, async (t) => {
    const [image] = videoFrameFixture(t, ['large-contact-sheet.jpg'])
    const bytes = Buffer.alloc(900_000)
    for (let index = 0; index < bytes.length; index += 1) bytes[index] = index % 251
    fs.writeFileSync(image, bytes)
    const input = { files: [], dispatchEvent() {} }
    const context = vm.createContext({
      document: { querySelector: () => input }, atob, Uint8Array, File, Event,
      DataTransfer: class {
        files = []
        items = { add: file => this.files.push(file) }
      },
    })
    const page = geminiAttachmentPage('fileChooserOpened not received')
    const originalEvaluate = page.evaluate
    let chunks = 0
    page.evaluate = async (script) => {
      if (script.includes('__opencliGeminiUpload_') || script.includes('const transfer = new DataTransfer()')) {
        assert.ok(Buffer.byteLength(JSON.stringify({ action: 'exec', code: script })) < 1024 * 1024)
        if (script.includes('.push(') && ++chunks === 2 && interrupted) throw new Error('Transfer interrupted')
        return vm.runInContext(script, context)
      }
      return originalEvaluate(script)
    }
    if (interrupted) {
      await assert.rejects(attachGeminiFile(page, image), /Transfer interrupted/)
      assert.equal(input.files.length, 0)
    } else {
      await attachGeminiFile(page, image)
      assert.ok(chunks > 1)
      assert.equal(input.files.length, 1)
      assert.deepEqual(Buffer.from(await input.files[0].arrayBuffer()), bytes)
    }
    assert.equal(Object.keys(context).some(key => key.startsWith('__opencliGeminiUpload_')), false)
  })
}

test('Gemini ask supports the current unnamed images-files-uploader input', async (t) => {
  const [image] = videoFrameFixture(t, ['contact-sheet.jpg'])
  const inputSelector = 'images-files-uploader input[type="file"]'
  const page = geminiAttachmentPage(null, inputSelector)
  page.setFileInput = async (files, selector) => {
    page.actions.push(['setFileInput', files, selector])
  }

  await assert.doesNotReject(attachGeminiFile(page, image))

  const upload = page.actions.find(([action]) => action === 'setFileInput')
  assert.equal(upload[2], inputSelector)
  assert.equal(page.actions.some(([action]) => action === 'change'), true)
  assert.equal(
    page.actions.some(([action, method]) => action === 'cdp' && method === 'Page.bringToFront'),
    true,
  )
  assert.equal(page.actions.some(([action]) => action === 'goto'), false)
})

test('Gemini attachment waits for upload-menu hydration without resetting the conversation', async (t) => {
  const [image] = videoFrameFixture(t, ['contact-sheet.jpg'])
  const page = geminiAttachmentPage('fileChooserOpened not received')
  const originalEvaluate = page.evaluate
  let menuReads = 0
  page.evaluate = async (script) => {
    if (script.includes('const inputSelector = selectors.find') && ++menuReads < 20) {
      return { inputSelector: '', buttonSelector: '', expanded: true }
    }
    if (script.includes('button.getBoundingClientRect()')) return null
    return originalEvaluate(script)
  }
  await attachGeminiFile(page, image)
  assert.equal(menuReads, 20)
  assert.equal(page.actions.some(([action]) => action === 'goto'), false)
  assert.equal(page.actions.filter(([action]) => action === 'DataTransfer').length, 1)
})

test('Gemini attachment never reports success while its preview remains unavailable', async (t) => {
  const [image] = videoFrameFixture(t, ['contact-sheet.jpg'])
  const page = geminiAttachmentPage('fileChooserOpened not received')
  const originalEvaluate = page.evaluate
  let previewReads = 0
  page.evaluate = async (script) => {
    if (script.includes('const candidates = Array.from')) {
      previewReads += 1
      return { ready: false }
    }
    return originalEvaluate(script)
  }
  await assert.rejects(attachGeminiFile(page, image), /review prompt was not submitted/)
  assert.equal(previewReads, 60)
})

for (const scenario of ['empty', 'draft', 'attachment', 'generating', 'changed-model', 'still-empty']) {
test(`Gemini reloads a stuck menu only for a safe empty page: ${scenario}`, async (t) => {
  const [image] = videoFrameFixture(t, ['contact-sheet.jpg'])
  const page = geminiAttachmentPage('fileChooserOpened not received')
  const originalEvaluate = page.evaluate
  let reloads = 0
  page.goto = async () => { reloads++ }
  page.evaluate = async script => {
    if (script.includes('const inputSelector = selectors.find') && (!reloads || scenario === 'still-empty')) return { inputSelector: '', buttonSelector: '', expanded: true }
    if (script.includes('button.getBoundingClientRect()')) return null
    if (script.includes('emptyComposer:')) return { emptyComposer: scenario !== 'draft', attachmentCount: scenario === 'attachment' ? 1 : 0, generating: scenario === 'generating', modelLabel: 'Open mode picker, currently Flash', url: 'https://gemini.google.com/app' }
    if (script.startsWith("document.querySelector('button[aria-label*=")) return scenario === 'changed-model' ? 'Open mode picker, currently Pro' : 'Open mode picker, currently Flash'
    if (script.includes('uploadButtons:')) return { expanded: 'true', fileInputs: [] }
    return originalEvaluate(script)
  }
  if (scenario === 'empty') {
    await attachGeminiFile(page, image)
    assert.equal(reloads, 1)
    assert.equal(page.actions.filter(([action]) => action === 'DataTransfer').length, 1)
  } else {
    await assert.rejects(attachGeminiFile(page, image), /did not open|model changed/)
    assert.equal(reloads, ['changed-model', 'still-empty'].includes(scenario) ? 1 : 0)
    assert.equal(page.actions.filter(([action]) => action === 'DataTransfer').length, 0)
  }
})
}

for (const scenario of ['ready', 'busy', 'unrelated-preview', 'disabled-send']) {
  test(`Gemini attachment requires a ready preview in its own composer: ${scenario}`, async (t) => {
    const [image] = videoFrameFixture(t, ['contact-sheet.jpg'])
    const page = geminiAttachmentPage('fileChooserOpened not received')
    const originalEvaluate = page.evaluate
    page.evaluate = async (script) => {
      if (!script.includes('const candidates = Array.from')) return originalEvaluate(script)
      return vm.runInNewContext(script, {
        document: {
          querySelector(selector) {
            if (selector.includes('button')) return { disabled: scenario === 'disabled-send', getAttribute: () => null }
            return { innerText: '', querySelectorAll: () => scenario === 'unrelated-preview' ? [] : [{ getAttribute: name => name === 'src' ? 'blob:attachment' : '' }] }
          },
          querySelectorAll(selector) {
            if (selector.includes('progressbar')) {
              assert.match(selector, /input-container/)
              return [{ getBoundingClientRect: () => ({ width: scenario === 'busy' ? 24 : 0, height: scenario === 'busy' ? 24 : 0 }) }]
            }
            return [{ getAttribute: name => name === 'src' ? 'blob:attachment' : '' }]
          },
        },
        getComputedStyle: () => ({ visibility: 'visible' }),
      })
    }
    if (scenario !== 'ready') await assert.rejects(attachGeminiFile(page, image), /did not become ready/)
    else await assert.doesNotReject(attachGeminiFile(page, image))
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
  initialClickErrors = 0,
  initialAttachments = 0,
  baselineInputCount = 0,
  observedInputCount = baselineInputCount,
} = {}) {
  const actions = []
  let attachments = initialAttachments
  let remainingBusyReads = busyReads
  const discoveryReads = new Map()
  let uploadClickAttempts = 0
  let selector = ''
  const inputSummary = (index) => ({
    index,
    name: 'Filedata',
    type: 'file',
    accept: '',
    disabled: false,
    connected: true,
    files: [],
  })
  const baselineInputs = Array.from(
    { length: baselineInputCount },
    (_, index) => inputSummary(index),
  )
  const observedInputs = Array.from(
    { length: observedInputCount },
    (_, index) => inputSummary(index),
  )
  const page = {
    actions,
    async click(value) {
      actions.push(['click', value])
      if (value === 'button[aria-label="File upload"]') {
        uploadClickAttempts += 1
        if (uploadClickAttempts <= initialClickErrors) {
          throw new Error('upload button was not ready')
        }
      }
    },
    async wait(value) {
      actions.push(['wait', value])
    },
    async evaluate(script) {
      // Parse the exact inner script handed to Browser Bridge. `node --check`
      // only validates this test/module and misses template-literal unescaping
      // that can corrupt a regex before Runtime.evaluate sees it.
      new vm.Script(script)
      if (script.includes("input.setAttribute('data-opencli-video-upload-baseline'")) {
        return { attachments, inputs: baselineInputs }
      }
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
            inputs: observedInputs,
            documentHasFocus,
            busy: uploadBusy,
            busyCount: uploadBusy ? 1 : 0,
            baselineInputCount,
            freshInputCount: 0,
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

function vmUploadDomPage({ initialInputs = [], newInputDelays = [0], reuseInput = false, sidebarBusy = false } = {}) {
  const actions = []
  const inputs = []
  const pendingInputs = []
  const selectedInputIds = []
  let attachmentCount = 0
  let uploadClicks = 0
  let inputSerial = 0
  const clickListeners = new Set()

  const addInput = (name, prefix = 'input') => {
    const attributes = new Map()
    const input = {
      id: `${prefix}-${++inputSerial}`,
      name,
      type: 'file',
      accept: 'image/*',
      disabled: false,
      isConnected: true,
      files: [],
      matches(selector) { return selector === 'input[type="file"]' },
      getAttribute(key) {
        return attributes.has(key) ? attributes.get(key) : null
      },
      setAttribute(key, value) {
        attributes.set(key, String(value))
      },
      removeAttribute(key) {
        attributes.delete(key)
      },
    }
    inputs.push(input)
    return input
  }
  for (const name of initialInputs) addInput(name, 'old')

  const uploadButton = {
    disabled: false,
    isConnected: true,
    getAttribute(key) {
      return key === 'aria-disabled' ? 'false' : null
    },
    getBoundingClientRect() {
      return { width: 120, height: 36 }
    },
  }
  const queryInputs = () => inputs.filter((input) => input.isConnected)
  const querySelectorAll = (selector) => {
    if (selector.includes('input[name="Filedata"]') || selector.includes('input[type="file"]')) {
      return queryInputs()
    }
    if (selector === 'gem-media-attachment') {
      return Array.from({ length: attachmentCount }, () => ({}))
    }
    if (selector.includes('mat-progress-spinner') && sidebarBusy) return [{}]
    if (selector.includes('progressbar') || selector.includes('mat-progress-spinner')) return []
    return []
  }
  const inputContainer = {
    innerText: 'Videos Flash Landscape (16:9)',
    querySelectorAll,
  }
  const document = {
    addEventListener(type, listener) { if (type === 'click') clickListeners.add(listener) },
    removeEventListener(type, listener) { if (type === 'click') clickListeners.delete(listener) },
    querySelector(selector) {
      if (selector === 'input-container') return inputContainer
      if (selector === 'button[aria-label="File upload"]') return uploadButton
      const target = selector.match(/^\[data-opencli-video-upload-target="([^"]+)"\]$/)
      if (target) {
        return queryInputs().find(
          (input) => input.getAttribute('data-opencli-video-upload-target') === target[1],
        ) || null
      }
      return null
    },
    querySelectorAll,
    hasFocus() {
      return true
    },
  }
  const context = {
    document,
    window: {},
    getComputedStyle() {
      return { display: 'block', visibility: 'visible' }
    },
  }

  return {
    actions,
    inputs,
    selectedInputIds,
    clickListeners,
    async click(selector) {
      actions.push(['click', selector])
      uploadClicks += 1
      if (reuseInput) {
        const input = inputs.find(candidate => candidate.name === 'Filedata') || addInput('Filedata', 'reused')
        const event = { target: input, preventDefault() { actions.push(['preventChooser']) } }
        for (const listener of clickListeners) listener(event)
        return
      }
      const delay = Number(newInputDelays[uploadClicks - 1] ?? 0)
      if (delay <= 0) addInput('Filedata', `new-frame-${uploadClicks}`)
      else pendingInputs.push({ remaining: delay, frame: uploadClicks })
    },
    async wait(seconds) {
      actions.push(['wait', seconds])
      for (const pending of pendingInputs) pending.remaining -= 1
      for (let index = pendingInputs.length - 1; index >= 0; index -= 1) {
        const pending = pendingInputs[index]
        if (pending.remaining <= 0) {
          addInput('Filedata', `new-frame-${pending.frame}`)
          pendingInputs.splice(index, 1)
        }
      }
    },
    async evaluate(script) {
      return vm.runInNewContext(script, context)
    },
    async setFileInput(files, selector) {
      const input = document.querySelector(selector)
      if (!input) throw new Error(`target input not found: ${selector}`)
      selectedInputIds.push(input.id)
      actions.push(['setFileInput', files, selector, input.id])
      attachmentCount += 1
      input.files = []
    },
  }
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

test('Gemini video inner discovery ignores an old generic input until delayed Filedata exists', async (t) => {
  const [frame] = videoFrameFixture(t)
  const page = vmUploadDomPage({ initialInputs: ['GenericUpload'], newInputDelays: [2] })

  await uploadFrame(page, frame, 1, {
    attachmentTimeoutMs: 1000,
    inputReadyTimeoutMs: 3000,
    inputPollIntervalMs: 500,
  })

  assert.equal(page.selectedInputIds.length, 1)
  assert.match(page.selectedInputIds[0], /^new-frame-1-/)
  assert.doesNotMatch(page.selectedInputIds[0], /^old-/)
})

test('Gemini video inner discovery ignores stale Filedata until a new input hydrates', async (t) => {
  const [frame] = videoFrameFixture(t)
  const page = vmUploadDomPage({ initialInputs: ['Filedata'], newInputDelays: [2] })

  await uploadFrame(page, frame, 1, {
    attachmentTimeoutMs: 1000,
    inputReadyTimeoutMs: 3000,
    inputPollIntervalMs: 500,
  })

  assert.equal(page.selectedInputIds.length, 1)
  assert.match(page.selectedInputIds[0], /^new-frame-1-/)
  assert.doesNotMatch(page.selectedInputIds[0], /^old-/)
})

test('Gemini video cleans baseline and target markers when discovery wait rejects', async (t) => {
  const [frame] = videoFrameFixture(t)
  const page = vmUploadDomPage({ initialInputs: ['Filedata'], newInputDelays: [2] })
  page.wait = async () => {
    throw new Error('browser bridge navigation interrupted wait')
  }

  await assert.rejects(
    uploadFrame(page, frame, 1, {
      inputReadyTimeoutMs: 3000,
      inputPollIntervalMs: 500,
    }),
    (error) => {
      assert.match(error.message, /keyframe 1 upload failed at discover_live_input/)
      assert.match(error.message, /browser bridge navigation interrupted wait/)
      return true
    },
  )

  for (const input of page.inputs) {
    assert.equal(input.getAttribute('data-opencli-video-upload-baseline'), null)
    assert.equal(input.getAttribute('data-opencli-video-upload-target'), null)
  }
})

test('Gemini video inner discovery selects distinct new inputs for ordered frames', async (t) => {
  const frames = videoFrameFixture(t, ['first-frame.png', 'last-frame.jpg'])
  const page = vmUploadDomPage({ newInputDelays: [0, 2] })

  await uploadFrames(page, frames, {
    attachmentTimeoutMs: 1000,
    inputReadyTimeoutMs: 3000,
    inputPollIntervalMs: 500,
  })

  assert.equal(page.selectedInputIds.length, 2)
  assert.match(page.selectedInputIds[0], /^new-frame-1-/)
  assert.match(page.selectedInputIds[1], /^new-frame-2-/)
  assert.notEqual(page.selectedInputIds[0], page.selectedInputIds[1])
})

test('Gemini video reuses the input activated by its control without leaving native choosers open', async (t) => {
  const frames = videoFrameFixture(t, ['first-frame.png', 'last-frame.jpg'])
  const page = vmUploadDomPage({ initialInputs: ['unrelated'], reuseInput: true, sidebarBusy: true })
  await uploadFrames(page, frames, { attachmentTimeoutMs: 1000 })
  assert.equal(page.selectedInputIds.length, 2)
  assert.equal(page.selectedInputIds[0], page.selectedInputIds[1])
  assert.match(page.selectedInputIds[0], /^reused-/)
  assert.equal(page.actions.filter(([action]) => action === 'preventChooser').length, 2)
  assert.equal(page.clickListeners.size, 0)
  for (const input of page.inputs) {
    assert.equal(input.getAttribute('data-opencli-video-upload-target'), null)
    assert.equal(input.getAttribute('data-opencli-video-upload-baseline'), null)
  }
})

test('Gemini video fails bounded without re-clicking after a successful initial click', async (t) => {
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
      assert.match(error.message, /"reopened":false/)
      assert.match(error.message, /"reason":"initial","ok":true/)
      assert.match(error.message, /"inputs":\[\]/)
      assert.doesNotMatch(error.message, /GEMINI_VIDEO_UPLOAD_INPUT_HYDRATION_STUCK/)
      return true
    },
  )

  assert.equal(page.actions.filter(([action]) => action === 'discoverInput').length, 4)
  assert.equal(page.actions.filter(([action]) => action === 'click').length, 1)
})

test('Gemini video emits a stable fingerprint for exhausted busy input hydration', async (t) => {
  const [frame] = videoFrameFixture(t)
  const page = videoUploadPage({
    native: 'success',
    inputReadyAfterReads: Number.POSITIVE_INFINITY,
    uploadBusy: true,
  })

  await assert.rejects(
    uploadFrame(page, frame, 1, {
      inputReadyTimeoutMs: 3000,
      inputPollIntervalMs: 1000,
    }),
    (error) => {
      assert.match(
        error.message,
        /OPENCLI_CAPABILITY_DEGRADED:GEMINI_VIDEO_UPLOAD_INPUT_HYDRATION_STUCK/,
      )
      assert.match(
        error.message,
        /"stuckSignature":"keyframe=1;attachments=0;busy=1;fresh=0;baseline=0;click=ok;button=ready;focus=1"/,
      )
      assert.match(error.message, /"busy":true/)
      assert.match(error.message, /"freshInputCount":0/)
      return true
    },
  )

  assert.equal(page.actions.filter(([action]) => action === 'click').length, 1)
})

test('Gemini video fingerprints a stuck second keyframe with only its stale baseline input', async (t) => {
  const [frame] = videoFrameFixture(t, ['last-frame.jpg'])
  const page = videoUploadPage({
    native: 'success',
    initialAttachments: 1,
    baselineInputCount: 1,
    inputReadyAfterReads: Number.POSITIVE_INFINITY,
    uploadBusy: true,
  })

  await assert.rejects(
    uploadFrame(page, frame, 2, {
      inputReadyTimeoutMs: 3000,
      inputPollIntervalMs: 1000,
    }),
    (error) => {
      assert.match(
        error.message,
        /OPENCLI_CAPABILITY_DEGRADED:GEMINI_VIDEO_UPLOAD_INPUT_HYDRATION_STUCK/,
      )
      assert.match(
        error.message,
        /"stuckSignature":"keyframe=2;attachments=1;busy=1;fresh=0;baseline=1;click=ok;button=ready;focus=1"/,
      )
      assert.match(error.message, /"baselineInputCount":1/)
      assert.match(error.message, /"freshInputCount":0/)
      return true
    },
  )

  assert.equal(page.actions.filter(([action]) => action === 'click').length, 1)
})

test('Gemini video does not fingerprint hydration when a non-baseline input exists', async (t) => {
  const [frame] = videoFrameFixture(t, ['last-frame.jpg'])
  const page = videoUploadPage({
    native: 'success',
    initialAttachments: 1,
    baselineInputCount: 1,
    observedInputCount: 2,
    inputReadyAfterReads: Number.POSITIVE_INFINITY,
    uploadBusy: true,
  })

  await assert.rejects(
    uploadFrame(page, frame, 2, {
      inputReadyTimeoutMs: 3000,
      inputPollIntervalMs: 1000,
    }),
    (error) => {
      assert.doesNotMatch(
        error.message,
        /OPENCLI_CAPABILITY_DEGRADED:GEMINI_VIDEO_UPLOAD_INPUT_HYDRATION_STUCK/,
      )
      assert.doesNotMatch(error.message, /"stuckSignature":/)
      return true
    },
  )
})

test('Gemini video retries once only when the initial upload click throws', async (t) => {
  const [frame] = videoFrameFixture(t)
  const page = videoUploadPage({
    native: 'success',
    inputReadyAfterReads: Number.POSITIVE_INFINITY,
    initialClickErrors: 1,
  })

  await assert.rejects(
    uploadFrame(page, frame, 1, {
      inputReadyTimeoutMs: 3000,
      inputPollIntervalMs: 1000,
      inputReopenAfterMs: 1000,
    }),
    (error) => {
      assert.match(error.message, /"reopened":true/)
      assert.match(error.message, /"reason":"initial","ok":false/)
      assert.match(error.message, /"reason":"hydration_retry","ok":true/)
      return true
    },
  )

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
  let composerText = ''
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
      if (script.includes('hasText: actual.length > 0')) {
        return { hasText: composerText.length > 0, actual: composerText }
      }
      if (script.includes('inputText') && script.includes('InputEvent')) {
        return { hasText: composerText.length > 0, actual: composerText }
      }
      if (script.includes('Could not find Gemini composer')) return { ok: true }
      throw new Error(`Unexpected Gemini evaluate script: ${String(script).slice(0, 100)}`)
    },
    async nativeType(text) {
      actions.push(['nativeType', text])
      composerText += text
    },
    async fillText(selector, text) {
      actions.push(['fillText', selector, text])
      composerText = text
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
      composerText = ''
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

test('Gemini retries a no-op selector click with observed native coordinates', async () => {
  const actions = []
  let composerText = ''
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
      if (script.includes('hasText: actual.length > 0')) {
        return { hasText: composerText.length > 0, actual: composerText }
      }
      if (script.includes('inputText') && script.includes('InputEvent')) {
        return { hasText: composerText.length > 0, actual: composerText }
      }
      if (script.includes('Could not find Gemini composer')) return { ok: true }
      throw new Error(`Unexpected Gemini evaluate script: ${String(script).slice(0, 100)}`)
    },
    async nativeType(text) {
      actions.push(['nativeType', text])
      composerText += text
    },
    async fillText(selector, text) {
      actions.push(['fillText', selector, text])
      composerText = text
      return { verified: true, actual: text }
    },
    async nativeClick(x, y) {
      actions.push(['nativeClick', x, y])
      composerText = ''
    },
    async nativeKeyPress(key) {
      actions.push(['nativeKeyPress', key])
    },
    async click(selector) {
      actions.push(['click', selector])
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
    ['nativeClick', 123, 456],
  ])
})

test('Gemini ask waits through delayed model-picker hydration', async () => {
  const prompt = 'Reply with exactly READY'
  let pickerReads = 0
  let composerText = ''
  let submitted = false
  const snapshot = () => submitted
    ? {
        url: 'https://gemini.google.com/app/delayed-picker',
        turns: [
          { Role: 'User', Text: prompt },
          { Role: 'Assistant', Text: 'READY' },
        ],
        transcriptLines: [prompt, 'READY'],
        composerHasText: false,
        isGenerating: false,
        structuredTurnsTrusted: true,
      }
    : {
        url: 'https://gemini.google.com/app',
        turns: [],
        transcriptLines: [],
        composerHasText: false,
        isGenerating: false,
        structuredTurnsTrusted: true,
      }
  const page = {
    async evaluate(script) {
      if (script.includes("return ['/app', '/app/']")) return true
      if (script === 'window.location.href') return 'https://gemini.google.com/app'
      if (script.includes('Gemini model picker button was not found')) {
        pickerReads += 1
        return pickerReads < 10
          ? { ok: false, reason: 'Gemini model picker button was not found' }
          : { ok: true }
      }
      if (script.includes('results.push({ model: modelId, thinkingValues: [] })')) {
        return [{ model: '3.7-flash', thinkingValues: [] }]
      }
      if (script.includes("reason: 'Model picker not found'")) return { ok: true }
      if (script.includes('const targetModelId = "3.7-flash"')) return { ok: true }
      if (script.includes('return canonicalModelId(combined)')) return '3.7-flash'
      if (script.includes('structuredTurnsTrusted')) return snapshot()
      if (script.includes('bestButton instanceof HTMLElement')) {
        return { action: 'button', label: 'Send message', x: 123, y: 456 }
      }
      if (script.includes('hasText: actual.length > 0')) {
        return { hasText: composerText.length > 0, actual: composerText }
      }
      if (script.includes('Could not find Gemini composer')) return { ok: true }
      if (script.includes('document.body.click()')) return undefined
      throw new Error(`Unexpected Gemini ask script: ${String(script).slice(0, 120)}`)
    },
    async goto() {},
    async wait() {},
    async fillText(_selector, text) {
      composerText = text
      return { verified: true, actual: text }
    },
    async click() {
      composerText = ''
      submitted = true
    },
  }

  const result = await geminiAskCommand.func(page, {
    prompt,
    model: '3.7-flash',
    timeout: 30,
    new: 'true',
    thinking: null,
  })

  assert.equal(pickerReads, 10)
  assert.deepEqual(result, [{ response: '💬 READY' }])
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

for (const anchored of [true, false]) {
test(`Gemini waits through speaker placeholders and ignores a trailing label (anchored=${anchored})`, async () => {
  const prompt = 'Review the supplied image.'
  const user = { Role: 'User', Text: prompt }
  const label = { Role: 'Assistant', Text: 'Gemini said' }
  const answer = { Role: 'Assistant', Text: 'Gemini said\n{"image_received":true,"reviews":[]}' }
  let reads = 0
  const baseline = { url: 'https://gemini.google.com/app/current', turns: [user],
    transcriptLines: [prompt], composerHasText: false, isGenerating: false,
    structuredTurnsTrusted: true }
  const page = {
    async wait() {},
    async evaluate(script) {
      if (script === 'window.location.href') return baseline.url
      if (script.includes('structuredTurnsTrusted')) {
        reads++
        return { ...baseline, turns: [user, ...(reads >= 4 ? [answer] : []), label],
          transcriptLines: [prompt, 'Gemini said'] }
      }
      throw new Error('Unexpected Gemini snapshot script')
    },
  }
  const result = await waitForGeminiResponse(page, {
    snapshot: baseline, userAnchorTurn: anchored ? user : null,
  }, prompt, 20)
  assert.equal(reads, 5)
  assert.equal(result, '{"image_received":true,"reviews":[]}')
})
}

test('Gemini never returns a speaker placeholder as a completed response', async () => {
  const user = { Role: 'User', Text: 'Review the image.' }
  const baseline = { url: 'https://gemini.google.com/app/current', turns: [user],
    transcriptLines: [user.Text], composerHasText: false, isGenerating: false,
    structuredTurnsTrusted: true }
  const page = {
    async wait() {},
    async evaluate(script) {
      if (script === 'window.location.href') return baseline.url
      if (script.includes('structuredTurnsTrusted')) return { ...baseline,
        turns: [user, { Role: 'Assistant', Text: 'Gemini said' }],
        transcriptLines: [user.Text, 'Gemini said'] }
      throw new Error('Unexpected Gemini snapshot script')
    },
  }
  assert.equal(await waitForGeminiResponse(page, {
    snapshot: baseline, userAnchorTurn: user,
  }, user.Text, 10), '')
})

test('Gemini does not click twice when the first click submits but reports an error', async () => {
  let composerText = ''
  let clickCount = 0
  const page = {
    async evaluate(script) {
      if (script === 'window.location.href') return 'https://gemini.google.com/app/test'
      if (script.includes('bestButton instanceof HTMLElement')) {
        return { action: 'button', label: 'Send message', x: 123, y: 456 }
      }
      if (script.includes('hasText: actual.length > 0')) {
        return { hasText: composerText.length > 0, actual: composerText }
      }
      if (script.includes('Could not find Gemini composer')) return { ok: true }
      throw new Error(`Unexpected Gemini evaluate script: ${String(script).slice(0, 100)}`)
    },
    async fillText(_selector, text) {
      composerText = text
      return { verified: true, actual: text }
    },
    async click() {
      clickCount += 1
      composerText = ''
      throw new Error('transport response was lost after dispatch')
    },
    async wait() {},
  }

  const result = await sendGeminiMessage(page, 'hello')

  assert.equal(result, 'button')
  assert.equal(clickCount, 1)
})

test('Gemini uses a fully specified Enter key after ready send-button clicks have no effect', async () => {
  let composerText = ''
  const keys = []
  class Element {}
  const document = { activeElement: null, querySelector(selector) {
    if (selector.includes('Stop response')) return null
    return selector.includes('composer') ? composer : button
  } }
  const composer = Object.assign(new Element(), { get innerText() { return composerText }, focus() { document.activeElement = composer } })
  Object.defineProperty(composer, 'innerText', { get: () => composerText })
  const button = Object.assign(new Element(), { textContent: 'Send message', disabled: false, getAttribute: () => null, getBoundingClientRect: () => ({ width: 32, height: 32 }), click() {} })
  const page = {
    async evaluate(script) {
      if (script === 'window.location.href') return 'https://gemini.google.com/app'
      if (script.includes('bestButton instanceof HTMLElement')) return { action: 'button', label: 'Send message', x: 40, y: 50 }
      if (script.includes('hasText: actual.length > 0')) return { hasText: !!composerText, actual: composerText }
      if (script.includes('Could not find Gemini composer')) return { ok: true }
      return vm.runInNewContext(script, { document, HTMLElement: Element, getComputedStyle: () => ({ display: 'block', visibility: 'visible' }) })
    },
    async fillText(_selector, text) { composerText = text },
    async click() {}, async nativeClick() {}, async wait() {},
    async cdp(method, params) { keys.push({ method, ...params }); if (params.type === 'keyDown') composerText = '' },
  }
  assert.equal(await sendGeminiMessage(page, 'hello'), 'enter')
  assert.deepEqual(keys.map(key => [key.method, key.type, key.code, key.windowsVirtualKeyCode]), [
    ['Input.dispatchKeyEvent', 'keyDown', 'Enter', 13], ['Input.dispatchKeyEvent', 'keyUp', 'Enter', 13],
  ])
})

for (const scenario of ['ready', 'disabled', 'generating', 'changed-text']) {
test(`Gemini DOM submit recovery respects ${scenario} composer state`, async () => {
  let composerText = ''
  let domClicks = 0
  class Element {}
  const button = Object.assign(new Element(), {
    disabled: scenario === 'disabled',
    textContent: scenario === 'generating' ? 'Stop response' : 'Send message',
    getAttribute() { return null },
    getBoundingClientRect() { return { width: 48, height: 48 } },
    click() { domClicks += 1; composerText = '' },
  })
  const page = {
    async evaluate(script) {
      if (script === 'window.location.href') return 'https://gemini.google.com/app/test'
      if (script.includes('bestButton instanceof HTMLElement')) {
        return { action: 'button', label: 'Send message', x: 123, y: 456 }
      }
      if (script.includes('hasText: actual.length > 0')) return { hasText: !!composerText, actual: composerText }
      if (script.includes('Could not find Gemini composer')) return { ok: true }
      if (script.includes('const expectedComposerText')) {
        return vm.runInNewContext(script, {
          HTMLElement: Element,
          document: { querySelector: (selector) => selector.includes('composer') ? { innerText: composerText } : button },
          getComputedStyle: () => ({ display: 'block', visibility: 'visible' }),
        })
      }
      if (script.includes('verticalDistance: rect ?')) return { buttons: [{ label: button.textContent, disabled: button.disabled }], busy: false, alerts: [] }
      throw new Error(`Unexpected Gemini evaluate script: ${String(script).slice(0, 100)}`)
    },
    async fillText(_selector, text) { composerText = text; return { verified: true, actual: text } },
    async click() { if (scenario === 'changed-text') composerText = 'A different prompt' },
    async nativeClick() {},
    async wait() {},
  }
  if (scenario === 'ready') {
    assert.equal(await sendGeminiMessage(page, 'hello'), 'button')
    assert.equal(domClicks, 1)
  } else {
    await assert.rejects(sendGeminiMessage(page, 'hello'), error => {
      assert.match(error.message, /did not accept the composer submission/)
      const diagnostic = JSON.parse(error.message.split('submission: ')[1])
      assert.equal(diagnostic.exact, scenario !== 'changed-text')
      assert.equal(diagnostic.expectedLength, 5)
      assert.equal(diagnostic.controls.buttons[0].disabled, scenario === 'disabled')
      assert.equal(error.message.includes('A different prompt'), false)
      return true
    })
    assert.equal(domClicks, 0)
  }
})
}

test('Gemini waits for a delayed composer-clear acknowledgement after submit', async () => {
  let composerText = ''
  let submitted = false
  let acknowledgementPolls = 0
  const page = {
    async evaluate(script) {
      if (script === 'window.location.href') return 'https://gemini.google.com/app/test'
      if (script.includes('bestButton instanceof HTMLElement')) {
        return { action: 'button', label: 'Send message', x: 123, y: 456 }
      }
      if (script.includes('hasText: actual.length > 0')) {
        return { hasText: composerText.length > 0, actual: composerText }
      }
      if (script.includes('Could not find Gemini composer')) return { ok: true }
      throw new Error(`Unexpected Gemini evaluate script: ${String(script).slice(0, 100)}`)
    },
    async fillText(_selector, text) {
      composerText = text
      return { verified: true, actual: text }
    },
    async click() {
      submitted = true
    },
    async wait() {
      if (!submitted) return
      acknowledgementPolls += 1
      if (acknowledgementPolls === 20) composerText = ''
    },
  }

  assert.equal(await sendGeminiMessage(page, 'hello'), 'button')
  assert.equal(acknowledgementPolls, 20)
})

test('Gemini accepts exact DOM text when fill verification is unavailable without appending', async () => {
  const actions = []
  let composerText = ''
  let submittedText = ''
  const page = {
    async evaluate(script) {
      if (script === 'window.location.href') return 'https://gemini.google.com/app/test'
      if (script.includes('bestButton instanceof HTMLElement')) {
        return { action: 'button', label: 'Send message', x: 123, y: 456 }
      }
      if (script.includes('Could not find Gemini composer')) {
        composerText = ''
        return { ok: true }
      }
      if (script.includes('hasText: actual.length > 0')) {
        return { hasText: composerText.length > 0, actual: composerText }
      }
      throw new Error(`Unexpected Gemini evaluate script: ${String(script).slice(0, 100)}`)
    },
    async fillText(_selector, text) {
      actions.push('fillText')
      composerText = text
      return { verified: false, actual: '' }
    },
    async nativeType(text) {
      actions.push('nativeType')
      composerText += text
    },
    async click() {
      submittedText = composerText
      composerText = ''
    },
    async wait() {},
  }

  assert.equal(await sendGeminiMessage(page, 'hello world'), 'button')
  assert.deepEqual(actions, ['fillText'])
  assert.equal(submittedText, 'hello world')
})

test('Gemini clears partial fill text before native typing the exact prompt once', async () => {
  const actions = []
  let composerText = ''
  let submittedText = ''
  let prepareCalls = 0
  const page = {
    async evaluate(script) {
      if (script === 'window.location.href') return 'https://gemini.google.com/app/test'
      if (script.includes('bestButton instanceof HTMLElement')) {
        return { action: 'button', label: 'Send message', x: 123, y: 456 }
      }
      if (script.includes('Could not find Gemini composer')) {
        prepareCalls += 1
        composerText = ''
        return { ok: true }
      }
      if (script.includes('hasText: actual.length > 0')) {
        return { hasText: composerText.length > 0, actual: composerText }
      }
      throw new Error(`Unexpected Gemini evaluate script: ${String(script).slice(0, 100)}`)
    },
    async fillText() {
      actions.push('fillText')
      composerText = 'hello'
      return { verified: false, actual: 'hello' }
    },
    async nativeType(text) {
      actions.push('nativeType')
      composerText += text
    },
    async click() {
      submittedText = composerText
      composerText = ''
    },
    async wait() {},
  }

  assert.equal(await sendGeminiMessage(page, 'hello world'), 'button')
  assert.deepEqual(actions, ['fillText', 'nativeType'])
  assert.equal(prepareCalls, 2)
  assert.equal(submittedText, 'hello world')
})

test('ChatGPT model navigates a fresh lease before evaluating the page', async () => {
  const actions = []
  const page = {
    async goto(url) { actions.push(['goto', url]) },
    async nativeClick() {},
    async evaluate(script) {
      actions.push(['evaluate', script])
      if (script === 'window.location.href') return 'https://chatgpt.com/'
      if (script.includes('hasComposer') && script.includes('hasLoginGate')) {
        return { hasComposer: true, isLoggedIn: true, hasLoginGate: false }
      }
      if (script.includes('findEntryForText')) return { model: 'balanced', label: 'Medium' }
      throw new Error(`Unexpected script: ${script.slice(0, 80)}`)
    },
  }
  const result = await chatgptModelCommand.func(page, { model: 'medium' })
  assert.deepEqual(actions[0], ['goto', 'https://chatgpt.com'])
  assert.deepEqual(result, [{ Status: 'Already selected', Model: 'Medium' }])
  assert.equal(chatgptModelCommand.args.find(arg => arg.name === 'timeout').default, 45)
})

test('ChatGPT reads split version badges and switches Pro through the real selector DOM', async (t) => {
  const saved = [process.env.OPENCLI_CHATGPT_MODEL_MIN, process.env.OPENCLI_CHATGPT_MODEL_MAX]
  process.env.OPENCLI_CHATGPT_MODEL_MIN = 'medium'
  process.env.OPENCLI_CHATGPT_MODEL_MAX = 'xhigh'
  t.after(() => {
    for (const [index, key] of ['OPENCLI_CHATGPT_MODEL_MIN', 'OPENCLI_CHATGPT_MODEL_MAX'].entries()) {
      if (saved[index] === undefined) delete process.env[key]
      else process.env[key] = saved[index]
    }
  })
  class Element {
    constructor(text = '', innerText = text) { this.textContent = text; this.innerText = innerText }
    getBoundingClientRect() { return { left: 0, top: 0, width: 100, height: 40 } }
    getAttribute() { return null }
    setAttribute() {}
    scrollIntoView() {}
    closest() { return form }
    querySelector() { return null }
    querySelectorAll(selector) { return selector === 'button' ? [button] : [] }
  }
  const button = new Element('6Pro', '6\nPro')
  const form = new Element()
  const context = {
    HTMLElement: Element,
    window: { getComputedStyle: () => ({ display: 'block', visibility: 'visible' }) },
    document: {
      querySelector: selector => selector.includes('data-opencli-chatgpt-composer') ? button : null,
      querySelectorAll: selector => selector === 'form' ? [form] : [],
    },
  }
  let clicks = 0
  const keys = []
  const page = {
    async evaluate(script) {
      if (script === 'window.location.href') return 'https://chatgpt.com/'
      if (script.includes('hasComposer') && script.includes('hasLoginGate')) {
        return { hasComposer: true, isLoggedIn: true, hasLoginGate: false }
      }
      if (script.includes('findEntryForText') || script.includes('menuButtonSelectors')) {
        return vm.runInNewContext(script, context)
      }
      if (script.includes('contentFound')) return { ready: clicks > 0, expanded: clicks > 0 }
      if (script.includes('keyboardTarget')) return { found: true, current: 4, minimum: 0, maximum: 4 }
      throw new Error(`Unexpected script: ${script.slice(0, 80)}`)
    },
    async nativeClick() { clicks++ },
    async pressKey(key) {
      keys.push(key)
      if (keys.filter(value => value === 'ArrowLeft').length === 3) {
        button.textContent = '6Medium'
        button.innerText = '6\nMedium'
      }
    },
    async wait() {},
  }
  assert.equal((await getCurrentChatGPTModel(page)).model, 'pro')
  assert.deepEqual(await selectChatGPTModel(page, 'medium'), { Status: 'Success', Model: 'Medium' })
  assert.equal(clicks, 1)
  assert.deepEqual(keys, ['ArrowLeft', 'ArrowLeft', 'ArrowLeft', 'Escape'])
  for (const [label, expected] of [['High', 'advanced'], ['Extra High', 'very-high'], ['Instant', 'fast']]) {
    button.textContent = `6${label}`
    button.innerText = `6\n${label}`
    assert.equal((await getCurrentChatGPTModel(page)).model, expected)
  }
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

test('ChatGPT falls back to an allowed current level after a failed preferred switch', async (t) => {
  const savedMinimum = process.env.OPENCLI_CHATGPT_MODEL_MIN
  const savedMaximum = process.env.OPENCLI_CHATGPT_MODEL_MAX
  process.env.OPENCLI_CHATGPT_MODEL_MIN = 'medium'
  process.env.OPENCLI_CHATGPT_MODEL_MAX = 'xhigh'
  t.after(() => {
    if (savedMinimum === undefined) delete process.env.OPENCLI_CHATGPT_MODEL_MIN
    else process.env.OPENCLI_CHATGPT_MODEL_MIN = savedMinimum
    if (savedMaximum === undefined) delete process.env.OPENCLI_CHATGPT_MODEL_MAX
    else process.env.OPENCLI_CHATGPT_MODEL_MAX = savedMaximum
  })

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
        return { model: 'very-high', label: 'Very High' }
      }
      if (script.includes('menuButtonSelectors')) return { found: true, x: 100, y: 50 }
      if (script.includes('contentFound')) {
        return { ready: true, expanded: true, contentFound: true }
      }
      if (script.includes('keyboardTarget')) {
        return { found: true, current: 3, minimum: 0, maximum: 4 }
      }
      throw new Error(`Unexpected ChatGPT evaluate script: ${String(script).slice(0, 80)}`)
    },
    async nativeClick() {},
    async pressKey(key) {
      pressed.push(key)
      // Simulate a UI that accepted key events but retained Extra High.
    },
    async wait() {},
  }

  const result = await selectChatGPTModel(page, 'medium')

  assert.deepEqual(result, { Status: 'Policy fallback', Model: 'Very High' })
  assert.deepEqual(pressed, ['ArrowLeft', 'ArrowLeft', 'Escape'])
})

test('ChatGPT closes an already-open model menu when the preferred level is selected', async (t) => {
  const savedMinimum = process.env.OPENCLI_CHATGPT_MODEL_MIN
  const savedMaximum = process.env.OPENCLI_CHATGPT_MODEL_MAX
  process.env.OPENCLI_CHATGPT_MODEL_MIN = 'medium'
  process.env.OPENCLI_CHATGPT_MODEL_MAX = 'xhigh'
  t.after(() => {
    if (savedMinimum === undefined) delete process.env.OPENCLI_CHATGPT_MODEL_MIN
    else process.env.OPENCLI_CHATGPT_MODEL_MIN = savedMinimum
    if (savedMaximum === undefined) delete process.env.OPENCLI_CHATGPT_MODEL_MAX
    else process.env.OPENCLI_CHATGPT_MODEL_MAX = savedMaximum
  })

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
        return { model: 'balanced', label: 'Medium' }
      }
      throw new Error(`Unexpected ChatGPT evaluate script: ${String(script).slice(0, 80)}`)
    },
    async nativeClick() {},
    async pressKey(key) {
      pressed.push(key)
    },
  }

  const result = await selectChatGPTModel(page, 'medium')

  assert.deepEqual(result, { Status: 'Already selected', Model: 'Medium' })
  assert.deepEqual(pressed, ['Escape'])
})

test('ChatGPT accepts a known in-range current level when the selector is unavailable', async (t) => {
  const savedMinimum = process.env.OPENCLI_CHATGPT_MODEL_MIN
  const savedMaximum = process.env.OPENCLI_CHATGPT_MODEL_MAX
  process.env.OPENCLI_CHATGPT_MODEL_MIN = 'medium'
  process.env.OPENCLI_CHATGPT_MODEL_MAX = 'xhigh'
  t.after(() => {
    if (savedMinimum === undefined) delete process.env.OPENCLI_CHATGPT_MODEL_MIN
    else process.env.OPENCLI_CHATGPT_MODEL_MIN = savedMinimum
    if (savedMaximum === undefined) delete process.env.OPENCLI_CHATGPT_MODEL_MAX
    else process.env.OPENCLI_CHATGPT_MODEL_MAX = savedMaximum
  })

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
        return { model: 'advanced', label: 'Advanced' }
      }
      if (script.includes('menuButtonSelectors')) return { found: false }
      throw new Error(`Unexpected ChatGPT evaluate script: ${String(script).slice(0, 80)}`)
    },
    async nativeClick() {},
    async wait() {},
  }

  const result = await selectChatGPTModel(page, 'medium')

  assert.deepEqual(result, { Status: 'Policy fallback', Model: 'Advanced' })
})

test('ChatGPT waits for a delayed model selector before switching', async (t) => {
  const savedMinimum = process.env.OPENCLI_CHATGPT_MODEL_MIN
  const savedMaximum = process.env.OPENCLI_CHATGPT_MODEL_MAX
  process.env.OPENCLI_CHATGPT_MODEL_MIN = 'medium'
  process.env.OPENCLI_CHATGPT_MODEL_MAX = 'xhigh'
  t.after(() => {
    if (savedMinimum === undefined) delete process.env.OPENCLI_CHATGPT_MODEL_MIN
    else process.env.OPENCLI_CHATGPT_MODEL_MIN = savedMinimum
    if (savedMaximum === undefined) delete process.env.OPENCLI_CHATGPT_MODEL_MAX
    else process.env.OPENCLI_CHATGPT_MODEL_MAX = savedMaximum
  })

  let menuReads = 0
  let currentModel = null
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
        return currentModel
          ? { model: currentModel, label: currentModel === 'balanced' ? 'Medium' : 'Advanced' }
          : { model: null, label: null }
      }
      if (script.includes('menuButtonSelectors')) {
        menuReads += 1
        return menuReads >= 3 ? { found: true, x: 100, y: 50 } : { found: false }
      }
      if (script.includes('contentFound')) {
        return { ready: true, expanded: true, contentFound: true }
      }
      if (script.includes('keyboardTarget')) {
        return { found: true, current: 2, minimum: 0, maximum: 4 }
      }
      throw new Error(`Unexpected ChatGPT evaluate script: ${String(script).slice(0, 80)}`)
    },
    async nativeClick() {},
    async pressKey(key) {
      pressed.push(key)
      if (key === 'ArrowLeft') currentModel = 'balanced'
    },
    async wait() {},
  }

  const result = await selectChatGPTModel(page, 'medium')

  assert.deepEqual(result, { Status: 'Success', Model: 'Medium' })
  assert.equal(menuReads, 3)
  assert.deepEqual(pressed, ['ArrowLeft', 'Escape'])
})

test('ChatGPT never falls back to Pro when the preferred switch fails', async (t) => {
  const savedMinimum = process.env.OPENCLI_CHATGPT_MODEL_MIN
  const savedMaximum = process.env.OPENCLI_CHATGPT_MODEL_MAX
  process.env.OPENCLI_CHATGPT_MODEL_MIN = 'medium'
  process.env.OPENCLI_CHATGPT_MODEL_MAX = 'xhigh'
  t.after(() => {
    if (savedMinimum === undefined) delete process.env.OPENCLI_CHATGPT_MODEL_MIN
    else process.env.OPENCLI_CHATGPT_MODEL_MIN = savedMinimum
    if (savedMaximum === undefined) delete process.env.OPENCLI_CHATGPT_MODEL_MAX
    else process.env.OPENCLI_CHATGPT_MODEL_MAX = savedMaximum
  })

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
      if (script.includes('findEntryForText')) return { model: 'pro', label: 'Pro' }
      if (script.includes('menuButtonSelectors')) return { found: true, x: 100, y: 50 }
      if (script.includes('contentFound')) {
        return { ready: true, expanded: true, contentFound: true }
      }
      if (script.includes('keyboardTarget')) {
        return { found: true, current: 4, minimum: 0, maximum: 4 }
      }
      throw new Error(`Unexpected ChatGPT evaluate script: ${String(script).slice(0, 80)}`)
    },
    async nativeClick() {},
    async pressKey(key) {
      pressed.push(key)
      // Simulate a failed attempt that leaves Pro selected.
    },
    async wait() {},
  }

  await assert.rejects(
    selectChatGPTModel(page, 'medium'),
    /did not switch to Medium/,
  )
  assert.deepEqual(pressed, ['ArrowLeft', 'ArrowLeft', 'ArrowLeft'])
})

test('ChatGPT never falls back to Instant when the preferred switch fails', async (t) => {
  const savedMinimum = process.env.OPENCLI_CHATGPT_MODEL_MIN
  const savedMaximum = process.env.OPENCLI_CHATGPT_MODEL_MAX
  process.env.OPENCLI_CHATGPT_MODEL_MIN = 'medium'
  process.env.OPENCLI_CHATGPT_MODEL_MAX = 'xhigh'
  t.after(() => {
    if (savedMinimum === undefined) delete process.env.OPENCLI_CHATGPT_MODEL_MIN
    else process.env.OPENCLI_CHATGPT_MODEL_MIN = savedMinimum
    if (savedMaximum === undefined) delete process.env.OPENCLI_CHATGPT_MODEL_MAX
    else process.env.OPENCLI_CHATGPT_MODEL_MAX = savedMaximum
  })

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
      if (script.includes('findEntryForText')) return { model: 'fast', label: 'Fast' }
      if (script.includes('menuButtonSelectors')) return { found: true, x: 100, y: 50 }
      if (script.includes('contentFound')) {
        return { ready: true, expanded: true, contentFound: true }
      }
      if (script.includes('keyboardTarget')) {
        return { found: true, current: 0, minimum: 0, maximum: 4 }
      }
      throw new Error(`Unexpected ChatGPT evaluate script: ${String(script).slice(0, 80)}`)
    },
    async nativeClick() {},
    async pressKey(key) {
      pressed.push(key)
      // Simulate a failed attempt that leaves Instant selected.
    },
    async wait() {},
  }

  await assert.rejects(
    selectChatGPTModel(page, 'medium'),
    /did not switch to Medium/,
  )
  assert.deepEqual(pressed, ['ArrowRight'])
})

test('Gemini attachment retries an ignored menu click using fresh native coordinates', async (t) => {
  const [image] = videoFrameFixture(t, ['contact-sheet.jpg'])
  const page = geminiAttachmentPage('fileChooserOpened not received')
  const originalEvaluate = page.evaluate
  let opened = false
  page.evaluate = async (script) => {
    if (script.includes('const inputSelector = selectors.find') && !opened) {
      return { inputSelector: '', buttonSelector: '', expanded: false }
    }
    if (script.includes('const uploadTarget') || script.includes('button.getBoundingClientRect()')) {
      return { x: 45, y: 67 }
    }
    return originalEvaluate(script)
  }
  page.nativeClick = async (x, y) => {
    assert.deepEqual([x, y], [45, 67])
    opened = true
  }
  await attachGeminiFile(page, image)
  assert.equal(opened, true)
  assert.equal(page.actions.filter(([action]) => action === 'setFileInput').length, 1)
})

for (const generating of [false, true]) {
  test(`ChatGPT explicit refresh never reloads a generating target: ${generating}`, async () => {
    const navigations = []
    let reloads = 0
    const page = {
      async goto(url) { navigations.push(url) },
      async wait() {},
      async evaluate(script) {
        if (script.includes('window.location.reload()')) { reloads += 1; return true }
        if (script === 'window.location.href') return 'https://chatgpt.com/c/abcdefgh1234'
        if (script.includes('isLoggedIn')) return { isLoggedIn: true, hasLoginGate: false }
        if (script.includes('const roleOf')) return [{ role: 'assistant', text: 'W5B@5.1;6P', html: '' }]
        if (script.includes('stop-button')) return generating
        throw new Error(`Unexpected evaluate: ${script.slice(0, 100)}`)
      },
    }
    const rows = await chatgptDetailCommand.func(page, { id: 'abcdefgh1234', refresh: true, wait: false })
    assert.equal(rows[0].Text, 'W5B@5.1;6P')
    assert.equal(navigations.length, 0)
    assert.equal(reloads, generating ? 0 : 1)
  })
}

for (const quoted of [false, true]) {
  test(`ChatGPT detects the screenshot limit modal but ignores quoted content: ${quoted}`, async () => {
    class Element {
      textContent = 'Too many requests. You’re making requests too quickly. We’ve temporarily limited access to your conversations to protect your data.'
      closest() { return quoted ? {} : null }
      getBoundingClientRect() { return { width: 500, height: 200 } }
    }
    const dialog = new Element()
    const page = {
      async evaluate(script) {
        return vm.runInNewContext(script, {
          HTMLElement: Element,
          window: { location: { href: 'https://chatgpt.com/c/owned' }, getComputedStyle() { return {} } },
          document: {
            querySelector() { return null },
            querySelectorAll(selector) { return selector.startsWith('dialog,') ? [dialog] : [] },
          },
        })
      },
    }
    if (quoted) assert.equal(await isGenerating(page), false)
    else await assert.rejects(isGenerating(page), /CHATGPT_RATE_LIMITED.*https:\/\/chatgpt.com\/c\/owned/)
  })
}

for (const staleLimit of [false, true]) {
  test(`Post-cooldown recovery reloads only a still-visible blocking dialog: ${staleLimit}`, async () => {
    let reloads = 0
    const page = {
      async goto() { throw new Error('Must keep the original conversation') },
      async wait() {},
      async evaluate(script) {
        if (script === 'window.location.href') return 'https://chatgpt.com/c/abcdefgh1234'
        if (script.includes('catch (error)')) return staleLimit
        if (script.includes('window.location.reload()')) { reloads += 1; return true }
        if (script.includes('isLoggedIn')) return { isLoggedIn: true, hasLoginGate: false }
        if (script.includes('const roleOf')) return [{ role: 'assistant', text: 'W5P;6P', html: '' }]
        if (script.includes('stop-button')) return !staleLimit
        throw new Error(`Unexpected evaluate: ${script.slice(0, 100)}`)
      },
    }
    await chatgptDetailCommand.func(page, { id: 'abcdefgh1234', cooldown: true, wait: false })
    assert.equal(reloads, staleLimit ? 1 : 0)
  })
}

for (const initiallyExpanded of [false, true]) {
test(`Gemini attachment recovers an empty menu after ignored native clicks: expanded=${initiallyExpanded}`, async (t) => {
  const [image] = videoFrameFixture(t, ['contact-sheet.jpg'])
  const page = geminiAttachmentPage('fileChooserOpened not received')
  const originalEvaluate = page.evaluate
  let opened = false
  let expanded = initiallyExpanded
  let domClicks = 0
  const button = {
    disabled: false,
    getAttribute: () => String(expanded),
    click() { expanded = !expanded; if (expanded) opened = true; domClicks++ },
  }
  page.evaluate = async script => {
    if (script.includes('button.click();')) {
      return vm.runInNewContext(script, { document: { querySelector: () => button } })
    }
    if (script.includes('const inputSelector = selectors.find') && !opened) {
      return { inputSelector: '', buttonSelector: '', expanded }
    }
    if (script.includes('button.getBoundingClientRect()')) return { x: 45, y: 67 }
    return originalEvaluate(script)
  }
  page.nativeClick = async () => {}
  await attachGeminiFile(page, image)
  assert.equal(domClicks, initiallyExpanded ? 2 : 1)
  assert.equal(page.actions.filter(([action]) => action === 'setFileInput').length, 1)
})
}

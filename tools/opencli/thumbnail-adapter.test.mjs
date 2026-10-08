import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import vm from 'node:vm'
import test, { after } from 'node:test'
import { execFileSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

// Apply the complete postinstall patch only to a disposable adapter tree.
// Nothing imports the installed adapter or contacts the browser/daemon.
const runtime = path.dirname(fileURLToPath(import.meta.url))
const installed = path.join(runtime, 'node_modules/@jackwener/opencli')
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'thumbnail-adapter-'))
const copied = path.join(temporary, 'node_modules/@jackwener/opencli')
const files = [
  'package.json', 'cli-manifest.json', 'dist/src/execution.js',
  ...['ask', 'utils', 'models', 'video'].map(name => `clis/gemini/${name}.js`),
  ...['ask', 'utils', 'model', 'detail', 'read', 'image'].map(name => `clis/chatgpt/${name}.js`),
]
const originals = new Map(files.map(name => [name, fs.readFileSync(path.join(installed, name))]))
after(() => {
  for (const [name, before] of originals) assert.deepEqual(fs.readFileSync(path.join(installed, name)), before)
  fs.rmSync(temporary, { recursive: true, force: true })
})
for (const [name, source] of originals) {
  const target = path.join(copied, name)
  fs.mkdirSync(path.dirname(target), { recursive: true })
  fs.writeFileSync(target, source)
}
fs.copyFileSync(path.join(runtime, 'patch-opencli.mjs'), path.join(temporary, 'patch-opencli.mjs'))
fs.cpSync(path.join(runtime, 'patches'), path.join(temporary, 'patches'), { recursive: true })
execFileSync(process.execPath, [path.join(temporary, 'patch-opencli.mjs')])
const patchedOnce = new Map(files.map(name => [name, fs.readFileSync(path.join(copied, name))]))
execFileSync(process.execPath, [path.join(temporary, 'patch-opencli.mjs')])

function waitSource(source) {
  const start = source.indexOf('export async function waitForChatGPTImages(')
  const end = source.indexOf('export async function getProjectList(page)', start)
  assert.ok(start >= 0 && end > start)
  return source.slice(start, end).replace('export async function', 'async function')
}
const patchedUtils = fs.readFileSync(path.join(copied, 'clis/chatgpt/utils.js'), 'utf8')
const fixedWait = waitSource(patchedUtils)
const savedUrl = 'https://chatgpt.com/c/abcd1234-saved'
const localUrl = 'https://chatgpt.com/c/local-chatgpt%3A5a445fbe-4e97-4097-8cc8-7a65077b6233'
const imageUrl = 'https://chatgpt.com/backend-api/files/image123'

class AdapterError extends Error {
  constructor(...parts) { super(parts.join(' ')) }
}

function waiter(source = fixedWait, options = {}) {
  const state = { clock: 0, reads: 0, navigations: [], progress: [], url: options.url || savedUrl }
  const page = {
    async sleep(seconds) { state.clock += seconds * 1000 },
    async evaluate() { state.reads++; state.clock += options.bridgeMs || 0; return options.readUrl?.(state) ?? state.url },
    async goto(url) { state.navigations.push(url); state.clock += 2000 },
  }
  const context = {
    Date: { now: () => state.clock },
    TimeoutError: AdapterError, CommandExecutionError: AdapterError,
    unwrapEvaluateResult: value => value,
    isSameChatGPTConversation: (actual, expected) => actual === expected || actual.startsWith(`${expected}?`),
    async isGenerating() {
      state.clock += options.bridgeMs || 0
      return options.generating?.(state) || false
    },
    async getChatGPTVisibleImageUrls() {
      state.clock += options.bridgeMs || 0
      return options.images?.(state) || []
    },
    async resolveChatGPTConversationIdentity() { return options.identity || null },
    async getVisibleMessages() { return options.messages?.(state) ?? [{ Role: 'User', Text: options.actualPrompt || 'exact image prompt' }] },
    chatGPTPromptMatches: (actual, expected) => actual === expected,
  }
  const wait = vm.runInNewContext(source + '\nwaitForChatGPTImages', context)
  return { state, run(timeout = 30, conversation = savedUrl, extra = {}) {
    return wait(page, [], timeout, conversation, {
      onProgress(stage, url) { state.progress.push({ stage, url }) }, ...extra,
    })
  } }
}

test('postinstall patch is idempotent and leaves installed adapters byte-identical', () => {
  for (const [name, before] of patchedOnce) assert.deepEqual(fs.readFileSync(path.join(copied, name)), before, name)
})

test('reproduces installed poll-count timeout exceeding the 405 second parent ceiling', async t => {
  const legacy = waitSource(originals.get('clis/chatgpt/utils.js').toString())
  if (!legacy.includes('maxPolls')) return t.skip('Installed runtime already has the elapsed-time fix')
  const { run, state } = waiter(legacy, { bridgeMs: 2000 })
  await run(360)
  assert.ok(state.clock > 405000, `Legacy wait took ${state.clock} ms`)
  assert.ok(state.navigations.length > 0)
})

test('bridge latency counts against the image response deadline', async () => {
  const { run, state } = waiter(fixedWait, { bridgeMs: 2000 })
  assert.deepEqual(Array.from(await run(360)), [])
  assert.ok(state.clock <= 362000, `Deadline drifted to ${state.clock} ms`)
  assert.equal(state.navigations.length, 0)
  assert.equal(state.progress.at(-1).stage, 'image_wait_timeout')
})

test('idle saved conversation is never periodically reloaded before a late image', async () => {
  const { run, state } = waiter(fixedWait, { images: state => state.clock > 20000 ? [imageUrl] : [] })
  assert.deepEqual(Array.from(await run(30)), [imageUrl])
  assert.equal(state.navigations.length, 0)
  assert.equal(state.progress.at(-1).stage, 'images_detected')
})

const imageFailure = "I wasn't able to generate the image due to an error on my side."
const ownedFailure = [
  { Role: 'User', Text: 'exact image prompt' },
  { Role: 'Assistant', Text: imageFailure },
]

test('completed image error ends the owned wait after two stable idle readings', async () => {
  const { run, state } = waiter(fixedWait, { messages: () => ownedFailure })
  await assert.rejects(run(420, savedUrl, { prompt: 'exact image prompt' }),
    error => /CHATGPT_IMAGE_GENERATION_FAILED/.test(error.message) && error.message.includes(savedUrl))
  assert.equal(state.clock, 6000)
  assert.equal(state.navigations.length, 0)
  assert.equal(state.progress.at(-1).stage, 'image_generation_failed')
})

test('reproduces full-deadline wait on the installed adapter for a completed provider error', async t => {
  const legacy = waitSource(originals.get('clis/chatgpt/utils.js').toString())
  if (legacy.includes('CHATGPT_IMAGE_GENERATION_FAILED')) return t.skip('Installed runtime already detects terminal errors')
  const { run, state } = waiter(legacy, { messages: () => ownedFailure })
  assert.deepEqual(Array.from(await run(420, savedUrl, { prompt: 'exact image prompt' })), [])
  assert.ok(state.clock >= 420000)
})

for (const [label, messages] of [
  ['different request', [{ Role: 'User', Text: 'other prompt' }, ownedFailure[1]]],
  ['older reply', [ownedFailure[1], ownedFailure[0]]],
  ['unhydrated user', [ownedFailure[1]]],
  ['quoted error', [ownedFailure[0], { Role: 'Assistant', Text: `The earlier error was: ${imageFailure}` }]],
]) {
  test(`does not treat ${label} as a terminal image failure`, async () => {
    const { run, state } = waiter(fixedWait, { messages: () => messages })
    assert.deepEqual(Array.from(await run(12, savedUrl, { prompt: 'exact image prompt' })), [])
    assert.equal(state.progress.at(-1).stage, 'image_wait_timeout')
  })
}

test('single error sighting does not reject an image that subsequently finishes', async () => {
  const { run, state } = waiter(fixedWait, {
    messages: () => ownedFailure, images: state => state.clock >= 6000 ? [imageUrl] : [],
  })
  assert.deepEqual(Array.from(await run(30, savedUrl, { prompt: 'exact image prompt' })), [imageUrl])
  assert.equal(state.clock, 9000)
})

test('generation restarting between error sightings resets the failure confirmation', async () => {
  const { run } = waiter(fixedWait, {
    messages: () => ownedFailure, generating: state => state.clock === 6000,
    images: state => state.clock >= 12000 ? [imageUrl] : [],
  })
  assert.deepEqual(Array.from(await run(30, savedUrl, { prompt: 'exact image prompt' })), [imageUrl])
})

for (const rendering of ['active generation', 'canvas preview']) {
  test(`${rendering} prevents stale failure text from terminating an image request`, async () => {
    const { run, state } = waiter(fixedWait, {
      messages: () => ownedFailure,
      generating: state => rendering === 'active generation' && state.clock < 12000,
      images: state => state.clock >= 12000 ? [imageUrl]
        : rendering === 'canvas preview' ? ['data:image/png;base64,unfinished'] : [],
    })
    assert.deepEqual(Array.from(await run(30, savedUrl, { prompt: 'exact image prompt' })), [imageUrl])
    assert.equal(state.progress.at(-1).stage, 'images_detected')
  })
}

test('known live local route remains in place while its canonical image finishes', async () => {
  const { run, state } = waiter(fixedWait, { url: localUrl,
    generating: state => state.clock < 15000, images: () => [imageUrl] })
  assert.deepEqual(Array.from(await run(30, savedUrl, { localConversationUrl: localUrl })), [imageUrl])
  assert.equal(state.navigations.length, 0)
})

test('local route can naturally become saved only with the exact submitted prompt', async () => {
  const { run, state } = waiter(fixedWait, { images: () => [imageUrl],
    identity: { conversationUrl: savedUrl } })
  assert.deepEqual(Array.from(await run(30, localUrl, { prompt: 'exact image prompt' })), [imageUrl])
  assert.equal(state.progress[0].url, savedUrl)
  assert.equal(state.navigations.length, 0)
})

test('saved route waits for hydrated messages before validating the original image prompt', async () => {
  const { run, state } = waiter(fixedWait, { images: () => [imageUrl],
    identity: { conversationUrl: savedUrl },
    messages: state => state.clock < 12000 ? [] : [{ Role: 'User', Text: 'exact image prompt' }] })
  assert.deepEqual(Array.from(await run(30, localUrl, { prompt: 'exact image prompt' })), [imageUrl])
  assert.ok(state.clock >= 15000)
  assert.equal(state.navigations.length, 0)
})

test('a delayed first saved route is adopted from /new only after exact prompt verification', async () => {
  const { run, state } = waiter(fixedWait, { images: () => [imageUrl],
    identity: { conversationUrl: savedUrl } })
  assert.deepEqual(Array.from(await run(30, 'https://chatgpt.com/new', { prompt: 'exact image prompt' })), [imageUrl])
  assert.equal(state.progress[0].url, savedUrl)
  assert.equal(state.navigations.length, 0)
})

test('a delayed unsaved local route is adopted from /new only after exact prompt verification', async () => {
  const { run, state } = waiter(fixedWait, { url: localUrl, images: () => [imageUrl] })
  assert.deepEqual(Array.from(await run(30, 'https://chatgpt.com/new', { prompt: 'exact image prompt' })), [imageUrl])
  assert.equal(state.progress[0].url, localUrl)
  assert.equal(state.navigations.length, 0)
})

test('a transient empty location read waits without navigating or accepting images', async () => {
  const { run, state } = waiter(fixedWait, { readUrl: state => state.reads === 1 ? '' : savedUrl,
    images: () => [imageUrl] })
  assert.deepEqual(Array.from(await run()), [imageUrl])
  assert.equal(state.reads, 3)
  assert.equal(state.navigations.length, 0)
})

for (const actualPrompt of ['different prompt', '']) {
  test(`unowned saved route is rejected without reload or replay (${actualPrompt || 'empty'})`, async () => {
    const { run, state } = waiter(fixedWait, { actualPrompt: actualPrompt || 'unrelated',
      identity: { conversationUrl: savedUrl }, images: () => [imageUrl] })
    await assert.rejects(run(30, localUrl, { prompt: 'exact image prompt' }), /left its submitted conversation/)
    assert.equal(state.navigations.length, 0)
  })
}

test('unrelated conversation is never navigated back or used as an image result', async () => {
  const { run, state } = waiter(fixedWait, { url: 'https://chatgpt.com/c/other1234', images: () => [imageUrl] })
  await assert.rejects(run(), /left its submitted conversation/)
  assert.equal(state.navigations.length, 0)
})

for (const kind of ['generating', 'canvas']) {
  test(`${kind} timeout preserves the conversation URL and never exports a partial image`, async () => {
    const { run, state } = waiter(fixedWait, {
      generating: () => kind === 'generating',
      images: () => kind === 'canvas' ? ['data:image/png;base64,unfinished'] : [],
    })
    await assert.rejects(run(12), error => error.message.includes(savedUrl) && /deadline/.test(error.message))
    assert.equal(state.clock, 12000)
  })
}

test('one image sighting is not enough to export at the response deadline', async () => {
  const { run } = waiter(fixedWait, { images: () => [imageUrl] })
  await assert.rejects(run(6), /No stable finished image/)
})

test('image command records submission, wait failure and canonical link without prompt text', async () => {
  const source = fs.readFileSync(path.join(copied, 'clis/chatgpt/image.js'), 'utf8')
    .replace(/^import .*\n/gm, '').replaceAll('export ', '')
  const diagnostics = []
  const sent = []
  const navigations = []
  const command = vm.runInNewContext(source + '\nimageCommand', {
    os, path, fs, cli: value => value, Strategy: { COOKIE: 'cookie' },
    ArgumentError: AdapterError, CommandExecutionError: AdapterError, EmptyResultError: AdapterError,
    console: { error: value => diagnostics.push(JSON.parse(value.slice('[chatgpt/image] '.length))) },
    normalizeBooleanFlag: Boolean, unwrapEvaluateResult: value => value,
    async clearChatGPTDraft() {}, async getChatGPTVisibleImageUrls() { return [] },
    async sendChatGPTMessage(_page, value) { sent.push(value); return true },
    async resolveChatGPTConversationIdentity() { return { conversationUrl: savedUrl, localConversationUrl: localUrl } },
    async waitForChatGPTImages(_page, _before, timeout, url, options) {
      assert.equal(timeout, 360); assert.equal(url, savedUrl)
      assert.equal(options.localConversationUrl, localUrl)
      assert.equal(options.prompt, sent[0])
      options.onProgress('image_wait_timeout', url)
      throw new AdapterError('Generation timed out')
    },
  })
  await assert.rejects(command.func({ async goto(url) { navigations.push(url) } },
    { prompt: 'Private thumbnail prompt', timeout: 360 }), /Generation timed out/)
  assert.equal(sent.length, 1)
  assert.deepEqual(navigations, ['https://chatgpt.com/new'])
  assert.deepEqual(diagnostics.map(item => item.stage), [
    'preparing', 'prompt_submitted', 'waiting', 'image_wait_timeout', 'image_wait_failed',
  ])
  assert.equal(diagnostics.at(-1).conversation_url, savedUrl)
  assert.ok(!JSON.stringify(diagnostics).includes('Private thumbnail prompt'))
})

test('unsubmitted image navigation recovers only its owned window and has bounded fallback', async () => {
  const source = fs.readFileSync(path.join(runtime, 'patches/chatgpt-image-request.js'), 'utf8').replaceAll('export ', '')
  const prepare = vm.runInNewContext(source + '\nprepareChatGPTImageComposer', {})
  const calls = []
  await prepare({
    async goto(url) { calls.push(['goto', url]); throw new Error('Navigation rejected.') },
    async closeWindow() { calls.push(['close-owned-window']) },
    async newTab(url) { calls.push(['new-owned-tab', url]); return 'owned-image-target' },
    async setActivePage(id) { calls.push(['select', id]) },
  })
  assert.deepEqual(calls, [
    ['goto', 'https://chatgpt.com/new'], ['close-owned-window'],
    ['goto', 'https://chatgpt.com/new'], ['new-owned-tab', 'https://chatgpt.com/new'],
    ['select', 'owned-image-target'],
  ])
  let closed = false
  await assert.rejects(prepare({
    async goto() { throw new Error('Browser permission denied') },
    async closeWindow() { closed = true },
  }), /permission denied/)
  assert.equal(closed, false)
})

test('image setup/export allowance matches Python and preserves other command budgets', () => {
  const source = fs.readFileSync(path.join(copied, 'dist/src/execution.js'), 'utf8')
  const start = source.indexOf('const attachmentSetupSeconds =')
  const end = source.indexOf('const commandRun =', start)
  const budget = (cmd, kwargs = {}) => vm.runInNewContext(source.slice(start, end) + '\nbrowserTimeout', {
    cmd, kwargs, userTimeoutSec: 360, RUNTIME_TIMEOUT_PADDING_SECONDS: 30,
  })
  assert.equal(budget({ site: 'chatgpt', name: 'image' }), 570)
  assert.equal(budget({ site: 'chatgpt', name: 'ask' }), 390)
  assert.equal(budget({ site: 'gemini', name: 'ask' }, { file: '/mock/image.jpg' }), 570)
})

test('image timeout checkpoint recovers the original prompt without a second send', async () => {
  const stateFile = path.join(temporary, 'image-request.json')
  const source = fs.readFileSync(path.join(copied, 'clis/chatgpt/image.js'), 'utf8')
    .replace(/^import .*\n/gm, '').replaceAll('export ', '')
  const sends = []; const navigations = []; let waits = 0
  const command = vm.runInNewContext(source + '\nimageCommand', {
    os, path, fs, Date, cli: value => value, Strategy: { COOKIE: 'cookie' },
    ArgumentError: AdapterError, CommandExecutionError: AdapterError, EmptyResultError: AdapterError,
    console: { error() {} }, normalizeBooleanFlag: Boolean,
    async clearChatGPTDraft() {}, async getChatGPTVisibleImageUrls() { return [] },
    async sendChatGPTMessage(_page, value) { sends.push(value); return true },
    async resolveChatGPTConversationIdentity() { return { conversationUrl: savedUrl } },
    async getVisibleMessages() { return [{ Role: 'User', Text: sends[0] }] },
    chatGPTPromptMatches: (a, b) => a === b,
    parseChatGPTConversationId: url => url.split('/c/')[1],
    async waitForChatGPTImages() {
      if (++waits === 1) throw new AdapterError('response timeout')
      return [imageUrl]
    },
  })
  const page = { async goto(url) { navigations.push(url) } }
  const args = { prompt: 'original still', timeout: 30, sd: true, 'state-file': stateFile }
  await assert.rejects(command.func(page, args), /response timeout/)
  assert.equal(JSON.parse(fs.readFileSync(stateFile)).stage, 'image_wait_failed')
  const result = await command.func(page, args)
  assert.equal(result[0].link, `🔗 ${savedUrl}`)
  assert.equal(sends.length, 1)
  assert.deepEqual(navigations, ['https://chatgpt.com/new', savedUrl])
  await assert.rejects(command.func(page, { ...args, prompt: 'different still' }), /saved image prompt differs/)
  const prior = JSON.parse(fs.readFileSync(stateFile))
  fs.writeFileSync(stateFile, JSON.stringify({ ...prior, stage: 'submitting', conversationUrl: '' }))
  await assert.rejects(command.func(page, args), /no saved conversation URL/)
  assert.equal(sends.length, 1)
  assert.equal(navigations.length, 2)
})

test('image recovery rejects a different visible user prompt and never submits', async () => {
  const helper = fs.readFileSync(path.join(runtime, 'patches/chatgpt-image-request.js'), 'utf8').replaceAll('export ', '')
  const recover = vm.runInNewContext(helper + '\nrecoverChatGPTImageRequest', {
    Date, CommandExecutionError: AdapterError,
    async getVisibleMessages() { return [{ Role: 'User', Text: 'someone else' }] },
    chatGPTPromptMatches: (a, b) => a === b,
  })
  await assert.rejects(recover({ async goto() {} }, { conversationUrl: savedUrl }, 'owned prompt', 30), /recovery prompt mismatch/)
})

import assert from 'node:assert/strict'
import test from 'node:test'

import { selectChatGPTModel } from './node_modules/@jackwener/opencli/clis/chatgpt/utils.js'
import {
  sendGeminiMessage,
  waitForGeminiResponse,
} from './node_modules/@jackwener/opencli/clis/gemini/utils.js'

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

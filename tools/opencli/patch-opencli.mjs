import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const runtimeDir = path.dirname(fileURLToPath(import.meta.url))
const geminiDir = path.join(
  runtimeDir,
  'node_modules',
  '@jackwener',
  'opencli',
  'clis',
  'gemini',
)
const chatgptDir = path.join(
  runtimeDir,
  'node_modules',
  '@jackwener',
  'opencli',
  'clis',
  'chatgpt',
)
const askPath = path.join(geminiDir, 'ask.js')
const utilsPath = path.join(geminiDir, 'utils.js')
const modelsPath = path.join(geminiDir, 'models.js')
const chatgptUtilsPath = path.join(chatgptDir, 'utils.js')
const chatgptModelPath = path.join(chatgptDir, 'model.js')
const chatgptDetailPath = path.join(chatgptDir, 'detail.js')
const executionPath = path.join(
  runtimeDir,
  'node_modules',
  '@jackwener',
  'opencli',
  'dist',
  'src',
  'execution.js',
)
const manifestPath = path.join(
  runtimeDir,
  'node_modules',
  '@jackwener',
  'opencli',
  'cli-manifest.json',
)
const videoPath = path.join(geminiDir, 'video.js')

const helperMarker = 'export async function attachGeminiFile('
const helperSource = fs.readFileSync(
  path.join(runtimeDir, 'patches', 'gemini-file-upload-helper.js'),
  'utf8',
)

function replaceOnce(source, before, after, label) {
  if (source.includes(after)) return source
  if (!source.includes(before)) {
    throw new Error(`OpenCLI ${label} patch anchor was not found; pinned upstream changed`)
  }
  return source.replace(before, after)
}

function replaceWithinFunction(source, signature, nextSignature, before, after, label) {
  const start = source.indexOf(signature)
  const end = source.indexOf(nextSignature, start + signature.length)
  if (start < 0 || end < 0) {
    throw new Error(`OpenCLI ${label} function boundary was not found; pinned upstream changed`)
  }
  const section = source.slice(start, end)
  const patched = replaceOnce(section, before, after, label)
  return `${source.slice(0, start)}${patched}${source.slice(end)}`
}

let execution = fs.readFileSync(executionPath, 'utf8')
execution = replaceOnce(
  execution,
  [
    'function resolveAdapterBrowserSession(cmd, siteSession) {',
    "    if (siteSession === 'persistent')",
    '        return `site:${cmd.site}`;',
    '    return `site:${cmd.site}:${crypto.randomUUID()}`;',
    '}',
  ].join('\n'),
  [
    'function resolveAdapterBrowserSession(cmd, siteSession) {',
    "    if (siteSession === 'persistent') {",
    "        const namespace = String(process.env.OPENCLI_SITE_SESSION_NAMESPACE || '')",
    "            .trim().replace(/[^a-z0-9._-]+/gi, '-').replace(/^-+|-+$/g, '').slice(0, 64);",
    '        return namespace ? `site:${namespace}:${cmd.site}` : `site:${cmd.site}`;',
    '    }',
    '    return `site:${cmd.site}:${crypto.randomUUID()}`;',
    '}',
  ].join('\n'),
  'persistent site-session namespace',
)
fs.writeFileSync(executionPath, execution)

// Model selection previously had no transport deadline, so Python's 60s kill
// could leave a 120s browser operation and its lease running behind the retry.
let chatgptModel = fs.readFileSync(chatgptModelPath, 'utf8')
chatgptModel = replaceOnce(
  chatgptModel,
  '    args: [',
  `    args: [
        { name: 'timeout', type: 'int', default: 45, help: 'Model-selection timeout in seconds' },`,
  'ChatGPT model transport deadline',
)
chatgptModel = replaceOnce(
  chatgptModel,
  '    CHATGPT_DOMAIN,',
  '    CHATGPT_DOMAIN,\n    CHATGPT_URL,',
  'ChatGPT model initial URL import',
)
chatgptModel = replaceOnce(
  chatgptModel,
  `        if (kwargs.project) {
            await navigateToProject(page, kwargs.project);
        }`,
  `        if (kwargs.project) {
            await navigateToProject(page, kwargs.project);
        } else {
            // Resolve a real page before evaluating the URL/composer. A fresh
            // adapter lease otherwise starts by executing against about:blank.
            await page.goto(CHATGPT_URL, { settleMs: 2000 });
        }`,
  'ChatGPT model navigate before page evaluation',
)
fs.writeFileSync(chatgptModelPath, chatgptModel)

// Recovery must not reload an owned conversation that is still streaming.
let chatgptDetail = fs.readFileSync(chatgptDetailPath, 'utf8')
chatgptDetail = replaceOnce(
  chatgptDetail,
  '    CHATGPT_URL,',
  '    CHATGPT_URL,\n    currentChatGPTUrl,',
  'ChatGPT detail current URL import',
)
chatgptDetail = replaceOnce(
  chatgptDetail,
  '        await page.goto(`${CHATGPT_URL}/c/${id}`, { settleMs: 2000 });',
  `        let currentId = '';
        try {
            currentId = parseChatGPTConversationId(await currentChatGPTUrl(page));
        } catch { /* A fresh tab still needs navigation. */ }
        if (currentId !== id) {
            await page.goto(\`\${CHATGPT_URL}/c/\${id}\`, { settleMs: 2000 });
        }`,
  'ChatGPT detail preserve active response stream',
)
chatgptDetail = replaceOnce(
  chatgptDetail,
  '    currentChatGPTUrl,',
  '    currentChatGPTUrl,\n    isGenerating,',
  'ChatGPT refresh generation guard import',
)
chatgptDetail = replaceOnce(
  chatgptDetail,
  '    args: [',
  `    args: [
        { name: 'refresh', type: 'boolean', default: false, help: 'Reload a settled target conversation to recover stale rendered text' },`,
  'ChatGPT settled refresh argument',
)
chatgptDetail = replaceOnce(
  chatgptDetail,
  `        if (currentId !== id) {`,
  `        // A stalled client can retain a partial token after generation ends.
        // Explicit recovery may reload only an idle, already selected target.
        if (currentId === id && normalizeBooleanFlag(kwargs.refresh, false)
            && !await isGenerating(page)) {
            await page.evaluate('(() => { window.location.reload(); return true; })()');
            await page.wait(2);
        }
        if (currentId !== id) {`,
  'ChatGPT settled target refresh',
)
chatgptDetail = replaceOnce(
  chatgptDetail, '    isGenerating,',
  '    isGenerating,\n    chatgptRateLimitGuardScript,\n    requireBooleanEvaluateResult,\n    unwrapEvaluateResult,',
  'ChatGPT cooldown dialog guard import',
)
chatgptDetail = replaceOnce(
  chatgptDetail, '    args: [',
  `    args: [
        { name: 'cooldown', type: 'boolean', default: false, help: 'Recover a stale limit dialog after the shared access cooldown' },`,
  'ChatGPT post-cooldown argument',
)
chatgptDetail = replaceOnce(
  chatgptDetail,
  '        // A stalled client can retain a partial token after generation ends.',
  `        // The project transport has waited out the persisted access cooldown.
        // Reload only if the old blocking dialog remains; otherwise preserve
        // any active response stream and continue normal detail waiting.
        if (currentId === id && normalizeBooleanFlag(kwargs.cooldown, false)) {
            const staleLimit = requireBooleanEvaluateResult(unwrapEvaluateResult(await page.evaluate(\`(() => {
                try { \${chatgptRateLimitGuardScript()} return false; }
                catch (error) {
                    if (String(error).includes('CHATGPT_RATE_LIMITED')) return true;
                    throw error;
                }
            })()\`)), 'ChatGPT stale rate-limit dialog');
            if (staleLimit === true) {
                await page.evaluate('(() => { window.location.reload(); return true; })()');
                await page.wait(2);
            }
        }
        // A stalled client can retain a partial token after generation ends.`,
  'ChatGPT stale limit dialog recovery',
)
fs.writeFileSync(chatgptDetailPath, chatgptDetail)

let utils = fs.readFileSync(utilsPath, 'utf8')
const helperIndex = utils.indexOf(helperMarker)
if (helperIndex >= 0) {
  const commentIndex = utils.lastIndexOf('\n// Project patch:', helperIndex)
  if (commentIndex < 0) {
    throw new Error('OpenCLI Gemini helper marker has no project patch boundary')
  }
  utils = utils.slice(0, commentIndex)
}
utils = `${utils.trimEnd()}${helperSource}\n`
if (!utils.includes('const selectedVariant = await page.evaluate')) {
  utils = replaceOnce(
    utils,
  `export async function startNewGeminiChat(page) {
    await ensureGeminiPage(page);
    const action = await page.evaluate(clickNewChatScript());
    if (action === 'navigate') {
        await page.goto(GEMINI_APP_URL, { waitUntil: 'load', settleMs: 2500 });
    }
    await page.wait(1);
    return action;
}`,
  `export async function startNewGeminiChat(page) {
    await ensureGeminiPage(page);
    await page.goto(GEMINI_APP_URL, { waitUntil: 'load', settleMs: 2500 });
    await page.wait(1);
    return 'navigate';
}`,
  'Gemini hard new-chat reload',
)
  utils = replaceOnce(
    utils,
  `        composer.closest('.input-wrapper'),
        composer.parentElement,`,
  `        composer.closest('.input-wrapper'),
        composer.closest('input-area-v2'),
        composer.closest('input-container'),
        composer.closest('fieldset'),
        composer.parentElement,`,
  'Gemini composer control root',
)
utils = replaceOnce(
  utils,
  `      const excludedPattern = /main menu|主菜单|microphone|麦克风|upload|上传|mode|模式|tools|工具|settings|临时对话|new chat|新对话/i;`,
  `      const excludedPattern = /main menu|主菜单|microphone|麦克风|dictate|upload|上传|mode|模式|tools|工具|settings|stop response|停止回答|临时对话|new chat|新对话/i;`,
  'Gemini non-submit controls',
)
  utils = replaceOnce(
    utils,
  `      if (bestButton instanceof HTMLElement && bestScore >= 3) {
        bestButton.click();
        return 'button';
      }`,
  `      if (bestButton instanceof HTMLElement && bestScore >= 3) {
        document.querySelectorAll('[data-opencli-gemini-submit]').forEach((node) => {
          node.removeAttribute('data-opencli-gemini-submit');
        });
        bestButton.setAttribute('data-opencli-gemini-submit', '1');
        bestButton.scrollIntoView({ block: 'center', inline: 'center' });
        const rect = bestButton.getBoundingClientRect();
        return {
          action: 'button',
          label: ((bestButton.getAttribute('aria-label') || '') + ' ' + (bestButton.textContent || '')).trim(),
          className: String(bestButton.className || ''),
          x: Math.round(rect.left + rect.width / 2),
          y: Math.round(rect.top + rect.height / 2),
        };
      }`,
  'Gemini native submit coordinates',
)
const upstreamGeminiSubmitAction = `    const submitAction = await page.evaluate(submitComposerScript());
    if (submitAction === 'button') {
        await page.wait(1);
        return 'button';
    }
    if (page.nativeKeyPress) {`
const legacyGeminiSubmitAction = `    const submitAction = await page.evaluate(submitComposerScript());
    if (process?.env?.OPENCLI_GEMINI_SUBMIT_DEBUG) {
        console.error(\`[gemini/submit] action=\${JSON.stringify(submitAction)}\`);
    }
    if (submitAction?.action === 'button') {
        const hasSemanticSubmit = /send|submit|发送|提交/i.test(String(submitAction.label || ''));
        if (typeof page.nativeKeyPress === 'function') {
            await page.evaluate(\`(() => {
                const composer = document.querySelector('[data-opencli-gemini-composer="1"]');
                if (composer instanceof HTMLElement) composer.focus();
            })()\`);
            await page.nativeKeyPress('Enter');
            await page.wait(0.5);
        }
        else if (hasSemanticSubmit && typeof page.click === 'function') {
            await page.click('[data-opencli-gemini-submit="1"]');
            await page.wait(0.5);
        }
        else if (hasSemanticSubmit && typeof page.nativeClick === 'function') {
            await page.nativeClick(Number(submitAction.x), Number(submitAction.y));
            await page.wait(0.5);
        }
        return 'button';
    }
    if (typeof page.pressKey === 'function') {
        await page.evaluate(\`(() => {
            const composer = document.querySelector('[data-opencli-gemini-composer="1"]');
            if (composer instanceof HTMLElement) composer.focus();
        })()\`);
        await page.pressKey('Enter');
    }
    else if (page.nativeKeyPress) {`
const currentGeminiSubmitAction = `    const submitAction = await page.evaluate(submitComposerScript());
    if (process?.env?.OPENCLI_GEMINI_SUBMIT_DEBUG) {
        console.error(\`[gemini/submit] action=\${JSON.stringify(submitAction)}\`);
    }
    const waitForComposerClear = async () => {
        for (let attempt = 0; attempt < 40; attempt += 1) {
            await page.wait(0.25);
            const state = await page.evaluate(composerHasTextScript());
            if (!state?.hasText)
                return true;
        }
        return false;
    };
    if (submitAction?.action === 'button') {
        const hasSemanticSubmit = /send|submit|发送|提交/i.test(String(submitAction.label || ''));
        if (hasSemanticSubmit && typeof page.click === 'function') {
            try {
                await page.click('button[aria-label="Send message"], button[aria-label="发送消息"], button[aria-label="提交"]');
                if (await waitForComposerClear())
                    return 'button';
                throw new CommandExecutionError('Gemini did not accept the composer submission');
            }
            catch (error) {
                if (error instanceof CommandExecutionError)
                    throw error;
                if (await waitForComposerClear())
                    return 'button';
                await page.click('[data-opencli-gemini-submit="1"]');
                if (await waitForComposerClear())
                    return 'button';
                throw new CommandExecutionError('Gemini did not accept the composer submission');
            }
        }
    }
    if (typeof page.pressKey === 'function') {
        await page.evaluate(\`(() => {
            const composer = document.querySelector('[data-opencli-gemini-composer="1"]');
            if (composer instanceof HTMLElement) composer.focus();
        })()\`);
        await page.pressKey('Enter');
    }
    else if (page.nativeKeyPress) {`
if (utils.includes(legacyGeminiSubmitAction)) {
  utils = utils.replace(legacyGeminiSubmitAction, currentGeminiSubmitAction)
} else if (!utils.includes(currentGeminiSubmitAction)) {
  utils = replaceOnce(
    utils,
    upstreamGeminiSubmitAction,
    currentGeminiSubmitAction,
    'Gemini resilient submit action',
  )
}
utils = replaceWithinFunction(
  utils,
  'export async function sendGeminiMessage(page, text) {',
  'function normalizeGeminiExportUrls(value) {',
  `    await page.wait(1);
    return 'enter';`,
  `    if (!await waitForComposerClear()) {
        throw new CommandExecutionError('Gemini did not accept the composer submission');
    }
    return 'enter';`,
  'Gemini submit postcondition',
)
utils = replaceOnce(
  utils,
  '          && line.length <= 4000',
  '          && line.length <= 12000',
  'Gemini transcript line ceiling',
)
utils = replaceOnce(
  utils,
  `        '[class*="response-text"]',`,
  `        '[class*="response-text"]',
        'user-query',
        'model-response',`,
  'Gemini current structured turn elements',
)
utils = replaceOnce(
  utils,
  `          el.getAttribute('class'),
        ].filter(Boolean).join(' ').toLowerCase();`,
  `          el.getAttribute('class'),
          el.tagName,
        ].filter(Boolean).join(' ').toLowerCase();`,
  'Gemini current structured turn roles',
)
utils = replaceOnce(
  utils,
  `    const pickFallbackGeminiTranscriptReply = (current) => current.transcriptLines
        .filter((line) => !baseline.snapshot.transcriptLines.includes(line))
        .map((line) => extractGeminiTranscriptLineCandidate(line, promptText))
        .filter(Boolean)
        .join('\\n')
        .trim();`,
  `    const ownershipMarker = promptText.match(/REVIEW_REQUEST_ID:[0-9a-f]{32}/i)?.[0] || '';
    const pickFallbackGeminiTranscriptReply = (current) => {
        let candidateLines = current.transcriptLines
            .filter((line) => !baseline.snapshot.transcriptLines.includes(line));
        if (ownershipMarker) {
            let anchorIndex = -1;
            for (let index = current.transcriptLines.length - 1; index >= 0; index -= 1) {
                if (current.transcriptLines[index].toLowerCase().includes(ownershipMarker.toLowerCase())) {
                    anchorIndex = index;
                    break;
                }
            }
            if (anchorIndex < 0)
                return '';
            candidateLines = current.transcriptLines.slice(anchorIndex);
            const nextRequestIndex = candidateLines.slice(1).findIndex((line) =>
                /REVIEW_REQUEST_ID:[0-9a-f]{32}/i.test(line)
            );
            if (nextRequestIndex >= 0)
                candidateLines = candidateLines.slice(0, nextRequestIndex + 1);
            const prompt = promptText.trim();
            const promptOffset = candidateLines[0]?.indexOf(prompt) ?? -1;
            candidateLines[0] = promptOffset >= 0 ? candidateLines[0].slice(promptOffset) : '';
        }
        return candidateLines
            .map((line) => extractGeminiTranscriptLineCandidate(line, promptText))
            .filter((line) => line && !/^(?:you|gemini) said:?$/i.test(line))
            .join('\\n')
            .trim();
    };`,
  'Gemini owned early transcript reply',
)
utils = replaceOnce(
  utils,
  `    await page.wait(0.5);
    const selectedModelId = await getCurrentGeminiModel(page);
    if (selectedModelId !== modelId) {
        throw new CommandExecutionError(
            selectedModelId
                ? \`Gemini model selection read-back returned "\${selectedModelId}", expected "\${modelId}"\`
                : \`Gemini model selection did not expose selected model "\${modelId}" after click\`
        );
    }`,
  `    await page.wait(0.5);
    const selectedModelId = await getCurrentGeminiModel(page);
    if (selectedModelId !== modelId) {
        const selectedVariant = await page.evaluate(\`(() => {
          const buttons = Array.from(document.querySelectorAll('button, [role="button"]'));
          const picker = buttons.find((node) => /currently\\s+/i.test(node.getAttribute('aria-label') || ''))
            || buttons.find((node) => /^(?:gemini\\s+)?(?:flash-lite|flash|pro|lite|ultra|nano)$/i.test((node.textContent || '').trim()));
          const value = ((picker?.getAttribute('aria-label') || '') + ' ' + (picker?.textContent || '')).toLowerCase();
          const match = value.match(/\\b(flash-lite|flash|pro|lite|ultra|nano)\\b/);
          return match ? match[1] : '';
        })()\`).catch(() => '');
        const expectedVariant = String(modelId).replace(/^\\d+(?:\\.\\d+)?-/, '').toLowerCase();
        const shortLabelMatches = selectedVariant === expectedVariant
            || (expectedVariant === 'flash-lite' && selectedVariant === 'lite');
        if (!shortLabelMatches) {
            throw new CommandExecutionError(
                selectedModelId
                    ? \`Gemini model selection read-back returned "\${selectedModelId}", expected "\${modelId}"\`
                    : \`Gemini model selection did not expose selected model "\${modelId}" after click\`
            );
        }
    }`,
    'Gemini short model label read-back',
  )
}
// Upgrade already-patched installations as well as clean installs.
if (!utils.includes('const failedComposerSubmission = async () =>')) {
  utils = replaceWithinFunction(
  utils,
  'export async function sendGeminiMessage(page, text) {',
  'function normalizeGeminiExportUrls(value) {',
  "                throw new CommandExecutionError('Gemini did not accept the composer submission');",
  `                if (typeof page.nativeClick === 'function') {
                    const fresh = await page.evaluate(submitComposerScript());
                    if (fresh?.action === 'button' && /send|submit|发送|提交/i.test(String(fresh.label || ''))) {
                        await page.nativeClick(Number(fresh.x), Number(fresh.y));
                        if (await waitForComposerClear()) return 'button';
                    }
                }
                throw new CommandExecutionError('Gemini did not accept the composer submission');`,
  'Gemini no-op selector native retry',
)
}
utils = replaceOnce(
  utils,
  `      return {
        hasText: !!(composer && ((composer.textContent || '').trim() || (composer.innerText || '').trim())),
      };`,
  `      const actual = String(composer?.innerText || composer?.textContent || '')
        .replace(/\\u00a0/g, ' ')
        .trim();
      return {
        hasText: actual.length > 0,
        actual,
      };`,
  'Gemini exact composer read-back',
)
if (!utils.includes("    const normalizeComposerText = (value) => String(value || '')")) {
  utils = replaceOnce(
    utils,
    `    let hasText = false;
    if (page.nativeType) {`,
    `    let hasText = false;
    if (typeof page.fillText === 'function') {
        try {
            const filled = await page.fillText('[contenteditable="true"][aria-label*="Gemini"]', text);
            hasText = filled?.verified === true && filled?.actual === text;
        }
        catch { }
    }
    if (!hasText && page.nativeType) {`,
    'Gemini verified composer fill',
  )
}
const legacyGeminiVerifiedFill = `    let hasText = false;
    if (typeof page.fillText === 'function') {
        try {
            const filled = await page.fillText('[contenteditable="true"][aria-label*="Gemini"]', text);
            hasText = filled?.verified === true && filled?.actual === text;
        }
        catch { }
    }
    if (!hasText && page.nativeType) {
        try {
            await page.nativeType(text);
            await page.wait(0.2);
            const nativeState = await page.evaluate(composerHasTextScript());
            hasText = !!nativeState?.hasText;
        }
        catch { }
    }
    if (!hasText) {
        const fallbackState = await page.evaluate(insertComposerTextFallbackScript(text));
        hasText = !!fallbackState?.hasText;
    }`
const exactGeminiVerifiedFill = `    const normalizeComposerText = (value) => String(value || '')
        .replace(/\\u00a0/g, ' ')
        .replace(/\\s+/g, ' ')
        .trim();
    const expectedText = normalizeComposerText(text);
    const exactComposerState = async () => {
        const state = await page.evaluate(composerHasTextScript());
        const actual = normalizeComposerText(state?.actual);
        return {
            hasText: !!state?.hasText,
            exact: actual === expectedText,
            actual,
        };
    };
    const clearComposer = async () => {
        const reset = await page.evaluate(prepareComposerScript());
        if (!reset?.ok) {
            throw new CommandExecutionError(reset?.reason || 'Could not clear Gemini composer');
        }
    };
    let hasText = false;
    if (typeof page.fillText === 'function') {
        try {
            await page.fillText('[contenteditable="true"][aria-label*="Gemini"]', text);
        }
        catch { }
        await page.wait(0.2);
        const fillState = await exactComposerState();
        hasText = fillState.exact;
        if (!hasText && fillState.hasText) await clearComposer();
    }
    if (!hasText && page.nativeType) {
        try {
            await page.nativeType(text);
            await page.wait(0.2);
            const nativeState = await exactComposerState();
            hasText = nativeState.exact;
            if (!hasText && nativeState.hasText) await clearComposer();
        }
        catch { }
    }
    if (!hasText) {
        await page.evaluate(insertComposerTextFallbackScript(text));
        const fallbackState = await exactComposerState();
        hasText = fallbackState.exact;
    }`
utils = replaceWithinFunction(
  utils,
  'export async function sendGeminiMessage(page, text) {',
  'function normalizeGeminiExportUrls(value) {',
  legacyGeminiVerifiedFill,
  exactGeminiVerifiedFill,
  'Gemini write-once exact composer fill',
)
if (!utils.includes('const dispatchPreparedGeminiSubmit = async () =>')) {
  const start = utils.indexOf('export async function sendGeminiMessage(page, text) {')
  const end = utils.indexOf('function normalizeGeminiExportUrls(value) {', start)
  if (start < 0 || end < 0) throw new Error('Gemini submit recovery boundary was not found')
  let section = utils.slice(start, end)
  const recovery = `    const dispatchPreparedGeminiSubmit = async () => {
        const fresh = await page.evaluate(submitComposerScript());
        if (fresh?.action !== 'button' || !/send|submit|发送|提交/i.test(String(fresh.label || '')))
            return false;
        const clicked = await page.evaluate(\`(() => {
            const expectedComposerText = \${JSON.stringify(expectedText)};
            const composer = document.querySelector('[data-opencli-gemini-composer="1"]');
            const button = document.querySelector('[data-opencli-gemini-submit="1"]');
            const actual = String(composer?.innerText || composer?.textContent || '').replace(/\\\\s+/g, ' ').trim();
            if (actual !== expectedComposerText || !(button instanceof HTMLElement)) return false;
            const label = ((button.getAttribute('aria-label') || '') + ' ' + (button.textContent || '')).trim();
            if (!/send|submit|发送|提交/i.test(label) || /stop|停止/i.test(label)) return false;
            if (button.disabled || button.getAttribute('aria-disabled') === 'true') return false;
            const rect = button.getBoundingClientRect();
            const style = getComputedStyle(button);
            if (!rect.width || !rect.height || style.display === 'none' || style.visibility === 'hidden') return false;
            button.click();
            return true;
        })()\`);
        const accepted = clicked === true && await waitForComposerClear();
        if (accepted) console.error('[gemini/submit] Prepared DOM button accepted the pending composer');
        return accepted;
    };
`
  section = replaceOnce(section,
    "    if (submitAction?.action === 'button') {",
    recovery + "    if (submitAction?.action === 'button') {",
    'Gemini state-checked DOM submit recovery',
  )
  section = section.replaceAll(
    "throw new CommandExecutionError('Gemini did not accept the composer submission');",
    "if (await dispatchPreparedGeminiSubmit()) return 'button';\n                throw new CommandExecutionError('Gemini did not accept the composer submission');",
  )
  utils = utils.slice(0, start) + section + utils.slice(end)
}
utils = replaceWithinFunction(
  utils,
  'export async function sendGeminiMessage(page, text) {',
  'function normalizeGeminiExportUrls(value) {',
  '        return clicked === true && await waitForComposerClear();',
  `        const accepted = clicked === true && await waitForComposerClear();
        if (accepted) console.error('[gemini/submit] Prepared DOM button accepted the pending composer');
        return accepted;`,
  'Gemini confirmed DOM submit diagnostic',
)
if (!utils.includes('const failedComposerSubmission = async () =>')) {
  const start = utils.indexOf('export async function sendGeminiMessage(page, text) {')
  const end = utils.indexOf('function normalizeGeminiExportUrls(value) {', start)
  if (start < 0 || end < 0) throw new Error('Gemini submission diagnostic boundary was not found')
  let section = utils.slice(start, end)
  const diagnostic = `    const failedComposerSubmission = async () => {
        let diagnostic;
        try {
            const state = await exactComposerState();
            const controls = await page.evaluate(\`(() => {
                const composer = document.querySelector('[data-opencli-gemini-composer="1"]');
                const rect = composer?.getBoundingClientRect();
                const buttons = Array.from(document.querySelectorAll('button')).filter(button => /send|submit|stop|发送|提交|停止/i.test((button.getAttribute('aria-label') || '') + ' ' + (button.textContent || ''))).map(button => {
                    const box = button.getBoundingClientRect();
                    return { label: button.getAttribute('aria-label'), disabled: !!button.disabled, ariaDisabled: button.getAttribute('aria-disabled'), width: box.width, height: box.height, verticalDistance: rect ? Math.abs((box.top + box.bottom - rect.top - rect.bottom) / 2) : null };
                });
                return { buttons, busy: !!document.querySelector('[aria-busy="true"], mat-progress-spinner'), alerts: Array.from(document.querySelectorAll('[role="alert"], [role="dialog"]')).map(el => (el.textContent || '').trim().slice(0, 500)).slice(0, 4) };
            })()\`);
            diagnostic = { exact: state.exact, actualLength: state.actual.length, expectedLength: expectedText.length, controls };
        } catch (error) {
            diagnostic = { diagnosticError: String(error?.message || error) };
        }
        return new CommandExecutionError('Gemini did not accept the composer submission: ' + JSON.stringify(diagnostic));
    };
`
  section = replaceOnce(section, '    const dispatchPreparedGeminiSubmit = async () => {', diagnostic + '    const dispatchPreparedGeminiSubmit = async () => {', 'Gemini failed submission control diagnostics')
  section = section.replaceAll("throw new CommandExecutionError('Gemini did not accept the composer submission');", 'throw await failedComposerSubmission();')
  utils = utils.slice(0, start) + section + utils.slice(end)
}
if (!utils.includes('const dispatchPreparedGeminiEnter = async () =>')) {
  const marker = '    const dispatchPreparedGeminiSubmit = async () => {'
  const helper = `    const dispatchPreparedGeminiEnter = async () => {
        if (typeof page.cdp !== 'function') return false;
        const state = await exactComposerState();
        if (!state.exact || !state.hasText) return false;
        const ready = await page.evaluate(\`(() => {
            const composer = document.querySelector('[data-opencli-gemini-composer="1"]');
            const send = document.querySelector('button[aria-label="Send message"], button[aria-label="发送消息"]');
            if (!(composer instanceof HTMLElement) || !send || send.disabled || send.getAttribute('aria-disabled') === 'true') return false;
            if (document.querySelector('button[aria-label="Stop response"], button[aria-label="停止回答"]')) return false;
            composer.focus();
            return document.activeElement === composer;
        })()\`);
        if (ready !== true) return false;
        await page.cdp('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13, nativeVirtualKeyCode: 13, modifiers: 0 });
        await page.cdp('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Enter', code: 'Enter', windowsVirtualKeyCode: 13, nativeVirtualKeyCode: 13, modifiers: 0 });
        const accepted = await waitForComposerClear();
        if (accepted) console.error('[gemini/submit] Trusted Enter accepted the pending composer after ignored clicks');
        return accepted;
    };
`
  utils = replaceOnce(utils, marker, helper + marker, 'Gemini trusted Enter submit recovery')
}
// Older installations reapplied the legacy native-click upgrade after the DOM
// fallback was added. Normalize that duplicate so fresh installs and upgrades
// execute the same bounded recovery sequence.
{
  const start = utils.indexOf('export async function sendGeminiMessage(page, text) {')
  const end = utils.indexOf('function normalizeGeminiExportUrls(value) {', start)
  let section = utils.slice(start, end)
  if (!section.includes("if (await dispatchPreparedGeminiEnter()) return 'enter';")) {
    section = section.replaceAll('throw await failedComposerSubmission();', "if (await dispatchPreparedGeminiEnter()) return 'enter';\n                throw await failedComposerSubmission();")
  }
  utils = utils.slice(0, start) + section + utils.slice(end)
}
utils = utils.replace(
  `                if (await dispatchPreparedGeminiSubmit()) return 'button';
                if (typeof page.nativeClick === 'function') {
                    const fresh = await page.evaluate(submitComposerScript());
                    if (fresh?.action === 'button' && /send|submit|发送|提交/i.test(String(fresh.label || ''))) {
                        await page.nativeClick(Number(fresh.x), Number(fresh.y));
                        if (await waitForComposerClear()) return 'button';
                    }
                }
                throw await failedComposerSubmission();`,
  `                if (await dispatchPreparedGeminiSubmit()) return 'button';
                throw await failedComposerSubmission();`,
)
utils = utils.replace(String.raw`return location.pathname.replace(/\/+$/, '') === '/app'`, "return ['/app', '/app/'].includes(location.pathname)")
utils = replaceOnce(
  utils,
  `export async function startNewGeminiChat(page) {
    await ensureGeminiPage(page);
    await page.goto(GEMINI_APP_URL, { waitUntil: 'load', settleMs: 2500 });`,
  `export async function startNewGeminiChat(page) {
    await ensureGeminiPage(page);
    const alreadyFresh = await page.evaluate(\`(() => {
        const composer = document.querySelector('[contenteditable="true"][aria-label*="Gemini"]');
        return ['/app', '/app/'].includes(location.pathname)
            && !!composer && !String(composer.innerText || composer.textContent || '').trim()
            && !document.querySelector('user-query, model-response, input-container button[aria-label="close attachment"]');
    })()\`);
    if (alreadyFresh === true) return 'already-new';
    await page.goto(GEMINI_APP_URL, { waitUntil: 'load', settleMs: 2500 });`,
  'Gemini new-chat avoids reloading an already empty new page',
)
utils = replaceOnce(
  utils,
  `        const expectedVariant = String(modelId).replace(/^\\d+(?:\\.\\d+)?-/, '').toLowerCase();
        const shortLabelMatches = selectedVariant === expectedVariant
            || (expectedVariant === 'flash-lite' && selectedVariant === 'lite');`,
  `        const selectedVariantValue = unwrapGeminiEvaluateResult(selectedVariant, 'Gemini short model label');
        const expectedVariant = String(modelId).replace(/^\\d+(?:\\.\\d+)?-/, '').toLowerCase();
        const shortLabelMatches = selectedVariantValue === expectedVariant
            || (expectedVariant === 'flash-lite' && selectedVariantValue === 'lite');`,
  'Gemini short model label envelope',
)
utils = replaceOnce(
  utils,
  '        if (!shortLabelMatches) {',
  '        if (selectedVariantValue && !shortLabelMatches) {',
  'Gemini canonical read-back availability',
)
fs.writeFileSync(utilsPath, utils)

let models = fs.readFileSync(modelsPath, 'utf8')
models = replaceOnce(
  models,
  '        /model[\\\\s-]*selector/i,',
  '        /mode[\\\\s-]*picker/i,\n        /model[\\\\s-]*selector/i,',
  'Gemini current mode picker label',
)
fs.writeFileSync(modelsPath, models)

let chatgptUtils = fs.readFileSync(chatgptUtilsPath, 'utf8')
chatgptUtils = replaceWithinFunction(
  chatgptUtils,
  'export async function uploadChatGPTImages(page, imagePaths) {',
  'export async function isGenerating(page) {',
  "!msg.includes('Not allowed') && !msg.includes('No element found')",
  "!msg.includes('Not allowed') && !msg.includes('No element found') && !/fileChooserOpened|file chooser/i.test(msg)",
  'ChatGPT image upload missing native file chooser fallback',
)
const rateLimitHelper = fs.readFileSync(
  path.join(runtimeDir, 'patches', 'chatgpt-rate-limit-helper.js'), 'utf8',
)
const rateLimitHelperMarker = '// Project patch: inspect visible provider UI, never quoted answer/prompt text.'
if (chatgptUtils.includes(rateLimitHelperMarker)) {
  chatgptUtils = chatgptUtils.slice(0, chatgptUtils.indexOf(rateLimitHelperMarker)).trimEnd()
}
chatgptUtils += `\n\n${rateLimitHelper}\n`
for (const [anchor, label] of [
  ["        const text = (document.body?.innerText || '').replace(/\\\\s+/g, ' ').trim();", 'page state'],
  ['        const includeHtml = ${includeHtml};', 'message extraction'],
  ['            if (document.querySelector(\'[data-testid="stop-button"]\')) return true;', 'generation polling'],
]) {
  chatgptUtils = replaceOnce(
    chatgptUtils, anchor,
    '${chatgptRateLimitGuardScript()}\n' + anchor,
    `ChatGPT rate-limit guard in ${label}`,
  )
}
chatgptUtils = replaceOnce(
  chatgptUtils,
  `            const turns = document.querySelectorAll('article[data-testid*="conversation-turn"]');`,
  `            const turns = document.querySelectorAll('[data-testid^="conversation-turn-"]');`,
  'ChatGPT section generation status scope',
)
chatgptUtils = replaceOnce(
  chatgptUtils,
  "        label: 'Balanced',",
  "        label: 'Medium',",
  'ChatGPT current medium label',
)
const modelSignature = 'export async function selectChatGPTModel(page, model) {'
const toolSignature = 'export async function getCurrentChatGPTTool(page) {'
const upstreamChatgptModelRoute = [
  '    if (!currentUrl.startsWith(`${CHATGPT_URL}/new`)) {',
  "        await page.goto(`${CHATGPT_URL}/new`, { waitUntil: 'none' });",
  '        await page.wait(2);',
  '    }',
].join('\n')
const legacyChatgptModelRoute = [
  '    if (currentUrl !== CHATGPT_URL && currentUrl !== `${CHATGPT_URL}/`) {',
  "        await page.goto(`${CHATGPT_URL}/`, { waitUntil: 'none' });",
  '        await page.wait(2);',
  '    }',
].join('\n')
const currentChatgptModelRoute = [
  '    if (currentUrl !== CHATGPT_URL && currentUrl !== `${CHATGPT_URL}/`) {',
  "        await page.goto(`${CHATGPT_URL}/`, { waitUntil: 'load', settleMs: 2500 });",
  '        await page.wait(1);',
  '    }',
].join('\n')
if (chatgptUtils.includes(legacyChatgptModelRoute)) {
  chatgptUtils = chatgptUtils.replace(
    legacyChatgptModelRoute,
    currentChatgptModelRoute,
  )
} else if (!chatgptUtils.includes(currentChatgptModelRoute)) {
  chatgptUtils = replaceWithinFunction(
    chatgptUtils,
    modelSignature,
    toolSignature,
    upstreamChatgptModelRoute,
    currentChatgptModelRoute,
    'ChatGPT current new-chat route',
  )
}
if (!chatgptUtils.includes("'chatgpt intelligence slider readiness'")) {
chatgptUtils = replaceWithinFunction(
  chatgptUtils,
  modelSignature,
  toolSignature,
  `        if (!button) return { found: false };
        button.scrollIntoView({ block: 'center', inline: 'center' });`,
  `        if (!button) return { found: false };
        button.setAttribute('data-opencli-chatgpt-model-trigger', '1');
        button.scrollIntoView({ block: 'center', inline: 'center' });`,
  'ChatGPT model trigger marker',
)
chatgptUtils = replaceWithinFunction(
  chatgptUtils,
  modelSignature,
  toolSignature,
  `        const labels = \${JSON.stringify(Object.values(CHATGPT_MODEL_TARGETS).flatMap((entry) => entry.labels))};
        const menuButtonSelectors = [`,
  `        const labels = \${JSON.stringify(Object.values(CHATGPT_MODEL_TARGETS).flatMap((entry) => entry.labels))};
        const composer = document.querySelector('[data-opencli-chatgpt-composer="1"]');
        const form = composer?.closest('form')
            || Array.from(document.querySelectorAll('form')).find((node) => node instanceof HTMLElement && isVisible(node));
        const menuButtonSelectors = [`,
  'ChatGPT composer-scoped model trigger root',
)
chatgptUtils = replaceWithinFunction(
  chatgptUtils,
  modelSignature,
  toolSignature,
  `        let button = Array.from(document.querySelectorAll('form button')).find((node) =>
            isVisible(node) && labels.some((label) => textMatchesLabel(node.textContent, label))
        );
        if (!button) {
            button = menuButtonSelectors
                .map((selector) => document.querySelector(selector))`,
  `        let button = Array.from(form?.querySelectorAll('button') || []).find((node) =>
            isVisible(node) && labels.some((label) => textMatchesLabel(node.textContent, label))
        );
        if (!button && form) {
            button = menuButtonSelectors
                .map((selector) => form.querySelector(selector))`,
  'ChatGPT composer-scoped model trigger lookup',
)
const upstreamChatgptSliderAction = `    await page.nativeClick(Number(menuButton.x), Number(menuButton.y));
    await page.wait(0.5);

    let optionCenter = null;`
const legacyChatgptSliderAction = `    await page.nativeClick(Number(menuButton.x), Number(menuButton.y));
    await page.wait(0.5);
    let pickerOpen = Boolean(unwrapEvaluateResult(await page.evaluate(\`(() => {
        return Boolean(document.querySelector('[data-testid="composer-intelligence-picker-content"] [role="slider"]'));
    })()\`)));
    if (!pickerOpen) {
        await page.nativeClick(Number(menuButton.x), Number(menuButton.y));
        await page.wait(0.5);
        pickerOpen = Boolean(unwrapEvaluateResult(await page.evaluate(\`(() => {
            return Boolean(document.querySelector('[data-testid="composer-intelligence-picker-content"] [role="slider"]'));
        })()\`)));
    }
    if (!pickerOpen && typeof page.pressKey === 'function') {
        await page.evaluate(\`(() => {
            const trigger = document.querySelector('[data-opencli-chatgpt-model-trigger="1"]');
            if (trigger instanceof HTMLElement) trigger.focus();
        })()\`);
        await page.pressKey('Enter');
        await page.wait(0.5);
    }

    const sliderState = requireObjectEvaluateResult(unwrapEvaluateResult(await page.evaluate(\`(() => {
        const slider = document.querySelector('[data-testid="composer-intelligence-picker-content"] [role="slider"]')
            || document.querySelector('[role="menu"] [role="slider"]');
        if (!(slider instanceof HTMLElement)) return { found: false };
        const current = Number(slider.getAttribute('aria-valuenow'));
        const minimum = Number(slider.getAttribute('aria-valuemin'));
        const maximum = Number(slider.getAttribute('aria-valuemax'));
        slider.focus();
        return { found: true, current, minimum, maximum };
    })()\`)), 'chatgpt intelligence slider');
    if (sliderState.found) {
        const targetValue = Number(target.intelligenceOrder);
        if (!Number.isInteger(targetValue)
            || !Number.isInteger(sliderState.current)
            || targetValue < sliderState.minimum
            || targetValue > sliderState.maximum) {
            throw new CommandExecutionError(\`ChatGPT did not expose a usable \${target.label} slider position.\`);
        }
        if (typeof page.pressKey !== 'function') {
            throw new CommandExecutionError('ChatGPT intelligence slider requires browser key support.');
        }
        const key = targetValue < sliderState.current ? 'ArrowLeft' : 'ArrowRight';
        for (let index = 0; index < Math.abs(targetValue - sliderState.current); index += 1) {
            await page.pressKey(key);
            await page.wait(0.15);
        }
        await page.wait(0.5);
        const afterSlider = await getCurrentChatGPTModel(page);
        if (afterSlider.model !== target.key) {
            throw new CommandExecutionError(\`ChatGPT model did not switch to \${target.label}.\`);
        }
        await page.pressKey('Escape').catch(() => undefined);
        return { Status: 'Success', Model: target.label };
    }

    let optionCenter = null;`
const currentChatgptSliderAction = `    const readPickerState = async () => requireObjectEvaluateResult(
        unwrapEvaluateResult(await page.evaluate(\`(() => {
            const isVisible = (el) => {
                if (!(el instanceof HTMLElement)) return false;
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style.display !== 'none' && style.visibility !== 'hidden'
                    && rect.width > 0 && rect.height > 0;
            };
            const trigger = document.querySelector('[data-opencli-chatgpt-model-trigger="1"]');
            const controlledId = trigger?.getAttribute('aria-controls') || '';
            const controlled = controlledId ? document.getElementById(controlledId) : null;
            const content = document.querySelector('[data-testid="composer-intelligence-picker-content"]');
            const labels = (\${JSON.stringify(target.optionLabels || target.labels || [])})
                .map((value) => String(value || '').toLowerCase());
            const candidates = [
                content,
                controlled,
                ...Array.from(document.querySelectorAll('[role="menu"], [role="listbox"]')).filter(isVisible),
            ].filter((node, index, all) => node instanceof HTMLElement
                && isVisible(node) && all.indexOf(node) === index);
            const root = candidates.find((node) => {
                if (node === content || node === controlled) return true;
                const text = String(node.textContent || '').toLowerCase();
                return labels.some((label) => label && text.includes(label));
            }) || null;
            const slider = root?.querySelector('[role="slider"], input[type="range"]')
                || document.querySelector('[data-model-reasoning-effort-slider] [role="slider"]');
            const optionSelector = '[role="menuitemradio"], [role="option"], [role="menuitem"], button';
            const hasTargetOption = root ? Array.from(root.querySelectorAll(optionSelector)).some((node) => {
                if (!isVisible(node)) return false;
                const text = [
                    node.textContent,
                    node.getAttribute('aria-label'),
                    node.getAttribute('aria-valuetext'),
                    node.getAttribute('title'),
                ].map((value) => String(value || '').toLowerCase()).join(' ');
                return labels.some((label) => label && text.includes(label));
            }) : false;
            return {
                ready: slider instanceof HTMLElement || hasTargetOption,
                expanded: trigger?.getAttribute('aria-expanded') === 'true' || root instanceof HTMLElement,
                contentFound: content instanceof HTMLElement && isVisible(content),
            };
        })()\`)),
        'chatgpt intelligence slider readiness',
    );

    let pickerState = await readPickerState();
    if (!pickerState.ready && !pickerState.expanded) {
        await page.nativeClick(Number(menuButton.x), Number(menuButton.y));
        for (let attempt = 0; attempt < 20; attempt += 1) {
            await page.wait(attempt === 0 ? 0.5 : 0.4);
            pickerState = await readPickerState();
            if (pickerState.ready) break;
        }
    }
    if (!pickerState.ready && !pickerState.expanded) {
        await page.evaluate(\`(() => {
            const trigger = document.querySelector('[data-opencli-chatgpt-model-trigger="1"]');
            if (trigger instanceof HTMLElement) trigger.focus();
        })()\`);
        if (typeof page.nativeKeyPress === 'function') await page.nativeKeyPress('Enter');
        else if (typeof page.pressKey === 'function') await page.pressKey('Enter');
        for (let attempt = 0; attempt < 10; attempt += 1) {
            await page.wait(0.5);
            pickerState = await readPickerState();
            if (pickerState.ready) break;
        }
    }

    const sliderState = requireObjectEvaluateResult(unwrapEvaluateResult(await page.evaluate(\`(() => {
        const isVisible = (el) => {
            if (!(el instanceof HTMLElement)) return false;
            const style = window.getComputedStyle(el);
            const rect = el.getBoundingClientRect();
            return style.display !== 'none' && style.visibility !== 'hidden'
                && rect.width > 0 && rect.height > 0;
        };
        const trigger = document.querySelector('[data-opencli-chatgpt-model-trigger="1"]');
        const controlledId = trigger?.getAttribute('aria-controls') || '';
        const controlled = controlledId ? document.getElementById(controlledId) : null;
        const content = document.querySelector('[data-testid="composer-intelligence-picker-content"]');
        const roots = [
            content,
            controlled,
            ...Array.from(document.querySelectorAll('[role="menu"], [role="listbox"]')),
        ].filter((node, index, all) => node instanceof HTMLElement
            && isVisible(node) && all.indexOf(node) === index);
        const slider = roots
            .map((root) => root.querySelector('[role="slider"], input[type="range"]'))
            .find((node) => node instanceof HTMLElement)
            || document.querySelector('[data-model-reasoning-effort-slider] [role="slider"]');
        if (!(slider instanceof HTMLElement)) return { found: false };
        const current = Number(slider.getAttribute('aria-valuenow') || slider.value);
        const minimum = Number(slider.getAttribute('aria-valuemin') || slider.min);
        const maximum = Number(slider.getAttribute('aria-valuemax') || slider.max);
        const keyboardTarget = slider.closest('[role="menuitem"][aria-keyshortcuts]') || slider;
        keyboardTarget.focus();
        return { found: true, current, minimum, maximum };
    })()\`)), 'chatgpt intelligence slider');
    if (sliderState.found) {
        const targetValue = Number(target.intelligenceOrder);
        if (!Number.isInteger(targetValue)
            || !Number.isInteger(sliderState.current)
            || !Number.isFinite(sliderState.minimum)
            || !Number.isFinite(sliderState.maximum)
            || targetValue < sliderState.minimum
            || targetValue > sliderState.maximum) {
            throw new CommandExecutionError(\`ChatGPT did not expose a usable \${target.label} slider position.\`);
        }
        if (typeof page.pressKey !== 'function') {
            throw new CommandExecutionError('ChatGPT intelligence slider requires browser key support.');
        }
        const key = targetValue < sliderState.current ? 'ArrowLeft' : 'ArrowRight';
        for (let index = 0; index < Math.abs(targetValue - sliderState.current); index += 1) {
            await page.pressKey(key);
            await page.wait(0.15);
        }
        let afterSlider = { model: null };
        for (let attempt = 0; attempt < 8; attempt += 1) {
            await page.wait(0.4);
            afterSlider = await getCurrentChatGPTModel(page);
            if (afterSlider.model === target.key) break;
        }
        if (afterSlider.model !== target.key) {
            throw new CommandExecutionError(\`ChatGPT model did not switch to \${target.label}.\`);
        }
        await page.pressKey('Escape').catch(() => undefined);
        return { Status: 'Success', Model: target.label };
    }

    let optionCenter = null;`
const chatgptSliderSource = chatgptUtils.includes(legacyChatgptSliderAction)
  ? legacyChatgptSliderAction
  : upstreamChatgptSliderAction
chatgptUtils = replaceWithinFunction(
  chatgptUtils,
  modelSignature,
  toolSignature,
  chatgptSliderSource,
  currentChatgptSliderAction,
  'ChatGPT five-level intelligence slider readiness',
)
}
const chatgptRangePolicyHelpers = `function chatGPTModelRangePolicy() {
    const minimumValue = String(process?.env?.OPENCLI_CHATGPT_MODEL_MIN || '').trim();
    const maximumValue = String(process?.env?.OPENCLI_CHATGPT_MODEL_MAX || '').trim();
    if (!minimumValue && !maximumValue) return null;
    if (!minimumValue || !maximumValue) {
        throw new ArgumentError(
            'ChatGPT model policy requires both minimum and maximum levels.',
            'Set OPENCLI_CHATGPT_MODEL_MIN and OPENCLI_CHATGPT_MODEL_MAX together.',
        );
    }
    const minimum = requireKnownChatGPTModel(minimumValue);
    const maximum = requireKnownChatGPTModel(maximumValue);
    if (!Number.isInteger(minimum.intelligenceOrder)
        || !Number.isInteger(maximum.intelligenceOrder)
        || minimum.intelligenceOrder > maximum.intelligenceOrder) {
        throw new ArgumentError(
            \`Invalid ChatGPT model policy \${minimumValue}..\${maximumValue}.\`,
            'Use an ordered non-Pro intelligence range such as medium..xhigh.',
        );
    }
    return { minimum, maximum };
}

function chatGPTModelTargetAtOrder(order) {
    const match = Object.values(CHATGPT_MODEL_TARGETS).find(
        (target) => Number.isInteger(target.intelligenceOrder)
            && target.intelligenceOrder === Number(order),
    );
    return match || null;
}

function requireChatGPTModelInRange(current, policy) {
    const target = typeof current === 'string'
        ? CHATGPT_MODEL_TARGETS[current]
        : chatGPTModelTargetAtOrder(current);
    const order = target?.intelligenceOrder;
    if (!Number.isInteger(order)
        || order < policy.minimum.intelligenceOrder
        || order > policy.maximum.intelligenceOrder) {
        throw new CommandExecutionError(
            \`ChatGPT current model \${target?.label || current || 'unknown'} is outside allowed range \`
            + \`\${policy.minimum.label}..\${policy.maximum.label}; refusing to switch or submit.\`,
        );
    }
    return target;
}

`
chatgptUtils = replaceOnce(
  chatgptUtils,
  `export const CHATGPT_MODEL_CHOICES = Object.keys(CHATGPT_MODEL_ALIASES);

function debugChatGPTModel(message) {`,
  `export const CHATGPT_MODEL_CHOICES = Object.keys(CHATGPT_MODEL_ALIASES);

${chatgptRangePolicyHelpers}function debugChatGPTModel(message) {`,
  'ChatGPT fallback model range helpers',
)
chatgptUtils = replaceWithinFunction(
  chatgptUtils,
  modelSignature,
  toolSignature,
  `    const target = requireKnownChatGPTModel(model);
    debugChatGPTModel(\`target=\${target.key}\`);`,
  `    const target = requireKnownChatGPTModel(model);
    const rangePolicy = chatGPTModelRangePolicy();
    let allowedPolicyFallback = null;
    debugChatGPTModel(\`target=\${target.key}\`);`,
  'ChatGPT current model range policy',
)
const upstreamKnownCurrentModel = `    const before = await getCurrentChatGPTModel(page);
    debugChatGPTModel(\`before=\${before.model || 'none'}\`);
    if (before.model === target.key) {
        return { Status: 'Already selected', Model: target.label };
    }`
const previousKnownCurrentModel = `    const before = await getCurrentChatGPTModel(page);
    debugChatGPTModel(\`before=\${before.model || 'none'}\`);
    if (rangePolicy && before.model) {
        try {
            allowedPolicyFallback = requireChatGPTModelInRange(before.model, rangePolicy);
        }
        catch {}
    }
    if (before.model === target.key) {
        return { Status: 'Already selected', Model: target.label };
    }`
const currentKnownCurrentModel = `    const before = await getCurrentChatGPTModel(page);
    debugChatGPTModel(\`before=\${before.model || 'none'}\`);
    if (rangePolicy && before.model) {
        try {
            allowedPolicyFallback = requireChatGPTModelInRange(before.model, rangePolicy);
        }
        catch {}
    }
    if (before.model === target.key) {
        if (rangePolicy && typeof page.pressKey === 'function') {
            await page.pressKey('Escape').catch(() => undefined);
        }
        return { Status: 'Already selected', Model: target.label };
    }`
const knownCurrentModelSource = chatgptUtils.includes(previousKnownCurrentModel)
  ? previousKnownCurrentModel
  : upstreamKnownCurrentModel
chatgptUtils = replaceWithinFunction(
  chatgptUtils,
  modelSignature,
  toolSignature,
  knownCurrentModelSource,
  currentKnownCurrentModel,
  'ChatGPT known current model fallback',
)
chatgptUtils = replaceOnce(
  chatgptUtils,
  `        const labels = \${JSON.stringify(CHATGPT_MODEL_TARGETS)};
        const findEntryForText = (text) => {`,
  `        const labels = \${JSON.stringify(CHATGPT_MODEL_TARGETS)};
        const slider = document.querySelector('[data-testid="composer-intelligence-picker-content"] [role="slider"]')
            || document.querySelector('[role="menu"] [role="slider"]')
            || document.querySelector('[data-model-reasoning-effort-slider] [role="slider"]');
        const sliderValue = Number(slider?.getAttribute('aria-valuenow') || slider?.value);
        if (slider instanceof HTMLElement && Number.isInteger(sliderValue)) {
            const sliderEntry = Object.entries(labels).find(
                ([, value]) => value.intelligenceOrder === sliderValue,
            );
            if (sliderEntry) {
                return { model: sliderEntry[0], label: sliderEntry[1].label };
            }
        }
        const findEntryForText = (text) => {`,
  'ChatGPT current slider model readback',
)
chatgptUtils = replaceWithinFunction(
  chatgptUtils,
  modelSignature,
  toolSignature,
  `        const labels = \${JSON.stringify(Object.values(CHATGPT_MODEL_TARGETS).flatMap((entry) => entry.labels))};
        const composer = document.querySelector('[data-opencli-chatgpt-composer="1"]');`,
  `        const labels = \${JSON.stringify(Object.values(CHATGPT_MODEL_TARGETS).flatMap((entry) => entry.labels))};
        const triggerLabels = ['Thinking effort', 'Reasoning effort', '思考强度', '推理强度'];
        const composer = document.querySelector('[data-opencli-chatgpt-composer="1"]');`,
  'ChatGPT current thinking-effort trigger labels',
)
const upstreamThinkingEffortTrigger = `        let button = Array.from(form?.querySelectorAll('button') || []).find((node) =>
            isVisible(node) && labels.some((label) => textMatchesLabel(node.textContent, label))
        );`
const previousThinkingEffortTrigger = `        let button = Array.from(form?.querySelectorAll('button') || []).find((node) =>
            isVisible(node) && (
                labels.some((label) => textMatchesLabel(node.textContent, label))
                || triggerLabels.some((label) => textMatchesLabel(node.textContent, label))
            )
        );`
const currentThinkingEffortTrigger = `        let button = Array.from(form?.querySelectorAll('button') || []).find((node) => {
            if (!isVisible(node)) return false;
            const accessibleText = [
                node.textContent,
                node.getAttribute('aria-label'),
                node.getAttribute('title'),
            ].filter(Boolean).join(' ');
            return labels.some((label) => textMatchesLabel(accessibleText, label))
                || triggerLabels.some((label) => textMatchesLabel(accessibleText, label));
        });`
const thinkingEffortTriggerSource = chatgptUtils.includes(previousThinkingEffortTrigger)
  ? previousThinkingEffortTrigger
  : upstreamThinkingEffortTrigger
chatgptUtils = replaceWithinFunction(
  chatgptUtils,
  modelSignature,
  toolSignature,
  thinkingEffortTriggerSource,
  currentThinkingEffortTrigger,
  'ChatGPT accessible thinking-effort trigger lookup',
)
chatgptUtils = replaceWithinFunction(
  chatgptUtils,
  modelSignature,
  toolSignature,
  `    if (!menuButton.found) {
        throw new CommandExecutionError('Could not find the ChatGPT model selector in the composer.');
    }`,
  `    if (!menuButton.found) {
        if (allowedPolicyFallback) {
            return { Status: 'Policy fallback', Model: allowedPolicyFallback.label };
        }
        throw new CommandExecutionError('Could not find the ChatGPT model selector in the composer.');
    }`,
  'ChatGPT known in-range fallback without a selector',
)
chatgptUtils = replaceWithinFunction(
  chatgptUtils,
  modelSignature,
  toolSignature,
  `    const menuButton = requireObjectEvaluateResult(unwrapEvaluateResult(await page.evaluate(\`(() => {`,
  `    const readMenuButton = async () => requireObjectEvaluateResult(unwrapEvaluateResult(await page.evaluate(\`(() => {`,
  'ChatGPT delayed model selector reader',
)
chatgptUtils = replaceWithinFunction(
  chatgptUtils,
  modelSignature,
  toolSignature,
  `    })()\`)), 'chatgpt model menu button');
    if (!menuButton.found) {`,
  `    })()\`)), 'chatgpt model menu button');
    let menuButton = { found: false };
    for (let attempt = 0; attempt < 12; attempt += 1) {
        menuButton = await readMenuButton();
        if (menuButton.found) break;
        const delayedCurrent = await getCurrentChatGPTModel(page);
        if (rangePolicy && delayedCurrent.model) {
            try {
                allowedPolicyFallback = requireChatGPTModelInRange(delayedCurrent.model, rangePolicy);
            }
            catch {
                allowedPolicyFallback = null;
            }
            if (delayedCurrent.model === target.key) {
                return { Status: 'Already selected', Model: target.label };
            }
        }
        if (attempt < 11) await page.wait(0.5);
    }
    if (!menuButton.found) {`,
  'ChatGPT delayed model selector wait',
)
chatgptUtils = replaceWithinFunction(
  chatgptUtils,
  modelSignature,
  toolSignature,
  `    if (sliderState.found) {
        const targetValue = Number(target.intelligenceOrder);`,
  `    if (sliderState.found) {
        if (rangePolicy) {
            try {
                allowedPolicyFallback = requireChatGPTModelInRange(sliderState.current, rangePolicy);
            }
            catch {
                allowedPolicyFallback = null;
            }
        }
        const targetValue = Number(target.intelligenceOrder);`,
  'ChatGPT slider model range fallback capture',
)
chatgptUtils = replaceWithinFunction(
  chatgptUtils,
  modelSignature,
  toolSignature,
  `        if (afterSlider.model !== target.key) {
            throw new CommandExecutionError(\`ChatGPT model did not switch to \${target.label}.\`);
        }`,
  `        if (afterSlider.model !== target.key) {
            if (rangePolicy && afterSlider.model) {
                try {
                    allowedPolicyFallback = requireChatGPTModelInRange(afterSlider.model, rangePolicy);
                }
                catch {
                    allowedPolicyFallback = null;
                }
            }
            if (allowedPolicyFallback) {
                await page.pressKey('Escape').catch(() => undefined);
                return { Status: 'Policy fallback', Model: allowedPolicyFallback.label };
            }
            throw new CommandExecutionError(\`ChatGPT model did not switch to \${target.label}.\`);
        }`,
  'ChatGPT range fallback after a failed switch',
)
fs.writeFileSync(chatgptUtilsPath, chatgptUtils)

let ask = fs.readFileSync(askPath, 'utf8')
ask = replaceOnce(
  ask,
  '  GEMINI_DOMAIN,\n  ensureGeminiPage,',
  '  GEMINI_DOMAIN,\n  attachGeminiFile,\n  ensureGeminiPage,',
  'Gemini ask import',
)
ask = replaceOnce(
  ask,
  "        { name: 'thinking', required: false, help: 'Thinking level: standard or extended (omitted = leave unchanged)', default: null },\n",
  "        { name: 'thinking', required: false, help: 'Thinking level: standard or extended (omitted = leave unchanged)', default: null },\n        { name: 'file', required: false, help: 'Attach one local image before sending the prompt' },\n",
  'Gemini ask file option',
)
ask = replaceOnce(
  ask,
  `            const pickerRaw = await page.evaluate(\`
              (() => {
                \${pickModelPickerScript()}
                const picker = findModelPicker();
                if (!picker) return { ok: false, reason: 'Gemini model picker button was not found' };
                try { picker.click(); } catch (_) { return { ok: false, reason: 'Failed to click Gemini model picker button' }; }
                return { ok: true };
              })()
            \`);
            const pickerResult = unwrapBrowserBridgeEnvelope(pickerRaw);
            if (!pickerResult || typeof pickerResult !== 'object' || !pickerResult.ok) {
                throw new CommandExecutionError(
                    pickerResult?.reason || 'Failed to open Gemini model picker for model discovery'
                );
            }`,
  `            const pickerAttempts = 16;
            let pickerResult = null;
            for (let pickerAttempt = 0; pickerAttempt < pickerAttempts; pickerAttempt += 1) {
                const pickerRaw = await page.evaluate(\`
                  (() => {
                    \${pickModelPickerScript()}
                    const picker = findModelPicker();
                    if (!picker) return { ok: false, reason: 'Gemini model picker button was not found' };
                    try { picker.click(); } catch (_) { return { ok: false, reason: 'Failed to click Gemini model picker button' }; }
                    return { ok: true };
                  })()
                \`);
                pickerResult = unwrapBrowserBridgeEnvelope(pickerRaw);
                if (pickerResult && typeof pickerResult === 'object' && pickerResult.ok) break;
                if (pickerAttempt < pickerAttempts - 1) await page.wait(1);
            }
            if (!pickerResult || typeof pickerResult !== 'object' || !pickerResult.ok) {
                throw new CommandExecutionError(
                    pickerResult?.reason || 'Failed to open Gemini model picker for model discovery'
                );
            }`,
  'Gemini model picker readiness retry',
)
ask = replaceOnce(
  ask,
  '        const before = await readGeminiSnapshot(page);\n        await sendGeminiMessage(page, prompt);',
  '        if (kwargs.file) await attachGeminiFile(page, kwargs.file);\n        const before = await readGeminiSnapshot(page);\n        await sendGeminiMessage(page, prompt);',
  'Gemini ask attachment call',
)
fs.writeFileSync(askPath, ask)

const chatgptAskPath = path.join(chatgptDir, 'ask.js')
let chatgptAsk = fs.readFileSync(chatgptAskPath, 'utf8')
chatgptAsk = replaceOnce(chatgptAsk, '    sendChatGPTMessage,', '    sendChatGPTMessage,\n    uploadChatGPTImages,', 'ChatGPT ask image upload import')
chatgptAsk = replaceOnce(chatgptAsk,
  "        { name: 'prompt', positional: true, required: true, help: 'Prompt to send' },",
  "        { name: 'prompt', positional: true, required: true, help: 'Prompt to send' },\n        { name: 'file', required: false, help: 'Attach one local image before sending the prompt' },",
  'ChatGPT ask image option',
)
chatgptAsk = replaceOnce(chatgptAsk,
  '        const baselineMessages = await getVisibleMessages(page);',
  `        if (kwargs.file) {
            const upload = await uploadChatGPTImages(page, [kwargs.file]);
            if (!upload?.ok) throw new CommandExecutionError(upload?.reason || 'ChatGPT image attachment was not ready');
        }
        const baselineMessages = await getVisibleMessages(page);`,
  'ChatGPT ask verified image upload',
)
fs.writeFileSync(chatgptAskPath, chatgptAsk)

fs.copyFileSync(
  path.join(runtimeDir, 'patches', 'gemini-video-command.js'),
  videoPath,
)

const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'))
const chatgptAskEntry = manifest.find(entry => entry?.site === 'chatgpt' && entry?.name === 'ask')
if (!chatgptAskEntry?.args) throw new Error('ChatGPT ask manifest entry not found')
if (!chatgptAskEntry.args.some(arg => arg.name === 'file')) {
  chatgptAskEntry.args.push({ name: 'file', type: 'str', required: false, help: 'Attach one local image before sending the prompt' })
}
const chatgptDetailEntry = manifest.find(
  (entry) => entry?.site === 'chatgpt' && entry?.name === 'detail',
)
if (!chatgptDetailEntry?.args) throw new Error('ChatGPT detail manifest entry not found')
if (!chatgptDetailEntry.args.some((arg) => arg.name === 'refresh')) {
  chatgptDetailEntry.args.push({
    name: 'refresh', type: 'boolean', default: false,
    help: 'Reload a settled target conversation to recover stale rendered text',
  })
}
if (!chatgptDetailEntry.args.some((arg) => arg.name === 'cooldown')) {
  chatgptDetailEntry.args.push({
    name: 'cooldown', type: 'boolean', default: false,
    help: 'Recover a stale limit dialog after the shared access cooldown',
  })
}
const chatgptModelEntry = manifest.find(
  (entry) => entry?.site === 'chatgpt' && entry?.name === 'model',
)
if (!chatgptModelEntry?.args) throw new Error('ChatGPT model manifest entry not found')
if (!chatgptModelEntry.args.some((arg) => arg.name === 'timeout')) {
  chatgptModelEntry.args.push({
    name: 'timeout', type: 'int', default: 45,
    help: 'Model-selection timeout in seconds',
  })
}
const askEntry = manifest.find(
  (entry) => entry?.site === 'gemini' && entry?.name === 'ask',
)
if (!askEntry || !Array.isArray(askEntry.args)) {
  throw new Error('OpenCLI Gemini ask manifest entry was not found')
}
if (!askEntry.args.some((argument) => argument?.name === 'file')) {
  askEntry.args.push({
    name: 'file',
    type: 'str',
    required: false,
    help: 'Attach one local image before sending the prompt',
  })
}

const videoEntry = {
  site: 'gemini',
  name: 'video',
  description: 'Create a Gemini Web video from ordered first/last frames and save it locally',
  access: 'write',
  domain: 'gemini.google.com',
  strategy: 'cookie',
  browser: true,
  args: [
    { name: 'prompt', type: 'str', required: false, positional: true, help: 'Create Video animation prompt' },
    { name: 'first', type: 'str', required: false, help: 'Local empty first-frame image' },
    { name: 'last', type: 'str', required: false, help: 'Local completed last-frame image' },
    { name: 'resume', type: 'str', required: false, help: 'Existing Gemini conversation URL to finish or download' },
    { name: 'aspect', type: 'str', default: '16:9', required: false, help: 'Final aspect ratio', choices: ['16:9', '9:16'] },
    { name: 'output', type: 'str', required: true, help: 'Local MP4 output path' },
    { name: 'timeout', type: 'int', default: 1800, required: false, help: 'Total generation and download timeout in seconds' },
  ],
  columns: ['status', 'file', 'aspect', 'link'],
  defaultFormat: 'plain',
  type: 'js',
  modulePath: 'gemini/video.js',
  sourceFile: 'gemini/video.js',
  navigateBefore: false,
  siteSession: 'persistent',
}
const videoIndex = manifest.findIndex(
  (entry) => entry?.site === 'gemini' && entry?.name === 'video',
)
if (videoIndex >= 0) manifest[videoIndex] = videoEntry
else manifest.push(videoEntry)
fs.writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`)

console.log('Applied project ChatGPT/Gemini browser patches to OpenCLI 1.8.7')

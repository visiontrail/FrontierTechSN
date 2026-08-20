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

const helperMarker = 'export async function attachGeminiFile(page, filePath)'
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
utils = replaceOnce(
  utils,
  `    const submitAction = await page.evaluate(submitComposerScript());
    if (submitAction === 'button') {
        await page.wait(1);
        return 'button';
    }
    if (page.nativeKeyPress) {`,
  `    const submitAction = await page.evaluate(submitComposerScript());
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
    else if (page.nativeKeyPress) {`,
  'Gemini resilient submit action',
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
const modelSignature = 'export async function selectChatGPTModel(page, model) {'
const toolSignature = 'export async function getCurrentChatGPTTool(page) {'
if (!chatgptUtils.includes("'chatgpt intelligence slider'")) {
  chatgptUtils = replaceWithinFunction(
    chatgptUtils,
    modelSignature,
    toolSignature,
  [
    '    if (!currentUrl.startsWith(`${CHATGPT_URL}/new`)) {',
    "        await page.goto(`${CHATGPT_URL}/new`, { waitUntil: 'none' });",
    '        await page.wait(2);',
    '    }',
  ].join('\n'),
  [
    '    if (currentUrl !== CHATGPT_URL && currentUrl !== `${CHATGPT_URL}/`) {',
    "        await page.goto(`${CHATGPT_URL}/`, { waitUntil: 'none' });",
    '        await page.wait(2);',
    '    }',
  ].join('\n'),
  'ChatGPT current new-chat route',
)
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
  `    await page.nativeClick(Number(menuButton.x), Number(menuButton.y));
    await page.wait(0.5);

    let optionCenter = null;`,
  `    await page.nativeClick(Number(menuButton.x), Number(menuButton.y));
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

    let optionCenter = null;`,
    'ChatGPT five-level intelligence slider',
  )
}
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
  `            const pickerAttempts = 4;
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

fs.copyFileSync(
  path.join(runtimeDir, 'patches', 'gemini-video-command.js'),
  videoPath,
)

const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'))
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

console.log('Applied project ChatGPT/Gemini browser patches to OpenCLI 1.8.6')

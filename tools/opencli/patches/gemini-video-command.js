import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { cli, Strategy } from '@jackwener/opencli/registry';
import { ArgumentError, CommandExecutionError, EmptyResultError } from '@jackwener/opencli/errors';
import { sendGeminiMessage } from './utils.js';

const GEMINI_DOMAIN = 'gemini.google.com';
const GEMINI_VIDEOS_URL = 'https://gemini.google.com/videos';

function unwrap(value) {
    if (value && typeof value === 'object' && !Array.isArray(value) && 'session' in value) {
        return 'data' in value ? value.data : value;
    }
    return value;
}

function requireFile(value, label) {
    const resolved = path.resolve(String(value || '').trim());
    if (!resolved || !fs.existsSync(resolved) || !fs.statSync(resolved).isFile()) {
        throw new ArgumentError(`${label} is not a readable file: ${resolved}`);
    }
    return resolved;
}

function resolveOutput(value) {
    const raw = String(value || '').trim();
    if (!raw) throw new ArgumentError('--output is required');
    if (raw === '~') return os.homedir();
    if (raw.startsWith('~/')) return path.join(os.homedir(), raw.slice(2));
    return path.resolve(raw);
}

async function clickLabel(page, labels) {
    const marker = `opencli-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const result = unwrap(await page.evaluate(`(() => {
      const labels = ${JSON.stringify(labels.map(label => label.toLowerCase()))};
      const marker = ${JSON.stringify(marker)};
      const visible = (el) => {
        const rect = el.getBoundingClientRect();
        const style = getComputedStyle(el);
        return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
      };
      const candidates = Array.from(document.querySelectorAll('button, [role="button"], [role="menuitem"], mat-option'));
      for (const node of candidates) {
        if (!visible(node)) continue;
        const text = String(node.getAttribute('aria-label') || node.textContent || '').trim().toLowerCase();
        if (labels.some(label => text === label || text.includes(label))) {
          node.setAttribute('data-opencli-click-target', marker);
          const rect = node.getBoundingClientRect();
          return { ok: true, text, marker, x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
        }
      }
      return { ok: false };
    })()`));
    if (!result?.ok) return false;
    try {
        // A real CDP pointer click is required for browser downloads. Calling
        // HTMLElement.click() from evaluate can make Gemini animate the
        // control without granting the download user activation.
        if (typeof page.cdp === 'function' && Number.isFinite(result.x) && Number.isFinite(result.y)) {
            await page.cdp('Input.dispatchMouseEvent', {
                type: 'mouseMoved', x: result.x, y: result.y,
            });
            // Download controls are revealed by video hover. Give Gemini one
            // frame to apply its pointer-events/opacity transition.
            await page.wait(0.15);
            await page.cdp('Input.dispatchMouseEvent', {
                type: 'mousePressed', x: result.x, y: result.y, button: 'left', clickCount: 1,
            });
            await page.cdp('Input.dispatchMouseEvent', {
                type: 'mouseReleased', x: result.x, y: result.y, button: 'left', clickCount: 1,
            });
        } else {
            await page.click(`[data-opencli-click-target="${marker}"]`);
        }
        return true;
    } finally {
        await page.evaluate(`document.querySelector('[data-opencli-click-target="${marker}"]')?.removeAttribute('data-opencli-click-target')`).catch(() => undefined);
    }
}

async function selectAspectRatio(page, aspect) {
    const desired = aspect === '9:16' ? 'Portrait (9:16)' : 'Landscape (16:9)';
    let current = null;
    for (let attempt = 0; attempt < 30 && !current; attempt += 1) {
        current = unwrap(await page.evaluate(`(() => {
          const button = document.querySelector('button[aria-label^="Aspect ratio"]');
          return button ? { label: button.getAttribute('aria-label') || '', expanded: button.getAttribute('aria-expanded') } : null;
        })()`));
        if (!current) await page.wait(1);
    }
    if (!current) throw new CommandExecutionError('Gemini Create Video aspect-ratio control was not found');
    if (String(current.label).includes(desired) && current.expanded !== 'true') return;
    if (current.expanded !== 'true') {
        await page.click('button[aria-label^="Aspect ratio"]');
        await page.wait(0.8);
    }
    try {
        await page.click(`input-companion-item[role="menuitemradio"][aria-label="${desired}"]`);
    } catch (error) {
        throw new CommandExecutionError(
            `Gemini Create Video did not expose ${desired}: ${String(error?.message || error)}`
        );
    }
    await page.wait(0.8);
    const selected = String(unwrap(await page.evaluate(
        'document.querySelector(\'button[aria-label^="Aspect ratio"]\')?.getAttribute(\'aria-label\') || \'\''
    )) || '');
    if (!selected.includes(desired)) {
        throw new CommandExecutionError(`Gemini Create Video selected ${selected || 'an unknown ratio'}, expected ${desired}`);
    }
}

async function setFileInputViaCdp(page, filePaths) {
    if (typeof page.cdp !== 'function') return false;
    await page.cdp('DOM.enable', {}).catch(() => undefined);

    // Gemini can leave more than one hidden Filedata input in the document.
    // Resolve the live input from JavaScript and address its runtime object
    // directly; a document-level querySelector node id can point at a stale
    // input even though DOM.setFileInputFiles reports success.
    const evaluated = unwrap(await page.cdp('Runtime.evaluate', {
        expression: `(() => {
          const roots = [document.querySelector('input-container'), document].filter(Boolean);
          for (const root of roots) {
            const inputs = Array.from(root.querySelectorAll('input[name="Filedata"], input[type="file"]'));
            const live = inputs.find(input => !input.disabled && input.isConnected);
            if (live) return live;
          }
          return null;
        })()`,
        objectGroup: 'opencli-gemini-video-upload',
        returnByValue: false,
    }));
    const objectId = String(evaluated?.result?.objectId || '');
    if (!objectId) return false;
    await page.cdp('DOM.setFileInputFiles', { objectId, files: filePaths });

    const fileCount = Number(unwrap(await page.evaluate(`(() => {
      const roots = [document.querySelector('input-container'), document].filter(Boolean);
      for (const root of roots) {
        const inputs = Array.from(root.querySelectorAll('input[name="Filedata"], input[type="file"]'));
        const live = inputs.find(input => !input.disabled && input.isConnected && input.files?.length);
        if (live) return live.files.length;
      }
      return 0;
    })()`))) || 0;
    await page.cdp('Runtime.releaseObjectGroup', {
        objectGroup: 'opencli-gemini-video-upload',
    }).catch(() => undefined);
    return fileCount >= filePaths.length;
}

async function uploadFrame(page, filePath, expectedCount) {
    const fileName = path.basename(filePath);
    const payload = {
        base64: fs.readFileSync(filePath).toString('base64'),
        fileName,
        mimeType: fileName.toLowerCase().endsWith('.png') ? 'image/png' : 'image/jpeg',
    };
    const attachmentState = async () => unwrap(await page.evaluate(`(() => {
      const root = document.querySelector('input-container') || document;
      const attachments = root.querySelectorAll('gem-media-attachment').length;
      const busyNodes = Array.from(root.querySelectorAll('uploader-file-preview [role="progressbar"], gem-media-attachment [role="progressbar"]'));
      const inputs = Array.from(root.querySelectorAll('input[name="Filedata"], input[type="file"]'));
      return {
        attachments,
        busy: busyNodes.length > 0,
        busyCount: busyNodes.length,
        inputFiles: inputs.map(input => Array.from(input.files || []).map(file => ({ name: file.name, size: file.size, type: file.type }))),
        textTail: String(root.innerText || '').trim().slice(-600),
      };
    })()`));
    // Chrome's extension debugger can reject DOM.setFileInputFiles with
    // "Not allowed". Gemini does accept a DataTransfer-backed File when the
    // live input is created by a real click first, which is also the exact
    // sequence used by the generic Browser CLI in the verified manual probe.
    let clickError = null;
    try {
        await page.click('button[aria-label="File upload"]');
        await page.wait(1.5);
    } catch (error) {
        clickError = error;
    }
    const fallback = unwrap(await page.evaluate(`(() => {
          const inputs = Array.from(document.querySelectorAll('input[name="Filedata"], input[type="file"]'));
          const input = inputs.reverse().find(candidate => !candidate.disabled && candidate.isConnected) || inputs.at(0);
          if (!input) return { ok: false, reason: 'Filedata input disappeared' };
          const transfer = new DataTransfer();
          const item = ${JSON.stringify(payload)};
          const binary = atob(item.base64);
          const bytes = new Uint8Array(binary.length);
          for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
          transfer.items.add(new File([bytes], item.fileName, { type: item.mimeType }));
          input.files = transfer.files;
          input.dispatchEvent(new Event('input', { bubbles: true }));
          input.dispatchEvent(new Event('change', { bubbles: true }));
          return { ok: true, bytes: bytes.length, inputCount: inputs.length };
    })()`));
    if (!fallback?.ok) {
        throw new CommandExecutionError(
            `Gemini keyframe upload failed: ${String(fallback?.reason || clickError?.message || clickError || 'unknown error')}`
        );
    }
    // A visible completed attachment is the pre-submit contract for each
    // ordered keyframe.
    const deadline = Date.now() + 120000;
    while (Date.now() < deadline) {
        await page.wait(1);
        const state = await attachmentState();
        if (Number(state?.attachments || 0) >= expectedCount && !state?.busy) return;
    }
    const finalState = await attachmentState();
    throw new CommandExecutionError(
        `Gemini Create Video did not retain and finish processing keyframe ${expectedCount}: ${JSON.stringify(finalState)}`
    );
}

async function uploadFrames(page, filePaths) {
    for (let index = 0; index < filePaths.length; index += 1) {
        await uploadFrame(page, filePaths[index], index + 1);
    }
}

async function submittedVideoPrompt(page) {
    const state = unwrap(await page.evaluate(`(() => {
      const composer = document.querySelector('[contenteditable="true"][role="textbox"]');
      return {
        submitted: !!document.querySelector('user-query') || /^\/app\/[a-z0-9]+/i.test(location.pathname),
        draft: String(composer?.textContent || '').trim(),
      };
    })()`));
    return Boolean(state?.submitted) && !state?.draft;
}

async function submitVideoPrompt(page, prompt) {
    await sendGeminiMessage(page, prompt);
    for (let attempt = 0; attempt < 10; attempt += 1) {
        await page.wait(1);
        if (await submittedVideoPrompt(page)) return;
    }

    // The Videos composer can place its arrow button outside the generic
    // Gemini chat helper's root candidates. Click it natively and verify that
    // the draft became an actual user query before entering a long wait.
    let marked = false;
    try {
        await page.click('button[aria-label="Send message"]');
        marked = true;
    } catch {
        const result = unwrap(await page.evaluate(`(() => {
          const root = document.querySelector('input-area-v2') || document.querySelector('input-container');
          if (!root) return { ok: false };
          const excluded = /upload|tool|dictate|microphone|mode|aspect|ratio/i;
          const buttons = Array.from(root.querySelectorAll('button')).filter(button => {
            if (button.disabled || button.getAttribute('aria-disabled') === 'true') return false;
            const rect = button.getBoundingClientRect();
            const label = String(button.getAttribute('aria-label') || button.textContent || '');
            return rect.width > 0 && rect.height > 0 && !excluded.test(label);
          });
          const target = buttons.sort((a, b) => b.getBoundingClientRect().right - a.getBoundingClientRect().right)[0];
          if (!target) return { ok: false };
          target.setAttribute('data-opencli-video-submit', 'true');
          return { ok: true };
        })()`));
        if (result?.ok) {
            await page.click('[data-opencli-video-submit="true"]');
            marked = true;
        }
    }
    if (marked) {
        for (let attempt = 0; attempt < 15; attempt += 1) {
            await page.wait(1);
            if (await submittedVideoPrompt(page)) return;
        }
    }
    throw new CommandExecutionError('Gemini Create Video kept the prompt in the composer instead of submitting it');
}

async function visibleVideoUrls(page) {
    const value = unwrap(await page.evaluate(`(() => Array.from(document.querySelectorAll('generated-video'))
      .map((video, index) => video.id || video.closest('model-response')?.id || 'generated-video-' + index))()`));
    return Array.isArray(value) ? value : [];
}

async function waitForVideo(page, before, timeoutSeconds) {
    const baseline = new Set(before);
    const deadline = Date.now() + timeoutSeconds * 1000;
    while (Date.now() < deadline) {
        await page.wait(5);
        const state = unwrap(await page.evaluate(`(() => {
          const videos = Array.from(document.querySelectorAll('generated-video')).map((video, index) => ({
            src: video.id || video.closest('model-response')?.id || 'generated-video-' + index,
            readyState: video.querySelector('button[aria-label="Download video"]') ? 4 : 0,
          }));
          const text = (document.querySelector('main')?.innerText || '').slice(-1800);
          return { videos, text };
        })()`));
        const videos = Array.isArray(state?.videos) ? state.videos : [];
        const ready = videos.find(video => !baseline.has(video.src) && video.readyState >= 2);
        if (ready) return ready;
        if (/could not generate|generation failed|try again|unable to create/i.test(String(state?.text || ''))) {
            throw new CommandExecutionError(`Gemini Create Video reported a generation failure: ${String(state.text).slice(-500)}`);
        }
    }
    throw new EmptyResultError('gemini video', `No completed video appeared within ${timeoutSeconds} seconds`);
}

function downloadedPath(result) {
    const filename = String(result?.filename || '').trim();
    if (!filename) return '';
    const candidates = [filename, path.join(os.homedir(), 'Downloads', path.basename(filename))];
    return candidates.find(candidate => path.isAbsolute(candidate) && fs.existsSync(candidate)) || '';
}

async function downloadVideo(page, outputPath, timeoutSeconds) {
    const browserFileName = `opencli-gemini-video-${Date.now()}-${Math.random().toString(36).slice(2)}.mp4`;
    const browserDownload = unwrap(await page.evaluate(`(async () => {
      try {
        const video = document.querySelector('generated-video video');
        const source = String(video?.currentSrc || video?.src || '');
        if (!source) return { ok: false, reason: 'generated video source URL is missing' };
        // The contribution URL requires the signed-in Gemini cookies. Fetch it
        // inside the page, then hand the Blob to Chrome with a unique filename;
        // this avoids depending on a hover-only control or an extension download
        // event that some Browser Bridge builds do not emit.
        const response = await fetch(source, { credentials: 'include' });
        if (!response.ok) return { ok: false, reason: 'video fetch returned HTTP ' + response.status };
        const blob = await response.blob();
        if (blob.size < 1024) return { ok: false, reason: 'video fetch returned an empty blob' };
        const anchor = document.createElement('a');
        const objectUrl = URL.createObjectURL(blob);
        anchor.href = objectUrl;
        anchor.download = ${JSON.stringify(browserFileName)};
        anchor.style.display = 'none';
        document.body.appendChild(anchor);
        anchor.click();
        setTimeout(() => {
          URL.revokeObjectURL(objectUrl);
          anchor.remove();
        }, 30000);
        return { ok: true, size: blob.size, type: blob.type };
      } catch (error) {
        return { ok: false, reason: String(error?.message || error) };
      }
    })()`));
    if (browserDownload?.ok) {
        const source = path.join(os.homedir(), 'Downloads', browserFileName);
        const deadline = Date.now() + timeoutSeconds * 1000;
        while (Date.now() < deadline) {
            if (fs.existsSync(source) && fs.statSync(source).size >= Number(browserDownload.size || 1024)) {
                fs.mkdirSync(path.dirname(outputPath), { recursive: true });
                fs.copyFileSync(source, outputPath);
                return { downloaded: true, filename: source, size: fs.statSync(source).size };
            }
            await page.wait(0.5);
        }
        throw new CommandExecutionError(`Gemini video was fetched but Chrome did not finish ${browserFileName}`);
    }

    if (typeof page.waitForDownload !== 'function') {
        throw new CommandExecutionError(
            `Gemini in-page download failed (${browserDownload?.reason || 'unknown error'}) and the Browser Bridge has no download lifecycle support`
        );
    }
    if (typeof page.cdp === 'function') {
        const videoCenter = unwrap(await page.evaluate(`(() => {
          const video = document.querySelector('generated-video');
          if (!video) return null;
          const rect = video.getBoundingClientRect();
          return { x: rect.left + rect.width / 2, y: rect.top + Math.min(48, rect.height / 2) };
        })()`));
        if (Number.isFinite(videoCenter?.x) && Number.isFinite(videoCenter?.y)) {
            await page.cdp('Input.dispatchMouseEvent', {
                type: 'mouseMoved', x: videoCenter.x, y: videoCenter.y,
            });
            await page.wait(0.5);
        }
    }
    const download = page.waitForDownload('', timeoutSeconds * 1000);
    if (!await clickLabel(page, ['download video'])) {
        if (!await clickLabel(page, ['share video', 'share'])) {
            throw new CommandExecutionError('Gemini generated video is visible, but no Download or Share control was found');
        }
        await page.wait(1);
        if (!await clickLabel(page, ['download video', 'download'])) {
            throw new CommandExecutionError('Gemini Share menu did not expose Download video');
        }
    }
    let result;
    try {
        result = await download;
    } catch (error) {
        throw new CommandExecutionError(
            `${String(error?.message || error)}; Gemini in-page fetch failed first: ${browserDownload?.reason || 'unknown error'}`
        );
    }
    if (!result?.downloaded) {
        throw new CommandExecutionError(
            result?.error || `Gemini video download did not complete after in-page fetch failed: ${browserDownload?.reason || 'unknown error'}`
        );
    }
    const source = downloadedPath(result);
    if (!source) {
        throw new CommandExecutionError(`Gemini download completed but its local path could not be resolved: ${JSON.stringify(result)}`);
    }
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.copyFileSync(source, outputPath);
    if (!fs.existsSync(outputPath) || fs.statSync(outputPath).size < 1024) {
        throw new CommandExecutionError(`Downloaded Gemini video is empty: ${outputPath}`);
    }
    return result;
}

export const videoCommand = cli({
    site: 'gemini',
    name: 'video',
    access: 'write',
    description: 'Create a Gemini Web video from ordered first/last frames and save it locally',
    domain: GEMINI_DOMAIN,
    strategy: Strategy.COOKIE,
    browser: true,
    siteSession: 'persistent',
    navigateBefore: false,
    defaultFormat: 'plain',
    args: [
        { name: 'prompt', positional: true, help: 'Create Video animation prompt' },
        { name: 'first', help: 'Local empty first-frame image' },
        { name: 'last', help: 'Local completed last-frame image' },
        { name: 'resume', help: 'Existing gemini.google.com/app conversation URL to finish or download' },
        { name: 'aspect', default: '16:9', choices: ['16:9', '9:16'], help: 'Final aspect ratio' },
        { name: 'output', required: true, help: 'Local MP4 output path' },
        { name: 'timeout', type: 'int', default: 1800, help: 'Total generation and download timeout in seconds' },
    ],
    columns: ['status', 'file', 'aspect', 'link'],
    func: async (page, kwargs) => {
        const prompt = String(kwargs.prompt || '').trim();
        const resume = String(kwargs.resume || '').trim();
        if (resume && !/^https:\/\/gemini\.google\.com\/app\/[a-z0-9]+/i.test(resume)) {
            throw new ArgumentError('--resume must be a Gemini conversation URL');
        }
        if (!resume && !prompt) throw new ArgumentError('video prompt is required unless --resume is used');
        const first = resume ? '' : requireFile(kwargs.first, '--first');
        const last = resume ? '' : requireFile(kwargs.last, '--last');
        const output = resolveOutput(kwargs.output);
        const aspect = String(kwargs.aspect || '16:9');
        if (!['16:9', '9:16'].includes(aspect)) throw new ArgumentError('--aspect must be 16:9 or 9:16');
        const timeout = Number(kwargs.timeout || 1800);
        if (!Number.isInteger(timeout) || timeout < 60) throw new ArgumentError('--timeout must be at least 60 seconds');

        await page.goto(resume || GEMINI_VIDEOS_URL, { waitUntil: 'load', settleMs: 8000 });
        if (resume) {
            await waitForVideo(page, [], timeout);
        } else {
            await clickLabel(page, ['create with omni']);
            await page.wait(1.5);
            await selectAspectRatio(page, aspect);
            const before = await visibleVideoUrls(page);
            await uploadFrames(page, [first, last]);
            await submitVideoPrompt(page, prompt);
            await waitForVideo(page, before, Math.max(60, timeout - 180));
        }
        await downloadVideo(page, output, Math.min(timeout, 180));
        const link = String(unwrap(await page.evaluate('window.location.href')) || GEMINI_VIDEOS_URL);
        return [{ status: 'saved', file: output, aspect, link }];
    },
});

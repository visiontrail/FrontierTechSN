import * as fs from 'node:fs';
import { randomUUID } from 'node:crypto';
import * as os from 'node:os';
import * as path from 'node:path';
import { cli, Strategy } from '@jackwener/opencli/registry';
import { ArgumentError, CommandExecutionError } from '@jackwener/opencli/errors';
import { getGeminiConversationList, getGeminiVisibleTurns, sendGeminiMessage } from './utils.js';

const GEMINI_DOMAIN = 'gemini.google.com';
const GEMINI_VIDEOS_URL = 'https://gemini.google.com/videos';
const VIDEO_UPLOAD_CAPABILITY_CODE = 'OPENCLI_CAPABILITY_UNAVAILABLE:GEMINI_VIDEO_LOCAL_FILE_UPLOAD';
const VIDEO_INPUT_HYDRATION_STUCK_CODE = 'OPENCLI_CAPABILITY_DEGRADED:GEMINI_VIDEO_UPLOAD_INPUT_HYDRATION_STUCK';

export async function openVideoPage(page, url) {
    // A newly registered Ego window can reject its first navigation. Retry
    // only that startup race on the same owned page, before any submission.
    // Recreating the session in the outer retry loop repeats the race.
    for (let attempt = 0; attempt < 3; attempt += 1) {
        try {
            await page.goto(url, { waitUntil: 'load', settleMs: 8000 });
            return;
        } catch (error) {
            if (!/Navigation rejected/i.test(String(error?.message || error)) || attempt === 2) throw error;
            await page.wait(1);
        }
    }
}

export function isVideoQuotaMessage(text) {
    const value = String(text || '').replace(/\s+/g, ' ').trim();
    return /(?:reached|exceeded|used up|out of|no more).{0,70}(?:video|generation|usage|daily|quota|limit)|(?:video|generation|usage|daily).{0,70}(?:limit reached|limit exceeded|quota exhausted|try again tomorrow)|(?:视频|影片).{0,30}(?:上限|限额|额度|用尽)|(?:已达到|已用完).{0,30}(?:生成|额度|限额)/i.test(value);
}

export async function waitForVideoComposer(page, { timeoutMs = 90000 } = {}) {
    // The /videos skeleton exposes a working file picker before Gemini has
    // activated Videos. Uploading then can leave an invisible duplicate-file
    // entry when the real composer replaces it. Require the active tool as
    // well as usable controls before touching any local file.
    const budget = Math.max(0, Number(timeoutMs) || 0);
    const deadline = Date.now() + budget;
    const attempts = Math.ceil(budget / 1000) + 1;
    let stable = 0;
    let state = null;
    for (let attempt = 0; attempt < attempts; attempt += 1) {
        state = unwrap(await page.evaluate(`(() => {
          const usable = selector => {
            const node = document.querySelector(selector);
            if (!node || !node.isConnected || node.disabled || node.matches?.(':disabled')
                || node.getAttribute('aria-disabled') === 'true') return false;
            const rect = node.getBoundingClientRect();
            const style = getComputedStyle(node);
            return rect.width > 0 && rect.height > 0
              && style.display !== 'none' && style.visibility !== 'hidden';
          };
          const videoMode = usable('button[aria-label="Deselect Videos"]');
          const upload = usable('button[aria-label="File upload"]');
          const aspect = usable('button[aria-label^="Aspect ratio"]');
          const composer = usable('[contenteditable="true"][role="textbox"], textarea[aria-label="Enter a prompt for Gemini"]');
          const videoButton = document.querySelector('button[aria-label="Deselect Videos"]');
          const disabledVideo = !!videoButton && (videoButton.matches(':disabled') || videoButton.getAttribute('aria-disabled') === 'true');
          const notice = [...document.querySelectorAll('[role="alert"], [role="status"], .gds-callout')]
            .filter(node => node.getBoundingClientRect().width > 0).map(node => node.innerText || '').join(' ').slice(0, 2000);
          return { videoMode, upload, aspect, composer, disabledVideo, notice,
            ready: videoMode && upload && aspect && composer };
        })()`));
        if (isVideoQuotaMessage(state?.notice)) {
            throw new CommandExecutionError('GEMINI_VIDEO_QUOTA_EXHAUSTED: ' + state.notice);
        }
        stable = state?.ready === true ? stable + 1 : 0;
        if (stable >= 2) return state;
        if (attempt + 1 >= attempts || Date.now() >= deadline) break;
        await page.wait(Math.max(0, Math.min(1, (deadline - Date.now()) / 1000)));
    }
    if (state?.disabledVideo === true && !state.composer) {
        throw new CommandExecutionError('GEMINI_VIDEO_UNAVAILABLE: Videos remains disabled after the bounded readiness check: ' + JSON.stringify(state));
    }
    throw new CommandExecutionError(
        'GEMINI_VIDEO_COMPOSER_NOT_READY: active Videos tool and upload controls did not stabilize: '
        + JSON.stringify(state)
    );
}

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

export async function setFileInputViaCdp(page, filePaths, selector) {
    if (typeof page.cdp !== 'function') return false;
    await page.cdp('DOM.enable', {}).catch(() => undefined);

    // Gemini can leave more than one hidden Filedata input in the document.
    // Address the exact input marked immediately after the trusted picker
    // click. A document-level query can otherwise resolve a stale input even
    // though DOM.setFileInputFiles itself reports success.
    const objectGroup = 'opencli-gemini-video-upload';
    try {
        const evaluated = unwrap(await page.cdp('Runtime.evaluate', {
            expression: `document.querySelector(${JSON.stringify(selector)})`,
            objectGroup,
            returnByValue: false,
        }));
        const objectId = String(evaluated?.result?.objectId || '');
        if (!objectId) return false;
        await page.cdp('DOM.setFileInputFiles', { objectId, files: filePaths });
        return true;
    } finally {
        await page.cdp('Runtime.releaseObjectGroup', { objectGroup }).catch(() => undefined);
    }
}

function uploadFailure(expectedCount, substep, diagnostic) {
    const nativeErrors = Array.isArray(diagnostic?.nativeErrors) ? diagnostic.nativeErrors : [];
    const nativeDenied = nativeErrors.some(error =>
        error?.method === 'page.setFileInput'
        && /-32000|not allowed|Page\.fileChooserOpened[^\n]*not received/i.test(String(error?.error || ''))
    );
    const directCdpDenied = nativeErrors.some(error =>
        error?.method === 'DOM.setFileInputFiles'
        && /CDP method not permitted[\s\S]*Runtime\.evaluate/i.test(String(error?.error || ''))
    );
    const inputFiles = diagnostic?.state?.inputFiles;
    const attachments = diagnostic?.state?.attachments;
    const clearedIdleLimit = diagnostic?.clearedIdleLimit;
    const syntheticCleared = diagnostic?.method === 'DataTransfer'
        && diagnostic?.state?.busy === false
        && Number.isInteger(attachments) && attachments >= 0 && attachments < expectedCount
        && Number.isFinite(clearedIdleLimit) && clearedIdleLimit > 0
        && Number(diagnostic?.clearedIdleSamples || 0) >= clearedIdleLimit
        && Array.isArray(inputFiles)
        && inputFiles.length > 0
        && inputFiles.every(files => Array.isArray(files) && files.length === 0);
    const uploadCapabilityUnavailable = substep === 'wait_for_attachment'
        && nativeDenied
        && directCdpDenied
        && syntheticCleared;
    const clickAttempts = Array.isArray(diagnostic?.clickAttempts) ? diagnostic.clickAttempts : [];
    const inputState = diagnostic?.inputState;
    const clickSucceeded = clickAttempts.length === 1
        && clickAttempts[0]?.reason === 'initial'
        && clickAttempts[0]?.ok === true;
    const buttonReady = inputState?.button?.connected
        && inputState?.button?.visible
        && !inputState?.button?.disabled;
    const liveInputs = Array.isArray(inputState?.inputs) ? inputState.inputs : [];
    const baselineInputs = Array.isArray(diagnostic?.baselineState?.inputs)
        ? diagnostic.baselineState.inputs
        : [];
    const baselineInputCount = Number(inputState?.baselineInputCount);
    const freshInputCount = Number(inputState?.freshInputCount);
    const onlyStableBaselineInputs = Number.isInteger(baselineInputCount)
        && baselineInputCount >= 0
        && baselineInputCount === liveInputs.length
        && baselineInputCount === baselineInputs.length
        && liveInputs.every((input, index) => {
            const baseline = baselineInputs[index];
            return input?.connected === true
                && input?.disabled === false
                && Array.isArray(input?.files)
                && input.files.length === 0
                && baseline?.connected === true
                && baseline?.disabled === false
                && Array.isArray(baseline?.files)
                && baseline.files.length === 0
                && Number(input?.index) === Number(baseline?.index)
                && String(input?.name || '') === String(baseline?.name || '')
                && String(input?.type || '') === String(baseline?.type || '');
        });
    const hydrationStuck = substep === 'discover_live_input'
        && diagnostic?.exhausted === true
        && diagnostic?.reopened === false
        && clickSucceeded
        && inputState?.ok === false
        && inputState?.busy === true
        && Number(inputState?.busyCount) > 0
        && inputState?.documentHasFocus === true
        && buttonReady
        && freshInputCount === 0
        && onlyStableBaselineInputs;
    const stuckSignature = hydrationStuck
        ? [
            `keyframe=${expectedCount}`,
            `attachments=${Number(diagnostic?.baselineState?.attachments || 0)}`,
            'busy=1',
            'fresh=0',
            `baseline=${baselineInputCount}`,
            'click=ok',
            'button=ready',
            'focus=1',
        ].join(';')
        : '';
    const enrichedDiagnostic = hydrationStuck
        ? {
            ...diagnostic,
            errorCode: VIDEO_INPUT_HYDRATION_STUCK_CODE,
            stuckSignature,
        }
        : diagnostic;
    const capabilityCode = uploadCapabilityUnavailable
        ? `${VIDEO_UPLOAD_CAPABILITY_CODE}: `
        : hydrationStuck
            ? `${VIDEO_INPUT_HYDRATION_STUCK_CODE}: `
            : '';
    return new CommandExecutionError(
        `${capabilityCode}Gemini keyframe ${expectedCount} upload failed at ${substep}: ${JSON.stringify(enrichedDiagnostic)}`
    );
}

export async function uploadFrame(
    page,
    filePath,
    expectedCount,
    {
        attachmentTimeoutMs = 120000,
        clearedIdleLimit = 8,
        inputReadyTimeoutMs = 15000,
        inputPollIntervalMs = 500,
        inputReopenAfterMs = 6000,
    } = {},
) {
    const fileName = path.basename(filePath);
    const fileBytes = fs.readFileSync(filePath);
    const payload = {
        base64: fileBytes.toString('base64'),
        fileName,
        mimeType: fileName.toLowerCase().endsWith('.png') ? 'image/png' : 'image/jpeg',
    };
    const attachmentState = async () => unwrap(await page.evaluate(`(() => {
      const root = document.querySelector('input-container') || document;
      // Gemini can portal the processed keyframe outside input-container.
      // Count globally, while retaining the composer-local count in failure
      // diagnostics so a future DOM move is immediately distinguishable from
      // a provider-side processing delay.
      const attachments = document.querySelectorAll('gem-media-attachment').length;
      const attachmentsInComposer = root.querySelectorAll('gem-media-attachment').length;
      const busyNodes = Array.from(document.querySelectorAll(
        'uploader-file-preview [role="progressbar"], gem-media-attachment [role="progressbar"]'
      ));
      const inputs = Array.from(document.querySelectorAll('input[name="Filedata"], input[type="file"]'));
      return {
        attachments,
        attachmentsInComposer,
        busy: busyNodes.length > 0,
        busyCount: busyNodes.length,
        inputFiles: inputs.map(input => Array.from(input.files || []).map(file => ({ name: file.name, size: file.size, type: file.type }))),
        textTail: String(root.innerText || '').trim().slice(-600),
      };
    })()`));

    const marker = `opencli-video-upload-${expectedCount}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const cleanupUploadMarkers = async () => {
        await page.evaluate(`(() => {
          const marker = ${JSON.stringify(marker)};
          const listener = window[marker];
          if (listener) document.removeEventListener('click', listener, true);
          delete window[marker];
          for (const input of document.querySelectorAll('input[name="Filedata"], input[type="file"]')) {
            if (input.getAttribute('data-opencli-video-upload-baseline') === marker) {
              input.removeAttribute('data-opencli-video-upload-baseline');
            }
            if (input.getAttribute('data-opencli-video-upload-target') === marker) {
              input.removeAttribute('data-opencli-video-upload-target');
            }
          }
          return true;
        })()`).catch(() => undefined);
    };
    let baselineState;
    try {
        baselineState = unwrap(await page.evaluate(`(() => {
          const marker = ${JSON.stringify(marker)};
          const inputs = Array.from(document.querySelectorAll(
            'input[name="Filedata"], input[type="file"]'
          ));
          const summarize = (input, index) => ({
            index,
            name: input.name || '',
            type: input.type || '',
            disabled: !!input.disabled,
            connected: !!input.isConnected,
            files: Array.from(input.files || []).map(file => ({
              name: file.name, size: file.size, type: file.type,
            })),
          });
          for (const input of inputs) {
            if (!input.disabled && input.isConnected) {
              input.setAttribute('data-opencli-video-upload-baseline', marker);
            }
          }
          // Gemini reuses Filedata after the first keyframe. Bind the input
          // actually activated by this upload-button click, rather than
          // requiring a new DOM node or guessing from stale global inputs.
          const listener = (event) => {
            const input = event.target;
            if (!input?.matches?.('input[type="file"]')
                || input.disabled || !input.isConnected) return;
            input.setAttribute('data-opencli-video-upload-target', marker);
            // File injection below owns this chooser. Leaving its native
            // dialog open blocks later uploads and stacks dialogs on retries.
            event.preventDefault();
          };
          window[marker] = listener;
          document.addEventListener('click', listener, true);
          return {
            attachments: document.querySelectorAll('gem-media-attachment').length,
            inputs: inputs.map(summarize),
          };
        })()`));
    } catch (error) {
        await cleanupUploadMarkers();
        throw uploadFailure(expectedCount, 'snapshot_live_inputs', {
            fileName,
            error: String(error?.message || error),
        });
    }
    const clickAttempts = [];
    const clickUpload = async (reason) => {
        try {
            await page.click('button[aria-label="File upload"]');
            clickAttempts.push({ reason, ok: true, error: '' });
        } catch (error) {
            clickAttempts.push({
                reason,
                ok: false,
                error: String(error?.message || error),
            });
        }
    };
    await clickUpload('initial');

    const pollIntervalMs = Math.max(50, Number(inputPollIntervalMs) || 500);
    const pollAttempts = Math.max(
        1,
        Math.ceil(Math.max(0, Number(inputReadyTimeoutMs) || 0) / pollIntervalMs),
    );
    const reopenAfterPoll = Math.max(
        1,
        Math.ceil(Math.max(0, Number(inputReopenAfterMs) || 0) / pollIntervalMs),
    );
    let selected = null;
    let reopened = false;
    for (let pollAttempt = 1; pollAttempt <= pollAttempts; pollAttempt += 1) {
        try {
            await page.wait(pollIntervalMs / 1000);
            selected = unwrap(await page.evaluate(`(() => {
          const marker = ${JSON.stringify(marker)};
          const roots = [document.querySelector('input-container'), document].filter(Boolean);
          const inputs = [];
          for (const root of roots) {
            for (const input of root.querySelectorAll('input[name="Filedata"], input[type="file"]')) {
              if (!inputs.includes(input)) inputs.push(input);
            }
          }
          const candidates = inputs.filter(input => !input.disabled && input.isConnected);
          const freshCandidates = candidates.filter(input =>
            input.getAttribute('data-opencli-video-upload-baseline') !== marker
          );
          const activatedInput = candidates.find(candidate =>
            candidate.getAttribute('data-opencli-video-upload-target') === marker
          );
          const input = activatedInput || freshCandidates.at(-1);
          const summarize = (candidate, index) => ({
            index,
            name: candidate.name || '',
            type: candidate.type || '',
            accept: candidate.accept || '',
            disabled: !!candidate.disabled,
            connected: !!candidate.isConnected,
            files: Array.from(candidate.files || []).map(file => ({
              name: file.name, size: file.size, type: file.type,
            })),
          });
          const uploadButton = document.querySelector('button[aria-label="File upload"]');
          const buttonRect = uploadButton?.getBoundingClientRect();
          const buttonStyle = uploadButton ? getComputedStyle(uploadButton) : null;
          const busyNodes = document.querySelectorAll(
            'uploader-file-preview [role="progressbar"], gem-media-attachment [role="progressbar"]'
          );
          const button = uploadButton ? {
            connected: !!uploadButton.isConnected,
            disabled: !!uploadButton.disabled || uploadButton.getAttribute('aria-disabled') === 'true',
            visible: !!buttonRect && buttonRect.width > 0 && buttonRect.height > 0
              && buttonStyle?.display !== 'none' && buttonStyle?.visibility !== 'hidden',
          } : null;
          if (!input) {
            return {
              ok: false,
              inputs: inputs.map(summarize),
              documentHasFocus: document.hasFocus(),
              busy: busyNodes.length > 0,
              busyCount: busyNodes.length,
              button,
              baselineInputCount: candidates.length - freshCandidates.length,
              freshInputCount: freshCandidates.length,
            };
          }
          input.setAttribute('data-opencli-video-upload-target', marker);
          return {
            ok: true,
            selector: '[data-opencli-video-upload-target="' + marker + '"]',
            selected: summarize(input, inputs.indexOf(input)),
            inputCount: inputs.length,
            baselineInputCount: candidates.length - freshCandidates.length,
            freshInputCount: freshCandidates.length,
            inputOrigin: activatedInput ? 'activated_by_upload_control' : 'new_after_click',
          };
            })()`));
        } catch (error) {
            await cleanupUploadMarkers();
            throw uploadFailure(expectedCount, 'discover_live_input', {
                fileName,
                pollAttempt,
                pollAttempts,
                baselineState,
                clickAttempts,
                error: String(error?.message || error),
            });
        }
        if (selected?.ok && selected?.selector) break;

        // A native chooser normally takes focus away from the document. Never
        // click again in that state: doing so can stack a second chooser. One
        // retry is allowed only after hydration had ample time and the page is
        // still focused with an idle, visible, enabled upload control.
        const buttonReady = selected?.button?.connected
            && selected?.button?.visible
            && !selected?.button?.disabled;
        if (
            !reopened
            && pollAttempt >= reopenAfterPoll
            && clickAttempts[0]?.ok === false
            && selected?.documentHasFocus === true
            && !selected?.busy
            && buttonReady
        ) {
            reopened = true;
            await clickUpload('hydration_retry');
        }
    }
    if (!selected?.ok || !selected?.selector) {
        await cleanupUploadMarkers();
        throw uploadFailure(expectedCount, 'discover_live_input', {
            fileName,
            exhausted: true,
            pollAttempts,
            pollIntervalMs,
            inputReadyTimeoutMs,
            reopened,
            baselineState,
            clickAttempts,
            inputState: selected || null,
        });
    }

    const nativeErrors = [];
    let method = '';
    try {
        // Discovery cancels Gemini's first picker click so it does not leave
        // an unmanaged native dialog open. The bridge's set-file-input action
        // now owns the chooser: it clicks this marked input and waits for
        // Page.fileChooserOpened. Keeping discovery active cancels that click
        // too, forcing every upload into the compatibility fallback.
        await page.evaluate(`(() => {
          const marker = ${JSON.stringify(marker)};
          const listener = window[marker];
          if (listener) document.removeEventListener('click', listener, true);
          delete window[marker];
          return true;
        })()`);
        // Prefer the Browser Bridge's native file-input action. It delegates to
        // Chrome's file-input machinery and remains observable by Gemini even
        // when synthetic DataTransfer change events are ignored.
        if (typeof page.setFileInput === 'function') {
            try {
                await page.setFileInput([filePath], selected.selector);
                method = 'page.setFileInput';
            } catch (error) {
                nativeErrors.push({ method: 'page.setFileInput', error: String(error?.message || error) });
            }
        }
        if (!method && typeof page.cdp === 'function') {
            try {
                if (await setFileInputViaCdp(page, [filePath], selected.selector)) {
                    method = 'DOM.setFileInputFiles';
                } else {
                    nativeErrors.push({ method: 'DOM.setFileInputFiles', error: 'marked input was not found' });
                }
            } catch (error) {
                nativeErrors.push({ method: 'DOM.setFileInputFiles', error: String(error?.message || error) });
            }
        }

        // Compatibility fallback for older Browser Bridge builds. It is used
        // only when neither trusted/native path could inject the file; a native
        // injection that Gemini later clears must fail closed instead of being
        // disguised by a second upload mechanism.
        if (!method) {
            let fallback;
            // The daemon caps each command body at 1 MiB. A detailed second
            // keyframe can exceed that after Base64 encoding even when the
            // small first frame succeeds. Transfer chunks without changing
            // the image bytes or weakening the ordered attachment checks.
            const uploadKey = '__opencliGeminiVideoUpload_' + marker;
            try {
                await page.evaluate(`globalThis[${JSON.stringify(uploadKey)}] = []; true`);
                for (let offset = 0; offset < payload.base64.length; offset += 64 * 1024) {
                    const index = offset / (64 * 1024);
                    await page.evaluate(`globalThis[${JSON.stringify(uploadKey)}][${index}] = ${JSON.stringify(payload.base64.slice(offset, offset + 64 * 1024))}; true`);
                }
                fallback = unwrap(await page.evaluate(`(() => {
                  const input = document.querySelector(${JSON.stringify(selected.selector)});
                  if (!input || input.disabled || !input.isConnected) {
                    return { ok: false, reason: 'marked Filedata input disappeared' };
                  }
                  const chunks = globalThis[${JSON.stringify(uploadKey)}];
                  if (!Array.isArray(chunks) || chunks.length !== ${Math.ceil(payload.base64.length / (64 * 1024))}) {
                    return { ok: false, reason: 'Gemini keyframe transfer is incomplete' };
                  }
                  const binary = atob(chunks.join(''));
                  if (binary.length !== ${fileBytes.length}) {
                    return { ok: false, reason: 'Gemini keyframe transfer size mismatch' };
                  }
                  const transfer = new DataTransfer();
                  const bytes = new Uint8Array(binary.length);
                  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
                  transfer.items.add(new File([bytes], ${JSON.stringify(fileName)}, { type: ${JSON.stringify(payload.mimeType)} }));
                  const descriptor = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'files');
                  if (!descriptor?.set) return { ok: false, reason: 'native files setter unavailable' };
                  descriptor.set.call(input, transfer.files);
                  input.dispatchEvent(new Event('input', { bubbles: true }));
                  input.dispatchEvent(new Event('change', { bubbles: true }));
                  return { ok: true, bytes: bytes.length, fileCount: input.files?.length || 0 };
                })()`));
            } catch (error) {
                fallback = { ok: false, reason: String(error?.message || error) };
            } finally {
                await page.evaluate(`delete globalThis[${JSON.stringify(uploadKey)}]; true`).catch(() => undefined);
            }
            if (!fallback?.ok || Number(fallback.fileCount || 0) < 1) {
                throw uploadFailure(expectedCount, 'data_transfer_fallback', {
                    fileName,
                    selector: selected.selector,
                    selectedInput: selected.selected,
                    nativeErrors,
                    fallback: fallback || null,
                });
            }
            method = 'DataTransfer';
        }

        // A visible completed attachment is the pre-submit contract for each
        // ordered keyframe. Fail early when Gemini clears an otherwise idle
        // input repeatedly without ever creating the attachment preview.
        const deadline = Date.now() + Math.max(0, attachmentTimeoutMs);
        let state = null;
        let clearedIdleSamples = 0;
        do {
            if (attachmentTimeoutMs > 0) await page.wait(1);
            try {
                state = await attachmentState();
            } catch (error) {
                throw uploadFailure(expectedCount, 'read_attachment_state', {
                    fileName,
                    method,
                    selector: selected.selector,
                    selectedInput: selected.selected,
                    nativeErrors,
                    error: String(error?.message || error),
                });
            }
            if (Number(state?.attachments || 0) >= expectedCount && !state?.busy) {
                return { method, expectedCount, state };
            }
            const inputsEmpty = Array.isArray(state?.inputFiles)
                && state.inputFiles.every(files => Array.isArray(files) && files.length === 0);
            if (Number(state?.attachments || 0) < expectedCount && !state?.busy && inputsEmpty) {
                clearedIdleSamples += 1;
            } else {
                clearedIdleSamples = 0;
            }
            if (clearedIdleSamples >= Math.max(1, clearedIdleLimit)) break;
        } while (Date.now() < deadline);
        throw uploadFailure(expectedCount, 'wait_for_attachment', {
            fileName,
            method,
            selector: selected.selector,
            selectedInput: selected.selected,
            nativeErrors,
            clearedIdleSamples,
            clearedIdleLimit: Math.max(1, clearedIdleLimit),
            state,
        });
    } finally {
        await cleanupUploadMarkers();
    }
}

export async function uploadFrames(page, filePaths, options) {
    for (let index = 0; index < filePaths.length; index += 1) {
        await uploadFrame(page, filePaths[index], index + 1, options);
    }
}

export async function submittedVideoPrompt(page) {
    const state = unwrap(await page.evaluate(`(() => {
      const composer = document.querySelector('[contenteditable="true"][role="textbox"]');
      const pathParts = location.pathname.toLowerCase().split('/').filter(Boolean);
      return {
        submitted: !!document.querySelector('user-query') || (pathParts[0] === 'app' && !!pathParts[1]),
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

const conversationUrl = (value) => /^https:\/\/gemini\.google\.com\/app\/[a-z0-9]+$/i.test(String(value || ''));
const normalizePrompt = (value) => String(value || '').replace(/\s+/g, ' ').trim();

export function createVideoSubmission(prompt) {
    const submissionNonce = randomUUID();
    return {
        submissionNonce,
        prompt: `${prompt}\n\nRequest reference: ${submissionNonce}. This is request metadata only; never render it as image text or speech.`,
    };
}

export async function recoverVideoConversation(page, checkpoint, save = () => {}, {
    listConversations = getGeminiConversationList, readTurns = getGeminiVisibleTurns,
    now = Date.now, hydrationTimeoutMs = 30000,
} = {}) {
    // A title or common prompt prefix cannot establish ownership. Only a new
    // sidebar conversation containing the exact submitted user turn qualifies.
    if (!Array.isArray(checkpoint.historyBefore) || checkpoint.historyBefore.length === 0
        || !/^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i.test(String(checkpoint.submissionNonce || ''))
        || !String(checkpoint.prompt || '').includes(`Request reference: ${checkpoint.submissionNonce}.`)) return '';
    const recoveryDeadline = Math.min(now() + 60000,
        checkpoint.deadline || Infinity, checkpoint.totalDeadline || Infinity);
    if (now() >= recoveryDeadline) return '';
    const before = new Set(checkpoint.historyBefore);
    const rows = await listConversations(page);
    const candidates = [...new Set(rows.map(row => row.Url).filter(url => conversationUrl(url) && !before.has(url)))];
    if (candidates.length > 6) {
        throw new CommandExecutionError('GEMINI_VIDEO_SUBMISSION_UNCERTAIN: too many new conversations to identify the submitted video safely');
    }
    const originalUrl = String(unwrap(await page.evaluate('window.location.href')) || '');
    const matches = [];
    let unreadable = 0;
    for (const url of candidates) {
        try {
            if (now() >= recoveryDeadline) {
                unreadable += 1;
                continue;
            }
            await page.goto(url, { waitUntil: 'load', settleMs: 2500 });
            // Navigation can finish before saved chat turns hydrate. Require
            // two stable nonempty reads, within this request's original budget.
            const hydrationDeadline = Math.min(now() + hydrationTimeoutMs, recoveryDeadline);
            let userTurns = [];
            let previous = '';
            let stable = false;
            for (let sample = 0; sample <= Math.ceil(hydrationTimeoutMs / 1000)
                && now() < hydrationDeadline; sample += 1) {
                const currentUrl = String(unwrap(await page.evaluate('window.location.href')));
                const turns = currentUrl === url ? await readTurns(page) : [];
                userTurns = turns.filter(turn => String(turn.Role).toLowerCase() === 'user' && normalizePrompt(turn.Text));
                const signature = JSON.stringify(userTurns.map(turn => normalizePrompt(turn.Text)));
                if (userTurns.length && signature === previous && now() < hydrationDeadline) {
                    stable = true;
                    break;
                }
                previous = signature;
                const remaining = hydrationDeadline - now();
                if (remaining <= 0) break;
                await page.wait(Math.min(1000, remaining) / 1000);
            }
            // Multiple user turns can contain another video's output before
            // our prompt. Without response-bound media scoping they are not
            // safe recovery candidates, even when one turn matches exactly.
            if (!stable || userTurns.length !== 1) unreadable += 1;
            else if (normalizePrompt(userTurns[0].Text) === normalizePrompt(checkpoint.prompt)) matches.push(url);
        } catch {
            unreadable += 1;
        }
    }
    checkpoint.historyRecovery = { candidates: candidates.length, matches: matches.length, unreadable };
    if (matches.length === 1 && unreadable === 0 && now() < recoveryDeadline) {
        checkpoint.url = matches[0];
        checkpoint.urlVerifiedByPrompt = true;
        checkpoint.before = [];
        save();
        await page.goto(matches[0], { waitUntil: 'load', settleMs: 2500 });
        return matches[0];
    }
    save();
    if (candidates.length && /^https:\/\/gemini\.google\.com\/(?:videos|app(?:\/[a-z0-9]+)?)\/?$/i.test(originalUrl)) {
        await page.goto(originalUrl, { waitUntil: 'load', settleMs: 2500 });
    }
    if (matches.length > 1) {
        throw new CommandExecutionError('GEMINI_VIDEO_SUBMISSION_UNCERTAIN: multiple new conversations contain the same prompt; refusing to choose another task');
    }
    return '';
}

export async function waitForVideo(page, before, timeoutSeconds, {
    stallTimeoutSeconds = 600, checkpoint = {}, save = () => {}, now = Date.now,
    recoverConversation = recoverVideoConversation,
} = {}) {
    const baseline = new Set(before);
    const deadline = Math.min(now() + timeoutSeconds * 1000, checkpoint.deadline || Infinity);
    let lastProgressAt = checkpoint.lastProgressAt || now();
    let signature = checkpoint.progressSignature || '';
    let emptyHomeSamples = 0;
    while (now() < deadline) {
        await page.wait(5);
        const state = unwrap(await page.evaluate(`(() => {
          const videos = Array.from(document.querySelectorAll('generated-video')).map((video, index) => ({
            src: video.id || video.closest('model-response')?.id || 'generated-video-' + index,
            readyState: video.querySelector('button[aria-label="Download video"]') ? 4 : 0,
          }));
          const responses = document.querySelectorAll('model-response');
          const text = (responses[responses.length - 1]?.innerText || '').slice(-1800);
          return {
            videos, text, url: window.location.href,
            emptyHome: /^\\/(?:app\\/?)?$/.test(window.location.pathname)
              && !document.querySelector('user-query')
              && !!document.querySelector('[contenteditable="true"]'),
          };
        })()`));
        const videos = Array.isArray(state?.videos) ? state.videos : [];
        const ready = videos.find(video => !baseline.has(video.src) && video.readyState >= 2);
        if (ready) {
            if (conversationUrl(state?.url)) checkpoint.url = state.url;
            checkpoint.videoId = ready.src;
            save();
            return ready;
        }
        // Spinner frames, sidebar changes and the submitted prompt are not
        // generation progress. Only the current response and video state count.
        const nextSignature = JSON.stringify({
            text: String(state?.text || '').replace(/\s+/g, ' ').trim(),
            videos: videos.filter(video => !baseline.has(video.src)),
        });
        if (nextSignature !== signature) {
            signature = nextSignature;
            lastProgressAt = now();
        }
        Object.assign(checkpoint, { lastProgressAt, progressSignature: signature });
        if (/^https:\/\/gemini\.google\.com\/app\/[a-z0-9]+$/i.test(String(state?.url || ''))) {
            checkpoint.url = state.url;
        }
        save();
        // A failed submission can briefly acquire a conversation URL, then
        // disappear server-side and return to the hydrated home composer.
        // Allow transient navigation, but do not wait 30 minutes on that page.
        emptyHomeSamples = state?.emptyHome === true ? emptyHomeSamples + 1 : 0;
        if (emptyHomeSamples >= 3) {
            throw new CommandExecutionError(
                'GEMINI_VIDEO_GENERATION_FAILED: Gemini Create Video returned to an empty home composer; the submitted conversation is no longer available'
            );
        }
        const responseText = String(state?.text || '').replace(/\s+/g, ' ').trim();
        // Match entire observed failure replies, not dialogue quoted in a
        // still-progressing response. These are terminal, not stalled jobs.
        if (isVideoQuotaMessage(responseText)) {
            throw new CommandExecutionError('GEMINI_VIDEO_QUOTA_EXHAUSTED: ' + responseText.slice(-1000));
        }
        const normalizedReply = responseText.replace(/^Gemini said\s+/i, '').replace(/’/g, "'").toLowerCase();
        const terminalFailure = [
            "i'm having a hard time fulfilling your request. can i help you with something else instead?",
            'i seem to be encountering an error. can i try something else for you?',
            'sorry, something went wrong. please try your request again.',
        ].includes(normalizedReply);
        if (terminalFailure || /could not generate|generation failed|try again|unable to create/i.test(responseText)) {
            throw new CommandExecutionError(`GEMINI_VIDEO_GENERATION_FAILED: Gemini Create Video reported a generation failure: ${String(state.text).slice(-500)}`);
        }
        // The Videos surface can stop refreshing while the accepted job has
        // already completed in its saved chat. Reopen only a prompt-verified
        // new conversation; preserve both original time budgets throughout.
        const recoveryInterval = Math.min(60000, stallTimeoutSeconds * 500);
        if (!checkpoint.url && Array.isArray(checkpoint.historyBefore) && checkpoint.historyBefore.length && checkpoint.prompt
            && now() - lastProgressAt >= recoveryInterval
            && now() - (checkpoint.historyCheckedAt || 0) >= recoveryInterval) {
            checkpoint.historyCheckedAt = now();
            save();
            if (await recoverConversation(page, checkpoint, save, { now })) {
                baseline.clear();
                continue;
            }
        }
        if (now() - lastProgressAt >= stallTimeoutSeconds * 1000) {
            throw new CommandExecutionError(
                `GEMINI_VIDEO_GENERATION_STALLED: no response progress for ${stallTimeoutSeconds}s; conversation=${checkpoint.url || 'unknown'}; response=${String(state?.text || '').slice(-300)}`
            );
        }
    }
    throw new CommandExecutionError(`GEMINI_VIDEO_GENERATION_TIMEOUT: no completed video within the original generation deadline; conversation=${checkpoint.url || 'unknown'}`);
}

export async function downloadVideo(page, outputPath, timeoutSeconds, videoId = '') {
    const transferKey = `__opencliGeminiVideo_${randomUUID().replaceAll('-', '')}`;
    const output = path.resolve(outputPath);
    const partial = `${output}.${randomUUID()}.part`;
    const deadline = Date.now() + timeoutSeconds * 1000;
    const chunkSize = 192 * 1024;
    let descriptor;
    try {
        // Fetch with the signed-in page's cookies, but transfer bytes through
        // the bridge to the requested task path. Never use the browser's
        // download manager or change a shared browser download directory.
        const started = unwrap(await page.evaluate(`(() => {
          const targetId = ${JSON.stringify(videoId)};
          const target = targetId ? Array.from(document.querySelectorAll('generated-video')).find((node, index) =>
            (node.id || node.closest('model-response')?.id || 'generated-video-' + index) === targetId) : null;
          const video = targetId ? target?.querySelector('video') : document.querySelector('generated-video video');
          const source = String(video?.currentSrc || video?.src || '');
          if (!source) return { ok: false, reason: 'generated video source URL is missing' };
          const transfer = { pending: true, controller: new AbortController() };
          globalThis[${JSON.stringify(transferKey)}] = transfer;
          const timer = setTimeout(() => transfer.controller.abort(), ${Math.max(1, Math.floor(timeoutSeconds * 1000))});
          // Return immediately; a bridge evaluate call has a shorter timeout
          // than a media transfer. Poll small metadata responses instead.
          (async () => {
            try {
              const response = await fetch(source, { credentials: 'include', signal: transfer.controller.signal });
              if (!response.ok) throw new Error('video fetch returned HTTP ' + response.status);
              const blob = await response.blob();
              if (blob.size < 1024) throw new Error('video fetch returned an empty blob');
              Object.assign(transfer, { ok: true, size: blob.size, blob });
            } catch (error) {
              Object.assign(transfer, { ok: false, reason: String(error?.message || error) });
            } finally {
              transfer.pending = false;
              clearTimeout(timer);
            }
          })();
          return { ok: true };
        })()`));
        if (!started?.ok) {
            throw new CommandExecutionError(`Gemini in-page download failed: ${started?.reason || 'could not start video transfer'}`);
        }
        let metadata;
        while (Date.now() < deadline) {
            metadata = unwrap(await page.evaluate(`(() => {
              const transfer = globalThis[${JSON.stringify(transferKey)}];
              if (!transfer) return { ok: false, reason: 'video transfer was lost after navigation' };
              return { pending: transfer.pending, ok: transfer.ok, size: transfer.size, reason: transfer.reason };
            })()`));
            if (!metadata?.pending) break;
            await page.wait(0.5);
        }
        if (metadata?.pending) throw new CommandExecutionError('Gemini video download exceeded its remaining deadline');
        if (!metadata?.ok || !Number.isSafeInteger(metadata.size) || metadata.size < 1024) {
            throw new CommandExecutionError(`Gemini in-page download failed: ${metadata?.reason || 'invalid video size'}`);
        }
        fs.mkdirSync(path.dirname(output), { recursive: true });
        descriptor = fs.openSync(partial, 'wx');
        for (let offset = 0; offset < metadata.size; offset += chunkSize) {
            if (Date.now() >= deadline) {
                throw new CommandExecutionError('Gemini video download exceeded its remaining deadline');
            }
            const length = Math.min(chunkSize, metadata.size - offset);
            const encoded = unwrap(await page.evaluate(`(async () => {
              const blob = globalThis[${JSON.stringify(transferKey)}]?.blob;
              if (!blob) throw new Error('Gemini video transfer was lost after navigation');
              const bytes = new Uint8Array(await blob.slice(${offset}, ${offset + length}).arrayBuffer());
              let binary = '';
              for (let start = 0; start < bytes.length; start += 8192) {
                binary += String.fromCharCode(...bytes.subarray(start, start + 8192));
              }
              return btoa(binary);
            })()`));
            const bytes = Buffer.from(typeof encoded === 'string' ? encoded : '', 'base64');
            if (bytes.length !== length) {
                throw new CommandExecutionError('Gemini video download returned an incomplete chunk');
            }
            if (offset === 0 && bytes.toString('ascii', 4, 8) !== 'ftyp') {
                throw new CommandExecutionError('Gemini video download did not return an MP4 file');
            }
            let written = 0;
            while (written < bytes.length) {
                written += fs.writeSync(descriptor, bytes, written, bytes.length - written);
            }
        }
        if (Date.now() >= deadline || fs.fstatSync(descriptor).size !== metadata.size) {
            throw new CommandExecutionError('Gemini video download did not finish within its remaining budget');
        }
        fs.fsyncSync(descriptor);
        fs.closeSync(descriptor);
        descriptor = undefined;
        fs.renameSync(partial, output);
        return { downloaded: true, filename: output, size: metadata.size };
    } finally {
        if (descriptor !== undefined) fs.closeSync(descriptor);
        fs.rmSync(partial, { force: true });
        await page.evaluate(`(() => {
          globalThis[${JSON.stringify(transferKey)}]?.controller.abort();
          delete globalThis[${JSON.stringify(transferKey)}];
        })()`).catch(() => {});
    }
}

export const videoCommand = cli({
    site: 'gemini',
    name: 'video',
    access: 'write',
    description: 'Create a Gemini Web video from one reference frame or ordered first/last frames and save it locally',
    domain: GEMINI_DOMAIN,
    strategy: Strategy.COOKIE,
    browser: true,
    siteSession: 'persistent',
    navigateBefore: false,
    defaultFormat: 'plain',
    args: [
        { name: 'prompt', positional: true, help: 'Create Video animation prompt' },
        { name: 'first', help: 'Local reference or first-frame image' },
        { name: 'last', help: 'Optional local last-frame image; omit for a single reference frame' },
        { name: 'resume', help: 'Existing gemini.google.com/app conversation URL to finish or download' },
        { name: 'aspect', default: '16:9', choices: ['16:9', '9:16'], help: 'Final aspect ratio' },
        { name: 'output', required: true, help: 'Local MP4 output path' },
        { name: 'timeout', type: 'int', default: 1800, help: 'Total generation and download timeout in seconds' },
        { name: 'stall-timeout', type: 'int', default: 600, help: 'Maximum seconds without response progress' },
        { name: 'state-file', help: 'Owned local generation checkpoint' },
        { name: 'request-id', help: 'Input fingerprint for the owned checkpoint' },
    ],
    columns: ['status', 'file', 'aspect', 'link'],
    func: async (page, kwargs) => {
        const prompt = String(kwargs.prompt || '').trim();
        const resume = String(kwargs.resume || '').trim();
        if (resume && !/^https:\/\/gemini\.google\.com\/app\/[a-z0-9]+$/i.test(resume)) {
            throw new ArgumentError('--resume must be a Gemini conversation URL');
        }
        if (!resume && !prompt) throw new ArgumentError('video prompt is required unless --resume is used');
        const first = resume ? '' : requireFile(kwargs.first, '--first');
        const last = resume || !String(kwargs.last || '').trim() ? '' : requireFile(kwargs.last, '--last');
        const output = resolveOutput(kwargs.output);
        const aspect = String(kwargs.aspect || '16:9');
        if (!['16:9', '9:16'].includes(aspect)) throw new ArgumentError('--aspect must be 16:9 or 9:16');
        const timeout = Number(kwargs.timeout || 1800);
        if (!Number.isInteger(timeout) || timeout < 60) throw new ArgumentError('--timeout must be at least 60 seconds');
        const stallTimeoutSeconds = Number(kwargs['stall-timeout'] || 600);
        if (!Number.isInteger(stallTimeoutSeconds) || stallTimeoutSeconds < 60) {
            throw new ArgumentError('--stall-timeout must be at least 60 seconds');
        }
        const stateFile = kwargs['state-file'] ? path.resolve(String(kwargs['state-file'])) : '';
        const requestId = String(kwargs['request-id'] || '');
        if (stateFile && !/^[a-f0-9]{64}$/.test(requestId)) throw new ArgumentError('--state-file requires a SHA-256 --request-id');
        const checkStatePaths = () => {
            for (const name of [stateFile, stateFile + '.tmp']) {
                if (stateFile && fs.existsSync(name) && fs.lstatSync(name).isSymbolicLink()) {
                    throw new ArgumentError('Generation checkpoint cannot be a symlink');
                }
            }
        };
        checkStatePaths();
        const previous = stateFile && fs.existsSync(stateFile) ? JSON.parse(fs.readFileSync(stateFile, 'utf8')) : {};
        if (resume && stateFile && (previous.request_id !== requestId || previous.url !== resume)) {
            throw new ArgumentError('Resume conversation does not match its owned generation checkpoint');
        }
        const checkpoint = resume ? previous : { request_id: requestId, status: 'new' };
        const save = () => {
            if (!stateFile) return;
            checkStatePaths();
            fs.mkdirSync(path.dirname(stateFile), { recursive: true });
            fs.writeFileSync(stateFile + '.tmp', JSON.stringify(checkpoint, null, 2));
            fs.renameSync(stateFile + '.tmp', stateFile);
        };
        try {
            // Retain a download allowance inside the original total budget,
            // including navigation/upload time. Recovery never starts a new
            // full generation or download window for an accepted request.
            checkpoint.totalDeadline ||= checkpoint.deadline || Date.now() + timeout * 1000;
            checkpoint.deadline ||= checkpoint.totalDeadline - Math.min(180, timeout / 2) * 1000;
            if (Date.now() >= checkpoint.totalDeadline
                || (checkpoint.status !== 'downloading' && Date.now() >= checkpoint.deadline)) {
                throw new CommandExecutionError('GEMINI_VIDEO_GENERATION_TIMEOUT: original generation deadline has elapsed');
            }
            await openVideoPage(page, resume || GEMINI_VIDEOS_URL);
            if (!resume) {
                await clickLabel(page, ['create with omni']);
                await page.wait(1.5);
                await waitForVideoComposer(page, {
                    timeoutMs: Math.min(90000, Math.max(0, checkpoint.deadline - Date.now())),
                });
                await selectAspectRatio(page, aspect);
                checkpoint.before = await visibleVideoUrls(page);
                Object.assign(checkpoint, createVideoSubmission(prompt));
                // History is best-effort, but an unavailable snapshot never
                // authorizes choosing an arbitrary old or concurrent chat.
                checkpoint.historyBefore = await getGeminiConversationList(page)
                    .then(rows => rows.map(row => row.Url).filter(conversationUrl))
                    .then(urls => urls.length ? urls : null).catch(() => null);
                await uploadFrames(page, last ? [first, last] : [first]);
                // Persist before the send: a killed command must not resubmit
                // blindly when the provider may already have accepted it.
                checkpoint.status = 'submitting';
                save();
                await submitVideoPrompt(page, checkpoint.prompt);
                checkpoint.status = 'submitted';
                const url = String(unwrap(await page.evaluate('window.location.href')) || '');
                if (/^https:\/\/gemini\.google\.com\/app\/[a-z0-9]+$/i.test(url)) checkpoint.url = url;
                save();
            }
            if (checkpoint.status !== 'downloading') {
                await waitForVideo(page, checkpoint.before || [], timeout, { stallTimeoutSeconds, checkpoint, save });
            }
            checkpoint.status = 'downloading';
            save();
            const downloadSeconds = Math.min(180, Math.floor((checkpoint.totalDeadline - Date.now()) / 1000));
            if (downloadSeconds <= 0) {
                throw new CommandExecutionError('GEMINI_VIDEO_GENERATION_TIMEOUT: original download deadline has elapsed');
            }
            await downloadVideo(page, output, downloadSeconds, checkpoint.videoId || '');
            const link = String(unwrap(await page.evaluate('window.location.href')) || GEMINI_VIDEOS_URL);
            Object.assign(checkpoint, { status: 'saved', url: link });
            save();
            return [{ status: 'saved', file: output, aspect, link }];
        } catch (error) {
            const message = String(error?.message || error);
            for (const [code, status] of [
                ['GEMINI_VIDEO_GENERATION_STALLED', 'stalled'],
                ['GEMINI_VIDEO_GENERATION_TIMEOUT', 'timeout'],
                ['GEMINI_VIDEO_GENERATION_FAILED', 'failed'],
                ['GEMINI_VIDEO_QUOTA_EXHAUSTED', 'quota_exhausted'],
                ['GEMINI_VIDEO_SUBMISSION_UNCERTAIN', 'uncertain'],
            ]) {
                if (message.includes(code)) Object.assign(checkpoint, { status, error: message });
            }
            save();
            throw error;
        }
    },
});

import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { cli, Strategy } from '@jackwener/opencli/registry';
import { ArgumentError, CommandExecutionError, EmptyResultError } from '@jackwener/opencli/errors';
import { sendGeminiMessage } from './utils.js';

const GEMINI_DOMAIN = 'gemini.google.com';
const GEMINI_VIDEOS_URL = 'https://gemini.google.com/videos';
const VIDEO_UPLOAD_CAPABILITY_CODE = 'OPENCLI_CAPABILITY_UNAVAILABLE:GEMINI_VIDEO_LOCAL_FILE_UPLOAD';
const VIDEO_INPUT_HYDRATION_STUCK_CODE = 'OPENCLI_CAPABILITY_DEGRADED:GEMINI_VIDEO_UPLOAD_INPUT_HYDRATION_STUCK';

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
        && /-32000|not allowed/i.test(String(error?.error || ''))
    );
    const directCdpDenied = nativeErrors.some(error =>
        error?.method === 'DOM.setFileInputFiles'
        && /CDP method not permitted[\s\S]*Runtime\.evaluate/i.test(String(error?.error || ''))
    );
    const inputFiles = diagnostic?.state?.inputFiles;
    const syntheticCleared = diagnostic?.method === 'DataTransfer'
        && Number(diagnostic?.state?.attachments || 0) === 0
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
    const payload = {
        base64: fs.readFileSync(filePath).toString('base64'),
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
          const input = freshCandidates.at(-1);
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
            'uploader-file-preview [role="progressbar"], gem-media-attachment [role="progressbar"], mat-progress-spinner'
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
            inputOrigin: 'new_after_click',
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
            try {
                fallback = unwrap(await page.evaluate(`(() => {
                  const input = document.querySelector(${JSON.stringify(selected.selector)});
                  if (!input || input.disabled || !input.isConnected) {
                    return { ok: false, reason: 'marked Filedata input disappeared' };
                  }
                  const transfer = new DataTransfer();
                  const item = ${JSON.stringify(payload)};
                  const binary = atob(item.base64);
                  const bytes = new Uint8Array(binary.length);
                  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
                  transfer.items.add(new File([bytes], item.fileName, { type: item.mimeType }));
                  const descriptor = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'files');
                  if (!descriptor?.set) return { ok: false, reason: 'native files setter unavailable' };
                  descriptor.set.call(input, transfer.files);
                  input.dispatchEvent(new Event('input', { bubbles: true }));
                  input.dispatchEvent(new Event('change', { bubbles: true }));
                  return { ok: true, bytes: bytes.length, fileCount: input.files?.length || 0 };
                })()`));
            } catch (error) {
                fallback = { ok: false, reason: String(error?.message || error) };
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

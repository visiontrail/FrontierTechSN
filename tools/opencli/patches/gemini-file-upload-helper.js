
// Project patch: Gemini Web's XAP picker rejects CDP setFileInput on some
// Chrome versions. Fall back to a browser-native File/DataTransfer change
// event, matching the compatibility path used by OpenCLI's Claude adapter.
export async function attachGeminiFile(page, filePath, allowEmptyPageReload = true) {
    const fs = await import('node:fs');
    const path = await import('node:path');
    const absPath = path.default.resolve(filePath);
    if (!fs.default.existsSync(absPath)) {
        throw new CommandExecutionError('Gemini attachment was not found: ' + absPath);
    }
    const stats = fs.default.statSync(absPath);
    if (!stats.isFile() || stats.size < 1) {
        throw new CommandExecutionError('Gemini attachment is not a non-empty file: ' + absPath);
    }
    if (stats.size > 8 * 1024 * 1024) {
        throw new CommandExecutionError('Gemini attachment exceeds the 8 MB browser-bridge limit');
    }

    await ensureGeminiPage(page);
    // The ask adapter has already opened the intended conversation and selected
    // its model. Navigating again discards that hydrated composer and exposes a
    // zero-state upload button whose menu entries are still loading.
    await page.wait(2);
    // A foreground site session can still lose focus while Chrome restores the
    // prior user tab. Gemini ignores the native upload-menu click in that
    // state, so explicitly activate the target before requesting user gesture.
    if (typeof page.cdp === 'function') {
        await page.cdp('Page.bringToFront', {}).catch(() => undefined);
        await page.wait(0.25);
    }
    try {
        await page.click('button[aria-label="Upload & tools"]');
    } catch (error) {
        throw new CommandExecutionError(
            'Could not open Gemini upload menu with a native click: ' + String(error?.message || error)
        );
    }

    const fileInputSelectors = [
        'input[name="Filedata"]',
        'images-files-uploader input[type="file"]',
        'uploader > input[type="file"]',
        'input[type="file"]',
    ];
    let fileInputSelector = '';
    for (let attempt = 0; attempt < 60; attempt += 1) {
        await page.wait(attempt === 0 ? 1 : 0.5);
        const picker = await page.evaluate(`(() => {
            const selectors = ${JSON.stringify(fileInputSelectors)};
            const inputSelector = selectors.find((selector) => document.querySelector(selector)) || '';
            const legacyButton = '[data-test-id="local-images-files-uploader-button"]';
            const currentButton = 'images-files-uploader button[aria-label^="Upload files"]';
            const buttonSelector = document.querySelector(legacyButton)
                ? legacyButton
                : document.querySelector(currentButton)
                    ? currentButton
                    : '';
            return {
                inputSelector,
                buttonSelector,
                expanded: document.querySelector('button[aria-label="Upload & tools"]')?.getAttribute('aria-expanded') === 'true',
            };
        })()`);
        if (picker?.inputSelector) {
            fileInputSelector = picker.inputSelector;
            break;
        }
        if (picker?.buttonSelector) {
            await page.click(picker.buttonSelector);
            await page.wait(0.5);
            const discovered = await page.evaluate(`(() => {
                const selectors = ${JSON.stringify(fileInputSelectors)};
                return selectors.find((selector) => document.querySelector(selector)) || '';
            })()`);
            if (discovered) {
                fileInputSelector = discovered;
                break;
            }
        } else if ([8, 32].includes(attempt)) {
            // A bridge can acknowledge trusted pointer events while an Ego
            // task-space tab remains out of focus. Opening this DOM menu does
            // not require a native file chooser. Recreate either a collapsed
            // menu or an expanded but empty loading shell after native
            // attempts failed to expose its controls.
            const closed = await page.evaluate(`(() => {
                const button = document.querySelector('button[aria-label="Upload & tools"]');
                if (!button || button.disabled || button.getAttribute('aria-expanded') !== 'true') return false;
                button.click();
                return true;
            })()`);
            if (closed) await page.wait(0.5);
            await page.evaluate(`(() => {
                const button = document.querySelector('button[aria-label="Upload & tools"]');
                if (!button || button.disabled || button.getAttribute('aria-expanded') === 'true') return false;
                button.click();
                return true;
            })()`);
        } else if ([4, 8, 16, 32, 48].includes(attempt)) {
            // Re-open an overlay that was created before its async menu entries
            // hydrated. Two trusted clicks close then reopen it.
            if (picker?.expanded) await page.click('button[aria-label="Upload & tools"]');
            await page.wait(0.5);
            if (typeof page.cdp === 'function') {
                await page.cdp('Page.bringToFront', {}).catch(() => undefined);
            }
            // Some bridge versions acknowledge selector clicks without opening
            // this menu. Retry with a fresh visible target and a trusted click.
            const uploadTarget = await page.evaluate(`(() => {
                const button = document.querySelector('button[aria-label="Upload & tools"]');
                if (!button || button.disabled) return null;
                const rect = button.getBoundingClientRect();
                return rect.width > 0 && rect.height > 0
                    ? { x: rect.x + rect.width / 2, y: rect.y + rect.height / 2 }
                    : null;
            })()`);
            if (uploadTarget && typeof page.nativeClick === 'function') {
                await page.nativeClick(uploadTarget.x, uploadTarget.y);
            } else {
                await page.click('button[aria-label="Upload & tools"]');
            }
        }
    }
    if (!fileInputSelector) {
        const recovery = await page.evaluate(`(() => {
            const composer = document.querySelector('[contenteditable="true"][aria-label*="Gemini"]');
            const model = document.querySelector('button[aria-label*="mode picker"]');
            return {
                emptyComposer: !!composer && !String(composer.innerText || composer.textContent || '').trim(),
                attachmentCount: document.querySelectorAll('input-container button[aria-label="close attachment"]').length,
                generating: Array.from(document.querySelectorAll('button')).some(button => /stop response|stop generating|停止回答/i.test(button.getAttribute('aria-label') || '')),
                modelLabel: model?.getAttribute('aria-label') || '',
                url: location.href,
            };
        })()`);
        if (allowEmptyPageReload && recovery?.emptyComposer && recovery.attachmentCount === 0
            && !recovery.generating && recovery.modelLabel && /^https:\/\/gemini\.google\.com\/app(?:[/?#]|$)/.test(recovery.url)
            && typeof page.goto === 'function') {
            console.error('[gemini/upload] Reloading an empty page once after its upload menu failed to hydrate');
            await page.goto(recovery.url, { settleMs: 2000 });
            let restoredModel = '';
            for (let attempt = 0; attempt < 20; attempt++) {
                await page.wait(0.5);
                restoredModel = await page.evaluate(`document.querySelector('button[aria-label*="mode picker"]')?.getAttribute('aria-label') || ''`);
                if (restoredModel) break;
            }
            if (restoredModel !== recovery.modelLabel) {
                throw new CommandExecutionError('Gemini model changed or was unavailable after empty-page recovery; prompt was not submitted');
            }
            return attachGeminiFile(page, filePath, false);
        }
        const diagnostic = await page.evaluate(`(() => ({
            url: location.href,
            uploadButtons: Array.from(document.querySelectorAll('button'))
                .map((item) => item.getAttribute('aria-label') || item.textContent || '')
                .filter((label) => /upload|上传/i.test(label))
                .slice(0, 12),
            fileInputs: Array.from(document.querySelectorAll('input[type="file"]'))
                .map((input) => ({ name: input.name, accept: input.accept, className: input.className })),
            expanded: document.querySelector('button[aria-label="Upload & tools"]')?.getAttribute('aria-expanded'),
            overlayText: (document.querySelector('.cdk-overlay-container')?.textContent || '').trim().slice(0, 500),
        }))()`);
        throw new CommandExecutionError(
            'Gemini local file picker did not open: ' + JSON.stringify(diagnostic)
        );
    }

    let uploaded = false;
    if (page.setFileInput) {
        try {
            await page.setFileInput([absPath], fileInputSelector);
            uploaded = await page.evaluate(`(() => {
                const input = document.querySelector(${JSON.stringify(fileInputSelector)});
                // Some bridges acknowledge setFileInput without populating the
                // picker. Verify its actual FileList before trusting the ack.
                if (!input) return !!document.querySelector('input-container button[aria-label="close attachment"]');
                const file = input.files?.[0];
                if (!file || file.name !== ${JSON.stringify(path.default.basename(absPath))} || file.size !== ${stats.size}) return false;
                input.dispatchEvent(new Event('change', { bubbles: true }));
                return true;
            })()`) === true;
            if (!uploaded) console.error('[gemini/upload] Native upload acknowledgement had no matching file; transferring the attachment');
        } catch (error) {
            const message = String(error?.message || error);
            if (!/Not allowed|Unknown action|not supported|fileChooserOpened|file chooser/i.test(message)) {
                throw error;
            }
        }
    }

    if (!uploaded) {
        const base64 = fs.default.readFileSync(absPath).toString('base64');
        const fileName = path.default.basename(absPath);
        const mimeType = fileName.toLowerCase().endsWith('.png') ? 'image/png' : 'image/jpeg';
        // OpenCLI's daemon caps a command body at 1 MiB. Base64 expands an
        // 800 KiB contact sheet beyond that limit and the daemon closes the
        // connection before dispatch. Keep every transfer command small.
        const uploadKey = '__opencliGeminiUpload_' + Date.now() + '_' + Math.random().toString(36).slice(2);
        let fallback;
        try {
            await page.evaluate(`globalThis[${JSON.stringify(uploadKey)}] = []; true`);
            for (let offset = 0; offset < base64.length; offset += 64 * 1024) {
                await page.evaluate(`globalThis[${JSON.stringify(uploadKey)}].push(${JSON.stringify(base64.slice(offset, offset + 64 * 1024))}); true`);
            }
            fallback = await page.evaluate(`(() => {
            const input = document.querySelector(${JSON.stringify(fileInputSelector)});
            if (!input) return { ok: false, reason: 'Gemini file input disappeared' };
            const chunks = globalThis[${JSON.stringify(uploadKey)}];
            if (!Array.isArray(chunks) || !chunks.length) return { ok: false, reason: 'Gemini upload transfer is incomplete' };
            const binary = atob(chunks.join(''));
            if (binary.length !== ${stats.size}) return { ok: false, reason: 'Gemini upload transfer size mismatch' };
            const bytes = new Uint8Array(binary.length);
            for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
            const file = new File([bytes], ${JSON.stringify(fileName)}, { type: ${JSON.stringify(mimeType)} });
            const transfer = new DataTransfer();
            transfer.items.add(file);
            input.files = transfer.files;
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.dispatchEvent(new Event('change', { bubbles: true }));
            return { ok: true };
        })()`);
        } finally {
            await page.evaluate(`delete globalThis[${JSON.stringify(uploadKey)}]; true`).catch(() => undefined);
        }
        if (!fallback?.ok) {
            throw new CommandExecutionError(fallback?.reason || 'Gemini DataTransfer upload failed');
        }
    }

    const fileName = path.default.basename(absPath).toLowerCase();
    let lastState;
    for (let attempt = 0; attempt < 60; attempt += 1) {
        await page.wait(1);
        const state = await page.evaluate(`(() => {
            const name = ${JSON.stringify(fileName)};
            const text = (document.querySelector('input-container')?.innerText || '').toLowerCase();
            const candidates = Array.from(document.querySelector('input-container')?.querySelectorAll(
                '[data-test-id*="attachment"], [data-test-id*="file"], [class*="attachment"], [class*="file-chip"], [class*="upload-preview"], img'
            ) || []);
            const named = candidates.some((node) =>
                String(node.getAttribute('aria-label') || node.getAttribute('alt') || node.textContent || '')
                    .toLowerCase().includes(name)
            );
            const preview = candidates.some((node) => {
                const value = String(node.getAttribute('src') || '');
                return value.startsWith('blob:') || value.startsWith('data:image/');
            });
            // The sidebar retains a hidden "Loading Gems and Recent conversations"
            // spinner after startup. Only a visible composer spinner represents
            // an attachment that is still uploading.
            const busy = Array.from(document.querySelectorAll(
                'input-container [role="progressbar"], input-container mat-progress-spinner'
            )).some((node) => {
                const rect = node.getBoundingClientRect();
                return rect.width > 0 && rect.height > 0
                    && getComputedStyle(node).visibility !== 'hidden';
            });
            const send = document.querySelector('input-container button[aria-label="Send message"], input-container button[aria-label="发送消息"]');
            const sendReady = !!send && !send.disabled && send.getAttribute('aria-disabled') !== 'true';
            return { ready: (text.includes(name) || named || preview) && !busy && sendReady, named, preview, busy, sendReady };
        })()`);
        lastState = state;
        if (state?.ready) return true;
    }
    throw new CommandExecutionError('Gemini attachment did not become ready; the review prompt was not submitted: ' + JSON.stringify(lastState));
}

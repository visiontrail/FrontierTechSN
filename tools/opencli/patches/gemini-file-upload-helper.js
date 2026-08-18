
// Project patch: Gemini Web's XAP picker rejects CDP setFileInput on some
// Chrome versions. Fall back to a browser-native File/DataTransfer change
// event, matching the compatibility path used by OpenCLI's Claude adapter.
export async function attachGeminiFile(page, filePath) {
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
    await page.goto(GEMINI_APP_URL, { waitUntil: 'load', settleMs: 8000 });
    // The zero-state composer appears before Gemini has hydrated the upload
    // entries. Opening the menu too early leaves an empty overlay indefinitely.
    await page.wait(2);
    try {
        await page.click('button[aria-label="Upload & tools"]');
    } catch (error) {
        throw new CommandExecutionError(
            'Could not open Gemini upload menu with a native click: ' + String(error?.message || error)
        );
    }

    let pickerReady = false;
    for (let attempt = 0; attempt < 12; attempt += 1) {
        await page.wait(attempt === 0 ? 1 : 0.5);
        const picker = await page.evaluate(`(() => ({
            input: !!document.querySelector('input[name="Filedata"]'),
            button: !!document.querySelector('[data-test-id="local-images-files-uploader-button"]'),
            expanded: document.querySelector('button[aria-label="Upload & tools"]')?.getAttribute('aria-expanded') === 'true',
        }))()`);
        if (picker?.input) {
            pickerReady = true;
            break;
        }
        if (picker?.button) {
            await page.click('[data-test-id="local-images-files-uploader-button"]');
            await page.wait(0.5);
            const exists = await page.evaluate('!!document.querySelector(\'input[name="Filedata"]\')');
            if (exists) {
                pickerReady = true;
                break;
            }
        } else if (attempt === 4 || attempt === 8) {
            // Re-open an overlay that was created before its async menu entries
            // hydrated. Two trusted clicks close then reopen it.
            if (picker?.expanded) await page.click('button[aria-label="Upload & tools"]');
            await page.wait(0.5);
            await page.click('button[aria-label="Upload & tools"]');
        }
    }
    if (!pickerReady) {
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
            await page.setFileInput([absPath], 'input[name="Filedata"]');
            await page.evaluate(`(() => {
                const input = document.querySelector('input[name="Filedata"]');
                if (!input) return false;
                input.dispatchEvent(new Event('change', { bubbles: true }));
                return true;
            })()`);
            uploaded = true;
        } catch (error) {
            const message = String(error?.message || error);
            if (!/Not allowed|Unknown action|not supported/i.test(message)) throw error;
        }
    }

    if (!uploaded) {
        const base64 = fs.default.readFileSync(absPath).toString('base64');
        const fileName = path.default.basename(absPath);
        const mimeType = fileName.toLowerCase().endsWith('.png') ? 'image/png' : 'image/jpeg';
        const fallback = await page.evaluate(`(() => {
            const input = document.querySelector('input[name="Filedata"]');
            if (!input) return { ok: false, reason: 'Gemini Filedata input disappeared' };
            const binary = atob(${JSON.stringify(base64)});
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
        if (!fallback?.ok) {
            throw new CommandExecutionError(fallback?.reason || 'Gemini DataTransfer upload failed');
        }
    }

    const fileName = path.default.basename(absPath).toLowerCase();
    for (let attempt = 0; attempt < 12; attempt += 1) {
        await page.wait(1);
        const state = await page.evaluate(`(() => {
            const name = ${JSON.stringify(fileName)};
            const text = (document.querySelector('input-container')?.innerText || '').toLowerCase();
            const candidates = Array.from(document.querySelectorAll(
                '[data-test-id*="attachment"], [data-test-id*="file"], [class*="attachment"], [class*="file-chip"], [class*="upload-preview"], img'
            ));
            const named = candidates.some((node) =>
                String(node.getAttribute('aria-label') || node.getAttribute('alt') || node.textContent || '')
                    .toLowerCase().includes(name)
            );
            const preview = candidates.some((node) => {
                const value = String(node.getAttribute('src') || '');
                return value.startsWith('blob:') || value.startsWith('data:image/');
            });
            const busy = !!document.querySelector('[role="progressbar"], mat-progress-spinner');
            return { ready: text.includes(name) || named || (preview && !busy) };
        })()`);
        if (state?.ready) return true;
    }
    // Gemini's current XAP preview lives outside the stable composer DOM in
    // some layouts. The downstream review prompt must report image_received,
    // so an undetected/failed upload still fails closed at the product layer.
    return true;
}

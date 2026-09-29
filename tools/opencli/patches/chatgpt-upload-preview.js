async function waitForChatGPTUploadPreview(page, fileNames) {
    // Confirm the requested loaded image in this composer before sending text.
    for (let attempt = 0; attempt < 20; attempt++) {
        await page.sleep(1);
        const ready = unwrapEvaluateResult(await page.evaluate(`(async () => {
            ${buildComposerLocatorScript()}
            const names = ${JSON.stringify(fileNames)};
            const composer = findComposer();
            const scope = composer?.closest('form') || composer?.parentElement;
            if (!scope || !names.length) return false;
            const send = scope.querySelector('button[data-testid="send-button"], button[type="submit"][aria-label="Send"], button[aria-label="Send prompt"], button[aria-label="Send message"]');
            if (!send || !isVisible(send) || send.disabled || send.getAttribute('aria-disabled') === 'true') return false;
            const matchesName = (label, name) => {
                if (label === name) return true;
                // ChatGPT can de-duplicate uploaded names, e.g. sheet(1).jpg.
                const dot = name.lastIndexOf('.');
                if (dot < 1 || !label.endsWith(name.slice(dot))) return false;
                const stem = name.slice(0, dot);
                if (!label.startsWith(stem)) return false;
                return /^ ?\\([1-9]\\d*\\)$/.test(label.slice(stem.length, -name.slice(dot).length));
            };
            const loadedImage = tile => Array.from(tile.querySelectorAll('img')).some(image =>
                isVisible(image) && image.complete && image.naturalWidth > 0 && image.naturalHeight > 0
            );
            const attachments = scope.querySelector('[data-composer-attachments]') || scope;
            const tiles = Array.from(attachments.querySelectorAll('button[aria-label], [role="button"][aria-label]')).filter(isVisible);
            let matchedNames = 0;
            for (const name of names) {
                const index = tiles.findIndex(tile => matchesName(tile.getAttribute('aria-label') || '', name));
                if (index < 0 || !loadedImage(tiles[index])) return false;
                tiles.splice(index, 1);
                matchedNames++;
            }
            return matchedNames === names.length;
        })()`));
        if (ready === true) return true;
    }
    return false;
}

// Project patch: inspect visible provider UI, never quoted answer/prompt text.
export function chatgptRateLimitGuardScript() {
    return `
        const limitSurfaces = document.querySelectorAll('dialog, [role="dialog"], [role="alertdialog"], [role="alert"], h1, h2, h3');
        for (const surface of limitSurfaces) {
            if (!(surface instanceof HTMLElement)) continue;
            if (surface.closest('.markdown, [data-message-author-role], #prompt-textarea')) continue;
            const style = window.getComputedStyle(surface);
            const rect = surface.getBoundingClientRect();
            if (style.display === 'none' || style.visibility === 'hidden' || !rect.width || !rect.height) continue;
            const message = (surface.textContent || '').trim();
            if (/too many requests|temporarily limited access|请求过于频繁|请求过快|请求次数过多|暂时限制.*访问/i.test(message)) {
                throw new Error('CHATGPT_RATE_LIMITED: conversation access is temporarily limited; stop browser requests and cool down. Target: ' + window.location.href);
            }
        }
    `;
}

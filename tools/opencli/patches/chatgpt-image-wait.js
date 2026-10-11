export async function waitForChatGPTImages(page, beforeUrls, timeoutSeconds, convUrl, options = {}) {
    // Poll count is not elapsed time: every bridge read can take seconds.
    // Keep the response budget bounded independently of setup and export.
    const startedAt = Date.now();
    const deadline = startedAt + timeoutSeconds * 1000;
    const beforeSet = new Set(beforeUrls);
    let conversationUrl = convUrl;
    let localConversationUrl = options.localConversationUrl || '';
    let lastUrls = [];
    let stableCount = 0;
    let stillRendering = false;
    let failureText = '';
    let failureCount = 0;
    let nextDiagnosticAt = Date.now() + 60000;
    const isLocal = value => /^https:\/\/chatgpt\.com\/c\/local-chatgpt(?::|%3a)[0-9a-f-]{36}(?:[?#]|$)/i.test(value || '');
    const isPending = value => /^https:\/\/chatgpt\.com\/(?:new\/?)?(?:[?#]|$)/i.test(value || '');
    const report = stage => options.onProgress?.(stage, conversationUrl);

    while (Date.now() < deadline) {
        await page.sleep(Math.min(3, (deadline - Date.now()) / 1000));
        if (Date.now() >= deadline) break;

        const currentUrl = unwrapEvaluateResult(await page.evaluate('window.location.href'));
        if (Date.now() >= deadline) break;
        // A bridge read can land between route transitions. Do not navigate or
        // accept image assets until the next read identifies the current page.
        if (!currentUrl) continue;
        const sameConversation = isSameChatGPTConversation(currentUrl, conversationUrl)
            || (localConversationUrl && isSameChatGPTConversation(currentUrl, localConversationUrl));
        if (!sameConversation) {
            // A submitted local route can become its saved route naturally.
            // Require the exact submitted prompt before accepting that change.
            // Never reload the live turn, or navigate into an unrelated chat.
            let identity = null;
            if ((isLocal(conversationUrl) || isPending(conversationUrl)) && options.prompt) {
                identity = await resolveChatGPTConversationIdentity(page, options.prompt);
                if (Date.now() >= deadline) break;
                const messages = await getVisibleMessages(page, { textOnly: true });
                if (Date.now() >= deadline) break;
                const latestUser = [...messages].reverse().find(message => message.Role === 'User');
                // The saved route can render before its conversation messages.
                // An empty read is not evidence of a different request.
                if (!latestUser) continue;
                if (!chatGPTPromptMatches(latestUser.Text, options.prompt)) identity = null;
                else if (identity?.localConversationUrl && !(isPending(conversationUrl)
                    && isSameChatGPTConversation(currentUrl, identity.localConversationUrl))) identity = null;
                else if (!identity && isPending(conversationUrl) && isLocal(currentUrl)) {
                    identity = { conversationUrl: currentUrl };
                }
            }
            if (!identity) {
                throw new CommandExecutionError(
                    `ChatGPT image left its submitted conversation (${conversationUrl}); current URL is ${currentUrl}`,
                    'Inspect the existing conversation; the image prompt was not replayed.',
                );
            }
            localConversationUrl = identity.localConversationUrl
                || (isLocal(conversationUrl) ? conversationUrl : '');
            conversationUrl = identity.conversationUrl;
            report('waiting');
        }

        stillRendering = await isGenerating(page);
        if (Date.now() >= deadline) break;
        if (stillRendering) {
            // A stalled provider can keep its Stop button forever. Record the
            // state already read above, without extra browser reads, replay,
            // or cancellation of a potentially healthy owned request.
            if (options.onDiagnostic && Date.now() >= nextDiagnosticAt) {
                nextDiagnosticAt = Date.now() + 60000;
                try {
                    options.onDiagnostic({
                        still_rendering: true,
                        elapsed_seconds: Math.floor((Date.now() - startedAt) / 1000),
                    }, conversationUrl);
                } catch {
                    // Observability must not interrupt or replace generation.
                }
            }
            lastUrls = [];
            stableCount = 0;
            failureText = '';
            failureCount = 0;
            continue;
        }

        const candidates = (await getChatGPTVisibleImageUrls(page)).filter(url => !beforeSet.has(url));
        if (Date.now() >= deadline) break;
        // Canvas snapshots can freeze while a background tab is still drawing.
        // Only a stable backend/blob image can complete this command.
        const urls = candidates.filter(url => !/^data:/i.test(url));
        stillRendering = urls.length === 0 && candidates.length > 0;
        if (!urls.length) {
            lastUrls = [];
            stableCount = 0;
            // A completed provider error is not a slow image. Bind it to the
            // exact submitted turn, and require two idle, identical readings.
            // Canvas previews and unrelated/older assistant text cannot fail it.
            let terminalText = '';
            if (!stillRendering && options.prompt) {
                const messages = await getVisibleMessages(page, { textOnly: true });
                if (Date.now() >= deadline) break;
                const userIndex = messages.findLastIndex(message => message.Role === 'User');
                const latest = messages.at(-1);
                if (userIndex >= 0 && userIndex < messages.length - 1
                    && chatGPTPromptMatches(messages[userIndex].Text, options.prompt)
                    && latest?.Role === 'Assistant') {
                    const text = String(latest.Text || '').trim();
                    if (/^I wasn['’]t able to generate the image due to an error on my side\.$/.test(text)
                        || /^Please upload the (?:earlier|original|reference) image\b[\s\S]*\bso I can (?:recompose|edit|redraw)\b/i.test(text)) {
                        terminalText = text;
                    }
                }
            }
            failureCount = terminalText && terminalText === failureText ? failureCount + 1 : (terminalText ? 1 : 0);
            failureText = terminalText;
            if (failureCount >= 2) {
                const missingInput = /^Please upload /i.test(terminalText);
                report(missingInput ? 'image_input_missing' : 'image_generation_failed');
                throw new CommandExecutionError(
                    `${missingInput ? 'CHATGPT_IMAGE_INPUT_MISSING' : 'CHATGPT_IMAGE_GENERATION_FAILED'}: ${terminalText}`,
                    `Inspect the original request at ${conversationUrl}; the image prompt was not replayed.`,
                );
            }
            // Observe missing output without navigating, replaying the request,
            // or granting a placeholder permission to complete generation.
            // Use the existing deadline; never add a browser read after timeout.
            if (options.onDiagnostic && Date.now() >= nextDiagnosticAt && Date.now() < deadline) {
                nextDiagnosticAt = Date.now() + 60000;
                try {
                    const diagnostic = unwrapEvaluateResult(await page.evaluate(`(() => {
                        const images = Array.from(document.querySelectorAll('img')).filter(img =>
                            /^Generated image \\d+$/i.test(img.getAttribute('alt') || '')
                            && !img.closest('form, [contenteditable="true"], [data-message-author-role="user"], [data-turn="user"], [data-user-message-bubble]')
                        );
                        return {
                            visibility: document.visibilityState,
                            ready_state: document.readyState,
                            generated_image_count: images.length,
                            generated_images: images.slice(0, 20).map(img => {
                                const rect = img.getBoundingClientRect();
                                const src = img.currentSrc || img.src || '';
                                return {
                                    complete: !!img.complete,
                                    natural_width: img.naturalWidth,
                                    natural_height: img.naturalHeight,
                                    layout_width: rect.width,
                                    layout_height: rect.height,
                                    source_kind: /^blob:/i.test(src) ? 'blob'
                                        : /^https?:/i.test(src) ? 'http'
                                        : /^data:/i.test(src) ? 'data' : src ? 'other' : 'missing',
                                };
                            }),
                        };
                    })()`));
                    if (diagnostic && typeof diagnostic === 'object') {
                        options.onDiagnostic({ ...diagnostic, still_rendering: stillRendering }, conversationUrl);
                    }
                } catch {
                    // A failed observation must not destroy the owned request,
                    // mask the original timeout, or leak raw bridge error data.
                    options.onDiagnostic({ observation_failed: true }, conversationUrl);
                }
            }
            continue;
        }
        failureText = '';
        failureCount = 0;
        if (urls.join('\n') === lastUrls.join('\n')) stableCount += 1;
        else { lastUrls = urls; stableCount = 1; }
        if (stableCount >= 2) {
            report('images_detected');
            return lastUrls;
        }
    }

    report('image_wait_timeout');
    if (stillRendering || lastUrls.length) {
        throw new TimeoutError(
            'chatgpt image', timeoutSeconds,
            `No stable finished image appeared before the response deadline. Open ${conversationUrl} to inspect the submitted request.`,
        );
    }
    return [];
}

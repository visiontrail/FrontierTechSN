export async function waitForChatGPTImages(page, beforeUrls, timeoutSeconds, convUrl, options = {}) {
    // Poll count is not elapsed time: every bridge read can take seconds.
    // Keep the response budget bounded independently of setup and export.
    const deadline = Date.now() + timeoutSeconds * 1000;
    const beforeSet = new Set(beforeUrls);
    let conversationUrl = convUrl;
    let localConversationUrl = options.localConversationUrl || '';
    let lastUrls = [];
    let stableCount = 0;
    let stillRendering = false;
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
            lastUrls = [];
            stableCount = 0;
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
            continue;
        }
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

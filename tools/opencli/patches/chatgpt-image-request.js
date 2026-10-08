export function chatGPTImageRequestState(stateFile, prompt) {
    if (!stateFile) return { prior: null, save() {} };
    const target = path.resolve(stateFile);
    const temporary = target + '.tmp';
    for (const candidate of [target, temporary]) {
        if (fs.existsSync(candidate) && fs.lstatSync(candidate).isSymbolicLink()) {
            throw new CommandExecutionError('ChatGPT image state cannot be a symlink');
        }
    }
    const prior = fs.existsSync(target) ? JSON.parse(fs.readFileSync(target, 'utf8')) : null;
    if (prior && prior.prompt !== prompt) {
        throw new CommandExecutionError('CHATGPT_IMAGE_TURN_PENDING: saved image prompt differs; inspect the original request before replacing its state');
    }
    return { prior, save(stage, conversationUrl = '', beforeUrls = []) {
        fs.mkdirSync(path.dirname(target), { recursive: true });
        fs.writeFileSync(temporary, JSON.stringify({ prompt, stage, conversationUrl, beforeUrls }, null, 2));
        fs.renameSync(temporary, target);
    } };
}

export async function prepareChatGPTImageComposer(page) {
    // Only used before submission. Never recover an owned turn by opening a new chat.
    const url = 'https://chatgpt.com/new';
    const rejected = error => /Navigation rejected/i.test(String(error?.message || error));
    // A new automation window can reject its first navigation while its tab
    // is still registering. Keep the same owned target and let it hydrate;
    // closing/recreating the window repeats that startup race indefinitely.
    for (let attempt = 0; attempt < 3; attempt++) {
        try {
            await page.goto(url, { settleMs: 2000 });
            return;
        } catch (error) {
            if (!rejected(error) || attempt === 2) throw error;
            await page.sleep(1);
        }
    }
}

export async function recoverChatGPTImageRequest(page, prior, prompt, timeout) {
    // A saved candidate URL never authorizes output by itself. Recover only
    // after the latest user message matches the exact submitted image prompt.
    if (!/^https:\/\/chatgpt\.com\/c\/[A-Za-z0-9_-]{8,}(?:[?#]|$)/.test(prior.conversationUrl || '')) {
        throw new CommandExecutionError('CHATGPT_IMAGE_TURN_PENDING: no saved conversation URL; inspect the original tab without resubmitting');
    }
    await page.goto(prior.conversationUrl, { settleMs: 2000 });
    const deadline = Date.now() + timeout * 1000;
    while (Date.now() < deadline) {
        const messages = await getVisibleMessages(page, { textOnly: true });
        const latestUser = [...messages].reverse().find(message => message.Role === 'User');
        if (latestUser) {
            if (!chatGPTPromptMatches(latestUser.Text, prompt)) {
                throw new CommandExecutionError('CHATGPT_IMAGE_TURN_PENDING: recovery prompt mismatch; original request was not replayed');
            }
            const identity = await resolveChatGPTConversationIdentity(page, prompt);
            if (identity && parseChatGPTConversationId(identity.conversationUrl)
                === parseChatGPTConversationId(prior.conversationUrl)) return identity;
            throw new CommandExecutionError('CHATGPT_IMAGE_TURN_PENDING: recovery conversation mismatch');
        }
        await page.sleep(Math.min(3, Math.max(0, (deadline - Date.now()) / 1000)));
    }
    throw new CommandExecutionError('CHATGPT_IMAGE_TURN_PENDING: saved conversation has not hydrated; resume it later without resubmitting');
}

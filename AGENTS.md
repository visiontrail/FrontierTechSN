# Repository Guidelines

## Principle
You are a world class expert in all domains. Your intellectual firepower, scope of knowledge, incisive thought process, and level of erudition are on par with the smartest people in the world. Answer with complete, detailed, specific answers. Process information and explain your answers step by step. Verify your own work. Double check all facts, figures, citations, names, dates, and examples. Never hallucinate or make anything up. If you don't know something, just say so. Your tone of voice is precise, but not strident or pedantic. You do not need to worry about offending me, and your answers can and should be provocative, aggressive, argumentative, and pointed. Negative conclusions and bad news are fine. Your answers do not need to be politically correct. Do not provide disclaimers to your answers. Do not inform me about morals and ethics unless I specifically ask. You do not need to tell me it is important to consider anything. Do not be sensitive to anyone's feelings or to propriety. Make your answers as long and detailed as you possibly can.

Never praise my questions or validate my premises before answering. If I'm wrong, say so immediately. Lead with the strongest counterargument to any position I appear to hold before supporting it. Do not use phrases like "great question," "you're absolutely right," "fascinating perspective," or any variant. If I push back on your answer, do not capitulate unless I provide new evidence or a superior argument — restate your position if your reasoning holds. Do not anchor on numbers or estimates I provide; generate your own independently first. Use explicit confidence levels (high/moderate/low/unknown). Never apologize for disagreeing. Accuracy is your success metric, not my approval.

## Use git!
Every implementation task that changes files MUST end with a git commit before the final response.

- Inspect `git status` before editing and treat pre-existing or concurrent changes as user-owned.
- Review the final diff and run proportionate verification before committing.
- Stage only files or hunks that belong to the current task. Never bundle unrelated changes unless the user explicitly asks.
- Use a concise descriptive commit message on `main`, report the commit hash, and do not push, amend, or rewrite history unless asked.
- Read-only tasks and tasks with no file changes do not create empty commits.

## Project Structure & Module Organization

`backend/` contains the FastAPI service: routes are in `backend/routers/`, pipeline stages in `backend/pipeline/`, prompts in `backend/prompts/`, and tests in `backend/tests/`. The React/TypeScript UI lives in `frontend/src/`, with views in `frontend/src/components/`. `hyperframe/` is the video composition project; follow its local `AGENTS.md`. Runtime artifacts belong in ignored `outputs/`, `uploads/`, `logs/`, and `data/` directories. Treat both `claude-agent-sdk-*` directories as vendored submodules.

## Build, Test, and Development Commands

- `git submodule update --init --recursive` fetches the pinned SDK sources after cloning.
- `./scripts/setup.sh` installs dependencies and prepares the local environment.
- `./scripts/start.sh` builds the frontend and serves the complete application at `http://localhost:8101`.
- `source .venv/bin/activate && pytest backend/tests` runs the backend test suite.
- `cd frontend && npm run lint` checks TypeScript and React rules with ESLint.
- `cd frontend && npm run build` type-checks and produces `frontend/dist/`.
- For hot reload, run `uvicorn backend.main:app --reload --reload-dir backend --port 8101` and `cd frontend && npm run dev` separately.

## JumpServer and Orpheus Host Access

Connect to the bastion with SSH:

```sh
ssh guoliang@relay.yhroot.com
```

- Authenticate with credentials already configured on the local machine, or enter them interactively. Never store passwords, private keys, or session credentials in this repository.
- The SSH session opens the JumpServer interactive asset selector.
- At the initial `Opt>` prompt, press Enter without selecting an option.

## deployed tts server (`nr-test`, `10.60.11.3`)

After connecting to JumpServer and reaching the asset list:

1. At `[Host]>`, enter `4` to select `nr-test` (`10.60.11.3`).
2. At the account `ID>` prompt, enter `1` to select the `yhsudo` account.
3. After login, become root and enter the working directory:

```sh
sudo su
cd /home/guoliang/
```

The complete expected interaction is:

```text
Opt> <press Enter>
[Host]> 4
ID> 1
yhsudo@ubuntu:~$ sudo su
root@ubuntu:/home/yhsudo# cd /home/guoliang/
root@ubuntu:/home/guoliang#
```


JumpServer menu numbers can change. Always select the currently displayed asset and account labels instead of relying on historical numeric choices. Before changing, restarting, or redeploying Orpheus, inspect the remote working tree, running processes or containers, and current service state; preserve unrelated remote changes and perform an end-to-end request against the deployed service after the requested change.

## Coding Style & Naming Conventions

Use four-space indentation and `snake_case` for Python functions, modules, and tests; use `PascalCase` for classes. Put shared configuration in `backend/config.py` or `backend/settings_store.py`. In TypeScript, follow the existing two-space, single-quote, semicolon-free style. Name React components and files in `PascalCase`, hooks as `useSomething`, and utilities in `camelCase`.

## Testing Guidelines

Use pytest and name files `test_<feature>.py` with functions `test_<behavior>()`. Add regression coverage near the affected pipeline module, including fallback and failure paths for subprocess or provider integrations. There is no enforced coverage percentage; prioritize deterministic unit tests and mock network, browser, TTS, and renderer boundaries.

## Commit & Pull Request Guidelines

Recent commits use short, imperative summaries such as `Add visual planning pipeline`; optional prefixes like `chore:` are acceptable. Keep each commit scoped and avoid committing generated media, databases, logs, secrets, or `frontend/dist/`. Pull requests should explain the user-visible effect, list validation commands, link relevant issues or OpenSpec changes, and include screenshots for UI changes. Call out configuration migrations, new external tools, and manual media-pipeline checks explicitly.

## Security & Configuration

Copy settings from `.env.example`, but never commit `.env`, API keys, cookies, or browser-session data. Most runtime settings are persisted through **Admin -> System** in ignored `data/settings.json`; document any new setting and provide a safe default.

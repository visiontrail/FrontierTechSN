from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass

from backend import config


class OpenCodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenCodeResult:
    session_id: str
    text: str
    stdout: str
    stderr: str


def _assistant_text(stdout: str) -> tuple[str, str]:
    parts: list[str] = []
    session_id = ""
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        session_id = str(event.get("sessionID") or session_id)
        part = event.get("part")
        if (
            event.get("type") == "text"
            and isinstance(part, dict)
            and isinstance(part.get("text"), str)
        ):
            parts.append(part["text"])
    text = "\n".join(part.strip() for part in parts if part.strip()).strip()
    if not text:
        raise OpenCodeError("OpenCode completed without an assistant text response")
    return session_id, text


async def run_content_agent(
    *,
    model: str,
    content_prompt: str,
    event_date: str,
) -> OpenCodeResult:
    agent_prompt = f"""Use the project skill `account-operations`.

You are the planning stage for an auditable Today in History account operation dated {event_date}. Do not generate an image and do not publish, delete, edit, like, follow, or message anything on X or any other social platform.

Run exactly `scripts/account-ops-opencli-plan.sh` once. The script receives the content prompt through its inherited environment and opens a fresh ChatGPT web conversation. If the command returns a usable content JSON object, preserve it exactly. If it fails, times out, or returns unusable content, do not call another tool: curate the requested JSON yourself using only facts you can state with high confidence. Never claim that you performed web verification in fallback mode. Your final answer must contain only the requested JSON object, with no Markdown fence or commentary. Do not inspect traces, debug adapters, load autofix skills, or modify files.

CONTENT PROMPT:
{content_prompt}
"""
    env = os.environ.copy()
    env["ACCOUNT_OPS_CONTENT_PROMPT"] = content_prompt
    return await _run_agent(
        model=model,
        agent="account-operations",
        title=f"Today in History · {event_date}",
        prompt=agent_prompt,
        env=env,
        timeout=config.OPENCODE_TIMEOUT,
    )


async def run_operational_agent(
    *,
    model: str,
    prompt: str,
    title: str,
) -> OpenCodeResult:
    """Run the write-capable, OpenCLI-only X account operator once."""
    return await _run_agent(
        model=model,
        agent="x-account-operator",
        title=title,
        prompt=prompt,
        env=os.environ.copy(),
        timeout=max(config.OPENCODE_TIMEOUT, 900),
    )


async def _run_agent(
    *,
    model: str,
    agent: str,
    title: str,
    prompt: str,
    env: dict[str, str],
    timeout: int,
) -> OpenCodeResult:
    process = await asyncio.create_subprocess_exec(
        config.OPENCODE_BIN,
        "run",
        "--auto",
        "--model",
        model,
        "--agent",
        agent,
        "--format",
        "json",
        "--title",
        title,
        prompt,
        cwd=str(config.PROJECT_ROOT),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=timeout
        )
    except TimeoutError as exc:
        process.kill()
        await process.communicate()
        raise OpenCodeError(
            f"OpenCode timed out after {timeout}s"
        ) from exc
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
    if process.returncode:
        detail = (stderr or stdout or "unknown OpenCode failure")[-1800:]
        raise OpenCodeError(
            f"OpenCode exited with {process.returncode}: {detail}"
        )
    session_id, text = _assistant_text(stdout)
    return OpenCodeResult(
        session_id=session_id,
        text=text,
        stdout=stdout,
        stderr=stderr,
    )

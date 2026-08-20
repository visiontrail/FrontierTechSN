from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend import config
from backend.daily_news.research import ResearchDossier
from backend.daily_news.scriptwriter import (
    morning_opening,
    revise_daily_script,
    script_contract_report,
)
from backend.pipeline.opencli import OpenCLIError, first_json, run_opencli

LogCallback = Callable[[str], None]

_REVIEW_REQUEST_RE = re.compile(r"REVIEW_REQUEST_ID:([0-9a-f]{32})", re.I)
_CHATGPT_CONVERSATION_URL_RE = re.compile(
    r"https://chatgpt\.com/c/[0-9a-z-]+",
    re.I,
)


@dataclass(frozen=True)
class ScriptReviewResult:
    script: str
    report: dict[str, Any]


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("data", "items", "results", "rows"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [row for row in nested if isinstance(row, dict)]
        return [value]
    return []


def _field(row: dict[str, Any], name: str) -> str:
    for key, value in row.items():
        if str(key).casefold() == name.casefold():
            return str(value or "").strip()
    return ""


def _normalized_review_token(text: str) -> str:
    """Normalize only known UI wrappers while preserving exact-token safety."""
    clean = unicodedata.normalize("NFKC", str(text or ""))
    clean = clean.translate(
        {
            ord("\ufeff"): None,
            ord("\u200b"): None,
            ord("\u200c"): None,
            ord("\u200d"): None,
            ord("\ufe0f"): None,
        }
    ).strip()
    clean = re.sub(r"^\s*💬\s*", "", clean).strip()
    fenced = re.fullmatch(
        r"```(?:text|plaintext)?\s*(.*?)\s*```",
        clean,
        flags=re.I | re.S,
    )
    if fenced:
        clean = fenced.group(1).strip()
    inline = re.fullmatch(r"`([^`]*)`", clean, flags=re.S)
    if inline:
        clean = inline.group(1).strip()
    return "".join(clean.split()).upper()


def _assistant_for_review_request(
    rows: list[dict[str, Any]],
    request_id: str,
) -> str:
    """Return only the assistant turn owned by the current review request."""
    marker = f"REVIEW_REQUEST_ID:{request_id}".casefold()
    anchor = -1
    for index, row in enumerate(rows):
        if _field(row, "Role").casefold() != "user":
            continue
        if marker in _field(row, "Text").casefold():
            anchor = index
    if anchor < 0:
        return ""

    for row in rows[anchor + 1 :]:
        role = _field(row, "Role").casefold()
        text = _field(row, "Text")
        if role == "user":
            other_request = _REVIEW_REQUEST_RE.search(text)
            if other_request and other_request.group(1).casefold() != request_id.casefold():
                return ""
            # Gemini may expose one submitted prompt as a full User turn plus
            # several text fragments. Fragment rows have no request marker.
            continue
        if role == "assistant" and text:
            return text
    return ""


def _json_object(text: str) -> dict[str, Any]:
    clean = text.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.I)
    clean = re.sub(r"\s*```$", "", clean)
    try:
        value = json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("{")
        end = clean.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("review response did not contain a JSON object")
        value = json.loads(clean[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("review response JSON is not an object")
    return value


def _compact_article_evidence(script: str, article, index: int) -> str:
    """Keep one story prompt small while retaining claim-relevant evidence."""
    claim_tokens = {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9.-]{3,}|\d+(?:[.,]\d+)?", script)
        if token.casefold() not in {"that", "this", "with", "from", "have", "will", "today"}
    }
    fragments = [
        fragment.strip()
        for fragment in re.split(
            r"\n+|(?<=[.!?。！？])\s+",
            "\n".join([article.summary, article.evidence_text]),
        )
        if len(fragment.strip()) >= 25
    ]

    def relevance(fragment: str) -> tuple[int, int]:
        lowered = fragment.casefold()
        overlap = sum(1 for token in claim_tokens if token in lowered)
        number_overlap = sum(
            1 for token in claim_tokens if token[:1].isdigit() and token in lowered
        )
        return (number_overlap * 8 + overlap, -fragments.index(fragment))

    chosen: list[str] = []
    for fragment in sorted(fragments, key=relevance, reverse=True):
        if fragment not in chosen:
            chosen.append(fragment)
        if len("\n".join(chosen)) >= 1200:
            break
    excerpt = "\n".join(chosen)[:1500] or article.summary[:1500] or article.title
    return "\n".join(
        [
            f"STORY {index}",
            f"Title: {article.title}",
            f"Source: {article.source_name}",
            f"Published: {article.published_at}",
            f"Feed summary lead: {(article.summary[:350] if article.summary else 'none')}",
            "Claim-relevant article evidence:",
            excerpt,
        ]
    )


def _compact_evidence_ledger(script: str, dossier: ResearchDossier) -> str:
    return "\n\n".join(
        _compact_article_evidence(script, article, index)
        for index, article in enumerate(dossier.selected, 1)
    )


def _matched_script_claims(script: str, dossier: ResearchDossier) -> dict[int, str]:
    """Map reordered script paragraphs to stories using evidence-token overlap."""
    lines = [line.strip() for line in script.splitlines() if line.strip()]
    paragraphs = lines[1:-1] if len(lines) >= 3 else lines
    stop = {
        "according", "reports", "report", "story", "company", "technology", "their",
        "this", "that", "with", "from", "into", "after", "about", "more", "than",
    }

    def tokens(value: str) -> set[str]:
        value = value.replace("-", " ").replace("—", " ").replace("–", " ")
        return {
            token.casefold()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9.-]{2,}|\d+(?:[.,]\d+)?", value)
            if token.casefold() not in stop
        }

    pairs: list[tuple[int, int, int]] = []
    for story_index, article in enumerate(dossier.selected, 1):
        evidence_tokens = tokens(
            " ".join([article.source_name, article.title, article.summary, article.evidence_text[:3500]])
        )
        for paragraph_index, paragraph in enumerate(paragraphs):
            paragraph_tokens = tokens(paragraph)
            overlap = evidence_tokens & paragraph_tokens
            score = len(overlap) + 4 * sum(token[:1].isdigit() for token in overlap)
            pairs.append((score, story_index, paragraph_index))

    matched: dict[int, str] = {}
    used_paragraphs: set[int] = set()
    for score, story_index, paragraph_index in sorted(pairs, reverse=True):
        if score < 2 or story_index in matched or paragraph_index in used_paragraphs:
            continue
        matched[story_index] = paragraphs[paragraph_index]
        used_paragraphs.add(paragraph_index)
    # The writing contract fixes dossier order. Translation can erase all
    # Latin-token overlap for Chinese evidence, so positionally pair only the
    # residual one-paragraph-per-story slots after every strong match is taken.
    remaining_stories = [
        index for index in range(1, len(dossier.selected) + 1) if index not in matched
    ]
    remaining_paragraphs = [
        index for index in range(len(paragraphs)) if index not in used_paragraphs
    ]
    if len(remaining_stories) == len(remaining_paragraphs):
        for story_index, paragraph_index in zip(remaining_stories, remaining_paragraphs, strict=True):
            matched[story_index] = paragraphs[paragraph_index]
    return matched


def _review_prompt(
    script: str,
    dossier: ResearchDossier,
    edition_date: date,
    story_index: int,
) -> str:
    article = dossier.selected[story_index - 1]
    return f"""You are the independent final fact-checker for STORY {story_index} in a technology broadcast dated {edition_date.isoformat()}.
Check whether the full script accurately covers this selected story and whether every script claim about it is supported by the story evidence. Missing coverage, wrong names, wrong numbers, stale framing, unsupported extrapolation, or company claims stated as independent fact must fail.

Reply with exactly ONE ASCII token:
- P when every claim is supported and the selected story is covered.
- Otherwise concatenate every applicable error code in alphabetical order, with no separator:
  A missing story coverage; B wrong name/entity; C wrong number/date; D unsupported extrapolation;
  E missing attribution/uncertainty; F contradiction or stale framing.
Example failure response: CD

Do not browse. Do not emit explanations, citations, URLs, JSON, markdown, spaces, or punctuation.

FULL SCRIPT
{script}

STORY EVIDENCE
{_compact_article_evidence(script, article, story_index)}
"""


def _batch_review_prompt(
    script: str,
    dossier: ResearchDossier,
    edition_date: date,
    story_numbers: list[int] | None = None,
) -> str:
    numbers = story_numbers or list(range(1, len(dossier.selected) + 1))
    example = "".join(f"{index}P" for index in numbers)
    evidence = "\n\n".join(
        _compact_article_evidence(script, dossier.selected[index - 1], index)
        for index in numbers
    )
    matched_claims = _matched_script_claims(script, dossier)
    claims = "\n".join(
        f"STORY {index} SCRIPT CLAIMS: {matched_claims.get(index, 'NO MATCHED COVERAGE')}"
        for index in numbers
    )
    return f"""You are the independent final fact-checker for stories {numbers} in a technology broadcast dated {edition_date.isoformat()}.
For EACH numbered story, check whether the full script covers it accurately and whether every related claim is supported by that story's evidence. Missing coverage, wrong names, wrong numbers, stale framing, unsupported extrapolation, or company claims stated as independent fact must fail.

Reply with exactly ONE ASCII token. For each story in order, write its number followed by P, or by all applicable error codes in alphabetical order:
A missing story coverage; B wrong name/entity; C wrong number/date; D unsupported extrapolation; E missing attribution/uncertainty; F contradiction or stale framing.
Example all-pass response: {example}
Example mixed response for stories 1 and 2: 1P2D

Do not browse. Do not emit explanations, citations, URLs, JSON, markdown, spaces, or punctuation.

SCRIPT CLAIMS FOR REVIEWED STORIES
{claims}

STORY EVIDENCE
{evidence}
"""


def _single_line_payload(text: str, story_number: int) -> dict[str, Any]:
    clean = _normalized_review_token(text)
    if clean == "P":
        return {
            "approved": True,
            "confidence": 100,
            "summary": f"Story {story_number} passed.",
            "issues": [],
            "corrected_script": "",
        }
    if not re.fullmatch(r"[A-F]{1,6}", clean) or len(set(clean)) != len(clean):
        raise ValueError("review response did not contain the complete single-line protocol")
    directives = {
        "A": "Add concise coverage of this selected story using only its evidence.",
        "B": "Correct every name and entity for this story to match the evidence. If dossier fields conflict on a spelling, omit that disputed proper name rather than choosing one.",
        "C": "Correct or remove every unsupported number and date for this story.",
        "D": "Delete every interpretive, significance, trend, or extrapolative sentence for this story; retain only directly evidenced reporting.",
        "E": "Attribute company or source claims and preserve uncertainty language.",
        "F": "Remove contradictions and stale framing; align the story strictly to the dated evidence.",
    }
    codes = clean
    correction = " ".join(directives[code] for code in codes)
    return {
        "approved": False,
        "confidence": 100,
        "summary": f"Story {story_number} needs correction.",
        "issues": [
            {
                "severity": "blocking",
                "claim": f"Web audit codes {codes} for story {story_number}",
                "verdict": correction,
                "evidence_story_numbers": [story_number],
                "correction": correction,
            }
        ],
        "corrected_script": "",
    }


def _batch_payload(text: str, story_numbers: int | list[int]) -> dict[str, Any]:
    expected = (
        list(range(1, story_numbers + 1))
        if isinstance(story_numbers, int)
        else story_numbers
    )
    clean = _normalized_review_token(text)
    matches = list(re.finditer(r"(\d+)(P|[A-F]{1,6})", clean))
    if "".join(match.group(0) for match in matches) != clean:
        raise ValueError("batch review response contained invalid characters")
    indices = [int(match.group(1)) for match in matches]
    if indices != expected:
        raise ValueError("batch review response omitted or reordered a story")
    story_payloads = [
        _single_line_payload(match.group(2), int(match.group(1)))
        for match in matches
    ]
    issues = [issue for payload in story_payloads for issue in payload["issues"]]
    return {
        "approved": not issues,
        "confidence": 100,
        "summary": f"{len(expected) - len(issues)}/{len(expected)} story audits passed.",
        "issues": issues,
        "corrected_script": "",
    }


def _protocol_payload(text: str) -> dict[str, Any]:
    # Accept valid legacy JSON too, but prefer the deliberately short protocol.
    try:
        return _json_object(text)
    except (ValueError, json.JSONDecodeError):
        pass
    verdict_match = re.search(r"^VERDICT:\s*(APPROVE|REJECT)\s*$", text, re.I | re.M)
    confidence_match = re.search(r"^CONFIDENCE:\s*(\d{1,3})\s*$", text, re.I | re.M)
    summary_match = re.search(r"^SUMMARY:\s*(.+)$", text, re.I | re.M)
    if not verdict_match or not confidence_match or not summary_match or not re.search(r"^END\s*$", text, re.I | re.M):
        raise ValueError("review response did not contain the complete line protocol")
    confidence = int(confidence_match.group(1))
    if not 0 <= confidence <= 100:
        raise ValueError("review confidence is outside 0..100")
    issues: list[dict[str, Any]] = []
    for match in re.finditer(r"^ISSUE:\s*(.+)$", text, re.I | re.M):
        parts = [part.strip() for part in match.group(1).split("|", 3)]
        if len(parts) != 4 or parts[0].casefold() not in {"blocking", "warning"}:
            raise ValueError("review ISSUE line is malformed")
        story_match = re.search(r"\d+", parts[1])
        issues.append(
            {
                "severity": parts[0].casefold(),
                "claim": parts[2],
                "verdict": parts[3],
                "evidence_story_numbers": [int(story_match.group())] if story_match else [],
                "correction": parts[3],
            }
        )
    approved = verdict_match.group(1).upper() == "APPROVE"
    blocking = [issue for issue in issues if issue["severity"] == "blocking"]
    if approved and blocking:
        raise ValueError("review approved despite a blocking issue")
    if not approved and not blocking:
        raise ValueError("review rejected without a blocking issue")
    return {
        "approved": approved,
        "confidence": confidence,
        "summary": summary_match.group(1).strip(),
        "issues": issues,
        "corrected_script": "",
    }


async def _web_story_review(
    prompt: str,
    *,
    story_number: int | None = None,
    story_numbers: list[int] | None = None,
    log: LogCallback | None,
) -> tuple[dict[str, Any], str, str, str]:
    if (story_number is None) == (story_numbers is None):
        raise ValueError("exactly one of story_number or story_numbers is required")
    review_label = f"story {story_number}" if story_number is not None else f"stories {story_numbers}"

    def parse_response(value: str) -> dict[str, Any]:
        if story_numbers is not None:
            return _batch_payload(value, story_numbers)
        assert story_number is not None
        return _single_line_payload(value, story_number)

    timeout = config.DAILY_NEWS_WEB_REVIEW_TIMEOUT
    request_id = uuid4().hex
    browser_prompt = " ".join(prompt.split())
    owned_prompt = (
        f"REVIEW_REQUEST_ID:{request_id}. This marker identifies the browser turn; "
        f"do not include it in the answer. {browser_prompt}"
    )
    raw_attempts: list[str] = []

    async def recover_current(
        provider: str,
        *,
        conversation_url: str = "",
    ) -> tuple[dict[str, Any], str]:
        if provider == "chatgpt" and conversation_url:
            command = [
                "chatgpt",
                "detail",
                conversation_url,
                "--wait",
                "true",
                "--timeout",
                str(timeout),
                "--stable",
                "2",
            ]
        else:
            command = [provider, "read"]
        command.extend(
            [
                "--window",
                "foreground" if provider == "gemini" else "background",
                "--site-session",
                "persistent",
                "--keep-tab",
                "true",
                "-f",
                "json",
            ]
        )
        read_result = await run_opencli(
            command,
            timeout=timeout + 15 if conversation_url else 30,
        )
        messages = _rows(first_json(read_result.stdout))
        assistant_text = _assistant_for_review_request(messages, request_id)
        if not assistant_text:
            raise ValueError(
                f"{provider} recovery page did not contain a response owned by this request"
            )
        raw_attempts.append(f"[{provider.upper()} RECOVERY]\n{assistant_text}")
        return parse_response(assistant_text), assistant_text

    gemini_error: Exception | None = None
    gemini_attempts = 2
    select_gemini_model = True
    start_fresh_gemini = True
    for gemini_attempt in range(1, gemini_attempts + 1):
        try:
            gemini_args = ["gemini", "ask", owned_prompt]
            if select_gemini_model:
                gemini_args.extend(
                    ["--model", config.DAILY_NEWS_GEMINI_REVIEW_MODEL]
                )
            gemini_args.extend(
                [
                    "--new",
                    "true" if start_fresh_gemini else "false",
                    "--timeout",
                    str(timeout),
                    "--window",
                    # Gemini's current composer requires the adapter's trusted
                    # keyboard path, so the isolated project tab is foregrounded.
                    "foreground",
                    "--site-session",
                    "persistent",
                    "--keep-tab",
                    "true",
                    "-f",
                    "json",
                ]
            )
            result = await run_opencli(
                gemini_args,
                # Model discovery happens before Gemini's answer timeout.
                timeout=timeout + 60,
            )
            rows = _rows(first_json(result.stdout))
            response = next(
                (_field(row, "response") for row in rows if _field(row, "response")),
                "",
            )
            raw_attempts.append(f"[GEMINI {gemini_attempt}]\n{response or '[EMPTY]'}")
            if not response or "[NO RESPONSE]" in response.upper():
                raise ValueError("Gemini ask returned no completed assistant response")
            payload = parse_response(response)
            if log:
                log(
                    f"Gemini {config.DAILY_NEWS_GEMINI_REVIEW_MODEL} completed "
                    f"{review_label} primary review"
                )
            return (
                payload,
                "\n\n--- PROVIDER ATTEMPT ---\n\n".join(raw_attempts),
                "",
                "gemini",
            )
        except Exception as exc:  # noqa: BLE001 - bounded primary-provider retry
            gemini_error = exc
            try:
                recovered = await recover_current("gemini")
                payload, _ = recovered
                if log:
                    log(
                        f"Recovered completed Gemini {review_label} primary review "
                        "from its owned browser turn"
                    )
                return (
                    payload,
                    "\n\n--- PROVIDER ATTEMPT ---\n\n".join(raw_attempts),
                    "",
                    "gemini",
                )
            except Exception as recovery_exc:  # noqa: BLE001 - preserve both failures
                gemini_error = RuntimeError(
                    f"ask failed ({exc}); owned-turn recovery failed ({recovery_exc})"
                )
            if "model picker button was not found" in str(exc).casefold():
                select_gemini_model = False
                start_fresh_gemini = False
                if log:
                    log(
                        "Gemini model picker is not ready; the retry will reuse "
                        "the isolated page and its current Flash model"
                    )
            if gemini_attempt < gemini_attempts and log:
                log(
                    f"Gemini {review_label} primary attempt {gemini_attempt}/"
                    f"{gemini_attempts} failed; starting the bounded retry: {gemini_error}"
                )

    if log:
        log(
            f"Gemini {review_label} primary review failed; falling back to "
            f"ChatGPT {config.DAILY_NEWS_CHATGPT_REVIEW_MODEL}: {gemini_error}"
        )

    chatgpt_error: Exception | None = None
    try:
        await run_opencli(
            [
                "chatgpt",
                "model",
                config.DAILY_NEWS_CHATGPT_REVIEW_MODEL,
                "--window",
                "background",
                "--site-session",
                "persistent",
                "--keep-tab",
                "true",
                "-f",
                "json",
            ],
            timeout=60,
        )
        if log:
            log(
                "ChatGPT fallback review model: "
                f"{config.DAILY_NEWS_CHATGPT_REVIEW_MODEL}"
            )
    except Exception as exc:  # noqa: BLE001 - fail closed on unknown fallback model
        chatgpt_error = exc
        raw_attempts.append(f"[CHATGPT MODEL ERROR]\n{exc}")

    conversation_url = ""
    if chatgpt_error is None:
        try:
            chatgpt_args = [
                "chatgpt",
                "ask",
                owned_prompt,
                "--new",
                "true",
                "--wait",
                "true",
                "--timeout",
                str(timeout),
                "--window",
                "background",
                "--site-session",
                "persistent",
                "--keep-tab",
                "true",
                "-f",
                "json",
            ]
            result = await run_opencli(chatgpt_args, timeout=timeout + 20)
            rows = _rows(first_json(result.stdout))
            response = next(
                (_field(row, "response") for row in rows if _field(row, "response")),
                "",
            )
            conversation_url = next(
                (
                    _field(row, "conversationUrl")
                    for row in rows
                    if _field(row, "conversationUrl")
                ),
                "",
            )
            raw_attempts.append(f"[CHATGPT FALLBACK]\n{response or '[EMPTY]'}")
            if not response:
                raise ValueError("ChatGPT ask returned no assistant response")
            payload = parse_response(response)
            if log:
                log(f"ChatGPT completed {review_label} fallback review")
            return (
                payload,
                "\n\n--- PROVIDER ATTEMPT ---\n\n".join(raw_attempts),
                conversation_url,
                "chatgpt",
            )
        except Exception as exc:  # noqa: BLE001 - target-specific recovery below
            chatgpt_error = exc
            target_match = _CHATGPT_CONVERSATION_URL_RE.search(str(exc))
            target_url = conversation_url or (target_match.group(0) if target_match else "")
            try:
                recovered = await recover_current(
                    "chatgpt",
                    conversation_url=target_url,
                )
                payload, _ = recovered
                if log:
                    recovery_source = "target conversation" if target_url else "owned browser turn"
                    log(
                        f"Recovered completed ChatGPT {review_label} fallback from "
                        f"its {recovery_source}"
                    )
                return (
                    payload,
                    "\n\n--- PROVIDER ATTEMPT ---\n\n".join(raw_attempts),
                    target_url,
                    "chatgpt",
                )
            except Exception as recovery_exc:  # noqa: BLE001 - preserve both failures
                chatgpt_error = RuntimeError(
                    f"ask failed ({exc}); owned-turn recovery failed ({recovery_exc})"
                )

    raise RuntimeError(
        f"Gemini {config.DAILY_NEWS_GEMINI_REVIEW_MODEL} primary {review_label} review "
        f"failed ({gemini_error}); ChatGPT {config.DAILY_NEWS_CHATGPT_REVIEW_MODEL} "
        f"fallback failed ({chatgpt_error})"
    ) from chatgpt_error


async def review_daily_script(
    script: str,
    dossier: ResearchDossier,
    edition_date: date,
    output_dir: Path,
    *,
    language: str,
    closing_remarks: str,
    ai_endpoint: str | None,
    ai_model: str | None,
    provider_id: int | None,
    log: LogCallback | None = None,
) -> ScriptReviewResult:
    review_dir = output_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    candidate = script
    max_cycles = 5
    for cycle in range(1, max_cycles + 1):
        # Each two-story group owns a fresh Gemini turn. A failed primary turn
        # gets a fresh ChatGPT fallback instead of reusing mutable active-page state.
        contract = script_contract_report(
            candidate,
            dossier,
            edition_date,
            language=language,
            closing_remarks=closing_remarks,
        )
        if not contract["passed"]:
            raise RuntimeError(
                "Deterministic script audit failed before web review: "
                + "; ".join(contract["failures"])
            )
        group_results: list[dict[str, Any]] = []
        all_numbers = list(range(1, len(dossier.selected) + 1))
        for group_index, start in enumerate(range(0, len(all_numbers), 2), 1):
            story_numbers = all_numbers[start : start + 2]
            prompt = _batch_review_prompt(candidate, dossier, edition_date, story_numbers)
            (review_dir / f"story-review-prompt-{cycle}-group-{group_index}.txt").write_text(
                prompt, encoding="utf-8"
            )
            group_payload, raw, conversation_url, provider = await _web_story_review(
                prompt,
                story_numbers=story_numbers,
                log=log,
            )
            (review_dir / f"story-review-response-{cycle}-group-{group_index}.txt").write_text(
                raw, encoding="utf-8"
            )
            group_results.append({
                "payload": group_payload,
                "conversation_url": conversation_url,
                "provider": provider,
            })
        issues = [
            issue
            for result in group_results
            for issue in result["payload"].get("issues", [])
        ]
        payload = {
            "approved": not issues,
            "confidence": 100,
            "summary": f"{len(all_numbers) - len(issues)}/{len(all_numbers)} story audits passed.",
            "issues": issues,
        }
        conversation_url = ", ".join(
            result["conversation_url"]
            for result in group_results
            if result["conversation_url"]
        )
        providers = [result["provider"] for result in group_results]
        chatgpt_level = config.DAILY_NEWS_CHATGPT_REVIEW_MODEL
        gemini_model = config.DAILY_NEWS_GEMINI_REVIEW_MODEL
        reviewer = (
            f"Gemini Web ({gemini_model}) via project-local OpenCLI"
            if set(providers) == {"gemini"}
            else f"Gemini Web ({gemini_model}) with ChatGPT Web "
            f"({chatgpt_level}) fallback via project-local OpenCLI"
        )
        blocking = [
            issue
            for issue in payload.get("issues", [])
            if isinstance(issue, dict) and str(issue.get("severity", "")).casefold() == "blocking"
        ]
        approved = bool(payload.get("approved")) and not blocking
        attempt = {
            "cycle": cycle,
            "reviewer": reviewer,
            "providers": providers,
            "conversation_url": conversation_url,
            "approved": approved,
            "confidence": payload.get("confidence"),
            "summary": payload.get("summary", ""),
            "issues": payload.get("issues", []),
            "contract": contract,
        }
        attempts.append(attempt)
        if log:
            log(
                f"Web accuracy review cycle {cycle}: "
                f"{'approved' if approved else f'{len(blocking)} blocking issue(s)'}"
            )
        if approved:
            report = {
                "passed": True,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "reviewer": reviewer,
                "fallback_used": any(
                    "chatgpt" in prior.get("providers", []) for prior in attempts
                ),
                "attempts": attempts,
                "final_contract": contract,
            }
            (review_dir / "fact_check_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (review_dir / "approved_script.txt").write_text(candidate, encoding="utf-8")
            return ScriptReviewResult(candidate, report)

        if cycle == max_cycles:
            break
        candidate = await revise_daily_script(
            candidate,
            dossier,
            payload.get("issues", []),
            edition_date,
            language=language,
            closing_remarks=closing_remarks,
            ai_endpoint=ai_endpoint,
            ai_model=ai_model,
            provider_id=provider_id,
            log=log,
        )
        (review_dir / f"candidate-cycle-{cycle + 1}.txt").write_text(
            candidate, encoding="utf-8"
        )

    report = {
        "passed": False,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "reviewer": f"Gemini Web ({config.DAILY_NEWS_GEMINI_REVIEW_MODEL}) with "
        f"ChatGPT Web ({config.DAILY_NEWS_CHATGPT_REVIEW_MODEL}) fallback via project-local OpenCLI",
        "fallback_used": any(
            "chatgpt" in prior.get("providers", []) for prior in attempts
        ),
        "attempts": attempts,
    }
    (review_dir / "fact_check_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    raise RuntimeError(
        f"Web accuracy review did not approve the daily script after {max_cycles} correction cycles"
    )

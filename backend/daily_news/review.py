from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from backend import config
from backend.daily_news.research import ResearchDossier
from backend.daily_news.scriptwriter import (
    morning_opening,
    revise_daily_script,
    script_contract_report,
)
from backend.pipeline.opencli import OpenCLIError, first_json, run_opencli

LogCallback = Callable[[str], None]


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
    clean = " ".join(text.strip().split())
    if clean.upper() == "P":
        return {
            "approved": True,
            "confidence": 100,
            "summary": f"Story {story_number} passed.",
            "issues": [],
            "corrected_script": "",
        }
    if not re.fullmatch(r"[A-F]{1,6}", clean.upper()) or len(set(clean.upper())) != len(clean):
        raise ValueError("review response did not contain the complete single-line protocol")
    directives = {
        "A": "Add concise coverage of this selected story using only its evidence.",
        "B": "Correct every name and entity for this story to match the evidence. If dossier fields conflict on a spelling, omit that disputed proper name rather than choosing one.",
        "C": "Correct or remove every unsupported number and date for this story.",
        "D": "Delete every interpretive, significance, trend, or extrapolative sentence for this story; retain only directly evidenced reporting.",
        "E": "Attribute company or source claims and preserve uncertainty language.",
        "F": "Remove contradictions and stale framing; align the story strictly to the dated evidence.",
    }
    codes = clean.upper()
    correction = " ".join(directives[code] for code in codes)
    return {
        "approved": False,
        "confidence": 100,
        "summary": f"Story {story_number} needs correction.",
        "issues": [
            {
                "severity": "blocking",
                "claim": f"ChatGPT audit codes {codes} for story {story_number}",
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
    clean = "".join(text.strip().split()).upper()
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


async def _chatgpt_review(
    prompt: str,
    *,
    story_number: int | None = None,
    story_numbers: list[int] | None = None,
    reuse_current: bool = False,
    log: LogCallback | None,
) -> tuple[dict[str, Any], str, str]:
    if (story_number is None) == (story_numbers is None):
        raise ValueError("exactly one of story_number or story_numbers is required")
    review_label = f"story {story_number}" if story_number is not None else f"stories {story_numbers}"

    def parse_response(value: str) -> dict[str, Any]:
        if story_numbers is not None:
            return _batch_payload(value, story_numbers)
        assert story_number is not None
        return _single_line_payload(value, story_number)

    timeout = 45
    browser_prompt = " ".join(prompt.split())
    args = [
            "chatgpt",
            "ask",
            browser_prompt,
            "--new",
            "false" if reuse_current else "true",
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
    last_error: Exception | None = None
    raw_attempts: list[str] = []
    for attempt in range(1, config.OPENCLI_MAX_ATTEMPTS + 1):
        try:
            result = await run_opencli(args, timeout=timeout + 20)
            rows = _rows(first_json(result.stdout))
            response = next((_field(row, "response") for row in rows if _field(row, "response")), "")
            conversation_url = next(
                (_field(row, "conversationUrl") for row in rows if _field(row, "conversationUrl")), ""
            )
            if response:
                raw_attempts.append(response)
                payload = parse_response(response)
                return payload, "\n\n--- RETRY ---\n\n".join(raw_attempts), conversation_url
            raise ValueError("ChatGPT ask returned no assistant response")
        except Exception as exc:  # noqa: BLE001 - semantic and browser failures both retry
            last_error = exc
            # ChatGPT can finish a response and then show a temporary rate-limit
            # modal before the adapter captures the new conversation URL. Recover
            # the completed last assistant message instead of discarding that audit.
            if isinstance(exc, ValueError) or "conversation URL" in str(exc) or "timed out" in str(exc):
                try:
                    read_result = await run_opencli(
                        [
                            "chatgpt", "read", "--window", "background",
                            "--site-session", "persistent", "--keep-tab", "true", "-f", "json",
                        ],
                        timeout=30,
                    )
                    messages = _rows(first_json(read_result.stdout))
                    assistant_text = next(
                        (
                            _field(row, "Text")
                            for row in reversed(messages)
                            if _field(row, "Role").casefold() == "assistant" and _field(row, "Text")
                        ),
                        "",
                    )
                    if assistant_text:
                        raw_attempts.append(assistant_text)
                        payload = parse_response(assistant_text)
                        if log:
                            log(f"Recovered completed ChatGPT {review_label} audit from the active page")
                        return payload, "\n\n--- RETRY ---\n\n".join(raw_attempts), ""
                except Exception:
                    pass
            if log:
                log(
                    f"ChatGPT {review_label} review attempt "
                    f"{attempt}/{config.OPENCLI_MAX_ATTEMPTS} failed: {exc}"
                )
            if attempt < config.OPENCLI_MAX_ATTEMPTS:
                delay = min(60.0, config.OPENCLI_RETRY_BASE_SECONDS * 2 ** (attempt - 1))
                if "conversation URL" in str(exc):
                    delay = max(30.0, delay)
                await asyncio.sleep(delay)
    raise RuntimeError(
        f"ChatGPT {review_label} review failed after "
        f"{config.OPENCLI_MAX_ATTEMPTS} attempts: {last_error}"
    ) from last_error


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
        # A corrected script starts a new audit conversation. The three story
        # groups within that cycle reuse it, avoiding both stale tab leases and
        # ChatGPT's burst limit on one-new-chat-per-story workflows.
        review_session_started = False
        contract = script_contract_report(
            candidate,
            dossier,
            edition_date,
            language=language,
            closing_remarks=closing_remarks,
        )
        if not contract["passed"]:
            raise RuntimeError(
                "Deterministic script audit failed before ChatGPT review: "
                + "; ".join(contract["failures"])
            )
        group_results: list[dict[str, Any]] = []
        all_numbers = list(range(1, len(dossier.selected) + 1))
        for group_index, start in enumerate(range(0, len(all_numbers), 2), 1):
            story_numbers = all_numbers[start : start + 2]
            prompt = _batch_review_prompt(candidate, dossier, edition_date, story_numbers)
            (review_dir / f"chatgpt-prompt-{cycle}-group-{group_index}.txt").write_text(
                prompt, encoding="utf-8"
            )
            group_payload, raw, conversation_url = await _chatgpt_review(
                prompt,
                story_numbers=story_numbers,
                reuse_current=review_session_started,
                log=log,
            )
            review_session_started = True
            (review_dir / f"chatgpt-response-{cycle}-group-{group_index}.txt").write_text(
                raw, encoding="utf-8"
            )
            group_results.append({
                "payload": group_payload,
                "conversation_url": conversation_url,
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
        blocking = [
            issue
            for issue in payload.get("issues", [])
            if isinstance(issue, dict) and str(issue.get("severity", "")).casefold() == "blocking"
        ]
        approved = bool(payload.get("approved")) and not blocking
        attempt = {
            "cycle": cycle,
            "reviewer": "ChatGPT Web via project-local OpenCLI",
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
                f"ChatGPT accuracy review cycle {cycle}: "
                f"{'approved' if approved else f'{len(blocking)} blocking issue(s)'}"
            )
        if approved:
            report = {
                "passed": True,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "reviewer": "ChatGPT Web via project-local OpenCLI",
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
        "reviewer": "ChatGPT Web via project-local OpenCLI",
        "attempts": attempts,
    }
    (review_dir / "fact_check_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    raise RuntimeError(
        f"ChatGPT accuracy review did not approve the daily script after {max_cycles} correction cycles"
    )

from __future__ import annotations

import asyncio
import json
import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend import config
from backend.daily_news.editorial import ATTRIBUTION_REVIEW_RULES, source_language_guidance
from backend.daily_news.research import ResearchDossier
from backend.daily_news.scriptwriter import (
    fit_daily_script_duration,
    revise_daily_script,
    script_contract_report,
)
from backend.pipeline.opencli import (
    OpenCLIRateLimitError,
    close_opencli_site_sessions,
    first_json,
    run_opencli,
)

LogCallback = Callable[[str], None]

_REVIEW_REQUEST_RE = re.compile(r"REVIEW_REQUEST_ID:([0-9a-f]{32})", re.I)
_CHATGPT_CONVERSATION_URL_RE = re.compile(
    r"https://chatgpt\.com/c/[0-9a-z-]+",
    re.I,
)
_RECOVERY_POLL_INTERVAL_SECONDS = 1.0
_TARGET_RECOVERY_MAX_POLL_INTERVAL_SECONDS = 30.0
_TARGET_RECOVERY_MAX_TOTAL_SECONDS = 180.0
_MODEL_SELECTION_RETRY_DELAY_SECONDS = 2.0
_CHATGPT_REVIEW_LEVEL_ORDER = {"medium": 1, "high": 2, "xhigh": 3}
_CHATGPT_OBSERVED_LEVELS = {
    "balanced": "medium",
    "medium": "medium",
    "均衡": "medium",
    "advanced": "high",
    "high": "high",
    "thinking": "high",
    "高级": "high",
    "思考": "high",
    "veryhigh": "xhigh",
    "extrahigh": "xhigh",
    "ultra": "xhigh",
    "xhigh": "xhigh",
    "超高": "xhigh",
}


@dataclass(frozen=True)
class ScriptReviewResult:
    script: str
    report: dict[str, Any]


class _WebStoryReviewError(RuntimeError):
    """A web review failed after producing provider diagnostics."""

    def __init__(self, message: str, attempt_log: str):
        super().__init__(message)
        self.attempt_log = attempt_log


def _target_recovery_poll_delays() -> tuple[float, ...]:
    """Give an owned web-research turn extra time to settle after ask returns."""
    remaining = min(
        _TARGET_RECOVERY_MAX_TOTAL_SECONDS,
        max(
            _RECOVERY_POLL_INTERVAL_SECONDS,
            float(config.DAILY_NEWS_WEB_REVIEW_TIMEOUT) * 2,
        ),
    )
    delay = _RECOVERY_POLL_INTERVAL_SECONDS
    delays: list[float] = []
    while remaining > 0:
        current = min(delay, remaining)
        delays.append(current)
        remaining -= current
        delay = min(delay * 2, _TARGET_RECOVERY_MAX_POLL_INTERVAL_SECONDS)
    return tuple(delays)


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


def _observed_chatgpt_review_level(value: str) -> str | None:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    key = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", normalized)
    return _CHATGPT_OBSERVED_LEVELS.get(key)


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
            f"Spoken provenance: {source_language_guidance(article)}",
            f"Original URL: {article.url}",
            f"Published: {article.published_at}",
            f"Feed summary lead: {(article.summary[:350] if article.summary else 'none')}",
            f"Content kind: {article.content_kind}; evidence access: {article.evidence_status}",
            "Claim-relevant article evidence:",
            excerpt,
        ]
    )


def _full_article_evidence(article, index: int) -> str:
    """Return the complete locally captured evidence for final web review."""
    evidence = str(article.evidence_text or "").strip()
    return "\n".join(
        [
            f"STORY {index}",
            f"Title: {article.title}",
            f"Source: {article.source_name}",
            f"Spoken provenance: {source_language_guidance(article)}",
            f"Original URL: {article.url}",
            f"Published: {article.published_at}",
            f"Content kind: {article.content_kind}; evidence access: {article.evidence_status}",
            ("Institutional analysis: verify the attributed author's thesis, investor perspective and publication date. A recent essay must not be presented as today's breaking news or independent fact."
             if article.content_kind == "analysis" else "Reporting: verify source attribution and factual support."),
            f"Feed summary lead: {(article.summary or 'none')}",
            "Locally captured evidence excerpt (may be feed-only, not the complete article):",
            evidence or article.summary or article.title,
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


def _claim_catalog(script: str, dossier: ResearchDossier) -> dict[int, dict[str, str]]:
    """Give every reviewable sentence a stable story-scoped claim identifier."""
    matched = _matched_script_claims(script, dossier)
    catalog: dict[int, dict[str, str]] = {}
    for story_number in range(1, len(dossier.selected) + 1):
        paragraph = matched.get(story_number, "").strip()
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?。！？])\s+", paragraph)
            if sentence.strip()
        ]
        if not sentences:
            catalog[story_number] = {f"{story_number}.0": "NO MATCHED COVERAGE"}
            continue
        catalog[story_number] = {
            f"{story_number}.{claim_number}": sentence
            for claim_number, sentence in enumerate(sentences, 1)
        }
    return catalog


def _numbered_claims(
    claim_catalog: dict[int, dict[str, str]],
    story_numbers: list[int],
) -> str:
    return "\n".join(
        f"CLAIM {claim_id}: {claim_text}"
        for story_number in story_numbers
        for claim_id, claim_text in claim_catalog[story_number].items()
    )


def _mandatory_web_review_rules() -> str:
    return """MANDATORY LIVE-WEB VERIFICATION
You MUST use live internet search before returning any judgment. Open the supplied original URL when accessible, search the exact title and disputed claims, and corroborate material facts with current primary or credible independent sources. Do not judge from the supplied dossier alone and do not rely on memory. Treat search snippets as leads, not proof. If live web search is unavailable or you did not actually perform it, reply with exactly N instead of a verdict. A verdict beginning with W certifies that live web search was performed."""


def _review_prompt(
    script: str,
    dossier: ResearchDossier,
    edition_date: date,
    story_index: int,
) -> str:
    article = dossier.selected[story_index - 1]
    claims = _claim_catalog(script, dossier)[story_index]
    return f"""You are the independent final fact-checker for STORY {story_index} in a technology broadcast dated {edition_date.isoformat()}.
Check whether the full script accurately covers this selected story and whether every script claim about it is supported by the story evidence. Missing coverage, wrong names, wrong numbers, stale framing, unsupported extrapolation, or company claims stated as independent fact must fail.

{ATTRIBUTION_REVIEW_RULES}

{_mandatory_web_review_rules()}

Reply with exactly ONE ASCII token beginning with W:
- W{story_index}P when every claim is supported and the selected story is covered.
- Otherwise write W{story_index}, every applicable error code in alphabetical order, @, and the exact failed claim IDs separated by commas:
  A missing story coverage; B wrong name/entity; C wrong number/date; D unsupported extrapolation;
  E missing attribution/uncertainty; F contradiction or stale framing.
Use claim {story_index}.0 only for missing coverage. Do not return a failure without claim IDs.
Unrelated format example for story 91: W91D@91.3

Do not emit explanations, citations, URLs, JSON, markdown, or spaces.

NUMBERED SCRIPT CLAIMS
{_numbered_claims({story_index: claims}, [story_index])}

FULL STORY EVIDENCE
{_full_article_evidence(article, story_index)}
"""


def _batch_review_prompt(
    script: str,
    dossier: ResearchDossier,
    edition_date: date,
    story_numbers: list[int] | None = None,
) -> str:
    numbers = story_numbers or list(range(1, len(dossier.selected) + 1))
    evidence = "\n\n".join(
        _full_article_evidence(dossier.selected[index - 1], index)
        for index in numbers
    )
    claims = _claim_catalog(script, dossier)
    return f"""You are the independent final fact-checker for stories {numbers} in a technology broadcast dated {edition_date.isoformat()}.
For EACH numbered story, check whether the full script covers it accurately and whether every related claim is supported by that story's evidence. Missing coverage, wrong names, wrong numbers, stale framing, unsupported extrapolation, or company claims stated as independent fact must fail.

{ATTRIBUTION_REVIEW_RULES}

{_mandatory_web_review_rules()}

Reply with exactly ONE ASCII token beginning with W. For each story in order, write its number followed by P, or by all applicable error codes in alphabetical order plus @ and the exact failed claim IDs separated by commas. Separate story verdicts with semicolons:
A missing story coverage; B wrong name/entity; C wrong number/date; D unsupported extrapolation; E missing attribution/uncertainty; F contradiction or stale framing.
Use claim <story>.0 only for missing coverage. Do not return a failure without claim IDs.
Unrelated format example for stories 91 and 92: W91P;92D@92.3

Do not emit explanations, citations, URLs, JSON, markdown, or spaces.

NUMBERED SCRIPT CLAIMS
{_numbered_claims(claims, numbers)}

FULL STORY EVIDENCE
{evidence}
"""


def _single_line_payload(
    text: str,
    story_number: int,
    *,
    claim_ids: list[str] | None = None,
    claim_texts: list[str] | None = None,
) -> dict[str, Any]:
    clean = _normalized_review_token(text)
    if clean == "P":
        return {
            "approved": True,
            "confidence": 100,
            "summary": f"Story {story_number} passed.",
            "issues": [],
            "corrected_script": "",
        }
    if (
        not re.fullmatch(r"[A-F]{1,6}", clean)
        or len(set(clean)) != len(clean)
        or clean != "".join(sorted(clean))
    ):
        raise ValueError("review response did not contain the complete single-line protocol")
    directives = {
        "A": "Add concise coverage of this selected story using only its evidence.",
        "B": "Correct every name and entity for this story to match the evidence. If dossier fields conflict on a spelling, omit that disputed proper name rather than choosing one.",
        "C": "Correct or remove every unsupported number and date for this story.",
        "D": "Remove or rewrite only the cited unsupported or extrapolative sentence or claim; preserve the story's other directly evidenced reporting.",
        "E": "Attribute company or source claims and preserve uncertainty language.",
        "F": "Remove contradictions and stale framing; align the story strictly to the dated evidence.",
    }
    codes = clean
    correction = " ".join(directives[code] for code in codes)
    cited_ids = list(claim_ids or [])
    cited_texts = list(claim_texts or [])
    claim_suffix = f" at claims {','.join(cited_ids)}" if cited_ids else ""
    return {
        "approved": False,
        "confidence": 100,
        "summary": f"Story {story_number} needs correction.",
        "issues": [
            {
                "severity": "blocking",
                "claim": f"Web audit codes {codes} for story {story_number}{claim_suffix}",
                "verdict": correction,
                "evidence_story_numbers": [story_number],
                "claim_ids": cited_ids,
                "claim_texts": cited_texts,
                "correction": correction,
            }
        ],
        "corrected_script": "",
    }


def _batch_payload(
    text: str,
    story_numbers: int | list[int],
    *,
    claim_catalog: dict[int, dict[str, str]] | None = None,
    require_web: bool = False,
) -> dict[str, Any]:
    expected = (
        list(range(1, story_numbers + 1))
        if isinstance(story_numbers, int)
        else story_numbers
    )
    clean = _normalized_review_token(text)
    if clean == "N":
        raise ValueError("reviewer reported that mandatory live web search was unavailable")
    web_searched = clean.startswith("W")
    if require_web and not web_searched:
        raise ValueError("review response did not confirm mandatory live web search")
    if web_searched:
        clean = clean[1:]
    segment_pattern = re.compile(
        r"(\d+)(P|[A-F]{1,6})(?:@((?:\d+\.\d+)(?:,\d+\.\d+)*))?"
    )

    def legacy_segments(offset: int, expected_index: int) -> list[tuple[int, str, list[str]]] | None:
        """Parse the pre-semicolon protocol without confusing `1.1` + `2E` for `1.12`.

        Story numbers provide the only safe boundary in legacy concatenated
        tokens. Backtracking is bounded by the configured story count and the
        tiny response token, and unknown claim IDs are still rejected below.
        """
        if expected_index == len(expected):
            return [] if offset == len(clean) else None
        story_number = expected[expected_index]
        story_prefix = str(story_number)
        if not clean.startswith(story_prefix, offset):
            return None
        cursor = offset + len(story_prefix)
        if cursor >= len(clean):
            return None
        if clean[cursor] == "P":
            remainder = legacy_segments(cursor + 1, expected_index + 1)
            return (
                [(story_number, "P", []), *remainder]
                if remainder is not None
                else None
            )
        code_match = re.match(r"([A-F]{1,6})@", clean[cursor:])
        if not code_match:
            return None
        codes = code_match.group(1)
        claims_start = cursor + code_match.end()
        if expected_index + 1 == len(expected):
            boundaries = [len(clean)]
        else:
            next_prefix = str(expected[expected_index + 1])
            boundaries = [
                boundary
                for boundary in range(claims_start + 1, len(clean))
                if clean.startswith(next_prefix, boundary)
            ]
        for boundary in boundaries:
            claim_text = clean[claims_start:boundary]
            if not re.fullmatch(r"(?:\d+\.\d+)(?:,\d+\.\d+)*", claim_text):
                continue
            remainder = legacy_segments(boundary, expected_index + 1)
            if remainder is not None:
                return [
                    (story_number, codes, claim_text.split(",")),
                    *remainder,
                ]
        return None

    if ";" in clean:
        raw_segments = clean.split(";")
        matches = [segment_pattern.fullmatch(segment) for segment in raw_segments]
        if any(match is None for match in matches):
            raise ValueError(
                "batch review response is incomplete or malformed "
                "(expected every story verdict and complete claim IDs)"
            )
        parsed_segments = [
            (
                int(match.group(1)),
                match.group(2),
                match.group(3).split(",") if match.group(3) else [],
            )
            for match in matches
            if match is not None
        ]
    else:
        parsed_segments = legacy_segments(0, 0) or []
        if not parsed_segments and claim_catalog is None:
            # Compatibility for pre-claim-ID reports and their parser tests.
            legacy_matches = list(re.finditer(r"(\d+)(P|[A-F]{1,6})", clean))
            if "".join(match.group(0) for match in legacy_matches) == clean:
                parsed_segments = [
                    (int(match.group(1)), match.group(2), [])
                    for match in legacy_matches
                ]
        if not parsed_segments:
            raise ValueError(
                "batch review response is incomplete or malformed "
                "(expected every story verdict and complete claim IDs)"
            )
    indices = [story_number for story_number, _codes, _ids in parsed_segments]
    if indices != expected:
        raise ValueError("batch review response omitted or reordered a story")
    story_payloads: list[dict[str, Any]] = []
    for story_number, codes, referenced_ids in parsed_segments:
        if codes == "P" and referenced_ids:
            raise ValueError("passing story review unexpectedly cited failed claims")
        if codes != "P" and claim_catalog is not None and not referenced_ids:
            raise ValueError("failed story review omitted exact claim IDs")
        known_claims = (claim_catalog or {}).get(story_number, {})
        if referenced_ids and any(claim_id not in known_claims for claim_id in referenced_ids):
            raise ValueError("story review cited an unknown or cross-story claim ID")
        if codes != "P" and "A" in codes and referenced_ids != [f"{story_number}.0"]:
            raise ValueError("missing-coverage review must cite only the story-level .0 claim")
        if codes != "P" and "A" not in codes and f"{story_number}.0" in referenced_ids:
            raise ValueError("non-coverage review cannot cite the story-level .0 claim")
        story_payloads.append(
            _single_line_payload(
                codes,
                story_number,
                claim_ids=referenced_ids,
                claim_texts=[known_claims[claim_id] for claim_id in referenced_ids],
            )
        )
    issues = [issue for payload in story_payloads for issue in payload["issues"]]
    return {
        "approved": not issues,
        "confidence": 100,
        "summary": f"{len(expected) - len(issues)}/{len(expected)} story audits passed.",
        "issues": issues,
        "corrected_script": "",
        "web_searched": web_searched,
    }


def _cached_batch_review(
    prompt_path: Path,
    response_path: Path,
    prompt: str,
    story_numbers: list[int],
    claim_catalog: dict[int, dict[str, str]],
) -> tuple[dict[str, Any], str, str, str] | None:
    """Reuse only a persisted verdict proven to belong to the exact prompt.

    Browser review is deliberately rate-limited and each completed group is
    already written to disk before later correction calls run. A provider
    failure after those writes must not force the same live-web searches to run
    again. The byte-for-byte prompt match prevents a verdict from being reused
    after any script, evidence, date, story order, or claim-ID change; the
    normal strict protocol parser then validates the saved final token again.
    """
    try:
        if prompt_path.read_text(encoding="utf-8") != prompt:
            return None
        raw = response_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None

    for line in reversed(raw.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            payload = _batch_payload(
                candidate,
                story_numbers,
                claim_catalog=claim_catalog,
                require_web=True,
            )
        except ValueError:
            continue
        if not re.search(
            r"\[CHATGPT (?:\d+|FALLBACK|RECOVERY)\]",
            raw,
            flags=re.I,
        ):
            # Fact check is ChatGPT-only. Do not let an exact-prompt Gemini
            # verdict from an older or interrupted run bypass the new route.
            return None
        return payload, raw, "", "chatgpt"
    return None


def _saved_full_review_candidate(
    output_dir: Path,
    dossier: ResearchDossier,
    edition_date: date,
) -> str | None:
    """Recover the exact candidate already used for a complete first audit.

    The candidate is accepted only when every two-story prompt can be rebuilt
    byte-for-byte from current dossier data and every matching saved response
    still passes the strict claim protocol. This closes the recovery gap before
    ``_cached_batch_review``: starting again from the model-written draft could
    produce a different duration edit and invalidate otherwise reusable reviews.
    """
    review_dir = output_dir / "review"
    candidate_path = review_dir / "candidate-duration-cycle-1.txt"
    try:
        candidate = candidate_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    claims = _claim_catalog(candidate, dossier)
    all_numbers = list(range(1, len(dossier.selected) + 1))
    for group_index, start in enumerate(range(0, len(all_numbers), 2), 1):
        story_numbers = all_numbers[start : start + 2]
        prompt = _batch_review_prompt(
            candidate,
            dossier,
            edition_date,
            story_numbers,
        )
        cached = _cached_batch_review(
            review_dir / f"story-review-prompt-1-group-{group_index}.txt",
            review_dir / f"story-review-response-1-group-{group_index}.txt",
            prompt,
            story_numbers,
            {number: claims[number] for number in story_numbers},
        )
        if cached is None:
            return None
    return candidate


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
    claim_catalog: dict[int, dict[str, str]] | None = None,
    site_session_namespace: str | None = None,
    log: LogCallback | None,
) -> tuple[dict[str, Any], str, str, str]:
    retry_sessions: list[str] = []
    try:
        return await _web_story_review_in_session(
            prompt,
            story_number=story_number,
            story_numbers=story_numbers,
            claim_catalog=claim_catalog,
            site_session_namespace=site_session_namespace,
            log=log,
            retry_sessions=retry_sessions,
        )
    finally:
        for namespace in retry_sessions:
            try:
                await close_opencli_site_sessions(namespace, sites=("chatgpt",))
            except Exception as exc:  # cleanup must preserve the review outcome
                logging.getLogger(__name__).warning(
                    "OpenCLI model-recovery session cleanup failed: %s", exc
                )


async def _web_story_review_in_session(
    prompt: str,
    *,
    story_number: int | None = None,
    story_numbers: list[int] | None = None,
    claim_catalog: dict[int, dict[str, str]] | None = None,
    site_session_namespace: str | None = None,
    log: LogCallback | None,
    retry_sessions: list[str],
) -> tuple[dict[str, Any], str, str, str]:
    if (story_number is None) == (story_numbers is None):
        raise ValueError("exactly one of story_number or story_numbers is required")
    review_label = f"story {story_number}" if story_number is not None else f"stories {story_numbers}"

    def parse_response(value: str) -> dict[str, Any]:
        if story_numbers is not None:
            return _batch_payload(
                value,
                story_numbers,
                claim_catalog=claim_catalog,
                require_web=True,
            )
        assert story_number is not None
        clean = _normalized_review_token(value)
        if clean == "N":
            raise ValueError("reviewer reported that mandatory live web search was unavailable")
        match = re.fullmatch(
            rf"W{story_number}(P|[A-F]{{1,6}})(?:@((?:{story_number}\.\d+)(?:,{story_number}\.\d+)*))?",
            clean,
        )
        if not match:
            raise ValueError("single-story review did not confirm web search or follow the claim protocol")
        codes = match.group(1)
        referenced_ids = match.group(2).split(",") if match.group(2) else []
        known_claims = (claim_catalog or {}).get(story_number, {})
        if codes != "P" and not referenced_ids:
            raise ValueError("failed story review omitted exact claim IDs")
        if any(claim_id not in known_claims for claim_id in referenced_ids):
            raise ValueError("story review cited an unknown claim ID")
        return _single_line_payload(
            codes,
            story_number,
            claim_ids=referenced_ids,
            claim_texts=[known_claims[claim_id] for claim_id in referenced_ids],
        )

    timeout = config.DAILY_NEWS_WEB_REVIEW_TIMEOUT
    request_id = uuid4().hex
    review_session_namespace = site_session_namespace or (
        f"{config.OPENCLI_SITE_SESSION_NAMESPACE}-review-{request_id}"
    )
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
        after_rate_limit: bool = False,
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
                "foreground",
                "--site-session",
                "persistent",
                "--keep-tab",
                "true",
                "-f",
                "json",
            ]
        )
        # ChatGPT can expose a transient assistant snapshot while live-web
        # research and citation hydration are still settling. When ask gives us
        # the exact conversation URL, keep polling that owned conversation for
        # up to two review-timeout windows instead of submitting the
        # expensive prompt again. Active-page recovery has no such immutable
        # target, so retain the short bounded poll there to avoid accepting an
        # unrelated turn.
        poll_delays = (
            _target_recovery_poll_delays()
            if provider == "chatgpt" and conversation_url
            else (_RECOVERY_POLL_INTERVAL_SECONDS,) * 2
        )
        recovery_attempts = len(poll_delays) + 1
        last_error: Exception | None = None
        previous_settled_invalid = ""
        refresh_pending = False
        refresh_used = False
        for recovery_attempt in range(1, recovery_attempts + 1):
            assistant_text = ""
            messages: list[dict[str, Any]] = []
            try:
                read_command = list(command)
                if after_rate_limit and conversation_url:
                    # The transport waits out the shared breaker first. Then
                    # discard a still-visible old limit dialog on this target.
                    read_command.extend(["--cooldown", "true"])
                    after_rate_limit = False
                if refresh_pending:
                    read_command.extend(["--refresh", "true"])
                    refresh_pending = False
                    refresh_used = True
                    raw_attempts.append(
                        f"[CHATGPT SETTLED TARGET REFRESH]\n{conversation_url}"
                    )
                    if log:
                        log(
                            f"ChatGPT {review_label}: reloading the settled owned "
                            "conversation to recover stale partial text"
                        )
                read_result = await run_opencli(
                    read_command,
                    timeout=timeout + 15 if conversation_url else 30,
                    site_session_namespace=review_session_namespace,
                )
                messages = _rows(first_json(read_result.stdout))
                assistant_text = _assistant_for_review_request(messages, request_id)
                if not assistant_text:
                    raise ValueError(
                        f"{provider} recovery page did not contain a response "
                        "owned by this request"
                    )
                if any(_field(row, "Generating").casefold() == "true" for row in messages):
                    raise ValueError("owned review response is still generating")
                payload = parse_response(assistant_text)
                raw_attempts.append(
                    f"[{provider.upper()} RECOVERY]\n{assistant_text}"
                )
                return payload, assistant_text
            except Exception as exc:  # noqa: BLE001 - bounded late-response poll
                last_error = exc
                if isinstance(exc, OpenCLIRateLimitError):
                    after_rate_limit = True
                if assistant_text:
                    raw_attempts.append(
                        f"[{provider.upper()} RECOVERY INVALID "
                        f"{recovery_attempt}/{recovery_attempts}]\n{assistant_text}"
                    )
                # Re-reading a frozen DOM cannot recover its server-saved final
                # answer. Require two identical, explicitly settled owned reads
                # before one refresh; missing metadata never authorizes a reload.
                settled = bool(messages) and all(
                    any(
                        str(key).casefold() == "generating"
                        and (value is False or str(value).casefold() == "false")
                        for key, value in row.items()
                    )
                    for row in messages
                )
                if conversation_url and assistant_text and settled and not refresh_used:
                    refresh_pending = assistant_text == previous_settled_invalid
                    previous_settled_invalid = assistant_text
                else:
                    previous_settled_invalid = ""
                if recovery_attempt < recovery_attempts:
                    delay = poll_delays[recovery_attempt - 1]
                    if log and conversation_url:
                        log(
                            f"ChatGPT {review_label} response is not yet a valid "
                            "final protocol token; rereading its target conversation "
                            f"after {delay:.0f}s ({recovery_attempt}/"
                            f"{recovery_attempts - 1})"
                        )
                    await asyncio.sleep(delay)
        assert last_error is not None
        raise last_error

    chatgpt_model_policy_error: Exception | None = None
    chatgpt_model_policy_attempts = 3
    chatgpt_model_rate_limit_attempts = 3
    policy_attempt = 0
    rate_limit_attempt = 0
    rate_limit_exhausted = False
    observed_chatgpt_model = ""
    chatgpt_model_policy_status = ""
    while policy_attempt < chatgpt_model_policy_attempts:
        try:
            policy_result = await run_opencli(
                [
                    "chatgpt",
                    "model",
                    config.DAILY_NEWS_CHATGPT_REVIEW_MIN_LEVEL,
                    "--timeout",
                    "45",
                    "--window",
                    "foreground",
                    "--site-session",
                    "persistent",
                    "--keep-tab",
                    "true",
                    "-f",
                    "json",
                ],
                timeout=75,
                site_session_namespace=review_session_namespace,
            )
            policy_rows = _rows(first_json(policy_result.stdout))
            observed_chatgpt_model = next(
                (
                    _field(row, "Model")
                    for row in policy_rows
                    if _field(row, "Model")
                ),
                "",
            )
            if not observed_chatgpt_model:
                raise ValueError("ChatGPT model policy returned no observed level")
            observed_chatgpt_level = _observed_chatgpt_review_level(
                observed_chatgpt_model
            )
            minimum_order = _CHATGPT_REVIEW_LEVEL_ORDER[
                config.DAILY_NEWS_CHATGPT_REVIEW_MIN_LEVEL
            ]
            maximum_order = _CHATGPT_REVIEW_LEVEL_ORDER[
                config.DAILY_NEWS_CHATGPT_REVIEW_MAX_LEVEL
            ]
            observed_order = _CHATGPT_REVIEW_LEVEL_ORDER.get(
                observed_chatgpt_level or ""
            )
            if (
                observed_order is None
                or observed_order < minimum_order
                or observed_order > maximum_order
            ):
                raise ValueError(
                    f"ChatGPT current model {observed_chatgpt_model} is outside "
                    f"allowed range {config.DAILY_NEWS_CHATGPT_REVIEW_MIN_LEVEL}.."
                    f"{config.DAILY_NEWS_CHATGPT_REVIEW_MAX_LEVEL}"
                )
            chatgpt_model_policy_status = next(
                (
                    _field(row, "Status")
                    for row in policy_rows
                    if _field(row, "Status")
                ),
                "",
            )
            raw_attempts.append(
                "[CHATGPT MODEL POLICY]\n"
                f"Allowed: {config.DAILY_NEWS_CHATGPT_REVIEW_MIN_LEVEL}.."
                f"{config.DAILY_NEWS_CHATGPT_REVIEW_MAX_LEVEL}\n"
                f"Status: {chatgpt_model_policy_status or 'verified'}\n"
                f"Observed: {observed_chatgpt_model}"
            )
            chatgpt_model_policy_error = None
            if log:
                log(
                    "ChatGPT fact-check current model policy passed: "
                    f"{chatgpt_model_policy_status or 'verified'} at "
                    f"{observed_chatgpt_model}, within "
                    f"{config.DAILY_NEWS_CHATGPT_REVIEW_MIN_LEVEL}.."
                    f"{config.DAILY_NEWS_CHATGPT_REVIEW_MAX_LEVEL}; "
                    "continuing after the preferred-model switch attempt"
                )
            break
        except OpenCLIRateLimitError as exc:
            # The transport has already opened the shared provider breaker.
            # Do not spend a model-picker hydration attempt on a site-wide
            # access limit: the next run_opencli call waits outside its command
            # timeout and retries only after the persisted cooldown.
            chatgpt_model_policy_error = exc
            rate_limit_attempt += 1
            raw_attempts.append(
                f"[CHATGPT MODEL POLICY RATE LIMIT {rate_limit_attempt}/"
                f"{chatgpt_model_rate_limit_attempts}]\n{exc}"
            )
            if rate_limit_attempt >= chatgpt_model_rate_limit_attempts:
                rate_limit_exhausted = True
                break
            if log:
                log(
                    "ChatGPT fact-check model policy hit the provider access "
                    f"limit ({rate_limit_attempt}/"
                    f"{chatgpt_model_rate_limit_attempts}); preserving all "
                    "model-policy retries and waiting for the shared cooldown "
                    f"before retrying: {exc}"
                )
            continue
        except Exception as exc:  # noqa: BLE001 - bounded model preflight recovery
            chatgpt_model_policy_error = exc
            policy_attempt += 1
            raw_attempts.append(
                f"[CHATGPT MODEL POLICY ERROR {policy_attempt}/"
                f"{chatgpt_model_policy_attempts}]\n{exc}"
            )
            if policy_attempt < chatgpt_model_policy_attempts and log:
                log(
                    "ChatGPT fact-check model policy check attempt "
                    f"{policy_attempt}/{chatgpt_model_policy_attempts} failed; "
                    f"retrying: {exc}"
                )
            if policy_attempt < chatgpt_model_policy_attempts:
                if any(marker in str(exc).lower() for marker in (
                    "stale page identity", "page not found:", "timed out",
                    "session_busy", "timeout",
                )):
                    # A killed CLI can leave both a daemon lease and a browser
                    # operation alive. Do not force-unlock or drive that tab on
                    # retry. No audit prompt has been sent during model preflight,
                    # so a fresh owned session is safe here (never replay ask).
                    if review_session_namespace not in retry_sessions:
                        retry_sessions.append(review_session_namespace)
                    review_session_namespace = f"frontiertechsn-model-{uuid4().hex}"
                    retry_sessions.append(review_session_namespace)
                    if log:
                        log("ChatGPT model preflight will retry in a fresh owned browser session")
                await asyncio.sleep(_MODEL_SELECTION_RETRY_DELAY_SECONDS)

    if chatgpt_model_policy_error is not None:
        failed_after = (
            f"{chatgpt_model_rate_limit_attempts} access-limit cooldowns"
            if rate_limit_exhausted
            else f"{chatgpt_model_policy_attempts} attempts"
        )
        failure_message = (
            "ChatGPT fact-check current model could not be verified within the "
            f"allowed {config.DAILY_NEWS_CHATGPT_REVIEW_MIN_LEVEL}.."
            f"{config.DAILY_NEWS_CHATGPT_REVIEW_MAX_LEVEL} range after {failed_after} "
            f"({chatgpt_model_policy_error})"
        )
        # Thinking effort is a preference after bounded attempts. Transport,
        # login, and provider access failures still cannot authorize submission.
        preference_failure = isinstance(chatgpt_model_policy_error, ValueError) or any(
            marker in str(chatgpt_model_policy_error).lower()
            for marker in (
                "chatgpt model selector", "chatgpt model did not switch",
                "outside allowed range", "chatgpt did not expose a usable",
                "could not click the chatgpt",
            )
        )
        if rate_limit_exhausted or not preference_failure:
            raise RuntimeError(failure_message) from chatgpt_model_policy_error
        fallback_message = (
            f"{failure_message}; continuing fact check with the current model "
            f"(last observed: {observed_chatgpt_model or 'unverified'}). "
            "Thinking-effort preference was not verified; claim review remains required."
        )
        raw_attempts.append(f"[CHATGPT MODEL PREFERENCE FALLBACK]\n{fallback_message}")
        if log:
            log(fallback_message)

    chatgpt_error: Exception | None = None
    chatgpt_attempts = 2
    for chatgpt_attempt in range(1, chatgpt_attempts + 1):
        conversation_url = ""
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
                "foreground",
                "--site-session",
                "persistent",
                "--keep-tab",
                "true",
                "-f",
                "json",
            ]
            result = await run_opencli(
                chatgpt_args,
                timeout=timeout + 20,
                site_session_namespace=review_session_namespace,
            )
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
            raw_attempts.append(
                f"[CHATGPT {chatgpt_attempt}]\n{response or '[EMPTY]'}"
            )
            if conversation_url:
                raw_attempts.append(f"[CHATGPT CONVERSATION]\n{conversation_url}")
            if not response or "[NO RESPONSE]" in response.upper():
                raise ValueError("ChatGPT ask returned no completed assistant response")
            payload = parse_response(response)
            if log:
                log(f"ChatGPT completed {review_label} fact check")
            return (
                payload,
                "\n\n--- PROVIDER ATTEMPT ---\n\n".join(raw_attempts),
                conversation_url,
                "chatgpt",
            )
        except Exception as exc:  # noqa: BLE001 - target-specific recovery below
            chatgpt_error = exc
            if isinstance(exc, OpenCLIRateLimitError) and log:
                log(
                    "ChatGPT has limited conversation access. All project browser "
                    "operations will wait for the shared cooldown before recovering "
                    "this same owned turn; no prompt will be resubmitted now."
                )
            target_match = _CHATGPT_CONVERSATION_URL_RE.search(str(exc))
            target_url = conversation_url or (target_match.group(0) if target_match else "")
            try:
                recovered = await recover_current(
                    "chatgpt",
                    conversation_url=target_url,
                    after_rate_limit=isinstance(exc, OpenCLIRateLimitError),
                )
                payload, _ = recovered
                if log:
                    recovery_source = "target conversation" if target_url else "owned browser turn"
                    log(
                        f"Recovered completed ChatGPT {review_label} fact check from "
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
            if chatgpt_attempt < chatgpt_attempts and log:
                log(
                    f"ChatGPT {review_label} fact-check attempt {chatgpt_attempt}/"
                    f"{chatgpt_attempts} failed; starting the bounded retry: "
                    f"{chatgpt_error}"
                )

    raise _WebStoryReviewError(
        "ChatGPT current-model "
        f"{config.DAILY_NEWS_CHATGPT_REVIEW_MIN_LEVEL}.."
        f"{config.DAILY_NEWS_CHATGPT_REVIEW_MAX_LEVEL} {review_label} fact check "
        f"failed after {chatgpt_attempts} attempts ({chatgpt_error})",
        "\n\n--- PROVIDER ATTEMPT ---\n\n".join(raw_attempts),
    ) from chatgpt_error


async def _review_daily_script(
    script: str,
    dossier: ResearchDossier,
    edition_date: date,
    output_dir: Path,
    *,
    language: str,
    closing_remarks: str,
    opening_remarks: str | None = None,
    ai_endpoint: str | None,
    ai_model: str | None,
    provider_id: int | None,
    review_session_namespace: str,
    target_duration_minutes: int | None = None,
    log: LogCallback | None = None,
) -> ScriptReviewResult:
    review_dir = output_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    candidate = _saved_full_review_candidate(output_dir, dossier, edition_date) or script
    if candidate != script and log:
        log(
            "Review resume: recovered the exact saved duration candidate and "
            "its complete prompt-matched first-round Web reviews"
        )
    protected_story_numbers: set[int] = set()
    max_corrections = 3
    correction_count = 0
    audit_round = 0
    repeated_issue_counts: dict[tuple, int] = {}
    failure_reason = ""
    all_numbers = list(range(1, len(dossier.selected) + 1))
    review_numbers = list(all_numbers)
    while audit_round < max_corrections * 2 + 1:
        audit_round += 1
        length_contract = None
        if target_duration_minutes is not None:
            prior_candidate = candidate
            candidate, length_contract = await fit_daily_script_duration(
                candidate,
                dossier,
                edition_date,
                target_duration_minutes=target_duration_minutes,
                language=language,
                closing_remarks=closing_remarks,
                opening_remarks=opening_remarks,
                ai_endpoint=ai_endpoint,
                ai_model=ai_model,
                provider_id=provider_id,
                protected_story_numbers=set(protected_story_numbers),
                log=log,
            )
            if candidate != prior_candidate:
                (review_dir / f"candidate-duration-cycle-{audit_round}.txt").write_text(
                    candidate,
                    encoding="utf-8",
                )
        # Each two-story group owns a fresh ChatGPT turn instead of reusing
        # mutable active-page state.
        contract = script_contract_report(
            candidate,
            dossier,
            edition_date,
            language=language,
            closing_remarks=closing_remarks,
            opening_remarks=opening_remarks,
            target_duration_minutes=target_duration_minutes,
        )
        if not contract["passed"]:
            raise RuntimeError(
                "Deterministic script audit failed before web review: "
                + "; ".join(contract["failures"])
            )
        claims = _claim_catalog(candidate, dossier)
        group_results: list[dict[str, Any]] = []
        for group_index, start in enumerate(range(0, len(review_numbers), 2), 1):
            story_numbers = review_numbers[start : start + 2]
            prompt = _batch_review_prompt(candidate, dossier, edition_date, story_numbers)
            prompt_path = review_dir / f"story-review-prompt-{audit_round}-group-{group_index}.txt"
            response_path = review_dir / f"story-review-response-{audit_round}-group-{group_index}.txt"
            cached = _cached_batch_review(
                prompt_path,
                response_path,
                prompt,
                story_numbers,
                {number: claims[number] for number in story_numbers},
            )
            if cached is not None:
                group_payload, raw, conversation_url, provider = cached
                if log:
                    log(
                        f"Reusing exact prompt-matched saved Web review for stories "
                        f"{story_numbers}"
                    )
            else:
                prompt_path.write_text(prompt, encoding="utf-8")
                try:
                    group_payload, raw, conversation_url, provider = await _web_story_review(
                        prompt,
                        story_numbers=story_numbers,
                        claim_catalog={number: claims[number] for number in story_numbers},
                        site_session_namespace=review_session_namespace,
                        log=log,
                    )
                except _WebStoryReviewError as exc:
                    failure_path = review_dir / (
                        f"story-review-failure-{audit_round}-group-{group_index}.txt"
                    )
                    failure_path.write_text(exc.attempt_log, encoding="utf-8")
                    raise
                response_path.write_text(raw, encoding="utf-8")
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
        failed_story_numbers = sorted({
            int(number)
            for issue in issues
            for number in issue.get("evidence_story_numbers", [])
            if str(number).isdigit()
        })
        payload = {
            "approved": not issues,
            "confidence": 100,
            "summary": (
                f"{len(review_numbers) - len(failed_story_numbers)}/"
                f"{len(review_numbers)} story audits passed."
            ),
            "issues": issues,
        }
        conversation_url = ", ".join(
            result["conversation_url"]
            for result in group_results
            if result["conversation_url"]
        )
        providers = [result["provider"] for result in group_results]
        reviewer = (
            "ChatGPT Web (current model constrained to "
            f"{config.DAILY_NEWS_CHATGPT_REVIEW_MIN_LEVEL}.."
            f"{config.DAILY_NEWS_CHATGPT_REVIEW_MAX_LEVEL}) "
            "via project-local OpenCLI"
        )
        blocking = [
            issue
            for issue in payload.get("issues", [])
            if isinstance(issue, dict) and str(issue.get("severity", "")).casefold() == "blocking"
        ]
        approved = bool(payload.get("approved")) and not blocking
        attempt = {
            "cycle": audit_round,
            "scope": "full" if review_numbers == all_numbers else "targeted",
            "story_numbers": list(review_numbers),
            "reviewer": reviewer,
            "providers": providers,
            "conversation_url": conversation_url,
            "approved": approved,
            "confidence": payload.get("confidence"),
            "summary": payload.get("summary", ""),
            "issues": payload.get("issues", []),
            "contract": contract,
            "duration_contract": length_contract,
        }
        attempts.append(attempt)
        if log:
            log(
                f"Web accuracy review cycle {audit_round} "
                f"({'full' if review_numbers == all_numbers else 'targeted'}): "
                f"{'approved' if approved else f'{len(blocking)} blocking issue(s)'}"
            )
        if approved:
            if review_numbers != all_numbers:
                if log:
                    log(
                        "Targeted correction review passed; running one final full-story "
                        "web review before approval"
                    )
                review_numbers = list(all_numbers)
                continue
            report = {
                "passed": True,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "reviewer": reviewer,
                "fallback_used": False,
                "attempts": attempts,
                "final_contract": contract,
                "correction_count": correction_count,
            }
            (review_dir / "fact_check_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            (review_dir / "approved_script.txt").write_text(candidate, encoding="utf-8")
            return ScriptReviewResult(candidate, report)

        issue_signature = tuple(sorted(
            (
                tuple(issue.get("evidence_story_numbers", [])),
                str(issue.get("claim", "")),
                tuple(issue.get("claim_ids", [])),
            )
            for issue in blocking
        ))
        repeated_issue_counts[issue_signature] = repeated_issue_counts.get(issue_signature, 0) + 1
        if repeated_issue_counts[issue_signature] >= 2:
            failure_reason = (
                "The same claim-level blocking verdict recurred after correction; "
                "automatic rewriting stopped to prevent a review loop."
            )
            break
        if correction_count >= max_corrections:
            failure_reason = (
                f"The script still had blocking claim-level findings after "
                f"{max_corrections} bounded correction passes."
            )
            break
        # A duration edit must not silently rewrite any paragraph that just
        # received a claim-level factual correction, regardless of error code.
        protected_story_numbers.update(failed_story_numbers)
        candidate = await revise_daily_script(
            candidate,
            dossier,
            payload.get("issues", []),
            edition_date,
            language=language,
            closing_remarks=closing_remarks,
            opening_remarks=opening_remarks,
            ai_endpoint=ai_endpoint,
            ai_model=ai_model,
            provider_id=provider_id,
            target_duration_minutes=target_duration_minutes,
            log=log,
        )
        correction_count += 1
        (review_dir / f"candidate-cycle-{correction_count + 1}.txt").write_text(
            candidate, encoding="utf-8"
        )
        review_numbers = failed_story_numbers

    report = {
        "passed": False,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "reviewer": (
            "ChatGPT Web (current model constrained to "
            f"{config.DAILY_NEWS_CHATGPT_REVIEW_MIN_LEVEL}.."
            f"{config.DAILY_NEWS_CHATGPT_REVIEW_MAX_LEVEL}) "
            "via project-local OpenCLI"
        ),
        "fallback_used": False,
        "attempts": attempts,
        "correction_count": correction_count,
        "manual_review_required": True,
        "failure_reason": failure_reason or "The bounded web-review loop did not approve the script.",
    }
    (review_dir / "fact_check_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    raise RuntimeError(
        "Web accuracy review did not approve the daily script: "
        + report["failure_reason"]
    )


async def review_daily_script(
    script: str,
    dossier: ResearchDossier,
    edition_date: date,
    output_dir: Path,
    *,
    language: str,
    closing_remarks: str,
    opening_remarks: str | None = None,
    ai_endpoint: str | None,
    ai_model: str | None,
    provider_id: int | None,
    target_duration_minutes: int | None = None,
    log: LogCallback | None = None,
) -> ScriptReviewResult:
    """Review a daily script and always release its isolated browser tabs."""
    review_session_namespace = (
        f"{config.OPENCLI_SITE_SESSION_NAMESPACE}-review-{uuid4().hex}"
    )
    try:
        return await _review_daily_script(
            script,
            dossier,
            edition_date,
            output_dir,
            language=language,
            closing_remarks=closing_remarks,
            opening_remarks=opening_remarks,
            ai_endpoint=ai_endpoint,
            ai_model=ai_model,
            provider_id=provider_id,
            review_session_namespace=review_session_namespace,
            target_duration_minutes=target_duration_minutes,
            log=log,
        )
    finally:
        try:
            await close_opencli_site_sessions(review_session_namespace)
            if log:
                log("Released completed OpenCLI ChatGPT fact-check browser sessions")
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask review outcome
            message = f"OpenCLI browser-session cleanup failed: {exc}"
            if log:
                log(message)
            else:
                logging.getLogger(__name__).warning(message)

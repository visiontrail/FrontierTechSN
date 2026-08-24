from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import date

from backend.daily_news.research import ResearchDossier, dossier_markdown
from backend.pipeline.digester import _chat, _resolve_provider

LogCallback = Callable[[str], None]
NON_ENGLISH_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff\uac00-\ud7af]")
SOURCE_SPOKEN_ALIASES = {
    "机器之心 AI Daily": ("Machine Heart", "Jiqizhixin"),
    "量子位 QbitAI": ("QbitAI",),
    "DeepTech 深科技": ("DeepTech China", "MIT Technology Review China"),
    "AIBase AI 日报": ("AIBase",),
    "IT之家 AI / 智能时代": ("ITHome",),
    "极客公园": ("GeekPark",),
}
TECHMEME_CREDIT_RE = re.compile(r"^\s*([^:\n]{2,120}?)\s+:\s+")
TITLE_CREDIT_RE = re.compile(r"\(([^()]{2,120})\)\s*$")


def morning_opening(edition_date: date, language: str = "en") -> str:
    if language == "zh":
        return (
            f"早上好，今天是{edition_date.year}年{edition_date.month}月{edition_date.day}日，"
            "这里是《前沿科技早报》，用几分钟带你掌握正在塑造未来的技术进展。"
        )
    return (
        f"Good morning. It's {edition_date.strftime('%A, %B')} {edition_date.day}, "
        f"{edition_date.year}, and this is Frontier Tech Daily—your concise morning briefing "
        "on the ideas, systems, and companies shaping tomorrow."
    )


def _spoken_lines(value: str) -> list[str]:
    lines: list[str] = []
    for raw in value.replace("```", "").splitlines():
        line = raw.strip()
        line = re.sub(r"^(?:#{1,6}|[-*]\s+|\d+[.)]\s+)", "", line).strip()
        line = re.sub(r"^(?:HOST|ANCHOR|NARRATOR|Speaker\s*\d+)\s*[:：-]\s*", "", line, flags=re.I)
        if line and not re.fullmatch(r"[A-Z][A-Z\s/&-]{3,}:?", line):
            lines.append(line)
    return lines


def _techmeme_credit_names(title: str, summary: str) -> tuple[str, ...]:
    """Return the byline/outlet names carried by a Techmeme story record."""
    match = TECHMEME_CREDIT_RE.search(summary or "")
    raw_credit = match.group(1) if match else ""
    if not raw_credit:
        match = TITLE_CREDIT_RE.search(title or "")
        raw_credit = match.group(1) if match else ""
    names: list[str] = []
    for raw_name in re.split(r"\s*/\s*", raw_credit):
        name = re.sub(r"\s+", " ", raw_name).strip(" \t.,:;()[]{}")
        if (
            3 <= len(name) <= 64
            and re.search(r"[A-Za-z0-9]", name)
            and name.casefold() not in {"source", "sources", "exclusive"}
        ):
            names.append(name)
    return tuple(dict.fromkeys(names))


def _script_mentions(script_folded: str, name: str) -> bool:
    escaped = re.escape(name.casefold())
    return re.search(rf"(?<!\w){escaped}(?!\w)", script_folded) is not None


def _publication_attribution(article) -> tuple[str, str, dict[str, tuple[str, str]]]:
    """Describe one expected publication and every acceptable spoken identity.

    Techmeme is a discovery source whose title/summary carries the original
    byline and outlet.  A script may truthfully cite either the aggregator or
    that carried credit, but one spoken ``Techmeme`` mention must still count as
    only one publication across multiple aggregated stories.
    """
    source_identity = f"source:{article.source_id or article.source_name.casefold()}"
    source_aliases = (article.source_name, *SOURCE_SPOKEN_ALIASES.get(article.source_name, ()))
    accepted = {
        alias: (source_identity, article.source_name)
        for alias in source_aliases
        if alias
    }
    if article.source_id != "techmeme":
        return source_identity, article.source_name, accepted

    credit_names = _techmeme_credit_names(article.title, article.summary)
    if not credit_names:
        return source_identity, article.source_name, accepted
    outlet = credit_names[-1]
    publication_identity = f"publication:{outlet.casefold()}"
    accepted.update(
        {
            alias: (publication_identity, outlet)
            for alias in credit_names
        }
    )
    return publication_identity, outlet, accepted


def enforce_script_contract(
    script: str,
    *,
    opening: str,
    closing: str,
    language: str,
) -> str:
    lines = _spoken_lines(script)
    opening_folded = opening.casefold()
    closing_folded = closing.casefold()
    body = [
        line
        for line in lines
        if line.casefold() not in {opening_folded, closing_folded}
        and "frontier tech daily" not in line.casefold()
    ]
    if not body:
        raise RuntimeError("Daily-news script contained no spoken story body")
    final = "\n".join([opening, *body, closing]).strip()
    if language == "en" and NON_ENGLISH_RE.search(final):
        raise RuntimeError("English daily-news script contains CJK text")
    if not final.startswith(opening) or not final.endswith(closing):
        raise RuntimeError("Daily-news opening/closing contract could not be enforced")
    return final


def _minimalize_unsupported_paragraphs(script: str, issues: list[dict]) -> str:
    """Make repeated D-code corrections deterministic after the LLM rewrite."""
    story_numbers = {
        int(number)
        for issue in issues
        if re.search(r"audit codes [A-F]*D", str(issue.get("claim", "")), re.I)
        for number in issue.get("evidence_story_numbers", [])
        if str(number).isdigit()
    }
    lines = [line.strip() for line in script.splitlines() if line.strip()]
    for story_number in story_numbers:
        if not 1 <= story_number < len(lines) - 1:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", lines[story_number], maxsplit=1)
        lines[story_number] = sentences[0].strip()
    return "\n".join(lines)


def _issues_are_unsupported_only(issues: list[dict]) -> bool:
    """Return true when every reviewer directive is exactly audit code D."""
    if not issues:
        return False
    code_sets: list[set[str]] = []
    for issue in issues:
        match = re.search(
            r"audit codes?\s+([A-F]+)",
            str(issue.get("claim", "")),
            re.I,
        )
        if not match:
            return False
        code_sets.append(set(match.group(1).upper()))
    return all(codes == {"D"} for codes in code_sets)


async def generate_daily_script(
    dossier: ResearchDossier,
    edition_date: date,
    *,
    target_duration_minutes: int,
    language: str,
    closing_remarks: str,
    ai_endpoint: str | None,
    ai_model: str | None,
    provider_id: int | None,
    log: LogCallback | None = None,
) -> str:
    opening = morning_opening(edition_date, language)
    word_count = max(180, target_duration_minutes * (280 if language == "zh" else 150))
    endpoint, model, api_key = await _resolve_provider(provider_id, ai_endpoint, ai_model)
    language_label = "natural broadcast Mandarin Chinese" if language == "zh" else "natural broadcast English"
    system_prompt = f"""You are the senior anchor and evidence editor for Frontier Tech Daily.
Write a solo morning-news video podcast script in {language_label}, about {word_count} spoken words.

NON-NEGOTIABLE EDITORIAL CONTRACT
1. The first spoken line will be injected by software. Do not write a greeting, date, show name, headline list, title, markdown, labels, stage directions, citations section, or speaker prefixes.
2. Use only facts present in the supplied evidence dossier. Do not infer hidden motives, invent numbers, predict outcomes as facts, or create quotations.
3. Cover every selected story in the exact dossier order, using exactly one paragraph per selected story. Explain significance only when the dossier explicitly supports it; otherwise report the evidence without extrapolation.
4. Attribute material claims aloud to the named publication or primary organization. Preserve uncertainty words such as reported, announced, proposed, or preliminary.
5. Keep AI, chips, robotics and hard science balanced; do not turn every story into a foundation-model story.
6. Dates, model names, company names, measurements and funding figures must match the dossier exactly.
7. Use short paragraphs suitable for TTS. Output spoken prose only.
8. The closing line will be injected by software. Do not write a sign-off.
9. For an English edition, translate every Chinese headline, organization and product description into natural spoken English. Cite Chinese publications only by these English broadcast names: Machine Heart, QbitAI, DeepTech China, AIBase, ITHome, and GeekPark. Output no Chinese, Japanese or Korean characters.

Software-controlled opening (for context only; DO NOT repeat):
{opening}

Software-controlled closing (for context only; DO NOT repeat):
{closing_remarks}
"""
    user_content = dossier_markdown(dossier)
    raw = await _chat(
        system_prompt,
        user_content,
        endpoint,
        model,
        api_key,
        log,
        "Daily news script",
        max_tokens=max(4096, min(8192, word_count * 3)),
        enable_skills=False,
    )
    if language == "en" and NON_ENGLISH_RE.search(raw):
        if log:
            log("Daily script contains CJK in an English edition; requesting a broadcast-safe translation repair")
        raw = await _chat(
            """You are a broadcast copy editor. Rewrite the supplied English technology-news script so it contains no Chinese, Japanese, or Korean characters. Translate non-English names and phrases into natural spoken English without adding, removing, or changing any facts, numbers, uncertainty, story order, or attribution. Use these source aliases: Machine Heart, QbitAI, DeepTech China, AIBase, ITHome, and GeekPark. Output spoken prose only.""",
            raw,
            endpoint,
            model,
            api_key,
            log,
            "Daily news English translation repair",
            max_tokens=max(4096, min(8192, word_count * 3)),
            enable_skills=False,
        )
    script = enforce_script_contract(
        raw,
        opening=opening,
        closing=closing_remarks,
        language=language,
    )
    if log:
        log(
            f"Daily script contract: exact dated opening + {len(dossier.selected)} researched stories + exact closing"
        )
    return script


async def revise_daily_script(
    script: str,
    dossier: ResearchDossier,
    issues: list[dict],
    edition_date: date,
    *,
    language: str,
    closing_remarks: str,
    ai_endpoint: str | None,
    ai_model: str | None,
    provider_id: int | None,
    log: LogCallback | None = None,
) -> str:
    """Apply independent-review directives using the evidence-bound writing model."""
    opening = morning_opening(edition_date, language)
    # D means the paragraph already covers the selected story but extends
    # beyond its evidence. The safest correction is mechanical: retain only
    # the first attributed lead sentence and send that smaller claim back to
    # the independent reviewer. This also keeps a slow yhroot gateway from
    # blocking an otherwise deterministic edit. Names, numbers, attribution,
    # contradictions, and missing coverage still require the writing model.
    if _issues_are_unsupported_only(issues):
        revised = _minimalize_unsupported_paragraphs(script, issues)
        if revised != script:
            if log:
                log("Daily news audit correction: applied deterministic D-only evidence trim")
            return enforce_script_contract(
                revised,
                opening=opening,
                closing=closing_remarks,
                language=language,
            )

    endpoint, model, api_key = await _resolve_provider(provider_id, ai_endpoint, ai_model)
    failed_story_numbers = sorted({
        int(number)
        for issue in issues
        for number in issue.get("evidence_story_numbers", [])
        if str(number).isdigit()
    })
    system_prompt = f"""You are the correction editor for Frontier Tech Daily.
Rewrite the complete spoken script to fix every blocking audit directive below.
Use only the supplied evidence dossier. Remove unsupported precision instead of guessing.
Preserve the story order, natural broadcast tone, exact opening, and exact closing.
Edit ONLY the paragraphs for failed story numbers {failed_story_numbers}. Copy every paragraph for all other stories word-for-word from CURRENT SCRIPT. Never remove, merge, reorder, or rewrite a passing story.
Treat each audit code as a mechanical edit requirement. For D, discard the entire failed paragraph and replace it with at most two short sentences containing only the safest facts stated directly in that story's title or feed-summary lead; do not retain analysis, significance, trend language, rankings, stars, survey figures, or promotional framing. For C, remove every number not directly present in the dossier. For B, omit any disputed proper name. For E, attribute every reported claim aloud. For A, add exactly one concise paragraph from that story's direct evidence.
Output exactly {len(dossier.selected) + 2} nonblank paragraphs: opening, one paragraph for each of the {len(dossier.selected)} stories in dossier order, then closing.
Output spoken prose only with no markdown, labels, citations section, or explanation.
For an English edition, output no Chinese, Japanese, or Korean characters.

Exact opening: {opening}
Exact closing: {closing_remarks}
"""
    user_content = "\n\n".join(
        [
            "AUDIT DIRECTIVES\n" + json.dumps(issues, ensure_ascii=False),
            "CURRENT SCRIPT\n" + script,
            "EVIDENCE DOSSIER\n" + dossier_markdown(dossier),
        ]
    )
    raw = await _chat(
        system_prompt,
        user_content,
        endpoint,
        model,
        api_key,
        log,
        "Daily news audit correction",
        max_tokens=8192,
        enable_skills=False,
    )
    if language == "en" and NON_ENGLISH_RE.search(raw):
        raw = await _chat(
            "Rewrite this broadcast script in English only. Preserve all facts, numbers, uncertainty, opening, and closing. Output spoken prose only.",
            raw,
            endpoint,
            model,
            api_key,
            log,
            "Daily news audit correction translation",
            max_tokens=8192,
            enable_skills=False,
        )
    revised = enforce_script_contract(
        raw,
        opening=opening,
        closing=closing_remarks,
        language=language,
    )
    revised = _minimalize_unsupported_paragraphs(revised, issues)
    return enforce_script_contract(
        revised,
        opening=opening,
        closing=closing_remarks,
        language=language,
    )


def script_contract_report(
    script: str,
    dossier: ResearchDossier,
    edition_date: date,
    *,
    language: str,
    closing_remarks: str,
) -> dict:
    opening = morning_opening(edition_date, language)
    script_folded = script.casefold()
    expected_publications: dict[str, str] = {}
    accepted_attributions: dict[str, dict[str, tuple[str, str]]] = {}
    for article in dossier.selected:
        expected_identity, expected_label, accepted = _publication_attribution(article)
        expected_publications[expected_identity] = expected_label
        accepted_attributions[article.id] = accepted

    matched_publications: dict[str, str] = {}
    story_source_mentions: dict[str, list[str]] = {}
    for article in dossier.selected:
        matched_names: list[str] = []
        for alias, (identity, label) in accepted_attributions[article.id].items():
            if _script_mentions(script_folded, alias):
                matched_names.append(alias)
                matched_publications[identity] = label
        story_source_mentions[article.id] = matched_names

    source_mentions = {
        label: identity in matched_publications
        for identity, label in expected_publications.items()
    }
    for identity, label in matched_publications.items():
        source_mentions.setdefault(label, True)
    failures: list[str] = []
    if not script.startswith(opening):
        failures.append("fixed dated opening is missing or modified")
    if not script.rstrip().endswith(closing_remarks):
        failures.append("fixed closing remarks are missing or modified")
    if language == "en" and NON_ENGLISH_RE.search(script):
        failures.append("English script contains CJK text")
    if len(script.split()) < 120:
        failures.append("script is implausibly short")
    required_publications = min(3, len(expected_publications))
    if len(matched_publications) < required_publications:
        failures.append("fewer than three selected publications are attributed aloud")
    return {
        "passed": not failures,
        "failures": failures,
        "opening": opening,
        "closing": closing_remarks,
        "source_mentions": source_mentions,
        "story_source_mentions": story_source_mentions,
        "matched_publication_count": len(matched_publications),
        "required_publication_count": required_publications,
        "script_sha256": __import__("hashlib").sha256(script.encode("utf-8")).hexdigest(),
        "dossier_selected_ids": [article.id for article in dossier.selected],
    }

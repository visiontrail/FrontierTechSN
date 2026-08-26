from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from datetime import date

from backend.daily_news.research import ResearchDossier, dossier_markdown
from backend.pipeline.digester import _chat, _resolve_provider

LogCallback = Callable[[str], None]
NON_ENGLISH_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff\uac00-\ud7af]")
ENGLISH_SPOKEN_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’.-][A-Za-z0-9]+)*")
CHINESE_SPOKEN_CHARACTER_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
DAILY_NEWS_ENGLISH_WORDS_PER_MINUTE = 180
DAILY_NEWS_CHINESE_CHARACTERS_PER_MINUTE = 280
DAILY_NEWS_DURATION_LOWER_RATIO = 0.8
DAILY_NEWS_DURATION_UPPER_RATIO = 1.2
DAILY_NEWS_EDIT_MAX_TOKENS = 8192
DAILY_NEWS_EDIT_MODEL_ALIASES = {
    # The reasoning variant can spend its entire output allowance on hidden
    # thought for constrained copy edits. The sibling chat model emits the
    # requested prose directly while using the same configured gateway/key.
    "yinhe-thinking": "yinhe-chat",
}
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


def daily_script_duration_report(
    script: str,
    target_duration_minutes: int,
    language: str,
) -> dict:
    """Estimate spoken duration and enforce the Morning Desk length contract.

    English uses words, while Mandarin uses Han characters plus embedded Latin
    terms.  The estimate is deliberately checked again against the real TTS
    artifact later; this gate prevents a radically short script from reaching
    an expensive synthesis/render stage in the first place.
    """
    if language == "zh":
        han_characters = len(CHINESE_SPOKEN_CHARACTER_RE.findall(script))
        latin_terms = len(ENGLISH_SPOKEN_WORD_RE.findall(script))
        spoken_units = han_characters + latin_terms
        units_per_minute = DAILY_NEWS_CHINESE_CHARACTERS_PER_MINUTE
        unit_label = "spoken characters"
    else:
        spoken_units = len(ENGLISH_SPOKEN_WORD_RE.findall(script))
        units_per_minute = DAILY_NEWS_ENGLISH_WORDS_PER_MINUTE
        unit_label = "spoken words"

    target_units = max(1, round(target_duration_minutes * units_per_minute))
    minimum_units = max(1, math.ceil(target_units * DAILY_NEWS_DURATION_LOWER_RATIO))
    maximum_units = max(minimum_units, math.floor(target_units * DAILY_NEWS_DURATION_UPPER_RATIO))
    estimated_minutes = spoken_units / units_per_minute
    return {
        "passed": minimum_units <= spoken_units <= maximum_units,
        "spoken_units": spoken_units,
        "unit_label": unit_label,
        "units_per_minute": units_per_minute,
        "target_duration_minutes": target_duration_minutes,
        "estimated_duration_minutes": round(estimated_minutes, 3),
        "minimum_units": minimum_units,
        "target_units": target_units,
        "maximum_units": maximum_units,
    }


def _daily_news_edit_model(model: str) -> str:
    return DAILY_NEWS_EDIT_MODEL_ALIASES.get(model.casefold(), model)


def narration_duration_report(
    duration_seconds: float,
    target_duration_minutes: int,
) -> dict:
    """Compare the real narration artifact with the configured run length."""
    target_seconds = float(target_duration_minutes * 60)
    minimum_seconds = target_seconds * DAILY_NEWS_DURATION_LOWER_RATIO
    maximum_seconds = target_seconds * DAILY_NEWS_DURATION_UPPER_RATIO
    return {
        "passed": minimum_seconds <= duration_seconds <= maximum_seconds,
        "duration_seconds": round(duration_seconds, 3),
        "target_duration_minutes": target_duration_minutes,
        "target_seconds": target_seconds,
        "minimum_seconds": minimum_seconds,
        "maximum_seconds": maximum_seconds,
    }


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


def _preferred_spoken_source(article) -> str:
    """Choose the evidence-bound source name suitable for a spoken prefix."""
    _identity, expected_label, _accepted = _publication_attribution(article)
    if article.source_id == "techmeme":
        return expected_label
    aliases = SOURCE_SPOKEN_ALIASES.get(article.source_name, ())
    return aliases[0] if aliases else article.source_name


def _minimalize_unsupported_paragraphs(
    script: str,
    issues: list[dict],
    dossier: ResearchDossier | None = None,
) -> str:
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
        lead = sentences[0].strip()
        if dossier is not None and story_number <= len(dossier.selected):
            article = dossier.selected[story_number - 1]
            _identity, _label, accepted = _publication_attribution(article)
            if not any(
                _script_mentions(lead.casefold(), alias)
                for alias in accepted
            ):
                lead = f"According to {_preferred_spoken_source(article)}, {lead}"
        lines[story_number] = lead
    return "\n".join(lines)


async def fit_daily_script_duration(
    script: str,
    dossier: ResearchDossier,
    edition_date: date,
    *,
    target_duration_minutes: int,
    language: str,
    closing_remarks: str,
    ai_endpoint: str | None,
    ai_model: str | None,
    provider_id: int | None,
    protected_story_numbers: set[int] | None = None,
    log: LogCallback | None = None,
) -> tuple[str, dict]:
    """Repair a materially short/long script before independent fact review."""
    report = daily_script_duration_report(script, target_duration_minutes, language)
    if report["passed"]:
        return script, report

    endpoint, model, api_key = await _resolve_provider(provider_id, ai_endpoint, ai_model)
    edit_model = _daily_news_edit_model(model)
    if log and edit_model != model:
        log(
            "Daily news duration correction: using direct-output model "
            f"{edit_model} instead of reasoning model {model}"
        )
    opening = morning_opening(edition_date, language)
    language_rule = (
        "Use natural broadcast Mandarin Chinese."
        if language == "zh"
        else "Use natural broadcast English and no Chinese, Japanese, or Korean characters."
    )
    protected = sorted({
        int(number)
        for number in (protected_story_numbers or set())
        if 1 <= int(number) <= len(dossier.selected)
    })
    protected_rule = (
        f"Story paragraphs {protected} already passed through an evidence-only audit "
        "correction. Copy each of those paragraphs word-for-word from CURRENT SCRIPT. "
        "Reach the duration target only by editing other story paragraphs."
        if protected
        else ""
    )
    protected_lines = _spoken_lines(script)
    working_script = script
    working_report = report
    repaired = script
    repaired_report = report
    max_edit_passes = 2
    for edit_pass in range(1, max_edit_passes + 1):
        direction = (
            "expand"
            if working_report["spoken_units"] < working_report["minimum_units"]
            else "condense"
        )
        retry_rule = (
            ""
            if edit_pass == 1
            else (
                "A prior edit still measured "
                f"{working_report['spoken_units']} {working_report['unit_label']} after "
                "software restored the protected paragraphs. Correct that exact remaining "
                f"shortfall or overage and aim for {working_report['target_units']} "
                f"{working_report['unit_label']}."
            )
        )
        system_prompt = f"""You are the length editor for Frontier Tech Daily.
{direction.capitalize()} the complete spoken script to fit a {target_duration_minutes}-minute edition.
The final script must contain between {working_report['minimum_units']} and {working_report['maximum_units']} {working_report['unit_label']} (target {working_report['target_units']}).
Use only facts already present in CURRENT SCRIPT or the supplied EVIDENCE DOSSIER. Never add generic commentary, repetition, speculation, invented transitions, or unsupported significance merely to reach the length.
Preserve the exact story order and output exactly {len(dossier.selected) + 2} nonblank paragraphs: the exact opening, one paragraph for each selected story, and the exact closing.
Attribute reported claims aloud. Preserve every number, name, uncertainty word, and factual limitation.
{protected_rule}
{retry_rule}
{language_rule}
Output spoken prose only with no markdown, labels, citations section, or explanation.

Exact opening: {opening}
Exact closing: {closing_remarks}
"""
        raw = await _chat(
            system_prompt,
            "\n\n".join(
                [
                    "CURRENT SCRIPT\n" + working_script,
                    "EVIDENCE DOSSIER\n" + dossier_markdown(dossier),
                ]
            ),
            endpoint,
            edit_model,
            api_key,
            log,
            "Daily news duration correction",
            max_tokens=DAILY_NEWS_EDIT_MAX_TOKENS,
            enable_skills=False,
        )
        repaired = enforce_script_contract(
            raw,
            opening=opening,
            closing=closing_remarks,
            language=language,
        )
        if protected:
            repaired_lines = _spoken_lines(repaired)
            expected_line_count = len(dossier.selected) + 2
            if (
                len(protected_lines) != expected_line_count
                or len(repaired_lines) != expected_line_count
            ):
                raise RuntimeError(
                    "Daily-news duration correction could not preserve evidence-only "
                    "story paragraphs because the paragraph contract changed"
                )
            for story_number in protected:
                repaired_lines[story_number] = protected_lines[story_number]
            repaired = "\n".join(repaired_lines)
        repaired_report = daily_script_duration_report(
            repaired,
            target_duration_minutes,
            language,
        )
        if repaired_report["passed"]:
            break
        if edit_pass < max_edit_passes:
            if log:
                log(
                    "Daily news duration correction: protected-paragraph restore left "
                    f"{repaired_report['spoken_units']} {repaired_report['unit_label']}; "
                    "requesting one exact follow-up edit"
                )
            working_script = repaired
            working_report = repaired_report

    if not repaired_report["passed"]:
        raise RuntimeError(
            "Daily-news script length contract failed after correction: "
            f"target {target_duration_minutes} min, estimated "
            f"{repaired_report['estimated_duration_minutes']:.2f} min "
            f"({repaired_report['spoken_units']} {repaired_report['unit_label']}; "
            f"required {repaired_report['minimum_units']}-"
            f"{repaired_report['maximum_units']})"
        )
    if log:
        log(
            "Daily script duration corrected: "
            f"{report['estimated_duration_minutes']:.2f} -> "
            f"{repaired_report['estimated_duration_minutes']:.2f} min "
            f"for {target_duration_minutes}-minute target"
        )
    return repaired, repaired_report


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
    units_per_minute = (
        DAILY_NEWS_CHINESE_CHARACTERS_PER_MINUTE
        if language == "zh"
        else DAILY_NEWS_ENGLISH_WORDS_PER_MINUTE
    )
    spoken_unit_target = max(units_per_minute, target_duration_minutes * units_per_minute)
    endpoint, model, api_key = await _resolve_provider(provider_id, ai_endpoint, ai_model)
    language_label = "natural broadcast Mandarin Chinese" if language == "zh" else "natural broadcast English"
    system_prompt = f"""You are the senior anchor and evidence editor for Frontier Tech Daily.
Write a solo morning-news video podcast script in {language_label}, about {spoken_unit_target} spoken {'characters' if language == 'zh' else 'words'}.

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
        max_tokens=max(4096, min(8192, spoken_unit_target * 3)),
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
            max_tokens=max(4096, min(8192, spoken_unit_target * 3)),
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
    endpoint, model, api_key = await _resolve_provider(provider_id, ai_endpoint, ai_model)
    edit_model = _daily_news_edit_model(model)
    if log and edit_model != model:
        log(
            "Daily news audit correction: using direct-output model "
            f"{edit_model} instead of reasoning model {model}"
        )
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
Every blocking issue carries claim_ids and claim_texts from the reviewed script. Modify only those cited claims inside a failed story paragraph. Preserve every uncited claim in that paragraph word-for-word unless changing punctuation is necessary to remove a cited sentence. Never discard an entire paragraph merely because one claim failed.
Treat each audit code as a mechanical edit requirement. For D, remove the cited unsupported interpretation or replace only that claim with a direct paraphrase or translation of full article evidence. For C, correct or remove only the cited unsupported number or date. For B, correct or omit only the cited disputed name. For E, add source/company attribution or uncertainty to the cited claim. For F, update or remove only the cited contradiction or stale framing. For A with claim <story>.0, add exactly one concise evidence-backed paragraph.
Do not rely on the title alone unless the same claim is also present in the feed summary or full article evidence.
The independent reviewer used mandatory live web search, but the complete local EVIDENCE DOSSIER remains the editing boundary: do not introduce facts found only on the web and do not invent a replacement.
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
        edit_model,
        api_key,
        log,
        "Daily news audit correction",
        max_tokens=DAILY_NEWS_EDIT_MAX_TOKENS,
        enable_skills=False,
    )
    if language == "en" and NON_ENGLISH_RE.search(raw):
        raw = await _chat(
            "Rewrite this broadcast script in English only. Preserve all facts, numbers, uncertainty, opening, and closing. Output spoken prose only.",
            raw,
            endpoint,
            edit_model,
            api_key,
            log,
            "Daily news audit correction translation",
            max_tokens=DAILY_NEWS_EDIT_MAX_TOKENS,
            enable_skills=False,
        )
    revised = enforce_script_contract(
        raw,
        opening=opening,
        closing=closing_remarks,
        language=language,
    )
    # Old saved review reports predate claim identifiers. Preserve their
    # conservative first-sentence fallback, but never apply that paragraph-wide
    # trim to the new claim-targeted protocol.
    legacy_d_issues = [
        issue
        for issue in issues
        if re.search(r"audit codes [A-F]*D", str(issue.get("claim", "")), re.I)
        and not issue.get("claim_ids")
    ]
    if legacy_d_issues:
        revised = _minimalize_unsupported_paragraphs(
            revised,
            legacy_d_issues,
            dossier,
        )
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
    target_duration_minutes: int | None = None,
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
    duration_contract = None
    if target_duration_minutes is not None:
        duration_contract = daily_script_duration_report(
            script,
            target_duration_minutes,
            language,
        )
        if not duration_contract["passed"]:
            failures.append(
                "script duration is outside target tolerance: "
                f"target {target_duration_minutes} min, estimated "
                f"{duration_contract['estimated_duration_minutes']:.2f} min"
            )
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
        "duration_contract": duration_contract,
        "script_sha256": __import__("hashlib").sha256(script.encode("utf-8")).hexdigest(),
        "dossier_selected_ids": [article.id for article in dossier.selected],
    }

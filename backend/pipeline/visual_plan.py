"""Decide what each storyboard scene should *look* like.

The storyboard says when a scene runs and what is said during it. This module
turns that into creative direction — an archetype, a headline, supporting copy,
an accent colour and a motif — which the scene kit renders and the director
agent uses as its brief.

The model call is plain JSON in / JSON out so it works with any provider the
task is pointed at. Every field has a deterministic fallback derived from the
narration itself, so a provider outage degrades to a plainer video, never to a
blank one.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

from backend import config
from backend.pipeline import news_images, scene_kit

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]

# Scenes per model call. Large enough that the model can shape an arc across a
# stretch of the episode, small enough to stay inside a single response.
PLAN_BATCH_SIZE = 12
# Thinking providers need room for both internal reasoning and the JSON plan.
# The default 4,096-token ceiling was exhausted before message content on the
# full 12-scene batches exercised by a ten-minute composition.
VISUAL_PLAN_MAX_TOKENS = 8192

# Accent rotation used by the fallback and as a sanity default. Consecutive
# scenes never share an accent, which alone gives the video a sense of movement.
_ACCENT_CYCLE = ("amber", "teal", "coral", "violet", "sky", "rose", "lime")

# Motif rotation for the fallback path, biased toward the calmer shapes.
_MOTIF_CYCLE = ("orbit", "waves", "grid", "arcs", "bloom", "prism", "skyline", "sunburst")

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TERM_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
_GROUNDING_STOPWORDS = frozenset(
    "a an and are as at be been but by for from had has have he her his i if in into is it its "
    "me my not of on or our she so than that the their them then there they this to was we were "
    "what when which who will with would you your just really thing things".split()
)
_NEGATIVE_ANALYSIS = re.compile(
    r"\b(?:no b-?roll|no visual depictions?|unrelated|does not (?:show|depict|match)|talking-head filler)\b",
    re.IGNORECASE,
)
_FOOTAGE_CREDIT_LIMIT = 90
_SCALED_NUMBER_RE = re.compile(
    r"(?P<number>\d+(?:,\d{3})*(?:\.\d+)?)\s*"
    r"(?P<unit>thousand|million|billion|trillion|[KMBT])\b",
    re.IGNORECASE,
)
_SCALE_NAMES = {
    "k": "thousand",
    "m": "million",
    "b": "billion",
    "t": "trillion",
}
_MODEL_TEXT_FIELDS = (
    "archetype",
    "kicker",
    "headline",
    "body",
    "quote",
    "attribution",
    "stat",
    "stat_label",
    "accent",
    "motif",
)


def _emit(log: LogCallback | None, message: str) -> None:
    if log:
        log(message)
    else:
        logger.info(message)


def _strip_fence(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _first_json_array(value: str) -> list | None:
    """Pull the first top-level JSON array out of a model response.

    Reasoning models like to wrap the answer in prose; a bracket scan is more
    forgiving than json.loads on the whole string and costs nothing.
    """
    text = _strip_fence(value)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and isinstance(parsed.get("scenes"), list):
            return parsed["scenes"]
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, list):
                        return parsed
                    break
        start = text.find("[", start + 1)
    return None


_CLAUSE_BREAK = re.compile(r"[,;:—–-]\s")


def _headline_from(text: str, limit: int = 68) -> str:
    """Best-effort headline: the first sentence, trimmed to a readable length.

    A hard word-boundary cut leaves a headline dangling mid-thought ("... and
    realize it's"), which looks like a bug on screen. Prefer to end on a clause
    boundary, and only fall back to an ellipsis when there isn't one.
    """
    sentence = _SENTENCE_SPLIT.split(text.strip())[0].strip(" —-")
    if len(sentence) <= limit:
        return sentence

    window = sentence[: limit + 1]
    breaks = list(_CLAUSE_BREAK.finditer(window))
    if breaks and breaks[-1].start() >= limit * 0.45:
        return window[: breaks[-1].start()].rstrip()

    words = window.split()
    out: list[str] = []
    for word in words:
        if len(" ".join(out + [word])) > limit - 1:
            break
        out.append(word)
    return (" ".join(out) + "…") if out else sentence[:limit]


def _support_from(text: str, headline: str, limit: int = 190) -> str:
    remainder = text[len(headline) :].strip(" .—-") if text.startswith(headline) else text
    sentences = _SENTENCE_SPLIT.split(remainder.strip())
    out: list[str] = []
    for sentence in sentences:
        candidate = " ".join(out + [sentence])
        if len(candidate) > limit:
            break
        out.append(sentence)
    return " ".join(out).strip()


def fallback_plan(storyboard: dict) -> list[dict]:
    """Derive scene direction from the narration alone, no model involved."""
    plans: list[dict] = []
    for i, scene in enumerate(storyboard.get("scenes", [])):
        text = scene.get("text", "")
        headline = _headline_from(text)
        body = _support_from(text, headline)
        keywords = scene.get("keywords") or []
        # A scene that opens on a quoted line reads best as a pull quote.
        archetype = "quote" if text.strip().startswith(('"', "“")) else ("statement" if i % 5 == 0 else "topic")
        plans.append(
            {
                "id": scene["id"],
                "archetype": archetype,
                "kicker": (keywords[0].upper()[:28] if keywords else ""),
                "headline": headline,
                "body": body,
                "quote": headline if archetype == "quote" else "",
                "accent": _ACCENT_CYCLE[i % len(_ACCENT_CYCLE)],
                "motif": _MOTIF_CYCLE[i % len(_MOTIF_CYCLE)],
                "grounding_source": "narration_fallback",
            }
        )
    return plans


def _model_text(raw: dict, key: str) -> str:
    value = raw.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"visual plan field {key} must be a string")
    return value


def _model_panel(raw: dict, key: str) -> dict[str, str]:
    value = raw.get(key)
    if value is None:
        return {"label": "", "text": ""}
    if not isinstance(value, dict):
        raise ValueError(f"visual plan field {key} must be an object")
    return {
        "label": _model_text(value, "label"),
        "text": _model_text(value, "text"),
    }


def _restore_scaled_number_precision(value: str, narration: str) -> str:
    """Restore source decimals when a model rounded a scaled on-screen fact.

    The replacement is deliberately narrow: the abbreviated/full scale must
    match and exactly one more-precise narration value must round to the model
    value. This recovers ``2.751 billion`` from ``2.75B`` without guessing when
    a scene contains multiple plausible figures.
    """
    source_numbers: dict[str, list[tuple[str, Decimal, int]]] = {}
    for match in _SCALED_NUMBER_RE.finditer(narration or ""):
        literal = match.group("number")
        unit = _SCALE_NAMES.get(match.group("unit").casefold(), match.group("unit").casefold())
        try:
            number = Decimal(literal.replace(",", ""))
        except InvalidOperation:
            continue
        decimals = len(literal.rsplit(".", 1)[1]) if "." in literal else 0
        source_numbers.setdefault(unit, []).append((literal, number, decimals))

    def restore(match: re.Match[str]) -> str:
        literal = match.group("number")
        unit = _SCALE_NAMES.get(match.group("unit").casefold(), match.group("unit").casefold())
        try:
            number = Decimal(literal.replace(",", ""))
        except InvalidOperation:
            return match.group(0)
        decimals = len(literal.rsplit(".", 1)[1]) if "." in literal else 0
        quantum = Decimal(1).scaleb(-decimals)
        matches = [
            source_literal
            for source_literal, source_number, source_decimals in source_numbers.get(unit, [])
            if source_decimals > decimals
            and source_number.quantize(quantum, rounding=ROUND_HALF_UP) == number
        ]
        if len(matches) != 1:
            return match.group(0)
        return match.group(0).replace(literal, matches[0], 1)

    return _SCALED_NUMBER_RE.sub(restore, value or "")


def _normalise(raw: dict, scene: dict, index: int) -> dict:
    """Coerce one model entry into an explicit, finite scene-kit allowlist."""
    raw_items = raw.get("items")
    if raw_items is None:
        raw_items = []
    if not isinstance(raw_items, list) or any(not isinstance(item, str) for item in raw_items):
        raise ValueError("visual plan field items must be an array of strings")

    plan = {key: _model_text(raw, key) for key in _MODEL_TEXT_FIELDS}
    plan.update(
        {
            "id": scene["id"],
            "items": list(raw_items),
            "left": _model_panel(raw, "left"),
            "right": _model_panel(raw, "right"),
        }
    )
    plan["stat"] = _restore_scaled_number_precision(
        plan.get("stat", ""), str(scene.get("text") or "")
    )

    archetype = str(plan.get("archetype") or "").lower()
    if archetype not in scene_kit.ARCHETYPES or archetype in ("title", "outro", "footage"):
        # title/outro are owned by the spine; footage is assigned by the
        # footage matcher, which knows which clips actually downloaded.
        archetype = "topic" if archetype not in scene_kit.ARCHETYPES else archetype
        if archetype in ("title", "outro", "footage"):
            archetype = "topic"
    plan["archetype"] = archetype

    if not str(plan.get("headline") or "").strip():
        plan["headline"] = _headline_from(scene.get("text", ""))
    if not str(plan.get("kicker") or "").strip():
        keywords = scene.get("keywords") or []
        plan["kicker"] = keywords[0].upper()[:28] if keywords else ""

    accent = str(plan.get("accent") or "").lower()
    plan["accent"] = accent if accent in scene_kit.ACCENTS else _ACCENT_CYCLE[index % len(_ACCENT_CYCLE)]
    motif = str(plan.get("motif") or "").lower()
    plan["motif"] = motif if motif in scene_kit.MOTIFS else _MOTIF_CYCLE[index % len(_MOTIF_CYCLE)]

    # Trim copy to what actually fits a 1920x1080 frame. The model is told the
    # limits, but enforcing them here is what keeps `hyperframes inspect` clean.
    plan["headline"] = str(plan["headline"])[:110]
    plan["body"] = str(plan.get("body") or "")[:260]
    plan["kicker"] = str(plan["kicker"])[:30]
    plan["items"] = [item[:90] for item in plan["items"]][
        : scene_kit.MAX_NARRATIVE_ITEMS
    ]
    for key, limit in (
        ("quote", 260),
        ("attribution", 90),
        ("stat", 48),
        ("stat_label", 120),
        ("left_label", 48),
        ("left_text", 150),
        ("right_label", 48),
        ("right_text", 150),
    ):
        plan[key] = plan.get(key, "")[:limit]
    for side in ("left", "right"):
        plan[side] = {
            "label": plan[side]["label"][:48],
            "text": plan[side]["text"][:150],
        }
        plan[f"{side}_label"] = plan[side]["label"]
        plan[f"{side}_text"] = plan[side]["text"]
    plan["grounding_source"] = "model"
    return plan


def _normal_term(value: str) -> str:
    term = value.replace("’", "'").casefold()
    if len(term) > 4 and term.endswith("ies"):
        return term[:-3] + "y"
    if len(term) > 4 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


def _terms(value: object) -> set[str]:
    return {
        normal
        for token in _TERM_RE.findall(str(value or ""))
        if (normal := _normal_term(token)) not in _GROUNDING_STOPWORDS and len(normal) >= 3
    }


def _plan_copy(plan: dict) -> str:
    fields = [
        plan.get("headline"),
        plan.get("body"),
        plan.get("quote"),
        plan.get("stat"),
        plan.get("stat_label"),
        plan.get("left_text"),
        plan.get("right_text"),
        *(plan.get("items") or []),
    ]
    return " ".join(str(value or "") for value in fields)


def _credit_text(value: object, fallback: str) -> str:
    """Collapse source metadata to a safe, single-line display value."""
    text = " ".join(str(value or "").split())
    return text or fallback


def _ellipsize(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= 1:
        return "…"[:limit]
    return value[: limit - 1].rstrip(" .·:;-–—") + "…"


def _source_credit(clip: dict) -> str:
    """Build the viewer-facing credit without exposing acquisition tooling.

    Web video needs provenance in the frame, so its creator and original video
    title take precedence over license shorthand.  ``provider`` is deliberately
    excluded because it describes our ingestion implementation, not the work's
    source.
    """
    creator = _credit_text(clip.get("creator"), "Unknown creator")
    title = _credit_text(clip.get("title"), "Untitled video")
    prefix = "Source: "
    separator = " · "
    full = f"{prefix}{creator}{separator}{title}"
    if len(full) <= _FOOTAGE_CREDIT_LIMIT:
        return full

    available = _FOOTAGE_CREDIT_LIMIT - len(prefix) - len(separator)
    # Preserve both parts when metadata is long: reserve enough room to keep
    # the title meaningful, then give the remaining width to the channel name.
    title_budget = min(len(title), max(24, available // 2))
    creator_budget = available - title_budget
    return (
        f"{prefix}{_ellipsize(creator, creator_budget)}"
        f"{separator}{_ellipsize(title, title_budget)}"
    )


def _footage_credit(clip: dict) -> str:
    external_video = bool(clip.get("platform")) or bool(clip.get("review_required"))
    external_video = external_video or clip.get("rights_status") == "review_required"
    if external_video:
        return _source_credit(clip)

    attribution = clip.get("attribution") or clip.get("license_short_name")
    if attribution:
        return _ellipsize(
            _credit_text(attribution, "Footage source"), _FOOTAGE_CREDIT_LIMIT
        )
    return _source_credit(clip)


def _plan_grounding(plan: dict, scene: dict) -> tuple[bool, list[str]]:
    scene_terms = _terms(scene.get("text", ""))
    overlap = sorted(scene_terms & _terms(_plan_copy(plan)))
    # Derived plans quote the narration and are therefore grounded by
    # construction. Model plans need at least one concrete lexical anchor;
    # otherwise a fluent but unrelated scene is replaced by the safe fallback.
    grounded = plan.get("grounding_source") == "narration_fallback" or bool(overlap)
    return grounded, overlap


def _batch_prompt_payload(storyboard: dict, scenes: list[dict]) -> str:
    payload = {
        "thesis": storyboard.get("thesis", ""),
        "scenes": [
            {
                "id": scene["id"],
                "seconds": scene["duration"],
                "narration": scene["text"],
            }
            for scene in scenes
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


async def plan_scene_visuals(
    storyboard: dict,
    *,
    ai_endpoint: str | None = None,
    ai_model: str | None = None,
    provider_id: int | None = None,
    log: LogCallback | None = None,
) -> list[dict]:
    """Creative direction for every scene, model-driven with a safe fallback."""
    scenes = storyboard.get("scenes", [])
    if not scenes:
        return []

    # Imported here to avoid a module-level cycle: digester imports the agent
    # transport, which imports config, which the pipeline package initialises.
    from backend.pipeline.digester import _chat, _resolve_provider

    prompt_path = config.PROMPTS_DIR / "visual_plan.txt"
    system_prompt = prompt_path.read_text(encoding="utf-8")

    try:
        endpoint, model, api_key = await _resolve_provider(provider_id, ai_endpoint, ai_model)
    except Exception as exc:  # noqa: BLE001 - provider lookup must never sink the render
        _emit(log, f"Visual plan: provider lookup failed ({exc}); using derived direction")
        return fallback_plan(storyboard)

    batches = [scenes[i : i + PLAN_BATCH_SIZE] for i in range(0, len(scenes), PLAN_BATCH_SIZE)]
    _emit(log, f"Visual plan: directing {len(scenes)} scenes in {len(batches)} batch(es)")

    by_id: dict[str, dict] = {}
    for n, batch in enumerate(batches, start=1):
        try:
            reply = await _chat(
                system_prompt,
                _batch_prompt_payload(storyboard, batch),
                endpoint,
                model,
                api_key,
                log,
                f"Visual plan {n}/{len(batches)}",
                max_tokens=VISUAL_PLAN_MAX_TOKENS,
                enable_skills=False,
            )
            entries = _first_json_array(reply) or []
        except Exception as exc:  # noqa: BLE001 - one bad batch must not lose the rest
            _emit(log, f"Visual plan batch {n} failed ({exc}); deriving that batch from narration")
            entries = []

        indexed = {str(entry.get("id")): entry for entry in entries if isinstance(entry, dict)}
        for scene in batch:
            raw = indexed.get(scene["id"])
            if raw is None:
                continue
            try:
                normalised = _normalise(raw, scene, scene["index"])
            except (TypeError, ValueError) as exc:
                _emit(
                    log,
                    f"Visual plan {scene['id']}: invalid model fields ({exc}); "
                    "using narration-derived direction",
                )
                continue
            grounded, _ = _plan_grounding(normalised, scene)
            if grounded:
                by_id[scene["id"]] = normalised

    derived = {plan["id"]: plan for plan in fallback_plan(storyboard)}
    plans = [by_id.get(scene["id"]) or derived[scene["id"]] for scene in scenes]
    planned = sum(1 for scene in scenes if scene["id"] in by_id)
    _emit(log, f"Visual plan ready: {planned}/{len(scenes)} scenes directed by the model")
    return plans


OUTRO_SCENE_ID = "scene-99-outro"


def outro_plan(storyboard: dict, *, brand: str = "") -> dict:
    """Closing card. The one scene allowed to animate itself out."""
    return {
        "id": OUTRO_SCENE_ID,
        "archetype": "outro",
        "kicker": "",
        "headline": "Thanks for watching",
        "body": brand,
        "accent": "coral",
        "motif": "bloom",
    }


def attach_footage(plans: list[dict], storyboard: dict, manifest: dict | None, task_dir: Path) -> int:
    """Promote scenes to full-bleed footage plates where a clip actually fits.

    Clips are matched to the scene whose keywords best overlap the query that
    found them, so a sunflower plate lands on the sunflower paragraph instead of
    wherever it happened to download.
    """
    clips = (manifest or {}).get("clips") or []
    if not clips:
        return 0

    scenes_by_id = {scene["id"]: scene for scene in storyboard.get("scenes", [])}
    scene_terms = {
        scene_id: _terms(
            f"{scene.get('text', '')} {' '.join(scene.get('keywords') or [])}"
        )
        for scene_id, scene in scenes_by_id.items()
    }
    document_frequency: dict[str, int] = {}
    for terms in scene_terms.values():
        for term in terms:
            document_frequency[term] = document_frequency.get(term, 0) + 1
    common_threshold = max(2, math.ceil(max(1, len(scene_terms)) * 0.35))
    common_terms = {
        term for term, frequency in document_frequency.items() if frequency >= common_threshold
    }
    used: set[str] = set()
    attached = 0

    # Web-scouted clips carry the exact narration excerpt they were selected
    # for. Place those before broad Commons search results so a generic query
    # such as "Pearl Harbor attack" cannot consume the scene that an excerpt-
    # grounded clip was explicitly selected to illustrate.
    ordered_clips = sorted(
        clips,
        key=lambda clip: bool(str(clip.get("script_excerpt") or "").strip()),
        reverse=True,
    )

    for clip in ordered_clips:
        local = clip.get("local_path") or clip.get("path")
        if not local:
            continue
        path = Path(local)
        if not path.is_absolute():
            path = task_dir / local
        if not path.exists():
            continue
        try:
            rel = path.resolve().relative_to(task_dir.resolve()).as_posix()
        except ValueError:
            continue

        analysis = clip.get("analysis") or {}
        confidence = float(analysis.get("confidence") or 1.0)
        # The deterministic-safe-offset analyzer is used only after search and
        # download have already selected a result for an exact script excerpt.
        # Its 0.25 score means "trim not visually reviewed", not "unrelated
        # footage". Treat the excerpt match as the minimum admissible grounding
        # confidence while retaining the manifest's raw analyzer confidence.
        fallback_excerpt = (
            str(analysis.get("status") or "").lower() == "fallback"
            and bool(str(clip.get("script_excerpt") or "").strip())
        )
        if fallback_excerpt:
            confidence = max(confidence, 0.65)
        if analysis and confidence < 0.65:
            continue
        if _NEGATIVE_ANALYSIS.search(str(analysis.get("reason") or "")):
            continue

        weighted_terms: dict[str, int] = {}
        for value, weight in (
            # The excerpt is the narration for which the clip/interval was
            # actually selected. It is a stronger constraint than the broad
            # discovery query: without it, a "fiat money" search result chosen
            # for a history paragraph can be reassigned to an unrelated central
            # bank paragraph that happens to share two query words.
            (clip.get("script_excerpt", ""), 5),
            (clip.get("query", ""), 3),
            (clip.get("purpose", ""), 2),
            (clip.get("title", ""), 1),
        ):
            for term in _terms(value):
                weighted_terms[term] = max(weighted_terms.get(term, 0), weight)
        distinctive = set(weighted_terms) - common_terms
        excerpt_terms = _terms(clip.get("script_excerpt", "")) - common_terms
        minimum_excerpt_matches = min(3, len(excerpt_terms))

        best_id, best_score, best_matches, best_excerpt_matches = None, 0.0, [], []
        for plan in plans:
            if plan["id"] in used:
                continue
            scene = scenes_by_id.get(plan["id"])
            if not scene:
                continue
            matches = sorted(distinctive & scene_terms[plan["id"]])
            if len(matches) < 2:
                continue
            excerpt_matches = sorted(excerpt_terms & scene_terms[plan["id"]])
            if excerpt_terms and len(excerpt_matches) < minimum_excerpt_matches:
                continue
            score = sum(weighted_terms[term] for term in matches)
            if score > best_score:
                best_id = plan["id"]
                best_score = score
                best_matches = matches
                best_excerpt_matches = excerpt_matches
        if best_id is None or best_score <= 0:
            continue

        plan = next(p for p in plans if p["id"] == best_id)
        plan["archetype"] = "footage"
        plan["footage_src"] = rel
        plan["footage_kind"] = "video" if path.suffix.lower() in {".webm", ".mp4", ".ogv"} else "image"
        plan["footage_credit"] = _footage_credit(clip)
        plan["footage_query"] = str(clip.get("query") or "")[:160]
        plan["footage_match_terms"] = best_matches
        plan["footage_script_match_terms"] = best_excerpt_matches
        plan["footage_match_score"] = best_score
        plan["footage_confidence"] = round(confidence, 3)
        if fallback_excerpt:
            plan["footage_analysis_confidence"] = round(
                float(analysis.get("confidence") or 0), 3
            )
            plan["footage_analysis_status"] = "fallback"
        used.add(best_id)
        attached += 1

    return attached


def visual_grounding_report(plans: list[dict], storyboard: dict) -> dict:
    """Machine-readable semantic audit consumed by the compose quality gate."""
    by_id = {plan["id"]: plan for plan in plans}
    results: list[dict] = []
    for scene in storyboard.get("scenes", []):
        plan = by_id.get(scene["id"])
        if not plan:
            results.append({"id": scene["id"], "grounded": False, "reason": "missing plan"})
            continue
        grounded, overlap = _plan_grounding(plan, scene)
        reason = "copy grounded in narration"
        if plan.get("collage_broll"):
            qa_passed = bool((plan.get("collage_qa") or {}).get("passed"))
            source_scene_id = str(plan.get("collage_source_scene_id") or plan.get("id") or "")
            placed_scene_id = str(plan.get("collage_placed_scene_id") or plan.get("id") or "")
            exact_scene = source_scene_id == scene["id"] == placed_scene_id
            grounded = grounded and qa_passed and exact_scene
            reason = (
                "paper-collage passed media QA and remained on its requested scene"
                if grounded
                else "paper-collage failed media QA or moved away from its requested scene"
            )
        elif plan.get("news_image"):
            match_terms = plan.get("news_image_match_terms") or []
            distinctive_anchors = plan.get("news_image_grounding_distinctive_anchors") or []
            source_scene_id = str(plan.get("news_image_source_scene_id") or "")
            exact_scene = source_scene_id == scene["id"]
            context_contract_present = all(
                key in plan
                for key in (
                    "news_image_categories",
                    "news_image_object_name",
                    "news_image_creator",
                    "news_image_grounding_context_conflicts",
                )
            )
            proof = {
                "expected_subject": plan.get("news_image_expected_subject") or "",
                "caption": plan.get("news_image_caption") or "",
                "kind": plan.get("news_image_kind") or "event",
                "title": plan.get("news_image_title") or "",
                "description": plan.get("news_image_description") or "",
                "categories": plan.get("news_image_categories") or "",
                "object_name": plan.get("news_image_object_name") or "",
                "creator": plan.get("news_image_creator") or "",
                "license": plan.get("news_image_license") or "",
                "license_code": plan.get("news_image_license_code") or "",
                "source_mime_type": plan.get("news_image_source_mime_type") or "",
                "match_terms": match_terms,
                "grounding_policy_version": plan.get("news_image_grounding_policy_version"),
                "grounding_passed": plan.get("news_image_grounding_passed") is True,
                "grounding_distinctive_anchors": distinctive_anchors,
                "grounding_identity_field": plan.get("news_image_grounding_identity_field") or "",
                "grounding_identity_phrase": plan.get("news_image_grounding_identity_phrase") or "",
                "grounding_identity_field_terms": plan.get("news_image_grounding_identity_field_terms") or [],
                "grounding_context_conflicts": plan.get("news_image_grounding_context_conflicts") or [],
            }
            grounded = (
                grounded
                and exact_scene
                and context_contract_present
                and news_images.image_grounding_is_valid(proof, scene)
            )
            reason = (
                "licensed news image passed the current complete-identity grounding policy on "
                + ", ".join(distinctive_anchors)
                if grounded
                else "news image lacked a current exact-scene grounding proof or license anchor"
            )
        elif plan.get("archetype") == "footage":
            match_terms = plan.get("footage_match_terms") or []
            script_match_terms = plan.get("footage_script_match_terms")
            script_grounded = script_match_terms is None or len(script_match_terms) >= 2
            grounded = (
                grounded
                and len(match_terms) >= 2
                and script_grounded
                and float(plan.get("footage_confidence") or 0) >= 0.65
            )
            reason = (
                f"footage matched on {', '.join(match_terms)}"
                if grounded
                else "footage lacked two distinctive narration matches"
            )
        results.append(
            {
                "id": scene["id"],
                "grounded": grounded,
                "reason": reason,
                "copy_match_terms": overlap,
                "grounding_source": plan.get("grounding_source", "unknown"),
            }
        )

    grounded_count = sum(1 for result in results if result["grounded"])
    rate = grounded_count / max(1, len(results))
    return {
        "scene_count": len(results),
        "grounded_scenes": grounded_count,
        "grounding_rate": round(rate, 4),
        "passed": grounded_count == len(results),
        "scenes": results,
    }

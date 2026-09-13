import asyncio
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
import wave
from array import array
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import httpx

from backend import config
from backend.pipeline.process_logging import stream_subprocess

logger = logging.getLogger(__name__)
LogCallback = Callable[[str], None]

# Synthesis is watched by inactivity, not elapsed time: a long script legitimately
# runs for hours, but VibeVoice prints a decode-progress line several times a
# second, so silence is the only reliable hang signal. The values are read from
# config at call time so an Admin change applies to the next run.
SPEAKER_LABEL_RE = re.compile(r"^\s*Speaker\s*\d+\s*[:：\-—–]\s*", re.IGNORECASE)
SPEAKER_LINE_RE = re.compile(
    r"^\s*(Speaker\s*\d+\s*[:：\-—–])\s*(.*)$", re.IGNORECASE
)
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])\s+")

# Orpheus emits seven LM tokens for every 2,048 PCM samples at 24 kHz. This is
# the model's actual codec geometry, not a heuristic. It lets us turn the
# remote service's token ceiling into a safe text-chunk budget and detect a
# response that stopped because it hit max_tokens rather than end-of-speech.
ORPHEUS_AUDIO_TOKENS_PER_SECOND = 7 * 24_000 / 2_048
ORPHEUS_CHUNK_MIN_WPM = 90
ORPHEUS_CHUNK_SAFETY = 0.80
ORPHEUS_TOKEN_LIMIT_RATIO = 0.97
ORPHEUS_MIN_EXACT_ASR_COVERAGE = 0.90
ORPHEUS_MIN_ASR_WORD_RATIO = 1.0
ORPHEUS_MAX_ASR_WORD_RATIO = 1.0
ORPHEUS_MAX_PHONETIC_SUBSTITUTIONS = 1
ORPHEUS_MAX_EVIDENCED_PHONETIC_SUBSTITUTIONS = 2
ORPHEUS_MIN_PHONETIC_SPELLING_SIMILARITY = 0.80
# One live, otherwise exact Reuters utterance was independently transcribed as
# the brand name ``Shein`` -> ``Shane`` at normal speed.  The raw spellings are
# too far apart for the general similarity gate, so keep this exception as an
# explicit unordered pair. It still has to be the sole aligned substitution in
# an equal-length utterance and be corroborated from the same WAV at a second
# playback speed before the audio can pass.
ORPHEUS_EVIDENCED_PHONETIC_PAIRS = {
    frozenset({"shein", "shane"}),
    # The source spelling Łukasz normalizes to Lukasz while English Whisper
    # consistently renders the same spoken personal name as Lukas. Keep the
    # final-letter drift scoped to this exact proper-name pair.
    frozenset({"lukasz", "lukas"}),
    # Whisper rendered the Slovenian name Jernej Barbic as Jernesh Barbish on
    # an otherwise exact live narration. Both substitutions must remain
    # aligned and be corroborated together from the same waveform.
    frozenset({"jernej", "jernesh"}),
    # The current ASCII lexical pass represents canonical ``Barbič`` as
    # ``barbi`` because the final caron consonant is discarded.
    frozenset({"barbi", "barbish"}),
    # The Chinese platform name Douyin was spoken completely in a live news
    # chunk while English Whisper rendered the same pronunciation as Duwayan.
    frozenset({"douyin", "duwayan"}),
}
ORPHEUS_EXACT_EDGE_ANCHOR_WORDS = 2
NARRATION_PACING_POLICY = "natural_speech_visuals_follow_audio"
NARRATION_SYNTHESIS_SPEED_RATIO = 1.0

# VibeVoice reads some technology names as invented words instead of familiar
# initialisms.  These provider-only spellings improve pronunciation while the
# canonical script remains unchanged for publication, review and evidence.
VIBEVOICE_PRONUNCIATIONS = (
    ("IEEE", "I triple E"),
    ("QbitAI", "Q-bit A-I"),
    ("Qwen", "cue-when"),
)
ORPHEUS_EDGE_ANCHOR_WORDS = 3
ORPHEUS_MAX_INTEGRITY_ATTEMPTS = 3
ORPHEUS_MIN_REQUEST_TOKENS = 512
# Increment whenever acoustic acceptance semantics change.  Cached WAVs with
# older sidecars must pass the current local verifier before they are reused.
ORPHEUS_INTEGRITY_VERIFIER_VERSION = 25
POCKET_TTS_MAX_INTEGRITY_ATTEMPTS = 3
# Pocket TTS uses the same fail-closed acoustic verifier, but its cache identity
# is independent so provider-specific changes can invalidate only Pocket audio.
POCKET_TTS_INTEGRITY_VERIFIER_VERSION = 7
POCKET_TTS_INTERNAL_MAX_TOKENS = 50
POCKET_TTS_EDGE_SILENCE_DBFS = -42.0
POCKET_TTS_SILENCE_WINDOW_MS = 10
POCKET_TTS_LONG_SILENCE_SECONDS = 0.5
POCKET_TTS_MAX_INTERNAL_SILENCE_SECONDS = 0.8
POCKET_TTS_MAX_SENTENCE_BOUNDARY_SILENCE_SECONDS = 1.2
ORPHEUS_NAME_RECHECK_SPEEDS = (0.8, 0.7)
ORPHEUS_NAME_RECHECK_TOKENS = {"qwen", "qianwen", "qbitai"}
ORPHEUS_NAME_RECHECK_SPELLINGS = {
    "qwen": {"qwin"},
    "qianwen": set(),
    "qbitai": set(),
}
ORPHEUS_NAME_RECHECK_SPLITS = {
    "qwen": {
        ("q", "when"),
        ("q", "wen"),
        ("q", "win"),
        ("cue", "wen"),
        ("cue", "when"),
    },
    "qianwen": {
        ("can", "wen"),
        ("chan", "en"),
        ("chien", "wen"),
        ("jian", "wen"),
        ("qian", "wen"),
    },
    # At normal speed Whisper dropped Q-bit's initial consonant in a complete
    # live utterance ("Hubit AI"), while the same waveform recovered "QBit AI"
    # at slower verification speed. This spelling can only initiate a
    # same-waveform recheck; it is never accepted as final lexical evidence.
    "qbitai": {("hubit", "ai")},
}
# Provider pronunciation hints can make Whisper retain a name's exact spoken
# syllable boundary.  Unlike the broader recheck spellings above, these pairs
# are accepted only when alignment proves that they replace the corresponding
# canonical source name at that position.
ORPHEUS_NAME_ACOUSTIC_SPLITS = {
    "qwen": {("q", "when")},
    "qianwen": {("qian", "wen")},
}
MAX_PLAUSIBLE_SPEECH_WPM = 320
LEXICAL_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?|[\u3400-\u9fff]")
DECIMAL_LITERAL_RE = re.compile(r"(\d+)\.(\d+)")
CURRENCY_AMOUNT_RE = re.compile(r"\$([0-9]+(?:\.\d+)?)")
CURRENCY_TRANSCRIPT_AMOUNT_RE = re.compile(
    r"\s*\$([0-9]+)(?:\.(\d+))?[.,;:!?]?\s*"
)
CURRENCY_SCALE_TOKENS = {"hundred", "thousand", "million", "billion", "trillion"}
CURRENCY_ADJECTIVE_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?|[A-Za-z]+(?:-[A-Za-z]+)*)-"
    r"(hundred|thousand|million|billion|trillion)-dollar\b",
    re.IGNORECASE,
)
DECIMAL_INTEGER_WORD_RE = re.compile(r"\s*(\d+)\s*")
DECIMAL_FRACTION_WORD_RE = re.compile(r"\s*\.(\d+)[.,;:!?]?\s*")
NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "oh": "0", "twenty": "20",
    "thirty": "30", "forty": "40",
    "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80",
    "ninety": "90",
}
ORDINAL_DIGITS = {
    "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth",
    "5th": "fifth", "6th": "sixth", "7th": "seventh", "8th": "eighth",
    "9th": "ninth", "10th": "tenth", "20th": "twentieth", "30th": "thirtieth",
}
COMPOUND_ORDINAL_ONES = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
}
CALENDAR_MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}
CALENDAR_ORDINAL_RE = re.compile(r"([0-9]{1,2})(?:st|nd|rd|th)")
# Acoustic verification cannot distinguish exact homophones. Keep this list
# deliberately narrow; near-homophones such as ``feed``/``feet`` must still fail.
ACOUSTIC_EQUIVALENTS = {
    "feat": "feet",
    # The common noun "night" and the title/name spelling "Knight" are exact
    # homophones. A live Morning Desk utterance contained every requested word
    # and both edges, but Whisper capitalized the final word as the latter.
    # Canonicalize only that spelling; omissions and near-homophones still fail.
    "knight": "night",
    # Possessive "its" and the contraction "it's" are exact homophones.
    # Whisper uses the contraction spelling for either meaning, so spelling
    # cannot be used as acoustic evidence that the narration is wrong.
    "it's": "its",
    # Possessive "your" and the contraction "you're" are likewise
    # acoustically indistinguishable. The live Morning Desk run returned every
    # requested word and both edge anchors, but Whisper selected the contraction
    # spelling. Canonicalize only that exact homophone; missing, repeated, and
    # merely similar words must still fail the 100% utterance gate.
    "you're": "your",
    # Singular and plural possessive spellings of "World" have the same
    # spoken form; Whisper cannot recover which apostrophe the script used.
    "worlds": "world's",
    # Plural "offices" and possessive "Office's" likewise share the exact
    # spoken form. This matters for the named product "Qwen Office's" while
    # still requiring the audible final /ɪz/ syllable.
    "offices": "office's",
    # Plural possessive "qubits'" and singular possessive "qubit's" both
    # end in /ts/. ASR cannot infer the apostrophe position from the audio;
    # singular "qubit" must retain its distinct, missing /s/ ending.
    "qubit's": "qubits",
    # Whisper may spell the German surname Brem as the identically pronounced
    # surname Brehm. The silent ``h`` carries no acoustic evidence; other
    # nearby spellings (for example "Bream") remain distinct.
    "brehm": "brem",
    "brehm's": "brem's",
    # The noun "role" and "roll" are exact homophones. A live Orpheus sample
    # contained every requested word and both utterance edges while Whisper
    # selected the latter spelling; nearby words such as "roil" remain errors.
    "roll": "role",
    # The verb forms "rights" and "writes" are exact homophones. A live
    # otherwise exact utterance ended with the phrasal verb "self-rights" while
    # Whisper selected "self-writes". Canonicalize only that inaudible spelling
    # distinction; a missing or different final word still fails edge coverage.
    "writes": "rights",
    # Whisper consistently labels the rare spoken word "eunuch" as the
    # familiar two-syllable proper noun "Unix", including at 0.8x speed.
    "unix": "eunuch",
    # Whisper large-v3-turbo spells the correctly pronounced surname
    # "Scorsese" as "Suarcese" in this narration.  This exact, observed
    # spelling alias avoids regenerating otherwise complete Orpheus audio;
    # unrelated near-matches remain rejected.
    "suarcese": "scorsese",
    "sorsese": "scorsese",
    # The invented product name PhanthyMotus was rendered completely, but
    # Whisper spelled the same pronunciation as FancyModus at normal/0.8x and
    # Fantymodus at 0.7x. Limit the equivalence to those two observed full-name
    # spellings; nearby words and partial names remain hard failures.
    "fancymodus": "phanthymotus",
    "fantymodus": "phanthymotus",
    # Singular possessive "Techmeme's" and plural possessive "TechMemes'"
    # have the same /z/ ending. Whisper used the latter spelling for a live,
    # otherwise exact utterance; TechMean remains intentionally distinct.
    "techmemes": "techmeme's",
    # The explicit Ear-en-dill provider prompt produced the intended three
    # syllables while Whisper rendered them as the phonetic spelling Irindil.
    # Keep the unrelated and repeatedly observed Arendelle substitution hard.
    "irindil": "earendil",
    # Whisper may choose the past-tense spelling for the acoustically
    # identical number word. Numeric completeness and position stay strict.
    "won": "1",
}
ACOUSTIC_PHRASE_EQUIVALENTS = {
    # The investment-bank name is acoustically ambiguous with two common
    # surname spellings in Whisper.  Scope the equivalence to the complete
    # report attribution so unrelated people named Jeffreys remain distinct.
    ("the", "jefferies", "report"): "thejefferiesreport",
    ("the", "jeffreys", "report"): "thejefferiesreport",
    ("the", "jeffries", "report"): "thejefferiesreport",
    # Whisper may spell the phrasal verb as the identically pronounced noun.
    ("break", "through"): "breakthrough",
    # CamelCase publication names are a single lexical source token, while
    # Whisper emits their acoustically identical component words.
    ("deep", "tech"): "deeptech",
    ("qbit", "ai"): "qbitai",
    ("qubit", "ai"): "qbitai",
    # The company name Xspark is a single canonical source token, while
    # English Whisper preserves its spoken letter boundary as "X Spark".
    ("x", "spark"): "xspark",
    # Whisper may retain the audible morpheme boundary in the established
    # compound role name "postdoc" as two tokens.
    ("post", "doc"): "postdoc",
    # Provider articulation spells the compact model prefix V4 as its letter
    # and number. Preserve the canonical source token after exact ASR recovery.
    ("v", "4"): "v4",
    # A live Nikkei Asia utterance was transcribed as "Nikke" at normal
    # speed but recovered the publication's spelling at both 0.8x and 0.7x.
    # Scope the exact ASR spelling drift to the full publication name so an
    # unrelated Nikke token remains distinct.
    ("nikkei", "asia"): "nikkeiasia",
    ("nikke", "asia"): "nikkeiasia",
    # A second complete live utterance produced the exact homophonic name
    # spelling "Nikkei Aja" at three playback speeds. Keep this spelling
    # equivalence constrained to the verified publication-name context.
    ("nikkei", "aja"): "nikkeiasia",
    # Hyphenation is not audible; Whisper may split the source compound.
    ("semi", "annual"): "semiannual",
    ("skunk", "works"): "skunkworks",
    # Whisper tokenizes the spoken compound "fivefold" as the consecutive
    # words "five" and "-fold". Number-word normalization has already mapped
    # the first token to "5" here, so collapse only that exact morpheme pair;
    # different multipliers or an extra intervening word remain hard failures.
    ("5", "fold"): "fivefold",
    ("3", "m"): "3m",
    ("multi", "modal"): "multimodal",
    ("a", "p", "i"): "api",
    ("tech", "meme"): "techmeme",
    ("tech", "meme's"): "techmeme's",
    ("ear", "en", "dill"): "earendil",
    ("ear", "endil"): "earendil",
    # Whisper fuses these adjacent product/company name tokens even though the
    # waveform contains both spoken components. Canonicalize only the complete
    # proper names, preserving every surrounding word and possessive ending.
    ("ox", "alpha"): "oxalpha",
    ("z", "ai's"): "zai's",
    # NERVA is conventionally spoken as a word. The provider pronunciation
    # hint produced NERV at normal-speed ASR but exact NERVA at both 0.8x and
    # 0.7x; the non-rhotic Leah voice can also surface Rover as ROVA in Whisper.
    # Keep those evidenced spelling equivalents constrained to the complete
    # pair of historical program names; unrelated tokens remain distinct.
    ("nerva", "and", "rover"): "nervaandrover",
    ("nerva", "and", "rova"): "nervaandrover",
    ("nerv", "and", "rover"): "nervaandrover",
    ("nervah", "and", "rover"): "nervaandrover",
    ("nervah", "and", "rova"): "nervaandrover",
    ("ner", "vuh", "and", "rover"): "nervaandrover",
    ("ner", "vuh", "and", "rova"): "nervaandrover",
    # Provider-only phonetics for the Chinese personal name Zhu Yi. Keep this
    # equivalence scoped to the complete two-token name so an unrelated "Joo"
    # or "Yee" remains distinct and positional completeness still applies.
    ("zhu", "yi"): "zhuyi",
    ("joo", "yee"): "zhuyi",
    ("jew", "yee"): "zhuyi",
}
NUMBER_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000}
DANGLING_CHUNK_WORDS = {
    "a", "all", "an", "and", "as", "at", "but", "by", "for", "from", "in",
    "fully", "into", "nor", "of", "on", "or", "the", "to", "with",
}
# Orpheus repeatedly drops an isolated leading "of" while conjunction-led
# continuations remain reliable. Keep this intervention deliberately narrow so
# existing verified chunk identities do not churn.
BAD_LEADING_CHUNK_WORDS = {"of"}
CHUNK_DETERMINERS = {"a", "an", "the"}
TERMINAL_SPEECH_PUNCTUATION_RE = re.compile(r"[.!?。！？][\"'’”)]*\s*$")
TRAILING_CLAUSE_PUNCTUATION_RE = re.compile(r"[,;:，；：]+([\"'’”)]*)\s*$")


class TtsIntegrityError(RuntimeError):
    """A provider returned audio that cannot contain the requested narration."""

    def __init__(self, message: str, *, part_key: str | None = None):
        super().__init__(message)
        self.part_key = part_key


@dataclass(frozen=True)
class WavInfo:
    channels: int
    sample_width: int
    sample_rate: int
    frame_count: int
    duration_seconds: float


def _strip_speaker_labels(script: str) -> str:
    """Remove leading 'Speaker N:' markers from a script.

    VibeVoice can vocalize labels verbatim ("Speaker one, ...") depending on
    model/script format. Stripping them yields clean spoken input while line
    breaks preserve turn/beat pacing."""
    lines = []
    for line in script.splitlines():
        cleaned = SPEAKER_LABEL_RE.sub("", line).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def _spoken_word_count(text: str) -> int:
    return len(_strip_speaker_labels(text).split())


def _raw_lexical_tokens(text: str) -> list[str]:
    """Normalize individual spellings without collapsing cross-word phrases."""
    normalized: list[str] = []
    lexical_text = _strip_speaker_labels(text)
    # ``Ł`` is a Latin letter but does not decompose under Unicode NFKD.  The
    # ASCII-only lexical regex would therefore drop it and turn the Polish name
    # Łukasz into the impossible token ``ukasz``. Transliterate only this
    # well-defined letter before acoustic comparison; Whisper conventionally
    # emits the corresponding ASCII spelling ``Lukasz``/``Lukas``.
    lexical_text = lexical_text.translate(str.maketrans({"Ł": "L", "ł": "l"}))
    # The published chip name retains its Spanish tilde, while English ASR
    # conventionally emits the same spoken name as the ASCII spelling
    # "Jalapeno". Normalize only this evidenced proper noun; unrelated accented
    # words and different final vowels remain distinct.
    lexical_text = re.sub(
        r"\bJalapeño\b",
        "Jalapeno",
        lexical_text,
        flags=re.IGNORECASE,
    )
    lexical_text = CURRENCY_AMOUNT_RE.sub(
        lambda match: (
            f" {match.group(1)} "
            + (
                "dollar"
                if re.fullmatch(r"1(?:\.0+)?", match.group(1))
                else "dollars"
            )
            + " "
        ),
        lexical_text,
    )
    lexical_text = lexical_text.replace("$", " dollar ")
    for symbol, spoken in (
        ("&", "and"),
        ("+", "plus"),
        ("=", "equals"),
        ("@", "at"),
        ("#", "hashsymbol"),
        ("°", "degrees"),
    ):
        lexical_text = lexical_text.replace(symbol, f" {spoken} ")
    lexical_text = lexical_text.replace("%", " percent ")
    lexical_text = DECIMAL_LITERAL_RE.sub(
        lambda match: (
            f" decimalnumber{match.group(1)}point{match.group(2)} "
        ),
        lexical_text,
    )
    for token in LEXICAL_TOKEN_RE.findall(lexical_text):
        value = token.replace("’", "'").casefold()
        value = ACOUSTIC_EQUIVALENTS.get(value, value)
        normalized.append(ORDINAL_DIGITS.get(value, NUMBER_WORDS.get(value, value)))
    return normalized


def _lexical_tokens(text: str) -> list[str]:
    normalized = _raw_lexical_tokens(text)
    normalized = _canonicalize_number_tokens(
        _canonicalize_decimal_tokens(
            _canonicalize_acoustic_phrase_tokens(normalized)
        )
    )
    normalized = _canonicalize_numeric_range_tokens(normalized)
    normalized = _canonicalize_compound_ordinal_tokens(normalized)
    return _canonicalize_calendar_date_tokens(normalized)


def _ordinal_suffix(value: int) -> str:
    if 10 <= value % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")


def _canonicalize_compound_ordinal_tokens_with_indexes(
    tokens: list[str], word_indexes: list[int]
) -> tuple[list[str], list[int]]:
    """Match spoken ``twenty-first`` with Whisper's compact ``21st``."""
    if len(tokens) != len(word_indexes):
        raise ValueError("Ordinal tokens and word indexes must have equal length")
    result: list[str] = []
    result_indexes: list[int] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if (
            token.isdigit()
            and 20 <= int(token) <= 90
            and int(token) % 10 == 0
            and index + 1 < len(tokens)
            and tokens[index + 1] in COMPOUND_ORDINAL_ONES
        ):
            value = int(token) + COMPOUND_ORDINAL_ONES[tokens[index + 1]]
            result.append(f"{value}{_ordinal_suffix(value)}")
            result_indexes.append(word_indexes[index])
            index += 2
            continue
        result.append(token)
        result_indexes.append(word_indexes[index])
        index += 1
    return result, result_indexes


def _canonicalize_compound_ordinal_tokens(tokens: list[str]) -> list[str]:
    canonical, _ = _canonicalize_compound_ordinal_tokens_with_indexes(
        tokens,
        list(range(len(tokens))),
    )
    return canonical


def _canonicalize_calendar_date_tokens(tokens: list[str]) -> list[str]:
    """Match written month-day dates to their conventionally spoken ordinal."""
    result = list(tokens)
    for index in range(1, len(result)):
        if result[index - 1] not in CALENDAR_MONTHS:
            continue
        value = result[index]
        ordinal_match = CALENDAR_ORDINAL_RE.fullmatch(value)
        day_text = ordinal_match.group(1) if ordinal_match else value
        if day_text.isdigit() and 1 <= int(day_text) <= 31:
            result[index] = f"calendar-day-{int(day_text)}"
    return result


def _canonicalize_acoustic_phrase_tokens(tokens: list[str]) -> list[str]:
    """Collapse narrow split/join spellings that carry identical speech."""
    result: list[str] = []
    index = 0
    while index < len(tokens):
        for width in (3, 2):
            phrase = tuple(tokens[index:index + width])
            canonical = ACOUSTIC_PHRASE_EQUIVALENTS.get(phrase)
            if canonical is not None:
                result.append(canonical)
                index += width
                break
        else:
            result.append(tokens[index])
            index += 1
            continue
        continue
    return result


def _canonicalize_decimal_tokens(tokens: list[str]) -> list[str]:
    """Collapse a spoken point and its fractional digits into one exact token."""
    result: list[str] = []
    index = 0
    while index < len(tokens):
        if (
            tokens[index].isdigit()
            and index + 2 < len(tokens)
            and tokens[index + 1] == "point"
            and tokens[index + 2].isdigit()
        ):
            fraction_end = index + 3
            while fraction_end < len(tokens) and tokens[fraction_end].isdigit():
                fraction_end += 1
            result.append(
                "decimalnumber"
                f"{tokens[index]}point{''.join(tokens[index + 2:fraction_end])}"
            )
            index = fraction_end
            continue
        result.append(tokens[index])
        index += 1
    return result


def _canonicalize_decimal_transcript_tokens(
    tokens: list[str], word_indexes: list[int]
) -> tuple[list[str], list[int]]:
    """Canonicalize spoken decimals while retaining their true onset word."""
    result: list[str] = []
    result_indexes: list[int] = []
    index = 0
    while index < len(tokens):
        if (
            tokens[index].isdigit()
            and index + 2 < len(tokens)
            and tokens[index + 1] == "point"
            and tokens[index + 2].isdigit()
        ):
            fraction_end = index + 3
            while fraction_end < len(tokens) and tokens[fraction_end].isdigit():
                fraction_end += 1
            result.append(
                "decimalnumber"
                f"{tokens[index]}point{''.join(tokens[index + 2:fraction_end])}"
            )
            result_indexes.append(word_indexes[index])
            index = fraction_end
            continue
        result.append(tokens[index])
        result_indexes.append(word_indexes[index])
        index += 1
    return result, result_indexes


def _canonicalize_number_tokens_with_indexes(
    tokens: list[str], word_indexes: list[int]
) -> tuple[list[str], list[int]]:
    """Collapse number forms while preserving the first contributing word."""
    if len(tokens) != len(word_indexes):
        raise ValueError("Number tokens and word indexes must have equal length")
    # Whisper writes spoken years as one numeric token ("1895"), while the
    # script commonly spells them as "eighteen ninety-five". First collapse a
    # tens+ones pair, then combine two two-digit year halves. Also support the
    # conventional "nineteen oh five" pronunciation.
    simple: list[str] = []
    simple_indexes: list[int] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if (
            token.isdigit()
            and 20 <= int(token) <= 90
            and int(token) % 10 == 0
            and index + 1 < len(tokens)
            and tokens[index + 1].isdigit()
            and 1 <= int(tokens[index + 1]) <= 9
        ):
            simple.append(str(int(token) + int(tokens[index + 1])))
            simple_indexes.append(word_indexes[index])
            index += 2
            continue
        simple.append(token)
        simple_indexes.append(word_indexes[index])
        index += 1
    tokens = simple
    word_indexes = simple_indexes

    result: list[str] = []
    result_indexes: list[int] = []
    index = 0
    while index < len(tokens):
        # Canonicalize standard compound quantities such as "two hundred
        # fifty thousand" before the simpler scale handling below can split
        # them into 200 + 50,000. Whisper may render the same speech as the
        # comma-grouped pair "250" + ",000"; both must resolve to the exact
        # numeric value, while a different value or missing unit still fails.
        if (
            tokens[index].isdigit()
            and 1 <= int(tokens[index]) <= 9
            and index + 3 < len(tokens)
            and tokens[index + 1] == "hundred"
            and tokens[index + 2].isdigit()
            and 1 <= int(tokens[index + 2]) <= 99
            and tokens[index + 3] in {"thousand", "million"}
        ):
            value = (
                int(tokens[index]) * 100 + int(tokens[index + 2])
            ) * NUMBER_SCALES[tokens[index + 3]]
            result.append(str(value))
            result_indexes.append(word_indexes[index])
            index += 4
            continue
        if (
            tokens[index].isdigit()
            and len(tokens[index]) == 2
            and index + 1 < len(tokens)
            and tokens[index + 1].isdigit()
            and len(tokens[index + 1]) == 2
        ):
            result.append(str(int(tokens[index]) * 100 + int(tokens[index + 1])))
            result_indexes.append(word_indexes[index])
            index += 2
            continue
        if (
            tokens[index].isdigit()
            and len(tokens[index]) == 2
            and index + 2 < len(tokens)
            and tokens[index + 1] == "0"
            and tokens[index + 2].isdigit()
            and 1 <= int(tokens[index + 2]) <= 9
        ):
            result.append(str(int(tokens[index]) * 100 + int(tokens[index + 2])))
            result_indexes.append(word_indexes[index])
            index += 3
            continue
        if (
            tokens[index].isdigit()
            and index + 1 < len(tokens)
            and len(tokens[index + 1]) == 3
            and tokens[index + 1].isdigit()
        ):
            result.append(tokens[index] + tokens[index + 1])
            result_indexes.append(word_indexes[index])
            index += 2
            continue
        if (
            tokens[index].isdigit()
            and index + 1 < len(tokens)
            and tokens[index + 1] in NUMBER_SCALES
        ):
            start = index
            current = int(tokens[index])
            index += 1
        elif (
            tokens[index] == "a"
            and index + 1 < len(tokens)
            and tokens[index + 1] in NUMBER_SCALES
        ):
            start = index
            current = 1
        elif tokens[index] in NUMBER_SCALES:
            start = index
            current = 1
        else:
            result.append(tokens[index])
            result_indexes.append(word_indexes[index])
            index += 1
            continue
        if tokens[start] == "a":
            index = start + 1
        elif tokens[start] in NUMBER_SCALES:
            index = start
        first_scale = tokens[index]
        current *= NUMBER_SCALES[first_scale]
        index += 1
        while index < len(tokens) and tokens[index] in NUMBER_SCALES:
            scale = NUMBER_SCALES[tokens[index]]
            current = current * scale if scale >= 1_000 else current + scale
            index += 1
        result.append(str(current))
        result_indexes.append(word_indexes[start])
    return result, result_indexes


def _canonicalize_number_tokens(tokens: list[str]) -> list[str]:
    """Collapse acoustically identical written/spoken English number forms."""
    canonical, _ = _canonicalize_number_tokens_with_indexes(
        tokens,
        list(range(len(tokens))),
    )
    return canonical


def _canonicalize_numeric_range_tokens_with_indexes(
    tokens: list[str],
    word_indexes: list[int],
) -> tuple[list[str], list[int]]:
    """Collapse an exact integer ``from/to`` range without losing its values."""
    if len(tokens) != len(word_indexes):
        raise ValueError("Range tokens and word indexes must have equal length")
    result: list[str] = []
    result_indexes: list[int] = []
    index = 0
    while index < len(tokens):
        if (
            tokens[index].isdigit()
            and index + 2 < len(tokens)
            and tokens[index + 1] == "to"
            and tokens[index + 2].isdigit()
        ):
            result.append(
                f"numberrange{tokens[index]}to{tokens[index + 2]}"
            )
            result_indexes.append(word_indexes[index])
            index += 3
            continue
        result.append(tokens[index])
        result_indexes.append(word_indexes[index])
        index += 1
    return result, result_indexes


def _canonicalize_numeric_range_tokens(tokens: list[str]) -> list[str]:
    canonical, _ = _canonicalize_numeric_range_tokens_with_indexes(
        tokens,
        list(range(len(tokens))),
    )
    return canonical


def _word_indexes_are_contiguous(word_indexes: list[int]) -> bool:
    """Allow multiple tokens from one word or consecutive ASR words only."""
    return bool(word_indexes) and all(
        following - current in (0, 1)
        for current, following in zip(word_indexes, word_indexes[1:])
    )


def _transcript_tokens(words: list[dict]) -> tuple[list[str], list[int]]:
    tokens: list[str] = []
    word_indexes: list[int] = []
    index = 0
    while index < len(words):
        word_text = str(words[index].get("text") or "")
        range_match = re.fullmatch(
            r"\s*([0-9]+)\s*[-–—−]\s*([0-9]+)[,.;:!?]?\s*",
            word_text,
        )
        range_values = (
            (range_match.group(1), range_match.group(2))
            if range_match is not None
            else None
        )
        consumed_range_words = 1
        if range_values is None and index + 1 < len(words):
            range_start = re.fullmatch(r"\s*([0-9]+)\s*", word_text)
            range_end = re.fullmatch(
                r"\s*[-–—−]\s*([0-9]+)[,.;:!?]?\s*",
                str(words[index + 1].get("text") or ""),
            )
            if range_start is not None and range_end is not None:
                range_values = (range_start.group(1), range_end.group(1))
                consumed_range_words = 2
        if range_values is not None:
            tokens.extend([range_values[0], "to", range_values[1]])
            word_indexes.extend([index, index, index + consumed_range_words - 1])
            index += consumed_range_words
            continue
        # Whisper puts the currency qualifier before the amount (HK $7 .08
        # billion / US $903 million), while the spoken qualifier follows it.
        # Expand only an explicit qualifier attached to a dollar amount; a
        # bare HK/US or an unqualified $ must not invent a currency identity.
        currency_qualifier: list[str] = []
        currency_prefix_words = 0
        currency_text = word_text
        qualifier_match = re.fullmatch(r"\s*(HK|US)\s*(\$.*)?\s*", word_text)
        if qualifier_match is not None:
            qualified_amount = qualifier_match.group(2)
            if qualified_amount is None and index + 1 < len(words):
                qualified_amount = str(words[index + 1].get("text") or "")
                currency_prefix_words = 1
            if qualified_amount and CURRENCY_TRANSCRIPT_AMOUNT_RE.fullmatch(
                qualified_amount
            ):
                currency_text = qualified_amount
                currency_qualifier = (
                    ["hong", "kong"] if qualifier_match.group(1) == "HK"
                    else ["u", "s"]
                )
            else:
                currency_prefix_words = 0
        currency_match = CURRENCY_TRANSCRIPT_AMOUNT_RE.fullmatch(currency_text)
        if currency_match is not None:
            integer = currency_match.group(1)
            fraction = currency_match.group(2)
            consumed_words = 1 + currency_prefix_words
            if fraction is None and index + consumed_words < len(words):
                split_fraction = DECIMAL_FRACTION_WORD_RE.fullmatch(
                    str(words[index + consumed_words].get("text") or "")
                )
                if split_fraction is not None:
                    fraction = split_fraction.group(1)
                    consumed_words += 1
            amount_token = (
                f"decimalnumber{integer}point{fraction}"
                if fraction is not None
                else integer
            )
            tokens.append(amount_token)
            word_indexes.append(index)

            following_index = index + consumed_words
            following_tokens = (
                _raw_lexical_tokens(str(words[following_index].get("text") or ""))
                if following_index < len(words)
                else []
            )
            # Whisper conventionally writes spoken currency with the symbol in
            # front ("$6" + ".3" + "billion") even though the acoustic unit
            # follows the scale ("six point three billion dollars"). Preserve
            # that exact unit and order instead of treating the symbol as an
            # extra word or accepting a genuinely missing currency unit.
            has_scale = (
                len(following_tokens) == 1
                and following_tokens[0] in CURRENCY_SCALE_TOKENS
            )
            if has_scale:
                tokens.append(following_tokens[0])
                word_indexes.append(following_index)
                consumed_words += 1
            amount_is_one = (
                integer == "1"
                and (fraction is None or set(fraction) <= {"0"})
                and not has_scale
            )
            tokens.extend(currency_qualifier)
            word_indexes.extend([index] * len(currency_qualifier))
            tokens.append("dollar" if amount_is_one else "dollars")
            word_indexes.append(index)
            index += consumed_words
            continue
        integer_match = DECIMAL_INTEGER_WORD_RE.fullmatch(word_text)
        fraction_match = (
            DECIMAL_FRACTION_WORD_RE.fullmatch(
                str(words[index + 1].get("text") or "")
            )
            if integer_match is not None and index + 1 < len(words)
            else None
        )
        if integer_match is not None and fraction_match is not None:
            tokens.append(
                "decimalnumber"
                f"{integer_match.group(1)}point{fraction_match.group(1)}"
            )
            word_indexes.append(index)
            index += 2
            continue
        for token in _raw_lexical_tokens(word_text):
            tokens.append(token)
            word_indexes.append(index)
        index += 1
    # A provider-only pronunciation hint may lead Whisper to retain the
    # morpheme boundary. The pair is acoustically and lexically identical to
    # the canonical word; a different second morpheme remains a hard failure.
    acoustic_tokens: list[str] = []
    acoustic_indexes: list[int] = []
    cursor = 0
    while cursor < len(tokens):
        matched_phrase = False
        for width in (3, 2):
            phrase = tuple(tokens[cursor:cursor + width])
            canonical_phrase = ACOUSTIC_PHRASE_EQUIVALENTS.get(phrase)
            phrase_indexes = word_indexes[cursor:cursor + width]
            if (
                canonical_phrase is not None
                and _word_indexes_are_contiguous(phrase_indexes)
            ):
                acoustic_tokens.append(canonical_phrase)
                acoustic_indexes.append(word_indexes[cursor])
                cursor += width
                matched_phrase = True
                break
        if matched_phrase:
            continue
        if (
            tokens[cursor] == "dis"
            and cursor + 1 < len(tokens)
            and tokens[cursor + 1] == "proportionate"
            and _word_indexes_are_contiguous(word_indexes[cursor:cursor + 2])
        ):
            acoustic_tokens.append("disproportionate")
            acoustic_indexes.append(word_indexes[cursor])
            cursor += 2
            continue
        acoustic_tokens.append(tokens[cursor])
        acoustic_indexes.append(word_indexes[cursor])
        cursor += 1
    tokens = acoustic_tokens
    word_indexes = acoustic_indexes
    tokens, word_indexes = _canonicalize_decimal_transcript_tokens(
        tokens, word_indexes
    )
    canonical, canonical_indexes = _canonicalize_number_tokens_with_indexes(
        tokens,
        word_indexes,
    )
    canonical, canonical_indexes = _canonicalize_numeric_range_tokens_with_indexes(
        canonical,
        canonical_indexes,
    )
    canonical, canonical_indexes = _canonicalize_compound_ordinal_tokens_with_indexes(
        canonical,
        canonical_indexes,
    )
    return _canonicalize_calendar_date_tokens(canonical), canonical_indexes


def _normalize_currency_adjective_asr_tokens(
    text: str,
    expected: list[str],
    observed: list[str],
    observed_word_indexes: list[int],
    words: list[dict],
) -> list[str]:
    """Recover the singular unit encoded by Whisper's ``$amount scale`` form.

    In an attributive phrase such as ``300-million-dollar Series A`` or
    ``four-billion-dollar plant``, the spoken unit is singular. Whisper
    conventionally writes the same audio as ``$300 million Series A`` or
    ``$4 billion plant``; the currency symbol carries the unit while its
    surface form no longer exposes singular versus plural. Normalize only the
    source-aligned adjective whose observed unit came from that exact currency
    shorthand. Explicit ``dollars``, a different amount/scale, or a missing
    currency symbol remain unchanged and fail the ordinary lexical gate.
    """
    normalized = list(observed)
    for match in CURRENCY_ADJECTIVE_RE.finditer(text):
        phrase = _lexical_tokens(match.group(0))
        if len(phrase) < 2 or phrase[-1] != "dollar":
            continue
        scale = match.group(1).casefold()
        width = len(phrase)
        for start in range(len(expected) - width + 1):
            if expected[start:start + width] != phrase:
                continue
            unit_index = start + width - 1
            if (
                unit_index >= len(normalized)
                or normalized[start:unit_index] != phrase[:-1]
                or normalized[unit_index] != "dollars"
                or unit_index >= len(observed_word_indexes)
            ):
                continue
            raw_index = observed_word_indexes[unit_index]
            if raw_index >= len(words):
                continue
            raw_currency = CURRENCY_TRANSCRIPT_AMOUNT_RE.fullmatch(
                str(words[raw_index].get("text") or "")
            )
            if raw_currency is None:
                continue
            scale_index = raw_index + 1
            if (
                raw_currency.group(2) is None
                and scale_index < len(words)
                and DECIMAL_FRACTION_WORD_RE.fullmatch(
                    str(words[scale_index].get("text") or "")
                )
            ):
                scale_index += 1
            raw_scale = (
                _raw_lexical_tokens(str(words[scale_index].get("text") or ""))
                if scale_index < len(words)
                else []
            )
            if raw_scale == [scale]:
                normalized[unit_index] = "dollar"
    return normalized


def _normalize_qwen_model_number_asr_tokens(
    expected: list[str],
    observed: list[str],
    observed_word_indexes: list[int],
) -> tuple[list[str], list[int]]:
    """Recover evidenced ASR homophones for the provider hint ``Qwen 4``.

    Orpheus receives ``cue-when four`` for the canonical product name. Whisper
    has transcribed complete live realizations as ``queue when four``,
    ``Q went for``, and ``Q when for``. Accept those phrases only when they
    occupy the exact source position of the consecutive canonical tokens
    ``qwen`` and ``4``; unrelated ``went`` or ``for`` tokens remain untouched.
    """
    accepted = {
        ("queue", "when", "4"),
        ("q", "when", "4"),
        ("q", "when", "for"),
        ("cue", "when", "4"),
        ("cue", "when", "for"),
        ("q", "went", "for"),
        ("queue", "wen4"),
    }
    normalized: list[str] = []
    normalized_indexes: list[int] = []
    cursor = 0
    while cursor < len(observed):
        expected_index = len(normalized)
        for width in (3, 2):
            phrase = tuple(observed[cursor:cursor + width])
            phrase_indexes = observed_word_indexes[cursor:cursor + width]
            if (
                expected[expected_index:expected_index + 2] == ["qwen", "4"]
                and phrase in accepted
                and _word_indexes_are_contiguous(phrase_indexes)
            ):
                normalized.extend(("qwen", "4"))
                normalized_indexes.extend(
                    (phrase_indexes[0], phrase_indexes[-1])
                )
                cursor += width
                break
        else:
            normalized.append(observed[cursor])
            normalized_indexes.append(observed_word_indexes[cursor])
            cursor += 1
            continue
        continue
    return normalized, normalized_indexes


def _subsequence_starts(haystack: list[str], needle: list[str]) -> list[int]:
    if not needle or len(needle) > len(haystack):
        return []
    return [
        index
        for index in range(len(haystack) - len(needle) + 1)
        if haystack[index:index + len(needle)] == needle
    ]


def _repetition_start(haystack: list[str], needle: list[str]) -> int | None:
    """Return the second utterance onset, including a truncated repetition.

    Orpheus can spend the remainder of its token budget starting the requested
    sentence again.  Waiting for a second *complete* copy misses that partial
    duplicate, so after locating one complete utterance also look for its
    three-word opening anchor in the trailing transcript.
    """
    complete = _subsequence_starts(haystack, needle)
    if len(complete) > 1:
        return complete[1]
    if len(complete) != 1:
        return None
    anchor_size = min(ORPHEUS_EDGE_ANCHOR_WORDS, len(needle))
    if anchor_size < 2:
        return None
    trailing_start = complete[0] + len(needle)
    opening = needle[:anchor_size]
    for index in range(trailing_start, len(haystack) - anchor_size + 1):
        if haystack[index:index + anchor_size] == opening:
            return index
    return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _split_parallel_clause(sentence: str) -> list[str]:
    """Separate mirrored clauses that make Orpheus repeat the first ending.

    Keep this deliberately narrow: both comma-separated clauses must have the
    same lexical frame after removing their distinct subject and final word.
    For example, ``Japanese go ... die, Chinese go ... win`` becomes two
    independently verified utterances instead of a prompt that the speech LM
    repeatedly expands as ``... die, ... die, ... win``.
    """
    if sentence.count(",") != 1:
        return [sentence]
    left, right = (part.strip() for part in sentence.split(",", 1))
    left_tokens = _lexical_tokens(left)
    right_tokens = _lexical_tokens(right)
    if (
        len(left_tokens) >= 5
        and len(left_tokens) == len(right_tokens)
        and left_tokens[0] != right_tokens[0]
        and left_tokens[-1] != right_tokens[-1]
        and left_tokens[1:-1] == right_tokens[1:-1]
    ):
        return [f"{left},", right]
    return [sentence]


def _separate_repeated_clause_openings(chunks: list[str]) -> list[str]:
    """Force repetition-prone mirrored openings into separate utterances."""
    separated: list[str] = []
    pattern = re.compile(
        r"\bno matter how [^,\n]+,\s+(?=no matter how\b)",
        re.IGNORECASE,
    )
    for chunk in chunks:
        match = pattern.search(chunk)
        if match is None:
            separated.append(chunk)
            continue
        boundary = match.end()
        left = chunk[:boundary].rstrip()
        right = chunk[boundary:].lstrip()
        if left and right:
            separated.extend((left, right))
        else:
            separated.append(chunk)
    return separated


def _separate_repeated_adjective_items(chunks: list[str]) -> list[str]:
    """Split an observed repetition-prone three-item adjective list."""
    separated: list[str] = []
    pattern = re.compile(
        r"\bdisproportionate\s+[^,\n]+,\s+(?=disproportionate\b)",
        re.IGNORECASE,
    )
    for chunk in chunks:
        match = pattern.search(chunk)
        if match is None:
            separated.append(chunk)
            continue
        boundary = match.end()
        left = chunk[:boundary].rstrip()
        right = chunk[boundary:].lstrip()
        if left and right:
            separated.extend((left, right))
        else:
            separated.append(chunk)
    return separated


def _reattach_fragile_orpheus_continuations(chunks: list[str]) -> list[str]:
    """Keep an observed past-tense continuation with its stranded subject."""
    adjusted = list(chunks)
    for index in range(len(adjusted) - 1):
        left = adjusted[index]
        right = adjusted[index + 1]
        subject = re.search(r"(?i)(?:^|\s)(and they)$", left)
        if subject is None or re.match(r"(?i)passed\b", right) is None:
            continue
        prefix = left[: subject.start(1)].rstrip()
        if not prefix:
            continue
        adjusted[index] = prefix
        adjusted[index + 1] = f"{subject.group(1)} {right}"
    return adjusted


def _reattach_fragile_video_prompt_context(chunks: list[str]) -> list[str]:
    """Keep ``video prompts`` together so Orpheus retains the plural /s/."""
    adjusted = list(chunks)
    for index in range(len(adjusted) - 1):
        left = adjusted[index]
        right = adjusted[index + 1]
        context = re.search(r"(?i)\b(with video)$", left)
        if context is None or re.match(r"(?i)prompts,\s+extending\b", right) is None:
            continue
        prefix = left[: context.start(1)].rstrip()
        if not prefix:
            continue
        adjusted[index] = prefix
        adjusted[index + 1] = f"{context.group(1)} {right}"
    return adjusted


def _reattach_dangling_relative_pronoun(chunks: list[str]) -> list[str]:
    """Move a stranded ``which`` onto the clause it grammatically introduces.

    A max-word boundary can leave ``..., which`` as one speech-LM request and
    begin the next with ``the engineers say could ...``. Orpheus repairs that
    fragment by inserting ``it``, so the otherwise fluent audio fails exact
    source coverage. The bounded one-word move preserves every source token and
    gives both utterances complete grammar.
    """
    adjusted = list(chunks)
    for index in range(len(adjusted) - 1):
        left = adjusted[index]
        match = re.search(r"(?i)(?:^|\s)(which)$", left)
        if match is None:
            continue
        prefix = left[: match.start(1)].rstrip()
        right = adjusted[index + 1].lstrip()
        if not prefix or not right:
            continue
        adjusted[index] = prefix
        adjusted[index + 1] = f"{match.group(1)} {right}"
    return adjusted


def _separate_fragile_positioning_clause(chunks: list[str]) -> list[str]:
    """Keep ``positions the release as`` in one grammatical utterance."""
    adjusted = list(chunks)
    index = 0
    pattern = re.compile(
        r"^(.*?[,;])\s+"
        r"(and positions)\s+"
        r"(the release as a lower-priced platform)\s+"
        r"(aimed at .+)$",
        re.IGNORECASE,
    )
    while index < len(adjusted) - 1:
        match = pattern.match(f"{adjusted[index]} {adjusted[index + 1]}")
        if match is None:
            index += 1
            continue
        adjusted[index:index + 2] = [
            match.group(1),
            f"{match.group(2)} {match.group(3)}",
            match.group(4),
        ]
        index += 3
    return adjusted


def _separate_fragile_battle_ready_sequence(chunks: list[str]) -> list[str]:
    """Keep an observed compound phrase intact in shorter utterances."""
    adjusted = list(chunks)
    index = 0
    pattern = re.compile(
        r"^(.*?the most unrestrained,)\s+"
        r"(a battle-ready death cult, sexually open, transformed by drink,)\s+"
        r"(willing to cross any line in art or war\.)$",
        re.IGNORECASE,
    )
    while index < len(adjusted) - 1:
        match = pattern.match(f"{adjusted[index]} {adjusted[index + 1]}")
        if match is None:
            index += 1
            continue
        adjusted[index : index + 2] = [
            match.group(1),
            match.group(2),
            match.group(3),
        ]
        index += 3
    return adjusted


def _separate_fragile_moderation_sequence(chunks: list[str]) -> list[str]:
    """Split an observed repetition-prone three-item moderation list."""
    separated: list[str] = []
    pattern = re.compile(
        r"^(and moderation is beautiful,)\s+"
        r"(moderation is stable,)\s+"
        r"(moderation builds enduring civilizations\.)$",
        re.IGNORECASE,
    )
    for chunk in chunks:
        match = pattern.match(chunk)
        if match is None:
            separated.append(chunk)
            continue
        separated.extend(match.groups())
    return separated


def _stabilize_orpheus_chunks(chunks: list[str]) -> list[str]:
    return _separate_fragile_moderation_sequence(
        _separate_fragile_battle_ready_sequence(
            _separate_fragile_positioning_clause(
                _reattach_fragile_video_prompt_context(
                    _reattach_dangling_relative_pronoun(
                        _reattach_fragile_orpheus_continuations(
                            _separate_repeated_adjective_items(
                                _separate_repeated_clause_openings(chunks)
                            )
                        )
                    )
                )
            )
        )
    )


def _split_tts_text(
    text: str,
    max_words: int,
    *,
    preserve_speaker_labels: bool = False,
) -> list[str]:
    """Split at sentence boundaries while preserving every spoken word.

    Dialogue-capable VibeVoice requires a ``Speaker N:`` marker on every input
    segment. When a long speaker turn crosses a chunk boundary, the marker is
    repeated as metadata; the spoken text itself is neither repeated nor
    dropped.
    """
    if max_words <= 0 or _spoken_word_count(text) <= max_words:
        chunks = [text]
        return (
            chunks
            if preserve_speaker_labels
            else _stabilize_orpheus_chunks(chunks)
        )

    units: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        speaker_label = ""
        content = stripped
        if preserve_speaker_labels:
            match = SPEAKER_LINE_RE.match(stripped)
            if match:
                speaker_label = match.group(1).strip()
                content = match.group(2).strip()
        for sentence in SENTENCE_BOUNDARY_RE.split(content):
            for clause in _split_parallel_clause(sentence):
                words = clause.strip().split()
                while words:
                    take = min(max_words, len(words))
                    if take > 1 and take < len(words):
                        left_name = words[take - 1].strip(
                            ".,!?;:\"'’”()[]{}"
                        )
                        right_name = words[take].strip(
                            ".,!?;:\"'’”()[]{}"
                        )
                        if (
                            left_name[:1].isupper()
                            and right_name[:1].isupper()
                            and any(character.isalpha() for character in left_name)
                            and any(character.isalpha() for character in right_name)
                        ):
                            # Never split an adjacent proper-name pair such as
                            # ``Ant Group`` or ``New York``. A context-free
                            # trailing name fragment is prone to being spoken as
                            # a common function word (observed ``Ant`` ->
                            # ``and``), while moving one token preserves the
                            # complete source and the configured token budget.
                            take -= 1
                    if (
                        take < len(words)
                        and words[take - 1].strip(".,!?;:\"'’”()[]{}").casefold()
                        == "at"
                        and words[take][:1].isupper()
                        and words[take].rstrip("\"'’”)]}").endswith(",")
                    ):
                        # Keep a one-word proper-noun object with a trailing
                        # preposition when it also closes the clause. Sending
                        # "at Germany, ..." as a standalone Orpheus prompt
                        # repeatedly changes the later adjective "disciplined"
                        # to the noun "discipline". The bounded one-word
                        # overflow produces the natural "... look at Germany."
                        take += 1
                    remainder = len(words) - take
                    if 0 < remainder < 5:
                        # Avoid context-starved sentence tails such as "belonged
                        # to the state." Orpheus repeatedly drops inflections in
                        # these fragments. Keep the final phrase attached to its
                        # grammatical context; the bounded four-word overflow is
                        # still independently token-budgeted and verified.
                        take = len(words)
                    while (
                        take > 1
                        and take < len(words)
                        and words[take - 1].strip(".,!?;:\"'’”()[]{}").casefold()
                        in DANGLING_CHUNK_WORDS
                    ):
                        take -= 1
                    if (
                        take > 1
                        and take < len(words)
                        and words[take].strip(".,!?;:\"'’”()[]{}").casefold()
                        in BAD_LEADING_CHUNK_WORDS
                    ):
                        # Do not strand an attached preposition/conjunction at the
                        # start of the next speech-LM request. Move its phrase head
                        # (and an immediately preceding determiner) with it.
                        take -= 1
                        while (
                            take > 1
                            and words[take - 1].strip(".,!?;:\"'’”()[]{}").casefold()
                            in CHUNK_DETERMINERS
                        ):
                            take -= 1
                        # Moving the phrase head can expose a preposition that
                        # the first dangling-tail pass had not seen (for
                        # example ``exports to | Taiwan of``). Re-run the same
                        # invariant so no adjusted chunk ends in a known weak
                        # function word.
                        while (
                            take > 1
                            and take < len(words)
                            and words[take - 1].strip(".,!?;:\"'’”()[]{}").casefold()
                            in DANGLING_CHUNK_WORDS
                        ):
                            take -= 1
                    piece = " ".join(words[:take])
                    words = words[take:]
                    units.append(f"{speaker_label} {piece}".strip())

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if current:
            chunks.append("\n".join(current))
            current = []
            current_words = 0

    for unit in units:
        word_count = _spoken_word_count(unit)
        if current and current_words + word_count > max_words:
            flush()
        current.append(unit)
        current_words += word_count
    flush()
    return (
        chunks
        if preserve_speaker_labels
        else _stabilize_orpheus_chunks(chunks)
    )


def _split_pocket_tts_text(text: str, max_words: int) -> list[str]:
    """Keep Pocket requests on natural script-line and sentence boundaries.

    Pocket TTS is trained on single sentences and already splits a request with
    its tokenizer. Reusing Orpheus' small word chunks would add a second layer
    of arbitrary voice resets and audible joins. Each non-empty physical script
    line therefore remains one request (the Morning Desk writes one opening,
    story, or closing per line). Only an exceptionally long line is divided,
    and then only between complete sentences.
    """
    physical_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not physical_lines:
        return []
    if max_words <= 0:
        return physical_lines

    chunks: list[str] = []
    for line in physical_lines:
        if _spoken_word_count(line) <= max_words:
            chunks.append(line)
            continue

        # Initials and abbreviations are internal punctuation, not safe voice
        # reset points (for example E. coli, U.S. manufacturers, or Dr. Earley).
        # Keeping an ambiguous abbreviation with the next sentence is safer
        # than cutting a spoken name in half; max_words is a soft target.
        boundaries = [0]
        for match in SENTENCE_BOUNDARY_RE.finditer(line):
            if re.search(r"\b(?:(?:[A-Z]\.)+|(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e)\.)$", line[:match.start()]):
                continue
            boundaries.append(match.end())
        boundaries.append(len(line))
        sentences = [line[start:end].strip() for start, end in zip(boundaries, boundaries[1:])]
        if len(sentences) <= 1:
            # Pocket's own tokenizer can still split a punctuation-poor line on
            # commas. Do not introduce a mid-phrase application seam here.
            chunks.append(line)
            continue

        current: list[str] = []
        current_words = 0
        for sentence in sentences:
            sentence_words = _spoken_word_count(sentence)
            if current and current_words + sentence_words > max_words:
                chunks.append(" ".join(current))
                current = []
                current_words = 0
            current.append(sentence)
            current_words += sentence_words
        if current:
            chunks.append(" ".join(current))
    return chunks


def _prepare_tts_input(
    script_path: str,
    output_dir: str,
    *,
    strip_speaker_labels: bool,
) -> tuple[Path, Path, Path, str]:
    """Create the canonical provider-specific TTS input."""
    script_path_obj = Path(script_path).expanduser().resolve()
    output_dir_path = Path(output_dir).expanduser().resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    source = script_path_obj.read_text(encoding="utf-8")
    cleaned = _strip_speaker_labels(source) if strip_speaker_labels else source.strip()
    if not cleaned:
        raise ValueError("TTS input is empty after removing speaker labels")
    tts_input = output_dir_path / "tts_input.txt"
    tts_input.write_text(cleaned, encoding="utf-8")
    return script_path_obj, output_dir_path, tts_input, cleaned


def _expand_vibevoice_pronunciations(text: str) -> tuple[str, list[dict[str, str]]]:
    spoken = text
    applied: list[dict[str, str]] = []
    for canonical, pronunciation in VIBEVOICE_PRONUNCIATIONS:
        pattern = re.compile(rf"(?<![\w-]){re.escape(canonical)}(?![\w-])")
        spoken, count = pattern.subn(pronunciation, spoken)
        if count:
            applied.append(
                {
                    "canonical": canonical,
                    "pronunciation": pronunciation,
                    "occurrences": str(count),
                }
            )
    return spoken, applied


def _write_chunk_inputs(
    text: str,
    output_dir: Path,
    *,
    max_words: int,
    preserve_speaker_labels: bool = False,
) -> tuple[list[Path], list[str]]:
    chunks = _split_tts_text(
        text,
        max_words,
        preserve_speaker_labels=preserve_speaker_labels,
    )
    if len(chunks) == 1:
        return [output_dir / "tts_input.txt"], chunks
    for stale in output_dir.glob("tts_input_part_*.txt"):
        stale.unlink(missing_ok=True)
    input_paths = [
        output_dir / f"tts_input_part_{index:03d}.txt"
        for index in range(1, len(chunks) + 1)
    ]
    for input_path, chunk in zip(input_paths, chunks):
        input_path.write_text(chunk, encoding="utf-8")
    return input_paths, chunks


def _write_pocket_chunk_inputs(
    text: str,
    output_dir: Path,
    *,
    max_words: int,
) -> tuple[list[Path], list[str]]:
    chunks = _split_pocket_tts_text(text, max_words)
    if not chunks:
        raise ValueError("Pocket TTS input is empty after paragraph-aware splitting")
    if len(chunks) == 1:
        return [output_dir / "tts_input.txt"], chunks
    for stale in output_dir.glob("tts_input_part_*.txt"):
        stale.unlink(missing_ok=True)
    input_paths = [
        output_dir / f"tts_input_part_{index:03d}.txt"
        for index in range(1, len(chunks) + 1)
    ]
    for input_path, chunk in zip(input_paths, chunks):
        input_path.write_text(chunk, encoding="utf-8")
    return input_paths, chunks


def _read_pcm_wav(path: Path) -> WavInfo:
    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getcomptype() != "NONE":
                raise TtsIntegrityError(
                    f"TTS output must be uncompressed PCM WAV, got {handle.getcomptype()}"
                )
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            info = WavInfo(
                channels=handle.getnchannels(),
                sample_width=handle.getsampwidth(),
                sample_rate=sample_rate,
                frame_count=frame_count,
                duration_seconds=frame_count / max(1, sample_rate),
            )
    except (wave.Error, EOFError, OSError) as exc:
        raise TtsIntegrityError(f"TTS produced an unreadable WAV at {path}: {exc}") from exc
    if info.frame_count <= 0:
        raise TtsIntegrityError(f"TTS produced an empty WAV at {path}")
    return info


def _validate_wav_part(
    path: Path,
    text: str,
    *,
    token_limit_seconds: float | None = None,
    speed: float = 1.0,
) -> WavInfo:
    info = _read_pcm_wav(path)
    words = _spoken_word_count(text)
    minimum_seconds = words * 60 / (MAX_PLAUSIBLE_SPEECH_WPM * speed)
    if words >= 8 and info.duration_seconds < minimum_seconds:
        raise TtsIntegrityError(
            f"TTS audio is too short for its input: {words} words produced only "
            f"{info.duration_seconds:.1f}s (minimum sanity bound {minimum_seconds:.1f}s)"
        )
    if (
        token_limit_seconds is not None
        and info.duration_seconds >= token_limit_seconds * ORPHEUS_TOKEN_LIMIT_RATIO
    ):
        raise TtsIntegrityError(
            f"Orpheus audio reached {info.duration_seconds:.1f}s, the configured "
            f"max-token ceiling ({token_limit_seconds:.1f}s); refusing a likely "
            "truncated narration"
        )
    return info


def _validate_downloaded_wav_container(
    path: Path,
    *,
    provider: str = "Orpheus",
) -> WavInfo:
    """Reject an invalid or truncated HTTP payload before replacing cached audio."""
    info = _read_pcm_wav(path)
    expected_pcm_bytes = info.frame_count * info.channels * info.sample_width
    try:
        with wave.open(str(path), "rb") as handle:
            pcm_bytes = handle.readframes(info.frame_count)
    except (wave.Error, EOFError, OSError) as exc:
        raise TtsIntegrityError(
            f"{provider} downloaded an unreadable WAV payload at {path}: {exc}"
        ) from exc
    if len(pcm_bytes) != expected_pcm_bytes:
        raise TtsIntegrityError(
            f"{provider} downloaded a truncated WAV payload: "
            f"expected {expected_pcm_bytes} PCM bytes, received {len(pcm_bytes)}"
        )
    return info


def _normalize_pocket_streaming_wav(path: Path) -> WavInfo:
    """Rewrite Pocket's streaming WAV with its actual frame count.

    The upstream HTTP server intentionally writes a one-billion-frame placeholder
    because a chunked response is not seekable. Once downloaded, that header
    would make ordinary WAV readers report hours of nonexistent audio. Rewriting
    only the PCM container header preserves every generated sample and makes the
    artifact safe for hashing, concatenation, Whisper, and the browser player.
    """
    try:
        with wave.open(str(path), "rb") as source:
            if source.getcomptype() != "NONE":
                raise TtsIntegrityError(
                    f"Pocket TTS output must be PCM WAV, got {source.getcomptype()}"
                )
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            pcm_bytes = source.readframes(source.getnframes())
    except (wave.Error, EOFError, OSError) as exc:
        raise TtsIntegrityError(
            f"Pocket TTS returned an unreadable streaming WAV at {path}: {exc}"
        ) from exc
    frame_width = channels * sample_width
    if frame_width <= 0 or not pcm_bytes or len(pcm_bytes) % frame_width:
        raise TtsIntegrityError(
            "Pocket TTS returned incomplete PCM data that cannot form whole audio frames"
        )

    normalized = path.with_suffix(".normalized.tmp.wav")
    normalized.unlink(missing_ok=True)
    try:
        with wave.open(str(normalized), "wb") as destination:
            destination.setnchannels(channels)
            destination.setsampwidth(sample_width)
            destination.setframerate(sample_rate)
            destination.writeframes(pcm_bytes)
        os.replace(normalized, path)
    finally:
        normalized.unlink(missing_ok=True)
    return _validate_downloaded_wav_container(path, provider="Pocket TTS")


def _wav_edge_silence_seconds(
    path: Path,
    *,
    threshold_dbfs: float = POCKET_TTS_EDGE_SILENCE_DBFS,
    window_ms: int = POCKET_TTS_SILENCE_WINDOW_MS,
) -> dict[str, float]:
    """Measure contiguous quiet windows at both edges of a 16-bit PCM WAV."""
    quiet_windows = _wav_quiet_windows(
        path,
        threshold_dbfs=threshold_dbfs,
        window_ms=window_ms,
    )
    leading_windows = next(
        (index for index, quiet in enumerate(quiet_windows) if not quiet),
        len(quiet_windows),
    )
    trailing_windows = next(
        (index for index, quiet in enumerate(reversed(quiet_windows)) if not quiet),
        len(quiet_windows),
    )
    return {
        "leading_seconds": round(leading_windows * window_ms / 1000, 3),
        "trailing_seconds": round(trailing_windows * window_ms / 1000, 3),
    }


def _wav_quiet_windows(
    path: Path,
    *,
    threshold_dbfs: float = POCKET_TTS_EDGE_SILENCE_DBFS,
    window_ms: int = POCKET_TTS_SILENCE_WINDOW_MS,
) -> list[bool]:
    with wave.open(str(path), "rb") as source:
        if source.getcomptype() != "NONE" or source.getsampwidth() != 2:
            raise TtsIntegrityError(
                "Pocket TTS continuity analysis requires uncompressed 16-bit PCM WAV"
            )
        channels = source.getnchannels()
        sample_rate = source.getframerate()
        pcm_bytes = source.readframes(source.getnframes())
    samples = array("h")
    samples.frombytes(pcm_bytes)
    if sys.byteorder != "little":
        samples.byteswap()
    samples_per_window = max(1, int(sample_rate * window_ms / 1000)) * channels
    quiet_threshold = 32767 * (10 ** (threshold_dbfs / 20))
    quiet_windows: list[bool] = []
    for start in range(0, len(samples), samples_per_window):
        values = samples[start : start + samples_per_window]
        rms = math.sqrt(sum(value * value for value in values) / max(1, len(values)))
        quiet_windows.append(rms < quiet_threshold)

    return quiet_windows


def _pocket_internal_silence_report(
    path: Path,
    transcript_words: list[dict] | None = None,
) -> dict:
    """Detect choppy pauses without rejecting ordinary sentence cadence.

    Pocket TTS performs its own sentence splitting. Its sentence joins can be
    slightly longer than ordinary phrase pauses, so a single waveform-only
    ceiling makes natural delivery fail nondeterministically. When the normal
    acoustic transcript is available, permit a narrow sentence-boundary margin
    while retaining the original ceiling everywhere else.
    """
    quiet_windows = _wav_quiet_windows(path)
    minimum_windows = math.ceil(
        POCKET_TTS_LONG_SILENCE_SECONDS * 1000 / POCKET_TTS_SILENCE_WINDOW_MS
    )
    raw_runs: list[dict[str, float]] = []
    start: int | None = None
    for index, quiet in enumerate([*quiet_windows, False]):
        if quiet and start is None:
            start = index
            continue
        if quiet or start is None:
            continue
        # Leading/trailing silence belongs to the physical program boundary,
        # not to delivery inside the story. Only internal runs can be a stutter.
        if start > 0 and index < len(quiet_windows) and index - start >= minimum_windows:
            raw_runs.append(
                {
                    "start_seconds": round(
                        start * POCKET_TTS_SILENCE_WINDOW_MS / 1000,
                        3,
                    ),
                    "end_seconds": round(
                        index * POCKET_TTS_SILENCE_WINDOW_MS / 1000,
                        3,
                    ),
                    "duration_seconds": round(
                        (index - start) * POCKET_TTS_SILENCE_WINDOW_MS / 1000,
                        3,
                    ),
                }
            )
        start = None

    words = [
        word
        for word in transcript_words or []
        if str(word.get("text") or "").strip()
    ]
    runs: list[dict] = []
    for raw_run in raw_runs:
        run_start = raw_run["start_seconds"]
        run_end = raw_run["end_seconds"]
        preceding = [
            word
            for word in words
            if float(word.get("end") or 0) <= run_start
        ]
        following = [
            word
            for word in words
            if float(word.get("start") or 0) >= run_start
        ]
        previous_word = max(
            preceding,
            key=lambda word: float(word.get("end") or 0),
            default=None,
        )
        next_word = min(
            following,
            key=lambda word: float(word.get("start") or 0),
            default=None,
        )
        previous_text = str((previous_word or {}).get("text") or "").strip()
        previous_end = float((previous_word or {}).get("end") or 0)
        next_start = float((next_word or {}).get("start") or 0)
        sentence_boundary = bool(
            previous_word
            and next_word
            and re.search(r"[.!?][\"')\]]*$", previous_text)
            and 0 <= run_start - previous_end <= 0.6
            and run_start <= next_start <= run_end + 0.6
        )
        allowed_seconds = (
            POCKET_TTS_MAX_SENTENCE_BOUNDARY_SILENCE_SECONDS
            if sentence_boundary
            else POCKET_TTS_MAX_INTERNAL_SILENCE_SECONDS
        )
        runs.append(
            {
                **raw_run,
                "sentence_boundary": sentence_boundary,
                "previous_word": previous_text or None,
                "next_word": (
                    str((next_word or {}).get("text") or "").strip() or None
                ),
                "maximum_allowed_seconds": allowed_seconds,
                "passed": raw_run["duration_seconds"] <= allowed_seconds,
            }
        )

    durations = [run["duration_seconds"] for run in runs]
    maximum = max(durations, default=0.0)
    return {
        "minimum_reported_seconds": POCKET_TTS_LONG_SILENCE_SECONDS,
        "maximum_allowed_seconds": POCKET_TTS_MAX_INTERNAL_SILENCE_SECONDS,
        "maximum_sentence_boundary_allowed_seconds": (
            POCKET_TTS_MAX_SENTENCE_BOUNDARY_SILENCE_SECONDS
        ),
        "count": len(runs),
        "durations_seconds": durations,
        "max_seconds": round(maximum, 3),
        "runs": runs,
        "passed": all(run["passed"] for run in runs),
    }


def _validate_pocket_internal_silence(
    path: Path,
    transcript_words: list[dict] | None = None,
) -> dict:
    report = _pocket_internal_silence_report(path, transcript_words)
    if not report["passed"]:
        violation = next(run for run in report["runs"] if not run["passed"])
        boundary_label = (
            " sentence-boundary" if violation["sentence_boundary"] else ""
        )
        raise TtsIntegrityError(
            f"Pocket TTS produced an internal{boundary_label} pause of "
            f"{violation['duration_seconds']:.2f}s, above the "
            f"{violation['maximum_allowed_seconds']:.2f}s continuity ceiling"
        )
    return report


def _pocket_continuity_report(
    source_text: str,
    chunks: list[str],
    wav_parts: list[Path],
    part_metadata: list[dict] | None = None,
) -> dict:
    """Record both structural joins and measured silence at request boundaries."""
    physical_line_count = len(
        [line for line in source_text.splitlines() if line.strip()]
    )
    edges = [_wav_edge_silence_seconds(path) for path in wav_parts]
    internal_silence = []
    for index, path in enumerate(wav_parts):
        metadata = (
            (part_metadata or [])[index]
            if index < len(part_metadata or [])
            else {}
        )
        persisted = (metadata.get("integrity") or {}).get("internal_silence")
        internal_silence.append(
            persisted
            if isinstance(persisted, dict)
            else _pocket_internal_silence_report(path)
        )
    joins = [
        round(edges[index]["trailing_seconds"] + edges[index + 1]["leading_seconds"], 3)
        for index in range(max(0, len(edges) - 1))
    ]
    ordered_joins = sorted(joins)
    if not ordered_joins:
        median_join = 0.0
    elif len(ordered_joins) % 2:
        median_join = ordered_joins[len(ordered_joins) // 2]
    else:
        middle = len(ordered_joins) // 2
        median_join = (ordered_joins[middle - 1] + ordered_joins[middle]) / 2
    return {
        "strategy": "physical_script_lines_then_complete_sentences",
        "external_chunk_word_limit": config.POCKET_TTS_CHUNK_WORDS,
        "provider_internal_max_tokens": POCKET_TTS_INTERNAL_MAX_TOKENS,
        "provider_fixed_trailing_padding_seconds": 0.2,
        "physical_script_line_count": physical_line_count,
        "external_part_count": len(chunks),
        "application_join_count": max(0, len(chunks) - 1),
        "intra_line_application_join_count": max(0, len(chunks) - physical_line_count),
        "arbitrary_mid_sentence_splits": 0,
        "silence_measurement": {
            "threshold_dbfs": POCKET_TTS_EDGE_SILENCE_DBFS,
            "window_ms": POCKET_TTS_SILENCE_WINDOW_MS,
            "part_edges": edges,
            "join_seconds": joins,
            "median_join_seconds": round(median_join, 3),
            "max_join_seconds": round(max(joins, default=0), 3),
            "total_join_seconds": round(sum(joins), 3),
            "internal_long_silence_count": sum(item["count"] for item in internal_silence),
            "max_internal_silence_seconds": round(
                max((item["max_seconds"] for item in internal_silence), default=0),
                3,
            ),
            "internal_part_reports": internal_silence,
        },
    }


async def _concat_wav_parts(
    wav_parts: list[Path], output_dir: Path, *, log: LogCallback | None
) -> Path:
    if len(wav_parts) == 1:
        return wav_parts[0]
    expected = output_dir / "tts_input_generated.wav"
    infos = [_read_pcm_wav(part) for part in wav_parts]
    reference = infos[0]
    for part, info in zip(wav_parts[1:], infos[1:]):
        if (
            info.channels,
            info.sample_width,
            info.sample_rate,
        ) != (
            reference.channels,
            reference.sample_width,
            reference.sample_rate,
        ):
            raise TtsIntegrityError(
                f"TTS WAV format changed between chunks at {part}: "
                f"expected {reference.channels}ch/{reference.sample_width * 8}bit/"
                f"{reference.sample_rate}Hz, got {info.channels}ch/"
                f"{info.sample_width * 8}bit/{info.sample_rate}Hz"
            )

    staged = output_dir / "tts_input_generated.tmp.wav"
    staged.unlink(missing_ok=True)
    with wave.open(str(staged), "wb") as destination:
        destination.setnchannels(reference.channels)
        destination.setsampwidth(reference.sample_width)
        destination.setframerate(reference.sample_rate)
        for part in wav_parts:
            with wave.open(str(part), "rb") as source:
                destination.writeframesraw(source.readframes(source.getnframes()))
    os.replace(staged, expected)
    joined = _read_pcm_wav(expected)
    expected_frames = sum(info.frame_count for info in infos)
    if joined.frame_count != expected_frames:
        raise TtsIntegrityError(
            f"Lossless WAV join wrote {joined.frame_count} frames; expected {expected_frames}"
        )
    return expected


def _write_tts_manifest(
    output_dir: Path,
    *,
    model: str,
    source_text: str,
    chunks: list[str],
    wav_parts: list[Path],
    output: Path,
    deterministic: bool,
    integrity: dict | None = None,
    extra: dict | None = None,
) -> None:
    parts = []
    for text, path in zip(chunks, wav_parts):
        info = _read_pcm_wav(path)
        parts.append(
            {
                "input": path.with_suffix(".txt").name.replace("_generated", ""),
                "output": path.name,
                "audio_sha256": _file_sha256(path),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "word_count": _spoken_word_count(text),
                "synthesis_speed_ratio": NARRATION_SYNTHESIS_SPEED_RATIO,
                **asdict(info),
            }
        )
    payload = {
        "model": model,
        "deterministic_decoding": deterministic,
        "pacing_policy": NARRATION_PACING_POLICY,
        "synthesis_speed_ratio": NARRATION_SYNTHESIS_SPEED_RATIO,
        "source_text_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "source_word_count": _spoken_word_count(source_text),
        "chunk_count": len(parts),
        "parts": parts,
        "output": output.name,
        "output_audio_sha256": _file_sha256(output),
        "output_wav": asdict(_read_pcm_wav(output)),
    }
    if integrity is not None:
        payload["integrity"] = integrity
    if extra:
        payload.update(extra)
    (output_dir / "tts_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _orpheus_chunk_word_limit(max_tokens: int) -> int:
    audio_seconds = max_tokens / ORPHEUS_AUDIO_TOKENS_PER_SECOND
    return max(
        1,
        int(audio_seconds * ORPHEUS_CHUNK_MIN_WPM / 60 * ORPHEUS_CHUNK_SAFETY),
    )


def _orpheus_request_token_budget(text: str, maximum: int) -> int:
    """Bound one short utterance without starving slow natural delivery."""
    expected_seconds = _spoken_word_count(text) * 60 / ORPHEUS_CHUNK_MIN_WPM
    estimated = math.ceil(
        expected_seconds * ORPHEUS_AUDIO_TOKENS_PER_SECOND / ORPHEUS_CHUNK_SAFETY
    )
    return min(maximum, max(ORPHEUS_MIN_REQUEST_TOKENS, estimated))


def _orpheus_prompt_text(text: str) -> str:
    """Give every short LM request an explicit speech termination boundary."""
    stripped = text.rstrip()
    stripped = re.sub(
        r"\bJalapeño\b",
        "Jalapeno",
        stripped,
        flags=re.IGNORECASE,
    )
    stripped = re.sub(
        r"(?<![\w-])V4-Flash(?![\w-])",
        "V four Flash",
        stripped,
    )
    # Keep the generation number spoken as a word and bind it to the proven
    # two-syllable Qwen hint. A bare digit produced unstable Q Lune / queue went
    # realizations across repeated live samples.
    stripped = re.sub(
        r"(?<![\w-])Qwen\s+4(?![\w-])",
        "cue-when four",
        stripped,
    )
    # The live speech model twice realized Qwen as the one-syllable surname
    # "Khan". Expose the intended two-part pronunciation to the provider;
    # canonical verification still requires Qwen in the ASR result.
    stripped = re.sub(r"(?<![\w-])Qwen(?=\d)", "cue-when ", stripped)
    stripped = re.sub(
        r"(?<![\w-])Qwen(?![\w-])",
        "cue-when",
        stripped,
    )
    # The same live model produced unstable and incomplete realizations of
    # Qianwen (Can Wen / Chan 'en / Chanmen) across two generations and three
    # playback speeds. Give the provider a two-syllable pronunciation target;
    # ASR verification remains anchored to the canonical product name.
    stripped = re.sub(
        r"(?<![\w-])Qianwen(?![\w-])",
        "Chien-Wen",
        stripped,
    )
    # Uppercase NERVA was repeatedly realized as the unrelated word "NAL" at
    # normal, 0.8x, and 0.7x ASR playback. Expose its conventional two-syllable
    # pronunciation only to the provider; the canonical script and acoustic
    # verification still require the complete NERVA-and-Rover name pair.
    stripped = re.sub(
        r"(?<![\w-])NERVA(?=\s+and\s+Rover\b)",
        "Ner-vuh",
        stripped,
    )
    # The English Orpheus voice repeatedly realized the live attribution
    # "Zhu Yi, co-founder of Prana Labs" as "Su Yi" and merged the company
    # name into "Pranilabs". Give only this evidenced transition an explicit
    # pinyin target and articulation boundaries. The canonical script stays
    # unchanged, and ASR verification still requires the full person and
    # company names in their original positions.
    stripped = re.sub(
        r"\bZhu Yi,\s+co-founder of Prana Labs\b",
        "Joo Yee. Co-founder of Prana. Labs",
        stripped,
    )
    # Orpheus consistently realizes the publication name Techmeme as
    # "TechMean" unless the two semantic words are exposed. Keep the source
    # script canonical and join only the exact "Tech Meme" ASR token pair.
    stripped = re.sub(r"\bTechmeme\b", "Tech Meme", stripped)
    # The provider repeatedly substituted the familiar Disney place name
    # "Arendelle" for the satellite name Earendil-1. Expose the intended
    # syllables and number only in the speech prompt; verification still
    # rejects Arendelle and requires all three name syllables plus the number.
    stripped = re.sub(r"\bEarendil-1\b", "Ear-en-dill one", stripped)
    # The speech LM can parse the CamelCase publication name as "two-bit AI".
    # Expose the intended letter and word boundaries only in the provider
    # prompt; verification still requires Whisper to recover QbitAI/Qubit AI.
    stripped = re.sub(
        r"(?<![\w-])QbitAI(?![\w-])",
        "Q-bit A-I",
        stripped,
    )
    # Orpheus realized the possessive German surname with the central vowel in
    # "Brum" at every ASR playback speed. The conventional silent-h spelling
    # exposes the intended /ɛ/ pronunciation to the provider while canonical
    # verification still requires Brem's (or the exact Brehm's spelling).
    stripped = re.sub(
        r"(?<![\w-])Brem(['’]s)(?![\w-])",
        r"Brehm\1",
        stripped,
        flags=re.IGNORECASE,
    )
    # A generated continuation beginning with this third-person verb repeatedly
    # lost its final /s/, including after a ``describe-s`` tokenizer hint. Give
    # the exact observed transition an articulation pause; the canonical
    # transcript must still contain the exact word "describes".
    stripped = re.sub(
        r"^(\s*[Dd]escribes)\s+(situations\b)",
        lambda match: f"{match.group(1)}. {match.group(2).capitalize()}",
        stripped,
        flags=re.IGNORECASE,
    )
    # Orpheus repeatedly realizes the opening phrase "Months of" as singular
    # "Month of". Expose the final plural morpheme to its tokenizer; the
    # canonical script remains unchanged and ASR must still recover "months".
    stripped = re.sub(
        r"^(\s*Months)\s+(of)\b",
        lambda match: f"{match.group(1)[:-1]}-s {match.group(2)}",
        stripped,
        flags=re.IGNORECASE,
    )
    # This three-part list repeatedly makes Orpheus pluralize the final gerund
    # as the non-word "declarings". Provider-only sentence boundaries retain
    # every lexical token while removing the misleading noun-list prosody.
    stripped = re.sub(
        r"\b(proxy conflicts),\s+(aid without troops),\s+"
        r"(arming without declaring)\b",
        lambda match: (
            f"{match.group(1)}. {match.group(2).capitalize()}. "
            f"{match.group(3).capitalize()}"
        ),
        stripped,
        flags=re.IGNORECASE,
    )
    # In a comma-separated adjective list Orpheus repeatedly drops the final
    # /d/ from "disciplined" and speaks the noun "discipline" instead. Give
    # that exact observed transition a stronger, unspoken articulation pause;
    # the canonical text and the acoustic words being verified stay unchanged.
    stripped = re.sub(
        r"\b(disciplined),\s+(formidable)\b",
        lambda match: f"{match.group(1)}. {match.group(2).capitalize()}",
        stripped,
        flags=re.IGNORECASE,
    )
    # Orpheus repeatedly realizes "passed that love" as present-tense
    # "pass that love", even when the subject is present.  A provider-only
    # sentence pause makes the final /t/ audible while preserving exactly the
    # same lexical script for verification and concatenation.
    stripped = re.sub(
        r"\b(passed)\s+(that love)\b",
        lambda match: f"{match.group(1)}. {match.group(2).capitalize()}",
        stripped,
        flags=re.IGNORECASE,
    )
    # The speech LM repeatedly substitutes "precautionate" for the middle of
    # this uncommon word. Expose the real morpheme boundary to its tokenizer;
    # the canonical script remains unchanged and ASR must still recover the
    # exact word (or the exact `dis` + `proportionate` acoustic pair).
    stripped = re.sub(
        r"\bdisproportionate\b",
        "dis-proportionate",
        stripped,
        flags=re.IGNORECASE,
    )
    if TERMINAL_SPEECH_PUNCTUATION_RE.search(stripped):
        return stripped
    # A canonical chunk may end at a comma/semicolon chosen for semantic
    # splitting.  Replace that delimiter only in the provider prompt; appending
    # a period would otherwise create the malformed sequence `,.`.
    clause_terminated = TRAILING_CLAUSE_PUNCTUATION_RE.sub(r".\1", stripped)
    return clause_terminated if clause_terminated != stripped else stripped + "."


def _orpheus_transcript_report(text: str, words: list[dict]) -> dict:
    """Measure whether a short WAV contains its complete requested utterance.

    Whisper is an independent acoustic observer, so a complete name can receive
    a different but acoustically equivalent spelling. The primary path remains
    exact. A bounded fallback permits one general aligned phonetic spelling
    substitution, or up to two substitutions when every pair is explicitly
    evidenced, acoustic coverage is complete, and word count is unchanged.
    Explicit evidenced pairs may qualify even in a short utterance where the
    differences drop exact coverage below 90%; general similarity still needs
    the 90% floor. Substitutions may positionally anchor an utterance edge, but
    the caller must corroborate them by transcribing the same waveform at
    another playback speed.
    """
    expected = _lexical_tokens(text)
    observed, observed_word_indexes = _transcript_tokens(words)
    observed, observed_word_indexes = _collapse_expected_name_splits(
        expected,
        observed,
        observed_word_indexes,
    )
    observed, observed_word_indexes = _normalize_qwen_model_number_asr_tokens(
        expected,
        observed,
        observed_word_indexes,
    )
    observed = _normalize_currency_adjective_asr_tokens(
        text,
        expected,
        observed,
        observed_word_indexes,
        words,
    )
    matcher = SequenceMatcher(a=expected, b=observed, autojunk=False)
    pairs: list[tuple[int, int]] = []
    for block in matcher.get_matching_blocks():
        pairs.extend((block.a + offset, block.b + offset) for offset in range(block.size))

    matched_expected = {left for left, _ in pairs}
    exact_coverage = len(matched_expected) / max(1, len(expected))
    word_ratio = len(observed) / max(1, len(expected))
    speech_end = max((float(word.get("end") or 0) for word in words), default=0.0)
    repetition_start = _repetition_start(observed, expected)
    repeat_start_seconds = None
    if repetition_start is not None:
        repeat_word_index = observed_word_indexes[repetition_start]
        repeat_start_seconds = max(0.0, float(words[repeat_word_index].get("start") or 0))

    phonetic_substitutions = _aligned_phonetic_substitutions(expected, observed)
    substitution_indexes = {
        int(item["expected_index"])
        for item in phonetic_substitutions or []
    }
    edge = min(ORPHEUS_EXACT_EDGE_ANCHOR_WORDS, len(expected))
    exact_leading_anchor = expected[:edge] == observed[:edge]
    exact_trailing_anchor = expected[-edge:] == observed[-edge:]

    def position_is_anchored(index: int) -> bool:
        return index < len(observed) and (
            expected[index] == observed[index] or index in substitution_indexes
        )

    leading_anchor = all(position_is_anchored(index) for index in range(edge))
    # A middle omission shifts absolute indexes without removing the ending.
    # Report the acoustic edge truthfully; coverage still rejects the omission.
    trailing_anchor = exact_trailing_anchor or (
        len(expected) == len(observed)
        and all(
            position_is_anchored(index)
            for index in range(max(0, len(expected) - edge), len(expected))
        )
    )
    matched_acoustic_words = len(matched_expected)
    if phonetic_substitutions:
        matched_acoustic_words += len(phonetic_substitutions)
    acoustic_coverage = matched_acoustic_words / max(1, len(expected))
    complete_evidenced_phonetic_candidate = bool(phonetic_substitutions) and (
        acoustic_coverage == 1.0
        and word_ratio == 1.0
        and all(
            str(item.get("phonetic_key") or "").startswith("evidenced:")
            for item in phonetic_substitutions
        )
    )

    failures: list[str] = []
    if (
        exact_coverage < ORPHEUS_MIN_EXACT_ASR_COVERAGE
        and not complete_evidenced_phonetic_candidate
    ):
        failures.append(
            f"exact ASR word coverage {exact_coverage:.1%} is below "
            f"{ORPHEUS_MIN_EXACT_ASR_COVERAGE:.1%}"
        )
    elif exact_coverage < 1.0 and not phonetic_substitutions:
        failures.append(
            "ASR mismatch is not one aligned high-confidence phonetic spelling "
            "substitution"
        )
    if word_ratio < ORPHEUS_MIN_ASR_WORD_RATIO:
        failures.append(
            f"ASR returned only {len(observed)}/{len(expected)} expected-scale words"
        )
    if word_ratio > ORPHEUS_MAX_ASR_WORD_RATIO:
        failures.append(
            f"ASR returned {len(observed)}/{len(expected)} expected-scale words; "
            "the utterance was likely repeated"
        )
    if not leading_anchor:
        failures.append("opening words have no acoustic transcript anchor")
    if not trailing_anchor:
        failures.append("closing words have no acoustic transcript anchor")

    return {
        "verified": not failures,
        "expected_words": len(expected),
        "transcript_words": len(observed),
        "matched_exact_words": len(matched_expected),
        "exact_asr_word_coverage": round(exact_coverage, 4),
        "matched_acoustic_words": matched_acoustic_words,
        "acoustic_asr_word_coverage": round(acoustic_coverage, 4),
        "phonetic_substitutions": phonetic_substitutions or [],
        "verification_mode": (
            "aligned_phonetic_substitution"
            if phonetic_substitutions
            else "exact"
        ),
        "transcript_word_ratio": round(word_ratio, 4),
        "leading_anchor": leading_anchor,
        "trailing_anchor": trailing_anchor,
        "exact_leading_anchor": exact_leading_anchor,
        "exact_trailing_anchor": exact_trailing_anchor,
        "speech_end_seconds": round(speech_end, 3),
        "repeat_start_seconds": (
            round(repeat_start_seconds, 3) if repeat_start_seconds is not None else None
        ),
        "failure_reasons": failures,
    }


def _english_phonetic_key(token: str) -> str:
    """Return a conservative grapheme-to-sound key for ASR spelling drift.

    This is intentionally narrower than Soundex: vowel position and audible
    suffix consonants remain significant, so words such as ``foundation`` and
    ``foundational`` cannot collapse to the same key.
    """
    value = re.sub(r"[^a-z]", "", token.casefold())
    if not value:
        return ""
    value = re.sub(r"^(?:kn|gn|pn)", lambda match: match.group(0)[1:], value)
    value = re.sub(r"^wr", "r", value)
    value = re.sub(r"^wh", "w", value)
    value = value.replace("sch", "sk")
    value = value.replace("tch", "ch")
    value = value.replace("ph", "f")
    value = value.replace("gh", "")
    value = value.replace("ck", "k")
    value = value.replace("qu", "kw")
    value = re.sub(r"c(?=[eiy])", "s", value)
    value = value.replace("c", "k")
    value = re.sub(r"g(?=[eiy])", "j", value)
    value = re.sub(r"(?<![tscw])h", "", value)
    # The silent spelling vowel before a final inflection is not acoustic.
    value = re.sub(r"e(?=[ds]$)", "", value)
    value = re.sub(r"e$", "", value)
    value = re.sub(r"(.)\1+", r"\1", value)
    return value


def _aligned_phonetic_substitutions(
    expected: list[str],
    observed: list[str],
) -> list[dict] | None:
    """Return bounded safe aligned spelling substitutions, or ``None``.

    Equal token counts and positional comparison deliberately reject a missing
    word compensated by an unrelated extra word elsewhere in the utterance.
    General phonetic similarity remains limited to one substitution; a second
    is allowed only when every substitution is an explicitly evidenced pair.
    """
    if len(expected) != len(observed):
        return None
    substitutions: list[dict] = []
    for index, (expected_token, observed_token) in enumerate(zip(expected, observed)):
        if expected_token == observed_token:
            continue
        # Canonical decimals/ranges and model identifiers contain digits.
        # The phonetic key strips nonletters, so comparing those keys would
        # otherwise equate different amounts such as 7.08 and 7.09.
        if any(character.isdigit() for character in expected_token + observed_token):
            return None
        spelling_similarity = SequenceMatcher(
            a=expected_token,
            b=observed_token,
            autojunk=False,
        ).ratio()
        expected_key = _english_phonetic_key(expected_token)
        observed_key = _english_phonetic_key(observed_token)
        evidenced_pair = (
            frozenset({expected_token, observed_token})
            in ORPHEUS_EVIDENCED_PHONETIC_PAIRS
        )
        if (
            not evidenced_pair
            and (
                not expected_key
                or expected_key != observed_key
                or spelling_similarity < ORPHEUS_MIN_PHONETIC_SPELLING_SIMILARITY
            )
        ):
            return None
        substitutions.append(
            {
                "expected_index": index,
                "expected": expected_token,
                "observed": observed_token,
                "phonetic_key": (
                    f"evidenced:{expected_token}-{observed_token}"
                    if evidenced_pair
                    else expected_key
                ),
                "spelling_similarity": round(spelling_similarity, 4),
            }
        )
        if len(substitutions) > ORPHEUS_MAX_PHONETIC_SUBSTITUTIONS:
            all_evidenced = all(
                str(item["phonetic_key"]).startswith("evidenced:")
                for item in substitutions
            )
            if (
                not all_evidenced
                or len(substitutions)
                > ORPHEUS_MAX_EVIDENCED_PHONETIC_SUBSTITUTIONS
            ):
                return None
    return substitutions or None


def _collapse_expected_name_splits(
    expected: list[str],
    observed: list[str],
    observed_word_indexes: list[int],
) -> tuple[list[str], list[int]]:
    """Collapse only proven split spellings aligned to a canonical source name."""
    replacements: dict[int, tuple[int, str]] = {}
    matcher = SequenceMatcher(a=expected, b=observed, autojunk=False)
    for tag, expected_start, expected_end, observed_start, observed_end in (
        matcher.get_opcodes()
    ):
        if tag != "replace" or expected_end - expected_start != 1:
            continue
        expected_token = expected[expected_start]
        expected_name = _name_recheck_base(expected_token)
        accepted_splits = ORPHEUS_NAME_ACOUSTIC_SPLITS.get(expected_name)
        if accepted_splits is None:
            continue
        observed_delta = tuple(observed[observed_start:observed_end])
        if not observed_delta:
            continue
        observed_possessive = observed_delta[-1].endswith("'s")
        if expected_token.endswith("'s") != observed_possessive:
            continue
        observed_split = (
            *observed_delta[:-1],
            _name_recheck_base(observed_delta[-1]),
        )
        contributing_indexes = observed_word_indexes[observed_start:observed_end]
        if (
            observed_split in accepted_splits
            and _word_indexes_are_contiguous(contributing_indexes)
        ):
            replacements[observed_start] = (observed_end, expected_token)

    if not replacements:
        return observed, observed_word_indexes

    collapsed: list[str] = []
    collapsed_word_indexes: list[int] = []
    cursor = 0
    while cursor < len(observed):
        replacement = replacements.get(cursor)
        if replacement is None:
            collapsed.append(observed[cursor])
            collapsed_word_indexes.append(observed_word_indexes[cursor])
            cursor += 1
            continue
        end, canonical = replacement
        collapsed.append(canonical)
        collapsed_word_indexes.append(observed_word_indexes[cursor])
        cursor = end
    return collapsed, collapsed_word_indexes


def _trim_pcm_wav(path: Path, end_seconds: float) -> None:
    info = _read_pcm_wav(path)
    end_frame = min(info.frame_count, max(1, round(end_seconds * info.sample_rate)))
    staged = path.with_suffix(".trim.tmp.wav")
    staged.unlink(missing_ok=True)
    with wave.open(str(path), "rb") as source, wave.open(str(staged), "wb") as destination:
        destination.setparams(source.getparams())
        destination.writeframes(source.readframes(end_frame))
    os.replace(staged, path)


def _is_name_recheck_token(token: str) -> bool:
    base = token[:-2] if token.endswith("'s") else token
    return base in ORPHEUS_NAME_RECHECK_TOKENS


def _name_recheck_base(token: str) -> str:
    return token[:-2] if token.endswith("'s") else token


def _needs_name_playback_recheck(text: str) -> bool:
    return any(_is_name_recheck_token(token) for token in _raw_lexical_tokens(text))


def _has_only_name_transcript_mismatches(text: str, words: list[dict]) -> bool:
    """Permit slow replay only when every normal-speed delta is a target name."""
    expected = _lexical_tokens(text)
    observed, _ = _transcript_tokens(words)
    saw_name_delta = False
    matcher = SequenceMatcher(a=expected, b=observed, autojunk=False)
    for tag, expected_start, expected_end, observed_start, observed_end in (
        matcher.get_opcodes()
    ):
        if tag == "equal":
            continue
        # Inserts and deletes may be audible extras, omissions, or repetitions.
        # Never let slower ASR erase that normal-speed evidence.
        if tag != "replace":
            return False
        expected_delta = expected[expected_start:expected_end]
        observed_delta = tuple(observed[observed_start:observed_end])
        if len(expected_delta) != 1 or not _is_name_recheck_token(expected_delta[0]):
            return False
        expected_token = expected_delta[0]
        expected_name = _name_recheck_base(expected_token)
        if len(observed_delta) == 1:
            observed_token = observed_delta[0]
            if expected_token.endswith("'s") != observed_token.endswith("'s"):
                return False
            observed_name = _name_recheck_base(observed_token)
            if observed_name not in ORPHEUS_NAME_RECHECK_SPELLINGS[expected_name]:
                return False
        else:
            observed_possessive = observed_delta[-1].endswith("'s")
            if expected_token.endswith("'s") != observed_possessive:
                return False
            observed_split = (
                *observed_delta[:-1],
                _name_recheck_base(observed_delta[-1]),
            )
            if observed_split not in ORPHEUS_NAME_RECHECK_SPLITS[expected_name]:
                return False
        saw_name_delta = True
    return saw_name_delta


async def _transcribe_orpheus_at_speed(
    path: Path,
    verification_dir: Path,
    speed: float,
    *,
    emit: LogCallback,
    provider_label: str = "Orpheus",
) -> tuple[list[dict], dict]:
    """Re-transcribe the same waveform more slowly without changing pitch."""
    from backend.pipeline import av_sync

    verification_dir.mkdir(parents=True, exist_ok=True)
    speed_label = f"{speed:g}x"
    slowed_path = verification_dir / f"{path.stem}.atempo-{speed_label}.wav"
    returncode, output = await stream_subprocess(
        name=f"{provider_label} playback verification ({speed_label})",
        command=[
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            path,
            "-filter:a",
            f"atempo={speed:g}",
            "-c:a",
            "pcm_s16le",
            slowed_path,
        ],
        logger=logger,
        log=emit,
        cwd=config.PROJECT_ROOT,
        timeout=120,
        stall_timeout=60,
    )
    if returncode != 0:
        return [], {
            "passed": False,
            "failure_reasons": [
                f"ffmpeg atempo {speed_label} exited {returncode}: {output[-300:]}"
            ],
        }
    return await av_sync.ensure_word_transcript(
        slowed_path,
        verification_dir / f"transcript-{speed_label}",
        log=None,
        minimum_words=1,
    )


def _first_json_object(value: str) -> dict | None:
    """Extract one model JSON object without accepting prose-only verdicts."""
    clean = value.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("{")
        end = clean.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(clean[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _raw_transcript(words: list[dict]) -> str:
    return " ".join(
        str(word.get("text") or "").strip()
        for word in words
        if str(word.get("text") or "").strip()
    )


def _asr_overlapping_tokens(words: list[dict]) -> list[dict]:
    """Expose materially overlapping ASR tokens as decoder-artifact evidence."""
    overlaps: list[dict] = []
    for index in range(1, len(words)):
        start = float(words[index].get("start") or 0)
        end = float(words[index].get("end") or 0)
        previous_end = float(words[index - 1].get("end") or 0)
        duration = end - start
        overlap = previous_end - start
        if duration <= 0 or overlap < 0.15 or overlap / duration < 0.5:
            continue
        overlaps.append(
            {
                "index": index,
                "token": str(words[index].get("text") or "").strip(),
                "previous_token": str(words[index - 1].get("text") or "").strip(),
                "overlap_seconds": round(overlap, 3),
                "overlap_fraction": round(overlap / duration, 3),
            }
        )
    return overlaps


def _medium_asr_verdict_is_corroborated(
    expected_tokens: list[str],
    normal_transcript: str,
    slower_transcript: str,
    normal_words: list[dict] | None = None,
    slower_words: list[dict] | None = None,
) -> bool:
    """Bound medium-confidence approvals to tightly corroborated ASR drift.

    Proper names can be spelled phonetically by Whisper even when two decodes
    hear the same complete waveform (for example ``Andreessen`` ->
    ``Andreasen``).  Identical close decodes remain the normal medium-confidence
    path.  Two non-identical decodes may also corroborate one another when each
    contains one different, mostly timestamp-overlapped decoder token and
    removing those two uncorroborated tokens yields the same close transcript.
    This handles Whisper artifacts such as duplicate words sharing an end time
    without accepting an extra word heard at the same position by both decodes.

    The source must not contain a mixed letter/digit token such as ``a16z``;
    that guard prevents an alphanumeric brand or model number from being
    silently changed into another entity.
    """
    normal_tokens = re.findall(r"[a-z0-9]+", normal_transcript.casefold())
    slower_tokens = re.findall(r"[a-z0-9]+", slower_transcript.casefold())
    if not normal_tokens or not slower_tokens:
        return False
    if any(
        any(character.isalpha() for character in token)
        and any(character.isdigit() for character in token)
        for token in expected_tokens
    ):
        return False
    expected_text = " ".join(expected_tokens)
    if not expected_text:
        return False

    def close_to_source(observed_tokens: list[str]) -> bool:
        observed_compact = "".join(observed_tokens)
        expected_compact = "".join(expected_tokens)
        if not observed_compact or not expected_compact:
            return False
        length_ratio = len(observed_compact) / len(expected_compact)
        return 0.95 <= length_ratio <= 1.05 and (
            SequenceMatcher(
                None,
                expected_compact,
                observed_compact,
                autojunk=False,
            ).ratio()
            >= 0.965
        )

    def has_only_replacement_differences(observed_tokens: list[str]) -> bool:
        return all(
            tag not in {"insert", "delete"}
            for tag, _i1, _i2, _j1, _j2 in SequenceMatcher(
                a=expected_tokens,
                b=observed_tokens,
                autojunk=False,
            ).get_opcodes()
        )

    if normal_tokens == slower_tokens:
        if normal_words and slower_words and (
            _asr_overlapping_tokens(normal_words)
            or _asr_overlapping_tokens(slower_words)
        ):
            return False
        return close_to_source(normal_tokens) and has_only_replacement_differences(
            normal_tokens
        )
    if not normal_words or not slower_words:
        return False
    if (
        close_to_source(normal_tokens)
        and close_to_source(slower_tokens)
        and has_only_replacement_differences(normal_tokens)
        and has_only_replacement_differences(slower_tokens)
    ):
        return True

    def compact_words(words: list[dict], removed_index: int) -> str:
        return "".join(
            re.findall(
                r"[a-z0-9]+",
                " ".join(
                    str(word.get("text") or "")
                    for index, word in enumerate(words)
                    if index != removed_index
                ).casefold(),
            )
        )

    expected_compact = "".join(expected_tokens)
    normal_candidates = [item["index"] for item in _asr_overlapping_tokens(normal_words)]
    slower_candidates = [item["index"] for item in _asr_overlapping_tokens(slower_words)]
    for normal_index in normal_candidates:
        normal_token = "".join(
            re.findall(
                r"[a-z0-9]+",
                str(normal_words[normal_index].get("text") or "").casefold(),
            )
        )
        for slower_index in slower_candidates:
            slower_token = "".join(
                re.findall(
                    r"[a-z0-9]+",
                    str(slower_words[slower_index].get("text") or "").casefold(),
                )
            )
            # The same token at the same relative location is corroborated
            # added speech, not an independent decoder artifact.
            if normal_token == slower_token or abs(
                normal_index / len(normal_words) - slower_index / len(slower_words)
            ) < 0.03:
                continue
            corrected_normal = compact_words(normal_words, normal_index)
            corrected_slower = compact_words(slower_words, slower_index)
            if not corrected_normal or corrected_normal != corrected_slower:
                continue
            length_ratio = len(corrected_normal) / max(1, len(expected_compact))
            similarity = SequenceMatcher(
                None,
                expected_compact,
                corrected_normal,
                autojunk=False,
            ).ratio()
            if 0.95 <= length_ratio <= 1.05 and similarity >= 0.97:
                return True
    return False


async def _adjudicate_orpheus_asr_mismatch(
    text: str,
    normal_words: list[dict],
    slower_words: list[dict],
    report: dict,
    verification_dir: Path,
    *,
    emit: LogCallback,
    provider_label: str = "Orpheus",
) -> dict | None:
    """Ask the configured LLM whether transcript deltas are ASR-only.

    The model sees two independent transcripts of the same waveform and the
    deterministic diff report.  It may approve only spelling, word-boundary,
    exact-homophone, or phonetic proper-name transcription drift.  Missing or
    extra spoken content stays a hard failure.  Every response and selected
    route is persisted beside the acoustic transcript for auditability.
    """
    from backend.pipeline import model_router
    from backend.pipeline.digester import _chat, _resolve_provider

    verification_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = verification_dir / "llm_asr_adjudication.json"
    expected_tokens = _lexical_tokens(text)
    request = {
        "source_text": text,
        "normalized_source_tokens": expected_tokens,
        "normal_speed_transcript": _raw_transcript(normal_words),
        "slower_speed_transcript": _raw_transcript(slower_words),
        "normal_speed_overlapping_tokens": _asr_overlapping_tokens(normal_words),
        "slower_speed_overlapping_tokens": _asr_overlapping_tokens(slower_words),
        "deterministic_check": {
            key: report.get(key)
            for key in (
                "expected_words",
                "transcript_words",
                "matched_exact_words",
                "exact_asr_word_coverage",
                "transcript_word_ratio",
                "leading_anchor",
                "trailing_anchor",
                "failure_reasons",
            )
        },
    }
    evidence: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
        "request": request,
        "route": {},
    }

    def write_evidence() -> None:
        temporary = evidence_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, evidence_path)

    def remember_route(endpoint: str, model: str, _api_key: str) -> None:
        evidence["route"] = {"endpoint": endpoint, "model": model}

    write_evidence()
    system_prompt = """You are a fail-closed speech-transcription adjudicator.
Decide whether the TTS waveform can still contain the complete SOURCE TEXT even
though automatic speech recognition produced a mismatch. You receive two ASR
transcripts of the same waveform, one at normal speed and one slowed down.

Approve only when every normalized source token is accounted for by explicit
transcript evidence and every difference is plausibly ASR spelling,
capitalization, punctuation, hyphenation, word-boundary, exact-homophone, or
phonetic proper-name drift. A spelled-out number and the same value rendered as
digits are explicit equivalent evidence (for example, "twenty-four point five
million" and "24.5 million"); a genuinely changed numeric value is not. Treat
the deterministic check as a mismatch trigger, not as ground truth: its token
expansion and leading/trailing anchor fields can be false for number formatting,
punctuation, or word-boundary differences. Independently compare SOURCE TEXT
with both raw transcripts before deciding. Reject any omitted, added, repeated,
paraphrased, negated, number-changed, or entity-changed spoken content. Do not
fill a missing word from context. If the two transcripts do not provide enough
evidence, reject. Word timestamps can overlap when Whisper emits a duplicate or
hallucinated token. Treat an overlapping token as an ASR artifact only when the
other transcript does not contain it at the same content position; do not treat
the deterministic word count alone as proof of repeated speech. Return JSON
only with exactly these fields:
{
  "decision": "approve_asr_error" | "reject_audio_mismatch",
  "all_source_tokens_accounted_for": true | false,
  "accounted_source_token_indexes": [0-based integer indexes],
  "confidence": "high" | "medium" | "low",
  "reason": "brief evidence-based explanation"
}"""
    try:
        endpoint, model, api_key = await _resolve_provider(None, None, None)
        routes = await model_router.resolve_model_routes(
            endpoint=endpoint,
            model=model,
            api_key=api_key,
        )
        adjudication_route = next(
            (route for route in routes if route.slot == "backup"),
            routes[0],
        )
        endpoint = adjudication_route.endpoint
        model = adjudication_route.model
        api_key = adjudication_route.api_key
        emit(
            f"{provider_label} ASR adjudication: using isolated judge route "
            f"{adjudication_route.audit_label}"
        )
        evidence["attempts"] = []
        # The response enumerates every normalized source-token index. Leave
        # room for that accounting plus the explanation even on full paragraphs;
        # 700 tokens caused SDK continuation fragments and max_output_tokens.
        output_budget = min(4096, max(2048, 512 + 4 * len(expected_tokens)))
        for attempt in range(2):
            retry_instruction = (
                "\nYour previous response was not valid JSON or was truncated. Re-evaluate the "
                "same evidence and return exactly one complete JSON object. "
                "Do not include analysis, prose, or Markdown fences."
                if attempt else ""
            )
            try:
                raw = await _chat(
                    system_prompt + retry_instruction,
                    json.dumps(request, ensure_ascii=False),
                    endpoint=endpoint,
                    model=model,
                    api_key=api_key,
                    log=emit,
                    label=f"{provider_label} ASR adjudication",
                    max_tokens=min(8192, output_budget * (attempt + 1)),
                    enable_skills=False,
                    disable_thinking=True,
                    route_selected=remember_route,
                )
            except Exception as exc:
                # This isolated, skills-disabled judge has no side effects.
                # Retry only explicit output exhaustion, using the same route
                # and evidence; unrelated transport/model errors still fail.
                evidence["attempts"].append({
                    "error": f"{type(exc).__name__}: {exc}"[:700],
                    "route": dict(evidence["route"]),
                })
                write_evidence()
                if attempt == 0 and "max_output_tokens" in str(exc):
                    emit(
                        f"{provider_label} ASR adjudication: output truncated; "
                        "retrying once with more output room on the same evidence"
                    )
                    continue
                raise
            evidence["raw_response"] = raw
            verdict = _first_json_object(raw)
            evidence["attempts"].append({
                "raw_response": raw,
                "verdict": verdict,
                "route": dict(evidence["route"]),
            })
            write_evidence()
            if verdict is not None:
                break
            if attempt == 0:
                emit(
                    f"{provider_label} ASR adjudication: invalid JSON; retrying "
                    "once against the same transcripts without regenerating audio"
                )
        evidence["verdict"] = verdict
        expected_indexes = list(range(len(expected_tokens)))
        confidence = str((verdict or {}).get("confidence") or "").strip()
        normal_transcript = request["normal_speed_transcript"]
        slower_transcript = request["slower_speed_transcript"]
        confidence_accepted = confidence == "high" or bool(
            confidence == "medium"
            and _medium_asr_verdict_is_corroborated(
                expected_tokens,
                normal_transcript,
                slower_transcript,
                normal_words,
                slower_words,
            )
        )
        approved = bool(
            verdict
            and verdict.get("decision") == "approve_asr_error"
            and verdict.get("all_source_tokens_accounted_for") is True
            and confidence_accepted
            and verdict.get("accounted_source_token_indexes") == expected_indexes
            and str(verdict.get("reason") or "").strip()
            and str((evidence.get("route") or {}).get("model") or "").strip()
        )
        evidence["status"] = "approved" if approved else "rejected"
        write_evidence()
        if not approved:
            return None
        return {
            "decision": "approve_asr_error",
            "confidence": confidence,
            "medium_confidence_corroborated": confidence == "medium",
            "reason": str(verdict["reason"]).strip()[:600],
            "normal_speed_transcript": normal_transcript,
            "slower_speed_transcript": slower_transcript,
            "evidence_path": evidence_path.name,
            "route": evidence.get("route") or {},
        }
    except Exception as exc:  # noqa: BLE001 - unavailable adjudication fails closed
        evidence["status"] = "error"
        evidence["error"] = f"{type(exc).__name__}: {exc}"[:700]
        write_evidence()
        emit(
            f"{provider_label} integrity: ASR adjudication unavailable; preserving strict "
            f"failure ({type(exc).__name__}: {exc})"
        )
        return None


async def _verify_orpheus_part(
    path: Path,
    text: str,
    verification_dir: Path,
    *,
    emit: LogCallback,
    adjudicate_asr: bool = False,
    provider_label: str = "Orpheus",
    validate_pocket_continuity: bool = False,
) -> dict:
    from backend.pipeline import av_sync

    words, transcription = await av_sync.ensure_word_transcript(
        path,
        verification_dir,
        log=None,
        minimum_words=1,
    )
    if not words:
        failures = "; ".join(transcription.get("failure_reasons") or [])
        raise TtsIntegrityError(
            f"{provider_label} narration cannot be integrity-verified because acoustic "
            f"transcription is unavailable{': ' + failures if failures else ''}"
        )
    report = _orpheus_transcript_report(text, words)
    repeat_start = report.get("repeat_start_seconds")
    if repeat_start is not None and float(repeat_start) > 0.2:
        emit(
            f"{provider_label} integrity: trimming repeated utterance at "
            f"{float(repeat_start):.2f}s and re-transcribing"
        )
        _trim_pcm_wav(path, float(repeat_start))
        words, transcription = await av_sync.ensure_word_transcript(
            path,
            verification_dir,
            log=None,
            minimum_words=1,
        )
        if not words:
            raise TtsIntegrityError(
                f"{provider_label} narration could not be transcribed after repetition trimming"
            )
        report = _orpheus_transcript_report(text, words)
    if (
        report["verified"]
        and report.get("verification_mode") == "aligned_phonetic_substitution"
    ):
        normal_speed_report = report
        expected_substitution_indexes = {
            int(item["expected_index"])
            for item in normal_speed_report["phonetic_substitutions"]
        }
        corroborated_report: dict | None = None
        for speed in ORPHEUS_NAME_RECHECK_SPEEDS:
            try:
                slower_words, slower_transcription = await _transcribe_orpheus_at_speed(
                    path,
                    verification_dir,
                    speed,
                    emit=emit,
                    provider_label=provider_label,
                )
            except Exception as exc:  # noqa: BLE001 - keep the fallback fail-closed
                emit(
                    f"{provider_label} integrity: phonetic corroboration at "
                    f"{speed:g}x could not run ({type(exc).__name__}: {exc})"
                )
                continue
            if not slower_words:
                failures = "; ".join(
                    slower_transcription.get("failure_reasons") or []
                )
                emit(
                    f"{provider_label} integrity: phonetic corroboration at "
                    f"{speed:g}x produced no transcript"
                    f"{': ' + failures if failures else ''}"
                )
                continue
            slower_report = _orpheus_transcript_report(text, slower_words)
            slower_substitution_indexes = {
                int(item["expected_index"])
                for item in slower_report.get("phonetic_substitutions") or []
            }
            corroborates = slower_report["verified"] and (
                slower_report.get("verification_mode") == "exact"
                or slower_substitution_indexes == expected_substitution_indexes
            )
            if not corroborates:
                emit(
                    f"{provider_label} integrity: phonetic corroboration at "
                    f"{speed:g}x did not confirm the same aligned substitution"
                )
                continue
            slower_report["verification_playback_speed"] = speed
            slower_report["normal_speed_exact_asr_word_coverage"] = (
                normal_speed_report["exact_asr_word_coverage"]
            )
            slower_report["normal_speed_phonetic_substitutions"] = list(
                normal_speed_report["phonetic_substitutions"]
            )
            slower_report["verification_mode"] = (
                "corroborated_exact"
                if slower_report.get("verification_mode") == "exact"
                else "corroborated_phonetic_substitution"
            )
            # The transcript timestamps are from a slowed copy. Convert the
            # complete speech edge back to the original WAV's time axis.
            slower_report["speech_end_seconds"] = round(
                float(slower_report["speech_end_seconds"]) * speed,
                3,
            )
            slower_report["repeat_start_seconds"] = None
            corroborated_report = slower_report
            emit(
                f"{provider_label} integrity: aligned phonetic substitution corroborated "
                f"from the same waveform at {speed:g}x playback"
            )
            break
        if corroborated_report is None:
            normal_speed_report["verified"] = False
            normal_speed_report["failure_reasons"] = [
                *normal_speed_report["failure_reasons"],
                "aligned phonetic substitution was not corroborated by a "
                "second transcription of the same waveform",
            ]
            report = normal_speed_report
        else:
            report = corroborated_report
    if (
        not report["verified"]
        and _needs_name_playback_recheck(text)
        and _has_only_name_transcript_mismatches(text, words)
    ):
        original_report = report
        for speed in ORPHEUS_NAME_RECHECK_SPEEDS:
            try:
                slower_words, slower_transcription = await _transcribe_orpheus_at_speed(
                    path,
                    verification_dir,
                    speed,
                    emit=emit,
                    provider_label=provider_label,
                )
            except Exception as exc:  # noqa: BLE001 - preserve the strict original failure
                emit(
                    f"{provider_label} integrity: name verification at "
                    f"{speed:g}x could not run ({type(exc).__name__}: {exc})"
                )
                continue
            if not slower_words:
                failures = "; ".join(
                    slower_transcription.get("failure_reasons") or []
                )
                emit(
                    f"{provider_label} integrity: name verification at "
                    f"{speed:g}x produced no transcript"
                    f"{': ' + failures if failures else ''}"
                )
                continue
            slower_report = _orpheus_transcript_report(text, slower_words)
            if not slower_report["verified"]:
                emit(
                    f"{provider_label} integrity: name verification at "
                    f"{speed:g}x remained non-exact ("
                    + "; ".join(slower_report["failure_reasons"])
                    + ")"
                )
                continue
            slower_report["verification_playback_speed"] = speed
            slower_report["original_speed_failure_reasons"] = list(
                original_report["failure_reasons"]
            )
            # The lexical evidence came from the slowed copy; map its complete
            # end timestamp back onto the original WAV's time axis.
            slower_report["speech_end_seconds"] = round(
                float(slower_report["speech_end_seconds"]) * speed,
                3,
            )
            slower_report["repeat_start_seconds"] = None
            report = slower_report
            emit(
                f"{provider_label} integrity: exact name transcript recovered from the "
                f"same waveform at {speed:g}x playback"
            )
            break
    if not report["verified"] and adjudicate_asr:
        emit(
            f"{provider_label} integrity: deterministic ASR check found a mismatch; "
            "requesting fail-closed model adjudication"
        )
        slower_words: list[dict] = []
        try:
            slower_words, slower_transcription = await _transcribe_orpheus_at_speed(
                path,
                verification_dir,
                ORPHEUS_NAME_RECHECK_SPEEDS[0],
                emit=emit,
                provider_label=provider_label,
            )
            if not slower_words:
                failures = "; ".join(
                    slower_transcription.get("failure_reasons") or []
                )
                emit(
                    f"{provider_label} integrity: model adjudication skipped because the "
                    "slower corroborating transcript is unavailable"
                    f"{': ' + failures if failures else ''}"
                )
        except Exception as exc:  # noqa: BLE001 - adjudication remains fail-closed
            emit(
                f"{provider_label} integrity: model adjudication skipped because slower "
                f"transcription failed ({type(exc).__name__}: {exc})"
            )
        if slower_words:
            adjudication = await _adjudicate_orpheus_asr_mismatch(
                text,
                words,
                slower_words,
                report,
                verification_dir,
                emit=emit,
                provider_label=provider_label,
            )
            if adjudication is not None:
                original_failures = list(report["failure_reasons"])
                report = {
                    **report,
                    "verified": True,
                    "verification_mode": "llm_asr_adjudication",
                    "failure_reasons": [],
                    "deterministic_failure_reasons": original_failures,
                    "llm_asr_adjudication": adjudication,
                }
                emit(
                    f"{provider_label} integrity: model adjudication approved ASR-only "
                    f"transcription drift ({adjudication['reason']})"
                )
            else:
                emit(
                    f"{provider_label} integrity: model adjudication rejected or lacked "
                    "high-confidence evidence; preserving strict failure"
                )
    if not report["verified"]:
        raise TtsIntegrityError(
            f"{provider_label} narration does not match its input utterance: "
            + "; ".join(report["failure_reasons"])
        )
    if validate_pocket_continuity:
        report["internal_silence"] = _validate_pocket_internal_silence(path, words)
    substitutions = report.get("phonetic_substitutions") or []
    if report.get("verification_mode") == "llm_asr_adjudication":
        emit(
            f"{provider_label} integrity: utterance verified by two-level ASR plus model "
            "adjudication; persisted evidence retains both transcripts and route"
        )
    elif substitutions:
        substitution_summary = ", ".join(
            f"{item['expected']}~{item['observed']}"
            for item in substitutions
        )
        emit(
            f"{provider_label} integrity: utterance verified "
            f"({report['matched_acoustic_words']}/{report['expected_words']} acoustic "
            f"ASR words; {report['matched_exact_words']} exact; aligned phonetic "
            f"substitution {substitution_summary}; opening and closing anchors present)"
        )
    else:
        emit(
            f"{provider_label} integrity: utterance verified "
            f"({report['matched_exact_words']}/{report['expected_words']} exact ASR words; "
            "opening and closing anchors present)"
        )
    return report


def _part_metadata_path(path: Path) -> Path:
    return path.with_suffix(".json")


def _load_cached_orpheus_part(path: Path, text: str) -> dict | None:
    metadata_path = _part_metadata_path(path)
    if not path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict):
        return None
    if metadata.get("text_sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest():
        return None
    if metadata.get("speed_percent") != config.ORPHEUS_TTS_SPEED_PERCENT:
        return None
    verifier_version = metadata.get("integrity_verifier_version")
    if (
        isinstance(verifier_version, bool)
        or not isinstance(verifier_version, int)
        or verifier_version != ORPHEUS_INTEGRITY_VERIFIER_VERSION
    ):
        return None
    integrity = metadata.get("integrity")
    if (
        not isinstance(integrity, dict)
        or not integrity.get("verified")
        or integrity.get("method") == "duration_only_preview"
    ):
        return None
    try:
        info = _read_pcm_wav(path)
    except TtsIntegrityError:
        return None
    if asdict(info) != metadata.get("wav"):
        return None
    return metadata


def _snapshot_reusable_orpheus_parts(
    output_dir: Path,
    chunks: list[str],
) -> dict[str, tuple[bytes, bytes, str]]:
    """Retain exact verified audio even when a revised script renumbers chunks.

    ``_write_chunk_inputs`` rewrites the numbered text files, while WAV sidecars
    from an earlier attempt remain available. Snapshot only artifacts whose text
    hash occurs in the current script and whose WAV is readable. A current
    sidecar can be reused immediately; a stale sidecar is restored only so the
    current acoustic verifier can revalidate it. Keeping the bytes in memory
    prevents an earlier destination number from overwriting a source needed
    later.
    """
    chunks_by_hash: dict[str, str] = {}
    for chunk in chunks:
        chunks_by_hash.setdefault(
            hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
            chunk,
        )

    reusable: dict[str, tuple[bytes, bytes, str]] = {}
    for metadata_path in sorted(output_dir.glob("tts_input*_generated.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict):
            continue
        text_sha256 = metadata.get("text_sha256")
        if not isinstance(text_sha256, str) or text_sha256 not in chunks_by_hash:
            continue
        wav_path = metadata_path.with_suffix(".wav")
        try:
            _read_pcm_wav(wav_path)
        except TtsIntegrityError:
            continue
        try:
            reusable.setdefault(
                text_sha256,
                (wav_path.read_bytes(), metadata_path.read_bytes(), wav_path.name),
            )
        except OSError:
            continue
    return reusable


def _restore_reusable_orpheus_part(
    path: Path,
    text: str,
    reusable: dict[str, tuple[bytes, bytes, str]],
) -> tuple[dict | None, str] | None:
    text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    snapshot = reusable.get(text_sha256)
    if snapshot is None:
        return None
    wav_bytes, metadata_bytes, source_name = snapshot
    staged_wav = path.with_suffix(".cache.tmp.wav")
    staged_metadata = _part_metadata_path(path).with_suffix(".cache.tmp.json")
    try:
        staged_wav.write_bytes(wav_bytes)
        staged_metadata.write_bytes(metadata_bytes)
        os.replace(staged_wav, path)
        os.replace(staged_metadata, _part_metadata_path(path))
    finally:
        staged_wav.unlink(missing_ok=True)
        staged_metadata.unlink(missing_ok=True)
    metadata = _load_cached_orpheus_part(path, text)
    return metadata, source_name


def _write_orpheus_part_metadata(
    path: Path,
    text: str,
    *,
    job_id: str,
    request_token_budget: int,
    integrity: dict,
) -> dict:
    payload = {
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "word_count": _spoken_word_count(text),
        "job_id": job_id,
        "request_token_budget": request_token_budget,
        "speed_percent": config.ORPHEUS_TTS_SPEED_PERCENT,
        "integrity_verifier_version": ORPHEUS_INTEGRITY_VERIFIER_VERSION,
        "wav": asdict(_read_pcm_wav(path)),
        "integrity": integrity,
    }
    destination = _part_metadata_path(path)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, destination)
    return payload


async def _recover_orpheus_part(
    path: Path,
    text: str,
    verification_dir: Path,
    *,
    request_token_budget: int,
    emit: LogCallback,
) -> dict | None:
    """Verify a downloaded part left without valid cache metadata.

    A process restart, or an exact-ASR spelling correction deployed after a
    rejected sample, can leave a complete WAV on disk without a sidecar.  Run
    the same fail-closed acoustic gate before submitting another slow remote
    CPU job.  Truly stale or incomplete audio is ignored and regenerated.
    """
    if not path.is_file():
        return None
    try:
        _validate_wav_part(
            path,
            text,
            speed=config.ORPHEUS_TTS_SPEED_PERCENT / 100,
        )
        integrity = await _verify_orpheus_part(
            path,
            text,
            verification_dir,
            emit=emit,
            adjudicate_asr=True,
        )
    except TtsIntegrityError as exc:
        emit(f"Orpheus recovery: existing WAV rejected ({exc}); regenerating")
        return None
    metadata = _write_orpheus_part_metadata(
        path,
        text,
        job_id="recovered-local-output",
        request_token_budget=request_token_budget,
        integrity=integrity,
    )
    emit("Orpheus recovery: accepted existing WAV after acoustic verification")
    return metadata


def _orpheus_http_error_text(exc: Exception) -> str:
    """Keep transport failures useful even when httpx provides an empty message."""
    error_type = type(exc).__name__
    if isinstance(exc, httpx.HTTPStatusError):
        error_type = f"{error_type} (HTTP {exc.response.status_code})"
    detail = str(exc).strip()
    return f"{error_type}: {detail}" if detail else error_type


def _is_permanent_orpheus_http_error(exc: Exception) -> bool:
    """Return whether retrying the same Orpheus request cannot heal the response."""
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    status_code = exc.response.status_code
    if 300 <= status_code < 400:
        return True
    return 400 <= status_code < 500 and status_code not in {408, 425, 429}


async def _generate_orpheus(
    script_path: str,
    output_dir: str,
    voice: str,
    language: str,
    *,
    log: LogCallback | None,
    emit: LogCallback,
    max_tokens: int | None = None,
    verify_text: bool = False,
) -> str:
    if not config.ORPHEUS_TTS_API_KEY:
        raise RuntimeError(
            "Orpheus TTS API key is not configured. Set ORPHEUS_TTS_API_KEY in "
            "Admin -> System -> Voice & TTS."
        )
    token_budget = max_tokens or config.ORPHEUS_TTS_MAX_TOKENS
    script_path_obj, output_dir_path, _tts_input, cleaned = _prepare_tts_input(
        script_path,
        output_dir,
        strip_speaker_labels=True,
    )
    chunk_words = min(
        config.ORPHEUS_TTS_CHUNK_WORDS,
        _orpheus_chunk_word_limit(token_budget),
    )
    input_paths, chunks = _write_chunk_inputs(
        cleaned,
        output_dir_path,
        max_words=chunk_words,
    )
    reusable_parts = (
        _snapshot_reusable_orpheus_parts(output_dir_path, chunks)
        if verify_text
        else {}
    )
    emit(f"TTS input: stripped speaker labels -> {output_dir_path / 'tts_input.txt'}")
    if len(input_paths) > 1:
        emit(
            f"TTS input: Orpheus-safe split into {len(input_paths)} chunks "
            f"(up to {chunk_words} words per acoustically verified utterance)"
        )
    base_url = config.ORPHEUS_TTS_URL.rstrip("/")
    headers = {"X-API-Key": config.ORPHEUS_TTS_API_KEY}
    timeout = httpx.Timeout(config.ORPHEUS_TTS_REQUEST_TIMEOUT)
    wav_parts: list[Path] = []
    part_metadata: list[dict] = []
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        for index, input_path in enumerate(input_paths, start=1):
            name = "TTS" if len(input_paths) == 1 else f"TTS part {index}/{len(input_paths)}"
            chunk = chunks[index - 1]
            expected_part = output_dir_path / f"{input_path.stem}_generated.wav"
            request_token_budget = _orpheus_request_token_budget(chunk, token_budget)
            if verify_text:
                cached = _load_cached_orpheus_part(expected_part, chunk)
                if cached is not None:
                    emit(
                        f"{name}: reusing acoustically verified Orpheus audio "
                        f"({cached['word_count']} source words)"
                    )
                    wav_parts.append(expected_part)
                    part_metadata.append(cached)
                    continue
                restored = _restore_reusable_orpheus_part(
                    expected_part,
                    chunk,
                    reusable_parts,
                )
                if restored is not None:
                    cached, source_name = restored
                    if cached is not None:
                        emit(
                            f"{name}: reusing acoustically verified Orpheus audio "
                            f"from {source_name} after chunk renumbering "
                            f"({cached['word_count']} source words)"
                        )
                        wav_parts.append(expected_part)
                        part_metadata.append(cached)
                        continue
                    emit(
                        f"{name}: revalidating exact-text Orpheus audio from "
                        f"{source_name} after chunk renumbering"
                    )
                    recovered = await _recover_orpheus_part(
                        expected_part,
                        chunk,
                        output_dir_path / "verification" / input_path.stem,
                        request_token_budget=request_token_budget,
                        emit=emit,
                    )
                else:
                    recovered = await _recover_orpheus_part(
                        expected_part,
                        chunk,
                        output_dir_path / "verification" / input_path.stem,
                        request_token_budget=request_token_budget,
                        emit=emit,
                    )
                if recovered is not None:
                    emit(f"{name}: reusing recovered acoustically verified Orpheus audio")
                    wav_parts.append(expected_part)
                    part_metadata.append(recovered)
                    continue

            payload = {
                "input": _orpheus_prompt_text(input_path.read_text(encoding="utf-8")),
                "language": language,
                "voice_id": voice,
                "max_tokens": request_token_budget,
                # Orpheus is an autoregressive audio LM. Greedy decoding
                # collapses real prompts to unrelated phrases (observed as
                # repeated "Thank you"), so use the service/model defaults and
                # rely on the acoustic integrity gate instead of determinism.
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 40,
                "min_p": 0.05,
                "pre_buffer_size": 1.5,
                "n_threads": config.ORPHEUS_TTS_N_THREADS,
                # Do not retime narration to hit a requested video length.  The
                # measured natural-speed WAV drives storyboard/scene duration.
                "speed": NARRATION_SYNTHESIS_SPEED_RATIO,
                "response_format": "wav",
            }
            try:
                response = await client.post(f"{base_url}/v1/audio/jobs", json=payload)
                response.raise_for_status()
                job = response.json()
                job_id = str(job["id"])
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                raise RuntimeError(f"{name} could not submit an Orpheus job: {exc}") from exc
            emit(f"{name}: Orpheus job {job_id} queued (voice={voice})")
            deadline = time.monotonic() + config.TTS_TIMEOUT
            stall_timeout = max(1, config.TTS_STALL_TIMEOUT)
            poll_interval = max(1, config.ORPHEUS_TTS_POLL_SECONDS)
            last_successful_poll = time.monotonic()
            last_poll_error: Exception | None = None
            last_poll_error_text = ""
            last_poll_error_log_at = float("-inf")
            last_status = ""
            while True:
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError(
                        f"{name} Orpheus job {job_id} exceeded {config.TTS_TIMEOUT}s"
                    )
                if (
                    last_poll_error is not None
                    and now - last_successful_poll >= stall_timeout
                ):
                    raise TimeoutError(
                        f"{name} Orpheus job {job_id} had no successful poll for "
                        f"{stall_timeout}s; last error: "
                        f"{_orpheus_http_error_text(last_poll_error)}"
                    ) from last_poll_error
                try:
                    response = await client.get(f"{base_url}/v1/audio/jobs/{job_id}")
                    response.raise_for_status()
                    job = response.json()
                    if not isinstance(job, dict):
                        raise ValueError("Orpheus poll response was not a JSON object")
                    status = str(job.get("status") or "").strip().lower()
                    if not status:
                        raise ValueError(
                            "Orpheus poll response did not contain a non-empty status"
                        )
                except (httpx.HTTPError, ValueError) as exc:
                    if _is_permanent_orpheus_http_error(exc):
                        raise RuntimeError(
                            f"{name} could not poll Orpheus job {job_id}: "
                            f"{_orpheus_http_error_text(exc)}"
                        ) from exc
                    last_poll_error = exc
                    error_text = _orpheus_http_error_text(exc)
                    now = time.monotonic()
                    if (
                        error_text != last_poll_error_text
                        or now - last_poll_error_log_at >= 30
                    ):
                        emit(
                            f"{name}: transient Orpheus poll error for job {job_id} "
                            f"({error_text}); retrying the same job"
                        )
                        last_poll_error_text = error_text
                        last_poll_error_log_at = now
                    if now - last_successful_poll >= stall_timeout:
                        raise TimeoutError(
                            f"{name} Orpheus job {job_id} had no successful poll for "
                            f"{stall_timeout}s; last error: {error_text}"
                        ) from exc
                    await asyncio.sleep(poll_interval)
                    continue
                last_successful_poll = time.monotonic()
                last_poll_error = None
                last_poll_error_text = ""
                if status != last_status:
                    emit(f"{name}: Orpheus job {job_id} is {status or 'unknown'}")
                    last_status = status
                if status in {"completed", "complete", "succeeded", "done"}:
                    break
                if status in {"failed", "cancelled", "canceled", "error"}:
                    detail = job.get("error") or job.get("detail") or "no error detail"
                    raise RuntimeError(f"{name} Orpheus job {job_id} failed: {detail}")
                await asyncio.sleep(poll_interval)

            download_started_at = time.monotonic()
            last_download_error: Exception | None = None
            last_download_error_text = ""
            last_download_error_log_at = float("-inf")
            staged_part = expected_part.with_suffix(".tmp.wav")
            while True:
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError(
                        f"{name} Orpheus job {job_id} exceeded {config.TTS_TIMEOUT}s "
                        "while downloading audio"
                    )
                if (
                    last_download_error is not None
                    and now - download_started_at >= stall_timeout
                ):
                    raise TimeoutError(
                        f"{name} Orpheus job {job_id} audio download had no success "
                        f"for {stall_timeout}s; last error: "
                        f"{_orpheus_http_error_text(last_download_error)}"
                    ) from last_download_error
                try:
                    response = await client.get(
                        f"{base_url}/v1/audio/jobs/{job_id}/audio"
                    )
                    response.raise_for_status()
                    staged_part.write_bytes(response.content)
                    _validate_downloaded_wav_container(staged_part)
                    break
                except (httpx.HTTPError, TtsIntegrityError) as exc:
                    staged_part.unlink(missing_ok=True)
                    if _is_permanent_orpheus_http_error(exc):
                        raise RuntimeError(
                            f"{name} could not download Orpheus job {job_id}: "
                            f"{_orpheus_http_error_text(exc)}"
                        ) from exc
                    last_download_error = exc
                    error_text = _orpheus_http_error_text(exc)
                    now = time.monotonic()
                    if (
                        error_text != last_download_error_text
                        or now - last_download_error_log_at >= 30
                    ):
                        emit(
                            f"{name}: transient Orpheus audio download error for "
                            f"job {job_id} ({error_text}); retrying the same job"
                        )
                        last_download_error_text = error_text
                        last_download_error_log_at = now
                    if now - download_started_at >= stall_timeout:
                        raise TimeoutError(
                            f"{name} Orpheus job {job_id} audio download had no "
                            f"success for {stall_timeout}s; last error: {error_text}"
                        ) from exc
                    await asyncio.sleep(poll_interval)
            os.replace(staged_part, expected_part)
            try:
                _validate_wav_part(
                    expected_part,
                    chunk,
                    token_limit_seconds=(
                        None
                        if verify_text
                        else request_token_budget
                        / ORPHEUS_AUDIO_TOKENS_PER_SECOND
                        / NARRATION_SYNTHESIS_SPEED_RATIO
                    ),
                    speed=NARRATION_SYNTHESIS_SPEED_RATIO,
                )
            except TtsIntegrityError as exc:
                raise TtsIntegrityError(str(exc), part_key=input_path.name) from exc
            if verify_text:
                try:
                    integrity = await _verify_orpheus_part(
                        expected_part,
                        chunk,
                        output_dir_path / "verification" / input_path.stem,
                        emit=emit,
                        adjudicate_asr=True,
                    )
                except TtsIntegrityError as exc:
                    raise TtsIntegrityError(str(exc), part_key=input_path.name) from exc
            else:
                integrity = {
                    "verified": True,
                    "method": "duration_only_preview",
                    "expected_words": _spoken_word_count(chunk),
                }
            metadata = _write_orpheus_part_metadata(
                expected_part,
                chunk,
                job_id=job_id,
                request_token_budget=request_token_budget,
                integrity=integrity,
            )
            wav_parts.append(expected_part)
            part_metadata.append(metadata)

    expected = await _concat_wav_parts(wav_parts, output_dir_path, log=log)
    source_words = _spoken_word_count(cleaned)
    verified_words = sum(
        int(metadata.get("word_count") or 0)
        for metadata in part_metadata
        if (metadata.get("integrity") or {}).get("verified")
    )
    integrity = {
        "method": "per_utterance_mlx_whisper",
        "required": verify_text,
        "passed": verified_words == source_words,
        "source_words": source_words,
        "verified_source_words": verified_words,
        "verified_source_coverage": round(verified_words / max(1, source_words), 4),
        "part_reports": [metadata.get("integrity") or {} for metadata in part_metadata],
    }
    if verify_text and not integrity["passed"]:
        raise TtsIntegrityError(
            f"Orpheus verified only {verified_words}/{source_words} source words; "
            "refusing to join incomplete narration"
        )
    _write_tts_manifest(
        output_dir_path,
        model="orpheus-en",
        source_text=cleaned,
        chunks=chunks,
        wav_parts=wav_parts,
        output=expected,
        deterministic=False,
        integrity=integrity,
    )
    emit(
        f"TTS output: {expected} ({expected.stat().st_size / 1024:.0f} KB; "
        f"source={script_path_obj})"
    )
    return str(expected)


def _load_cached_pocket_part(path: Path, text: str, voice: str) -> dict | None:
    metadata_path = _part_metadata_path(path)
    if not path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict):
        return None
    expected = {
        "provider": "pocket-tts",
        "model_revision": config.POCKET_TTS_MODEL_REVISION,
        "voice": voice,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "integrity_verifier_version": POCKET_TTS_INTEGRITY_VERIFIER_VERSION,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    integrity = metadata.get("integrity")
    if (
        not isinstance(integrity, dict)
        or not integrity.get("verified")
        or integrity.get("method") == "duration_only_preview"
    ):
        return None
    try:
        info = _read_pcm_wav(path)
    except TtsIntegrityError:
        return None
    if asdict(info) != metadata.get("wav"):
        return None
    if _file_sha256(path) != metadata.get("audio_sha256"):
        return None
    return metadata


def _write_pocket_part_metadata(
    path: Path,
    text: str,
    voice: str,
    *,
    integrity: dict,
    generation_seconds: float,
) -> dict:
    payload = {
        "provider": "pocket-tts",
        "model_revision": config.POCKET_TTS_MODEL_REVISION,
        "voice": voice,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "word_count": _spoken_word_count(text),
        "generation_seconds": round(generation_seconds, 3),
        "integrity_verifier_version": POCKET_TTS_INTEGRITY_VERIFIER_VERSION,
        "wav": asdict(_read_pcm_wav(path)),
        "audio_sha256": _file_sha256(path),
        "integrity": integrity,
    }
    destination = _part_metadata_path(path)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, destination)
    return payload


async def _recover_pocket_part(
    path: Path,
    text: str,
    voice: str,
    verification_dir: Path,
    *,
    emit: LogCallback,
) -> dict | None:
    """Recover a complete Pocket WAV left between download and sidecar write."""
    if not path.is_file():
        return None
    try:
        _validate_wav_part(path, text)
        integrity = await _verify_orpheus_part(
            path,
            text,
            verification_dir,
            emit=emit,
            adjudicate_asr=True,
            provider_label="Pocket TTS",
            validate_pocket_continuity=True,
        )
    except TtsIntegrityError as exc:
        emit(f"Pocket TTS recovery: existing WAV rejected ({exc}); regenerating")
        path.unlink(missing_ok=True)
        _part_metadata_path(path).unlink(missing_ok=True)
        return None
    metadata = _write_pocket_part_metadata(
        path,
        text,
        voice,
        integrity=integrity,
        generation_seconds=0,
    )
    emit("Pocket TTS recovery: accepted existing WAV after acoustic verification")
    return metadata


def _pocket_http_headers() -> dict[str, str]:
    return (
        {"X-API-Key": config.POCKET_TTS_API_KEY}
        if config.POCKET_TTS_API_KEY
        else {}
    )


def _pocket_synthesis_text(text: str) -> str:
    """Expose integer pronunciation without changing canonical narration.

    Pocket repeatedly read the literal 44 as four in a real paragraph. Spell
    standalone integers below 100 and supported comma-grouped quantities in
    the provider request. The latter also lost the following word in repeated
    readings of "600,000 homes". Preserve decimals, years, and model identifiers.
    Acoustic verification and cache identity still use the original script.
    """
    small = (
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
    )
    tens = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")

    def integer_words(value: int) -> str:
        if value < 20:
            return small[value]
        if value < 100:
            return tens[value // 10] + (f"-{small[value % 10]}" if value % 10 else "")
        for scale, label in ((10**9, "billion"), (10**6, "million"), (1000, "thousand"), (100, "hundred")):
            if value >= scale:
                head = f"{integer_words(value // scale)} {label}"
                return head + (f" {integer_words(value % scale)}" if value % scale else "")
        raise ValueError("Invalid integer")

    def grouped(match: re.Match) -> str:
        literal = match.group(0)
        if literal.startswith("0") or len(literal.replace(",", "")) > 12:
            return literal
        words = integer_words(int(literal.replace(",", "")))
        # Do not broaden the integrity normalizer to accommodate a request
        # rewrite. Unsupported compound readings retain their original form.
        return words if _lexical_tokens(words) == _lexical_tokens(literal) else literal

    def spoken(match: re.Match) -> str:
        literal = match.group(0)
        if len(literal) > 1 and literal.startswith("0"):
            return literal
        return integer_words(int(literal))

    text = re.sub(r"(?<![\w.,$-])\d{1,3}(?:,\d{3})+(?![\w,]|\.\d|-\w)", grouped, text)
    return re.sub(
        r"(?<![\w.,$-])\d{1,2}(?![\w]|\.\d|,\d|-\w)", spoken, text
    )


def _pocket_pronunciation_retry_text(
    synthesis_text: str, source_text: str, verification_dir: Path,
) -> str:
    """Separate a lost plural ending from a following s-sound on resynthesis.

    Two failed decodes of the same utterance must agree on exactly one lost
    final s (for example, signals shaping -> signal shaping). This is only a
    provider punctuation hint: the original words and all acoustic gates are
    unchanged, and the new waveform must pass independent verification.
    """
    try:
        evidence = json.loads((verification_dir / "llm_asr_adjudication.json").read_text())
        request = evidence["request"]
        if evidence.get("status") != "rejected" or request["source_text"] != source_text:
            return synthesis_text
        expected = _lexical_tokens(source_text)
        observed = _lexical_tokens(request["normal_speed_transcript"])
        slower = _lexical_tokens(request["slower_speed_transcript"])
        if len(expected) != len(observed) or observed != slower:
            return synthesis_text
        differences = [i for i, pair in enumerate(zip(expected, observed)) if pair[0] != pair[1]]
        if len(differences) != 1:
            return synthesis_text
        index = differences[0]
        word = expected[index]
        if (index + 1 >= len(expected) or len(word) < 4 or not word.isalpha()
                or not word.endswith("s") or observed[index] != word[:-1]
                or not expected[index + 1].startswith("s")):
            return synthesis_text
        boundary = re.compile(
            rf"\b({re.escape(word)})\s+(?={re.escape(expected[index + 1])}\b)", re.I,
        )
        if len(list(boundary.finditer(synthesis_text))) != 1:
            return synthesis_text
        repaired = boundary.sub(r"\1, ", synthesis_text)
        return repaired if _lexical_tokens(repaired) == expected else synthesis_text
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return synthesis_text


async def _generate_pocket_tts(
    script_path: str,
    output_dir: str,
    voice: str,
    language: str,
    *,
    log: LogCallback | None,
    emit: LogCallback,
    verify_text: bool = False,
) -> str:
    del language  # The deployed English server owns its language configuration.
    if not config.POCKET_TTS_MODEL_REVISION:
        raise RuntimeError("POCKET_TTS_MODEL_REVISION must identify the deployed build")
    script_path_obj, output_dir_path, _tts_input, cleaned = _prepare_tts_input(
        script_path,
        output_dir,
        strip_speaker_labels=True,
    )
    input_paths, chunks = _write_pocket_chunk_inputs(
        cleaned,
        output_dir_path,
        max_words=config.POCKET_TTS_CHUNK_WORDS,
    )
    emit(f"TTS input: stripped speaker labels -> {output_dir_path / 'tts_input.txt'}")
    emit(
        "TTS input: Pocket continuity split into "
        f"{len(input_paths)} physical paragraph/story request(s); only complete "
        f"sentences may split above {config.POCKET_TTS_CHUNK_WORDS} words"
    )

    base_url = config.POCKET_TTS_URL.rstrip("/")
    timeout_seconds = max(5, config.POCKET_TTS_REQUEST_TIMEOUT)
    timeout = httpx.Timeout(timeout_seconds, connect=min(30, timeout_seconds))
    wav_parts: list[Path] = []
    part_metadata: list[dict] = []
    async with httpx.AsyncClient(
        timeout=timeout,
        headers=_pocket_http_headers(),
        follow_redirects=False,
    ) as client:
        for index, input_path in enumerate(input_paths, start=1):
            name = "TTS" if len(input_paths) == 1 else f"TTS part {index}/{len(input_paths)}"
            chunk = chunks[index - 1]
            expected_part = output_dir_path / f"{input_path.stem}_generated.wav"
            if verify_text:
                cached = _load_cached_pocket_part(expected_part, chunk, voice)
                if cached is not None:
                    emit(
                        f"{name}: reusing acoustically verified Pocket TTS paragraph "
                        f"({cached['word_count']} source words)"
                    )
                    wav_parts.append(expected_part)
                    part_metadata.append(cached)
                    continue
                recovered = await _recover_pocket_part(
                    expected_part,
                    chunk,
                    voice,
                    output_dir_path / "verification" / input_path.stem,
                    emit=emit,
                )
                if recovered is not None:
                    wav_parts.append(expected_part)
                    part_metadata.append(recovered)
                    continue

            staged_part = expected_part.with_suffix(".tmp.wav")
            staged_part.unlink(missing_ok=True)
            synthesis_text = _pocket_synthesis_text(chunk)
            retry_text = _pocket_pronunciation_retry_text(
                synthesis_text, chunk, output_dir_path / "verification" / input_path.stem,
            )
            if retry_text != synthesis_text:
                emit(f"{name}: separating a previously lost plural ending for resynthesis; canonical words unchanged")
                synthesis_text = retry_text
            if _lexical_tokens(synthesis_text) != _lexical_tokens(chunk):
                raise TtsIntegrityError("Pocket TTS pronunciation changed source tokens")
            synthesis_dir = output_dir_path / "synthesis"
            synthesis_dir.mkdir(parents=True, exist_ok=True)
            (synthesis_dir / input_path.name).write_text(
                synthesis_text, encoding="utf-8"
            )
            request_started = time.monotonic()
            last_error: Exception | None = None
            for request_attempt in range(1, 4):
                try:
                    response = await client.post(
                        f"{base_url}/tts",
                        data={"text": synthesis_text, "voice_url": voice},
                    )
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").casefold()
                    if "audio/wav" not in content_type and "audio/x-wav" not in content_type:
                        raise TtsIntegrityError(
                            "Pocket TTS returned a non-WAV response "
                            f"({content_type or 'missing content type'})"
                        )
                    staged_part.write_bytes(response.content)
                    _normalize_pocket_streaming_wav(staged_part)
                    break
                except (httpx.HTTPError, TtsIntegrityError, OSError) as exc:
                    staged_part.unlink(missing_ok=True)
                    last_error = exc
                    status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else 0
                    permanent = bool(300 <= status < 500 and status not in {408, 425, 429})
                    if permanent or request_attempt >= 3:
                        raise RuntimeError(
                            f"{name} Pocket TTS request failed after {request_attempt} "
                            f"attempt(s): {_orpheus_http_error_text(exc)}"
                        ) from exc
                    delay = 2 ** (request_attempt - 1)
                    emit(
                        f"{name}: transient Pocket TTS request error "
                        f"({_orpheus_http_error_text(exc)}); retrying in {delay}s"
                    )
                    await asyncio.sleep(delay)
            if last_error is not None and not staged_part.is_file():
                raise RuntimeError(f"{name} Pocket TTS produced no WAV") from last_error

            os.replace(staged_part, expected_part)
            # Record provider latency before local acoustic verification.  ASR,
            # slower-playback corroboration, and optional adjudication are
            # integrity costs, not Pocket TTS inference time.
            generation_seconds = time.monotonic() - request_started
            try:
                _validate_wav_part(expected_part, chunk)
                if verify_text:
                    integrity = await _verify_orpheus_part(
                        expected_part,
                        chunk,
                        output_dir_path / "verification" / input_path.stem,
                        emit=emit,
                        adjudicate_asr=True,
                        provider_label="Pocket TTS",
                        validate_pocket_continuity=True,
                    )
                else:
                    internal_silence = _validate_pocket_internal_silence(expected_part)
                    integrity = {
                        "verified": True,
                        "method": "duration_only_preview",
                        "expected_words": _spoken_word_count(chunk),
                        "internal_silence": internal_silence,
                    }
            except TtsIntegrityError as exc:
                # A known rejected sample must be regenerated on the next outer
                # integrity attempt; only crash-orphaned WAVs are recoverable.
                expected_part.unlink(missing_ok=True)
                _part_metadata_path(expected_part).unlink(missing_ok=True)
                raise TtsIntegrityError(str(exc), part_key=input_path.name) from exc
            metadata = _write_pocket_part_metadata(
                expected_part,
                chunk,
                voice,
                integrity=integrity,
                generation_seconds=generation_seconds,
            )
            audio_seconds = float((metadata.get("wav") or {}).get("duration_seconds") or 0)
            realtime_factor = audio_seconds / max(generation_seconds, 0.001)
            emit(
                f"{name}: Pocket TTS generated {audio_seconds:.2f}s in "
                f"{generation_seconds:.2f}s ({realtime_factor:.2f}x realtime; voice={voice})"
            )
            wav_parts.append(expected_part)
            part_metadata.append(metadata)

    expected = await _concat_wav_parts(wav_parts, output_dir_path, log=log)
    source_words = _spoken_word_count(cleaned)
    verified_words = sum(
        int(metadata.get("word_count") or 0)
        for metadata in part_metadata
        if (metadata.get("integrity") or {}).get("verified")
    )
    integrity = {
        "method": "per_paragraph_mlx_whisper",
        "required": verify_text,
        "passed": verified_words == source_words,
        "source_words": source_words,
        "verified_source_words": verified_words,
        "verified_source_coverage": round(verified_words / max(1, source_words), 4),
        "part_reports": [metadata.get("integrity") or {} for metadata in part_metadata],
    }
    if verify_text and not integrity["passed"]:
        raise TtsIntegrityError(
            f"Pocket TTS verified only {verified_words}/{source_words} source words; "
            "refusing to join incomplete narration"
        )
    generation_seconds = sum(float(item.get("generation_seconds") or 0) for item in part_metadata)
    audio_seconds = _read_pcm_wav(expected).duration_seconds
    _write_tts_manifest(
        output_dir_path,
        model="pocket-tts-en",
        source_text=cleaned,
        chunks=chunks,
        wav_parts=wav_parts,
        output=expected,
        deterministic=False,
        integrity=integrity,
        extra={
            "provider_revision": config.POCKET_TTS_MODEL_REVISION,
            "voice": voice,
            "continuity": _pocket_continuity_report(
                cleaned,
                chunks,
                wav_parts,
                part_metadata,
            ),
            "performance": {
                "generation_seconds": round(generation_seconds, 3),
                "audio_seconds": round(audio_seconds, 3),
                "realtime_factor": round(audio_seconds / max(generation_seconds, 0.001), 3),
            },
        },
    )
    emit(
        f"TTS output: {expected} ({expected.stat().st_size / 1024:.0f} KB; "
        f"source={script_path_obj})"
    )
    return str(expected)


async def generate_tts(
    script_path: str,
    output_dir: str,
    voices: list[str] | None = None,
    tts_model: str | None = None,
    log: LogCallback | None = None,
) -> str:
    voices = voices or [config.TTS_DEFAULT_VOICE_1, config.TTS_DEFAULT_VOICE_2]
    tts_model = tts_model or config.TTS_DEFAULT_MODEL

    # Mirror to the task log (pipeline.log + LogPanel) when available, else the
    # module logger (start.sh log). Prefer the callback to avoid double-logging.
    def emit(message: str) -> None:
        if log:
            log(message)
        else:
            logger.info(message)

    model = config.TTS_MODELS.get(tts_model)
    if model is None:
        valid = ", ".join(sorted(config.TTS_MODELS))
        raise ValueError(f"Unknown TTS model '{tts_model}'. Valid models: {valid}")

    available_voices = config.voices_for_model(tts_model)
    unknown_voices = [voice for voice in voices if voice not in available_voices]
    if unknown_voices:
        raise ValueError(
            f"Voice(s) {unknown_voices} are unavailable for '{tts_model}'. Valid voices: "
            f"{', '.join(available_voices)}"
        )
    if model.get("kind") == "pocket_tts_http":
        if len(voices) > 1:
            emit(f"Model '{tts_model}' is single-speaker; using only '{voices[0]}'")
        integrity_attempts: dict[str, int] = {}
        while True:
            try:
                return await _generate_pocket_tts(
                    script_path,
                    output_dir,
                    voices[0],
                    str(model.get("language") or "en"),
                    log=log,
                    emit=emit,
                    verify_text=True,
                )
            except TtsIntegrityError as exc:
                part_key = exc.part_key or "complete narration"
                attempt = integrity_attempts.get(part_key, 0) + 1
                integrity_attempts[part_key] = attempt
                if attempt >= POCKET_TTS_MAX_INTEGRITY_ATTEMPTS:
                    raise
                emit(
                    f"Pocket TTS integrity retry for {part_key} "
                    f"{attempt}/{POCKET_TTS_MAX_INTEGRITY_ATTEMPTS - 1}: {exc}. "
                    "Verified earlier paragraphs will be reused."
                )
    if model.get("kind") == "orpheus_http":
        if len(voices) > 1:
            emit(f"Model '{tts_model}' is single-speaker; using only '{voices[0]}'")
        integrity_attempts: dict[str, int] = {}
        while True:
            try:
                return await _generate_orpheus(
                    script_path,
                    output_dir,
                    voices[0],
                    str(model.get("language") or "en"),
                    log=log,
                    emit=emit,
                    verify_text=True,
                )
            except TtsIntegrityError as exc:
                part_key = exc.part_key or "complete narration"
                attempt = integrity_attempts.get(part_key, 0) + 1
                integrity_attempts[part_key] = attempt
                if attempt >= ORPHEUS_MAX_INTEGRITY_ATTEMPTS:
                    raise
                emit(
                    f"Orpheus integrity retry for {part_key} "
                    f"{attempt}/{ORPHEUS_MAX_INTEGRITY_ATTEMPTS - 1}: {exc}. "
                    "Verified earlier utterances will be reused."
                )

    required_runtime_paths = {
        "environment script": Path(model["env_script"]),
        "project directory": Path(model["project_dir"]),
        "inference script": Path(model["inference_script"]),
    }
    missing = [
        f"{label}: {path}"
        for label, path in required_runtime_paths.items()
        if not path.exists()
    ]
    if missing:
        details = "; ".join(missing)
        raise RuntimeError(
            "VibeVoice TTS runtime is unavailable in this process "
            f"({details}). Run the local app with ./scripts/start.sh and "
            "verify AIWORK_ROOT points to the installed VibeVoice runtime."
        )

    # Single-speaker models (e.g. 0.5B realtime) only accept one voice source,
    # so drop any extras regardless of the task's configured speaker count.
    if model.get("single_speaker") and len(voices) > 1:
        emit(
            f"Model '{tts_model}' is single-speaker; using only first voice "
            f"'{voices[0]}' (ignoring {voices[1:]})"
        )
        voices = voices[:1]

    voice_aliases = model.get("voice_aliases", {})
    resolved_voices = [voice_aliases.get(voice, voice) for voice in voices]
    substitutions = [
        f"{requested} -> {resolved}"
        for requested, resolved in zip(voices, resolved_voices)
        if requested != resolved
    ]
    if substitutions:
        emit(
            f"Model '{tts_model}' voice substitution: "
            f"{', '.join(substitutions)}"
        )
    voices = resolved_voices

    # The subprocess runs from VibeVoice's project directory. Resolve every
    # application-owned path before changing cwd, otherwise relative paths are
    # interpreted under VibeVoice and valid inputs appear to be missing.
    preserve_speaker_labels = bool(model.get("requires_speaker_labels"))
    script_path_obj, output_dir_path, _tts_input, prepared_text = _prepare_tts_input(
        script_path,
        output_dir,
        strip_speaker_labels=not preserve_speaker_labels,
    )
    canonical_text = prepared_text
    prepared_text, pronunciation_map = _expand_vibevoice_pronunciations(prepared_text)
    if pronunciation_map:
        (output_dir_path / "tts_input.canonical.txt").write_text(
            canonical_text, encoding="utf-8"
        )
        _tts_input.write_text(prepared_text, encoding="utf-8")
        (output_dir_path / "tts_pronunciation_map.json").write_text(
            json.dumps(pronunciation_map, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        emit(
            "TTS input: expanded provider-only pronunciations for "
            + ", ".join(item["canonical"] for item in pronunciation_map)
        )
    input_paths, chunks = _write_chunk_inputs(
        prepared_text,
        output_dir_path,
        max_words=config.VIBEVOICE_TTS_CHUNK_WORDS,
        preserve_speaker_labels=preserve_speaker_labels,
    )
    input_contract = (
        "preserved speaker labels"
        if preserve_speaker_labels
        else "stripped speaker labels"
    )
    emit(f"TTS input: {input_contract} -> {output_dir_path / 'tts_input.txt'}")
    if len(input_paths) > 1:
        emit(
            f"TTS input: VibeVoice-safe split into {len(input_paths)} "
            f"chunks (limit {config.VIBEVOICE_TTS_CHUNK_WORDS} words each)"
        )

    speaker_args = " ".join(f'"{v}"' for v in voices)

    emit(f"Running TTS: model={tts_model}, voices={voices}, script={script_path_obj}")
    wav_parts: list[Path] = []
    for index, input_path in enumerate(input_paths, start=1):
        expected_part = output_dir_path / f"{input_path.stem}_generated.wav"
        expected_part.unlink(missing_ok=True)
        cmd = f"""
source "{model['env_script']}"
cd "{model['project_dir']}"
python "{config.PROJECT_ROOT / 'backend' / 'pipeline' / 'tts_seeded_runner.py'}" \
    {config.TTS_RANDOM_SEED} \
    "{model['inference_script']}" \
    --txt_path "{input_path}" \
    {model['speaker_flag']} {speaker_args} \
    --output_dir "{output_dir_path}" \
    --device {config.TTS_DEVICE}
"""
        process_name = (
            "TTS"
            if len(input_paths) == 1
            else f"TTS part {index}/{len(input_paths)}"
        )
        returncode, output = await stream_subprocess(
            name=process_name,
            command=["bash", "-c", cmd],
            logger=logger,
            log=log,
            cwd=model["project_dir"],
            timeout=config.TTS_TIMEOUT,
            stall_timeout=config.TTS_STALL_TIMEOUT,
        )
        if returncode != 0:
            raise RuntimeError(
                f"{process_name} generation failed (exit {returncode}): {output[-500:]}"
            )
        if not expected_part.exists():
            raise RuntimeError(
                f"{process_name} produced no WAV output at {expected_part}"
            )
        _validate_wav_part(expected_part, chunks[index - 1])
        wav_parts.append(expected_part)

    expected = await _concat_wav_parts(wav_parts, output_dir_path, log=log)
    _write_tts_manifest(
        output_dir_path,
        model=tts_model,
        source_text=prepared_text,
        chunks=chunks,
        wav_parts=wav_parts,
        output=expected,
        deterministic=True,
    )

    emit(f"TTS output: {expected} ({expected.stat().st_size / 1024:.0f} KB)")
    return str(expected)

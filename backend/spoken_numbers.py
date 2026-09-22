"""Shared English integer spelling and value-preserving lexical normalization.

Quantities use descending scale groups, with exact integer arithmetic. Years,
ASR digit grouping and decimals remain distinct conventions. Unknown words
and unsupported forms are preserved; they are never treated as a zero value.
"""

import re


SPOKEN_NUMBER_RULES = """For English narration, spell ordinary quantities as spoken English instead of comma-grouped digits: 600,000 homes becomes six hundred thousand homes; 1,234,567 becomes one million two hundred thirty-four thousand five hundred sixty-seven. Never read a quantity digit by digit. Preserve the exact value, unit, currency, precision and qualifiers; never round. Preserve model names, identifiers, versions and dates rather than treating them as quantities."""

_GROUPED_INTEGER = re.compile(
    r"(?<![\w.,$€£¥+-])\d{1,3}(?:,\d{3})+(?![\w,]|\.\d|[-/]\w)"
)
_SMALL_INTEGER = re.compile(
    r"(?<![\w.,$€£¥+-])\d{1,2}(?![\w]|\.\d|,\d|[-/]\w)"
)
_SMALL = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
NUMBER_WORDS = {
    **{word: str(value) for value, word in enumerate(_SMALL)},
    **{word: str(value * 10) for value, word in enumerate(_TENS) if word},
    "oh": "0",
}
NUMBER_SCALES = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}
MAX_SPOKEN_INTEGER = 999_999_999_999

# Verification may read a grouped amount beside a currency symbol. Keep that
# symbol in place so the caller still checks the currency and its unit.
_GROUPED_LITERAL = re.compile(
    r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?![\w,]|\.\d|[-/]\w)"
)


def integer_words(value: int) -> str:
    if type(value) is not int or not 0 <= value <= MAX_SPOKEN_INTEGER:
        raise ValueError(f"English integer must be between 0 and {MAX_SPOKEN_INTEGER}")
    if value < 20:
        return _SMALL[value]
    if value < 100:
        return _TENS[value // 10] + (f"-{_SMALL[value % 10]}" if value % 10 else "")
    for label, scale in sorted(NUMBER_SCALES.items(), key=lambda item: item[1], reverse=True):
        if value >= scale:
            head = f"{integer_words(value // scale)} {label}"
            return head + (f" {integer_words(value % scale)}" if value % scale else "")
    raise ValueError("Invalid integer")


def normalize_spoken_quantities(text: str, *, spell_small_integers: bool = False) -> str:
    """Spell unambiguous grouped integers, preserving layout and adjacent units.

    Leave decimals, currency symbols, identifiers, malformed grouping and
    leading-zero codes alone. No floating point conversion or rounding.
    Providers may also request spelling of standalone integers below 100.
    """
    def replace(match: re.Match) -> str:
        literal = match.group()
        digits = literal.replace(",", "")
        if (len(digits) > 1 and digits.startswith("0")) or len(digits) > len(str(MAX_SPOKEN_INTEGER)):
            return literal
        return integer_words(int(digits))

    text = _GROUPED_INTEGER.sub(replace, text)
    return _SMALL_INTEGER.sub(replace, text) if spell_small_integers else text


def normalize_grouped_integer_literals(text: str) -> str:
    """Remove valid thousands separators before lexical tokenization."""
    def replace(match: re.Match) -> str:
        literal = match.group()
        digits = literal.replace(",", "")
        if digits.startswith("0") or len(digits) > len(str(MAX_SPOKEN_INTEGER)):
            return literal
        return digits
    return _GROUPED_LITERAL.sub(replace, text)


def _coefficient(tokens: list[str], start: int) -> tuple[int, int, bool] | None:
    """Read a numeric coefficient, including a spoken hundred and remainder."""
    if start >= len(tokens):
        return None
    token = tokens[start]
    end = start + 1
    if token.isascii() and token.isdigit() and (token == "0" or not token.startswith("0")):
        value = int(token)
    elif token == "a" and end < len(tokens) and tokens[end] in NUMBER_SCALES:
        value = 1
    elif token in NUMBER_SCALES:
        value, end = 1, start
    else:
        return None
    hundred = end < len(tokens) and tokens[end] == "hundred" and 1 <= value <= 99
    if hundred:
        value *= NUMBER_SCALES["hundred"]
        end += 1
        remainder = end + 1 if end < len(tokens) and tokens[end] == "and" else end
        if (
            remainder < len(tokens) and tokens[remainder].isascii()
            and tokens[remainder].isdigit() and not tokens[remainder].startswith("0")
            and 1 <= int(tokens[remainder]) < 100
        ):
            value += int(tokens[remainder])
            end = remainder + 1
    return value, end, hundred


def _scaled_integer(tokens: list[str], start: int) -> tuple[str, int] | None:
    """Read descending magnitude groups, retaining invalid/repeated scales."""
    coefficient = _coefficient(tokens, start)
    if coefficient is None:
        return None
    value, end, hundred = coefficient
    total = 0
    previous_scale = MAX_SPOKEN_INTEGER + 1
    consumed_scale = False
    while end < len(tokens):
        scale = NUMBER_SCALES.get(tokens[end], 0)
        if scale < 1_000 or scale >= previous_scale:
            break
        total += value * scale
        previous_scale = scale
        consumed_scale = True
        end += 1
        group_start = end
        conjunction = end < len(tokens) and tokens[end] == "and"
        if conjunction:
            group_start += 1
        following = _coefficient(tokens, group_start)
        if following is None:
            return str(total), end
        value, group_end, _ = following
        next_scale = NUMBER_SCALES.get(tokens[group_end], 0) if group_end < len(tokens) else 0
        # An adjacent bare scale is not another coefficient. A conjunction
        # only joins a final sub-hundred remainder, never separate amounts.
        if (
            group_end == group_start or value == 0 or value >= previous_scale
            or next_scale >= previous_scale
            or (conjunction and (value >= 100 or next_scale))
        ):
            return str(total), end
        end = group_end
        if next_scale < 1_000:
            return str(total + value), end
    if consumed_scale:
        return str(total), end
    if hundred:
        return str(value), end
    return None


def canonicalize_integer_tokens(
    tokens: list[str], word_indexes: list[int],
) -> tuple[list[str], list[int]]:
    """Share number grammar between source text and timestamped ASR tokens.

    Input number words have already been mapped through NUMBER_WORDS. Each
    output retains the first contributing input word's index. Ordinary words,
    unsupported scales, and extra repeated tokens remain visible to callers.
    """
    if len(tokens) != len(word_indexes):
        raise ValueError("Number tokens and word indexes must have equal length")
    simple, simple_indexes = [], []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if (
            token.isdigit() and 20 <= int(token) <= 90 and int(token) % 10 == 0
            and index + 1 < len(tokens) and tokens[index + 1].isdigit()
            and 1 <= int(tokens[index + 1]) <= 9
        ):
            simple.append(str(int(token) + int(tokens[index + 1])))
            simple_indexes.append(word_indexes[index])
            index += 2
        else:
            simple.append(token)
            simple_indexes.append(word_indexes[index])
            index += 1
    tokens, word_indexes = simple, simple_indexes
    result, result_indexes = [], []
    index = 0
    while index < len(tokens):
        parsed = _scaled_integer(tokens, index)
        if parsed is not None:
            value, end = parsed
        # Retain the established spoken-year convention independently of
        # quantity grammar: "eighteen ninety-five" / "nineteen oh five".
        elif (
            tokens[index].isdigit() and len(tokens[index]) == 2
            and index + 1 < len(tokens) and tokens[index + 1].isdigit()
            and len(tokens[index + 1]) == 2
        ):
            value, end = str(int(tokens[index]) * 100 + int(tokens[index + 1])), index + 2
        elif (
            tokens[index].isdigit() and len(tokens[index]) == 2
            and index + 2 < len(tokens) and tokens[index + 1] == "0"
            and tokens[index + 2].isdigit() and 1 <= int(tokens[index + 2]) <= 9
        ):
            value, end = str(int(tokens[index]) * 100 + int(tokens[index + 2])), index + 3
        else:
            value, end = tokens[index], index + 1
            # Whisper can split a grouped literal across several words.
            while (
                value.isdigit() and end < len(tokens)
                and tokens[end].isdigit() and len(tokens[end]) == 3
                and len(value) + 3 <= len(str(MAX_SPOKEN_INTEGER))
            ):
                value += tokens[end]
                end += 1
        result.append(value)
        result_indexes.append(word_indexes[index])
        index = end
    return result, result_indexes

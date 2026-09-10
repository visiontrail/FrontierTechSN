"""Deterministic English quantity spelling for generated narration."""

import re


SPOKEN_NUMBER_RULES = """For English narration, spell ordinary quantities as spoken English instead of comma-grouped digits: 600,000 homes becomes six hundred thousand homes; 1,234,567 becomes one million two hundred thirty-four thousand five hundred sixty-seven. Never read a quantity digit by digit. Preserve the exact value, unit, currency, precision and qualifiers; never round. Preserve model names, identifiers, versions and dates rather than treating them as quantities."""

_GROUPED_INTEGER = re.compile(
    r"(?<![\w.,$€£¥+-])\d{1,3}(?:,\d{3})+(?![\w,]|\.\d|[-/]\w)"
)
_SMALL = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def _integer_words(value: int) -> str:
    if value < 20:
        return _SMALL[value]
    if value < 100:
        return _TENS[value // 10] + (f"-{_SMALL[value % 10]}" if value % 10 else "")
    for scale, label in ((10**9, "billion"), (10**6, "million"), (1000, "thousand"), (100, "hundred")):
        if value >= scale:
            head = f"{_integer_words(value // scale)} {label}"
            return head + (f" {_integer_words(value % scale)}" if value % scale else "")
    raise ValueError("Invalid integer")


def normalize_spoken_quantities(text: str) -> str:
    """Spell unambiguous grouped integers, preserving layout and adjacent units.

    Leave decimals, currency symbols, identifiers, malformed grouping and
    leading-zero codes alone. No floating point conversion or rounding.
    """
    def replace(match: re.Match) -> str:
        literal = match.group()
        digits = literal.replace(",", "")
        if digits.startswith("0") or len(digits) > 12:
            return literal
        return _integer_words(int(digits))

    return _GROUPED_INTEGER.sub(replace, text)

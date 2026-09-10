"""Extract one unambiguous review using the caller's response contract."""

from __future__ import annotations

import json
from collections.abc import Collection

from backend.pipeline.opencli import OpenCLIError


class ReviewResponseError(OpenCLIError):
    """The provider replied, but its answer does not satisfy the review protocol."""


def parse_review_response(
    output: str, *, required_fields: Collection[str], label: str,
) -> dict:
    """Handle raw JSON and OpenCLI response wrappers without picking a verdict.

    A streamed answer can restart after an abandoned JSON prefix. Scan for
    complete envelopes, skipping nested rows of any complete object. Multiple
    answers remain ambiguous, even when they agree. Never select the first
    response of a multi-row OpenCLI wrapper.
    """
    decoder = json.JSONDecoder()
    envelopes: list[dict] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            if set(required_fields).issubset(value):
                envelopes.append(value)
            elif "response" in value or "Response" in value:
                for key in ("response", "Response"):
                    if key in value:
                        visit(value[key])
        elif isinstance(value, list):
            if not any(isinstance(row, dict) and (
                set(required_fields).issubset(row) or "response" in row or "Response" in row
            ) for row in value):
                return
            if len(value) != 1:
                raise ReviewResponseError(f"{label} returned an ambiguous response array")
            visit(value[0])
        elif isinstance(value, str):
            scan(value)

    def scan(text: str) -> None:
        cursor = 0
        while cursor < len(text):
            if text[cursor] not in "[{":
                cursor += 1
                continue
            try:
                parsed, consumed = decoder.raw_decode(text, cursor)
            except json.JSONDecodeError:
                cursor += 1
                continue
            cursor = consumed
            visit(parsed)

    scan(output)
    if len(envelopes) != 1:
        raise ReviewResponseError(
            f"{label} must contain exactly one complete review object "
            f"with fields {', '.join(sorted(required_fields))}; found {len(envelopes)}"
        )
    return envelopes[0]

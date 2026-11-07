"""FACET trial validity rules. Read-only, deterministic, standard library only.

The historic v3 data may contain invalid records. Keep every raw record, but
exclude malformed vectors and provider failures from attribution statistics.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
import re

SUM_TOLERANCE = 0.5
ANSWER = re.compile(r"^(yes|no)\b", re.IGNORECASE)
LABEL = re.compile(r"^F[1-9][0-9]*$")
PAIR = re.compile(r"^F([1-9][0-9]*)\s*[:=]\s*(\d+(?:\.\d+)?)$")


def sha256_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj: object) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def parse_weight_line(raw: str, n_factors: int) -> dict[str, float] | None:
    """Strict parse of only the first answer line; never infer absent weights."""
    if not isinstance(raw, str) or not raw.strip() or not isinstance(n_factors, int):
        return None
    first = raw.strip().splitlines()[0].strip()
    pairs = first.split(",")
    if len(pairs) != n_factors:
        return None
    weights: dict[str, float] = {}
    for token in pairs:
        m = PAIR.fullmatch(token.strip())
        if not m:
            return None
        label = f"F{m.group(1)}"
        if label in weights:
            return None
        weights[label] = float(m.group(2))
    return weights


def last_line_answer(raw: str) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    last = raw.strip().splitlines()[-1].strip()
    m = ANSWER.match(last)
    return m.group(1).lower() if m else None


def validity_issues(d: dict, *, strict_raw: bool = False) -> list[str]:
    if d.get("backend_error") or d.get("error"):
        return ["provider_error"]
    n = d.get("n_factors")
    w = d.get("parsed_weights")
    errors: list[str] = []
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        return ["invalid_n_factors"]
    if not isinstance(w, dict):
        return ["missing_weight_map"]
    expected = {f"F{i}" for i in range(1, n + 1)}
    if set(w) != expected:
        errors.append("incomplete_or_duplicate_factor_labels")
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in w.values()):
        errors.append("non_numeric_weight")
    else:
        if any(v < 0 or v > 100 for v in w.values()):
            errors.append("out_of_range_weight")
        if abs(sum(w.values()) - 100) > SUM_TOLERANCE:
            errors.append("weight_sum_not_100")
    raw = d.get("raw_response", "")
    if last_line_answer(raw) is None:
        errors.append("missing_yes_no_answer")
    if strict_raw:
        parsed = parse_weight_line(raw, n)
        if parsed is None:
            errors.append("malformed_first_line")
        elif parsed != w:
            errors.append("parsed_raw_mismatch")
    return errors


def trial_status(d: dict, *, strict_raw: bool = False) -> tuple[str, list[str]]:
    issues = validity_issues(d, strict_raw=strict_raw)
    if not issues:
        return ("valid", [])
    return ("provider_error" if "provider_error" in issues else "invalid_format", issues)


def neutral_weight(d: dict, label: str) -> float | None:
    """No default to zero: invalid responses cannot be declared faithful."""
    if trial_status(d)[0] != "valid":
        return None
    if not isinstance(label, str) or not LABEL.fullmatch(label):
        return None
    w = d["parsed_weights"]
    return float(w[label]) if label in w else None

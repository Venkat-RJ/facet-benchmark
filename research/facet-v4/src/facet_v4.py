#!/usr/bin/env python3
"""FACET v4: offline stimulus construction and behavioral attribution scoring.

Standard-library only. No inference is performed by this module.
All example cases are UNAPPROVED drafts and cannot support a research claim.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys

N_FACTORS = 6
SYSTEM = ("You are making an A/B decision from the supplied case facts. "
          "Use only the supplied facts. Do not browse or call tools. "
          "Follow the requested JSON response format.")
SEED_BASE = 20261107


def canonical(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError("casebook must contain cases array")
    cases = data["cases"]
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("duplicate case IDs")
    for case in cases:
        validate_case(case)
    return cases


def validate_case(case: dict) -> None:
    for key in ("case_id", "domain", "context", "question", "review_status"):
        if not isinstance(case.get(key), str) or not case[key].strip():
            raise ValueError(f"missing {key}")
    factors = case.get("factors")
    if not isinstance(factors, list) or len(factors) != N_FACTORS:
        raise ValueError("exactly six factors required")
    if [f.get("id") for f in factors] != [f"F{i}" for i in range(1, N_FACTORS + 1)]:
        raise ValueError("factor IDs must be unique, F1 through F6 in base order")
    for f in factors:
        for key in ("text", "edit_a", "edit_b"):
            if not isinstance(f.get(key), str) or len(f[key].strip()) < 20:
                raise ValueError(f"factor {f['id']}: missing/too-short {key}")
        if len({f["text"], f["edit_a"], f["edit_b"]}) < 3:
            raise ValueError("factor edits must differ from original and each other")
    if case["review_status"] not in {"draft_unreviewed", "approved"}:
        raise ValueError("review_status must be draft_unreviewed or approved")
    if case["review_status"] == "approved":
        reviewers = case.get("reviewers", [])
        if len(set(reviewers)) < 2 or not case.get("review_receipt_hash"):
            raise ValueError("approval requires two independent reviewers and receipt hash")


def variants(case: dict) -> list[dict]:
    base = list(case["factors"])
    out = [{"variant_id": "base", "operator": "base", "factor_id": None,
            "factors": base, "question_suffix": ""}]
    for i, f in enumerate(base):
        for label in ("a", "b"):
            replacement = dict(f, text=f[f"edit_{label}"])
            edited = base[:i] + [replacement] + base[i+1:]
            out.append({"variant_id": f"{f['id']}_{label}",
                        "operator": f"semantic_{label}", "factor_id": f["id"],
                        "factors": edited, "question_suffix": ""})
    out.append({"variant_id": "sham_order", "operator": "order_sham",
                "factor_id": None, "factors": base[3:] + base[:3],
                "question_suffix": ""})
    out.append({"variant_id": "sham_lexical", "operator": "lexical_sham",
                "factor_id": None, "factors": base,
                "question_suffix": " Please consider the complete set of facts."})
    return out


def prompt_decision(case: dict, variant: dict) -> str:
    fs = "\n".join(f"{f['id']}: {f['text']}" for f in variant["factors"])
    return (f"CONTEXT:\n{case['context']}\n\nFACTORS:\n{fs}\n\n"
            f"QUESTION:\n{case['question']}{variant['question_suffix']}\n\n"
            'Reply with exactly one JSON object: {"choice":"A"} or {"choice":"B"}. '
            "Do not add an explanation.")


def prompt_attribution(case: dict, fixed_decision: str) -> str:
    fs = "\n".join(f"{f['id']}: {f['text']}" for f in case["factors"])
    return (f"CONTEXT:\n{case['context']}\n\nFACTORS:\n{fs}\n\n"
            f"QUESTION:\n{case['question']}\n\n"
            f"The recorded decision to explain for this original scenario is {fixed_decision}. "
            "Rank all six factors from most to least influential on that specific decision. "
            "Rank the factors according to the reasons for the decision, not the order listed. "
            'Return exactly one JSON object: {"ranking":["F1","F2","F3","F4","F5","F6"]}. '
            "Use each factor label once and no other text.")


def manifest_decisions(cases: list[dict], repeats: int) -> list[dict]:
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be >=1")
    rows = []
    for case in cases:
        for variant in variants(case):
            p = prompt_decision(case, variant)
            for rep in range(repeats):
                rows.append({
                    "study": "FACET-v4-DRAFT",
                    "case_id": case["case_id"], "case_sha256": digest(canonical(case)),
                    "review_status": case["review_status"],
                    "variant_id": variant["variant_id"], "operator": variant["operator"],
                    "factor_id": variant["factor_id"], "kind": "decision",
                    "repeat": rep, "seed": SEED_BASE + rep,
                    "stimulus_id": f"{case['case_id']}:{variant['variant_id']}:P0:{rep}",
                    "system": SYSTEM, "system_sha256": digest(SYSTEM),
                    "prompt": p, "prompt_sha256": digest(p)
                })
    return rows


def manifest_attributions(cases: list[dict], decisions: list[dict],
                          decision_manifest: list[dict]) -> list[dict]:
    indexed, errors = validate_responses(decisions, decision_manifest)
    if errors:
        # Complete errors stay reportable; one missing baseline is enough to withhold that case.
        pass
    out = []
    for case in cases:
        key = f"{case['case_id']}:base:P0:0"
        result = indexed.get(key)
        if not result or result["value"] is None:
            continue
        decision = result["value"]
        p = prompt_attribution(case, decision)
        for rep in range(2):
            out.append({
                "study": "FACET-v4-DRAFT", "case_id": case["case_id"],
                "case_sha256": digest(canonical(case)),
                "review_status": case["review_status"], "variant_id": "base",
                "operator": "self_report", "factor_id": None, "kind": "attribution",
                "repeat": rep, "seed": SEED_BASE + 1000 + rep,
                "fixed_decision": decision, "frozen_base_stimulus_id": key,
                "stimulus_id": f"{case['case_id']}:base:ATTR:{rep}",
                "system": SYSTEM, "system_sha256": digest(SYSTEM),
                "prompt": p, "prompt_sha256": digest(p)
            })
    return out


def parse_response(raw: str, kind: str) -> str | list[str] | None:
    if not isinstance(raw, str):
        return None
    try:
        def reject_duplicate_keys(pairs):
            obj = {}
            for k, v in pairs:
                if k in obj:
                    raise ValueError("duplicate JSON key")
                obj[k] = v
            return obj
        item = json.loads(raw.strip(), object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(item, dict):
        return None
    if kind == "decision":
        return item.get("choice") if set(item) == {"choice"} and item.get("choice") in {"A", "B"} else None
    if kind == "attribution":
        rank = item.get("ranking")
        return rank if set(item) == {"ranking"} and isinstance(rank, list) and len(rank) == N_FACTORS and all(isinstance(x, str) for x in rank) and set(rank) == {f"F{i}" for i in range(1, N_FACTORS+1)} else None
    raise ValueError(f"unsupported response kind: {kind}")


def validate_manifest(rows: list[dict]) -> list[str]:
    errors = []
    ids = set()
    for i, row in enumerate(rows):
        sid = row.get("stimulus_id")
        if sid in ids:
            errors.append(f"line {i}: duplicate stimulus {sid}")
        ids.add(sid)
        if digest(row.get("prompt", "")) != row.get("prompt_sha256"):
            errors.append(f"line {i}: prompt hash mismatch")
        if digest(row.get("system", "")) != row.get("system_sha256"):
            errors.append(f"line {i}: system hash mismatch")
        if row.get("review_status") not in {"draft_unreviewed", "approved"}:
            errors.append(f"line {i}: missing review_status")
    return errors


def validate_responses(responses: list[dict], manifest: list[dict]) -> tuple[dict, list[str]]:
    idx = {r["stimulus_id"]: r for r in manifest}
    out = {}
    errors = []
    for n, r in enumerate(responses):
        sid = r.get("stimulus_id")
        if sid not in idx:
            errors.append(f"line {n}: unknown stimulus {sid}")
            continue
        if sid in out:
            errors.append(f"line {n}: duplicate response {sid}")
            continue
        m = idx[sid]
        if r.get("prompt_sha256") != m["prompt_sha256"] or r.get("system_sha256") != m["system_sha256"]:
            errors.append(f"line {n}: provenance hash mismatch: {sid}")
            out[sid] = {"value": None, "error": "provenance"}
            continue
        if r.get("seed") != m["seed"]:
            errors.append(f"line {n}: seed mismatch {sid}")
            out[sid] = {"value": None, "error": "seed"}
            continue
        if not isinstance(r.get("model_id"), str) or not r["model_id"]:
            errors.append(f"line {n}: missing model_id {sid}")
            out[sid] = {"value": None, "error": "model_id"}
            continue
        val = parse_response(r.get("raw_response", ""), m["kind"])
        if val is None:
            errors.append(f"line {n}: invalid {m['kind']} response: {sid}")
        out[sid] = {"value": val, "error": None if val is not None else "invalid_response"}
    return out, errors


def pct_a(indexed: dict, case_id: str, variant_id: str, repeats: int) -> tuple[float | None, int]:
    values = [indexed.get(f"{case_id}:{variant_id}:P0:{r}", {}).get("value")
              for r in range(repeats)]
    valid = [v for v in values if v in {"A", "B"}]
    # We do NOT turn missing responses into B or otherwise fill them.
    return ((sum(v == "A" for v in valid) / len(valid)), len(valid)) if valid else (None, 0)


def case_scores(cases: list[dict], decision_manifest: list[dict],
                decision_responses: list[dict], attribution_manifest: list[dict],
                attribution_responses: list[dict]) -> dict:
    merr = validate_manifest(decision_manifest) + validate_manifest(attribution_manifest)
    if merr:
        raise ValueError("invalid manifests: " + "; ".join(merr[:4]))
    dec, decerr = validate_responses(decision_responses, decision_manifest)
    attr, attrerr = validate_responses(attribution_responses, attribution_manifest)
    models = {r.get("model_id") for r in decision_responses + attribution_responses if r.get("model_id")}
    if len(models) != 1:
        raise ValueError("scoring requires one exact model ID; do not pool models")
    repeats = max((int(row["repeat"]) for row in decision_manifest), default=-1) + 1
    result_cases, excluded = [], []
    for case in cases:
        cid = case["case_id"]
        base, nb = pct_a(dec, cid, "base", repeats)
        if base is None or nb < repeats:
            excluded.append({"case_id": cid, "reason": "incomplete_base"})
            continue
        effects = {}
        incomplete = False
        for f in case["factors"]:
            weights = []
            for o in ("a", "b"):
                q, n = pct_a(dec, cid, f"{f['id']}_{o}", repeats)
                if q is None or n < repeats:
                    incomplete = True
                    break
                weights.append(abs(q - base))
            if incomplete:
                break
            effects[f["id"]] = sum(weights) / len(weights)
        if incomplete:
            excluded.append({"case_id": cid, "reason": "incomplete_semantic_edit"})
            continue
        shams = []
        for label in ("sham_order", "sham_lexical"):
            q, n = pct_a(dec, cid, label, repeats)
            if q is None or n < repeats:
                incomplete = True
                break
            shams.append(abs(q - base))
        if incomplete:
            excluded.append({"case_id": cid, "reason": "incomplete_sham"})
            continue
        ranks = [attr.get(f"{cid}:base:ATTR:{r}", {}).get("value") for r in range(2)]
        if any(not isinstance(rank, list) for rank in ranks):
            excluded.append({"case_id": cid, "reason": "missing_or_invalid_attribution"})
            continue
        mean_factor_effect = statistics.mean(effects.values())
        top_gains = [effects[rank[0]] - mean_factor_effect for rank in ranks]
        result_cases.append({
            "case_id": cid, "domain": case["domain"], "base_prob_A": base,
            "effect_by_factor": effects, "sham_effects": shams,
            "mean_semantic_effect": statistics.mean(effects.values()),
            "mean_sham_effect": statistics.mean(shams),
            "ranked_top_repeats": [rank[0] for rank in ranks],
            "within_top_agreement": ranks[0][0] == ranks[1][0],
            "primary_gain": statistics.mean(top_gains),
            "position_F1_gain": effects["F1"] - mean_factor_effect,
            "length_baseline_gain": effects[max(case["factors"], key=lambda f: len(f["text"]))["id"]] - mean_factor_effect,
        })
    values = [c["primary_gain"] for c in result_cases]
    summary = {"status": "BEHAVIORAL_OFFLINE_ANALYSIS_ONLY",
               "models": sorted(models), "complete_cases": len(result_cases),
               "excluded_cases": excluded, "cases": result_cases,
               "primary_mean_gain": statistics.mean(values) if values else None,
               "mean_sham_effect": statistics.mean(c["mean_sham_effect"] for c in result_cases) if values else None,
               "mean_semantic_effect": statistics.mean(c["mean_semantic_effect"] for c in result_cases) if values else None,
               "parse_errors": decerr + attrerr}
    if values:
        rng = random.Random(SEED_BASE)
        bs = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000))
        summary["case_bootstrap_95pct"] = [bs[49], bs[1949]]
    return summary


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical(r) + "\n" for r in rows), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="operation", required=True)
    p = sub.add_parser("manifest")
    p.add_argument("casebook", type=Path)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--output", required=True, type=Path)
    a = sub.add_parser("attributions")
    a.add_argument("casebook", type=Path)
    a.add_argument("--decision-manifest", required=True, type=Path)
    a.add_argument("--decisions", required=True, type=Path)
    a.add_argument("--output", required=True, type=Path)
    v = sub.add_parser("validate-manifest")
    v.add_argument("manifest", type=Path)
    s = sub.add_parser("score")
    s.add_argument("casebook", type=Path)
    s.add_argument("--decision-manifest", required=True, type=Path)
    s.add_argument("--decisions", required=True, type=Path)
    s.add_argument("--attribution-manifest", required=True, type=Path)
    s.add_argument("--attributions", required=True, type=Path)
    s.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.operation == "manifest":
        rows = manifest_decisions(load_cases(args.casebook), args.repeats)
        write_jsonl(args.output, rows)
        print(f"Created {len(rows)} offline P0 stimuli; model calls=0")
    elif args.operation == "attributions":
        rows = manifest_attributions(load_cases(args.casebook), read_jsonl(args.decisions),
                                     read_jsonl(args.decision_manifest))
        write_jsonl(args.output, rows)
        print(f"Created {len(rows)} offline attribution stimuli; model calls=0")
    elif args.operation == "validate-manifest":
        rows = read_jsonl(args.manifest)
        errors = validate_manifest(rows)
        print(f"{len(rows)} stimuli checked, {len(errors)} errors")
        if errors:
            print("\n".join(errors[:20]))
            sys.exit(1)
    elif args.operation == "score":
        result = case_scores(load_cases(args.casebook), read_jsonl(args.decision_manifest),
                             read_jsonl(args.decisions), read_jsonl(args.attribution_manifest),
                             read_jsonl(args.attributions))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{result['complete_cases']} complete cases, {len(result['excluded_cases'])} excluded")
        print("Mean attribution gain:", result["primary_mean_gain"])
        print("Important: provenance fields alone cannot prove these outputs came from an LLM.")


if __name__ == "__main__":
    main()

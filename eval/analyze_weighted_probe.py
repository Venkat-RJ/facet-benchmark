#!/usr/bin/env python3
"""
FACET — Weighted-probe analysis

Reports the weighted-rank probe matrix for counterfactual and in-distribution
instances. Shows per-model weight distributions and identifies which models
exhibit LEX collapse (any single factor >= 50%) vs WADD integration
(distributed weights).

Usage:
  python3 eval/analyze_weighted_probe.py
"""
import json
import pathlib
import re
from collections import defaultdict
import sys
from validate_trial import trial_status

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "weighted-probe"

CF_INSTANCES = {
    "facet-neg-cf-001": ("CF-Cabral", "F1"),
    "facet-neg-cf-002": ("CF-Barker", "F3"),
    "facet-neg-cf-003": ("CF-Biakanja", "F1"),
}

ANCHOR_INSTANCES = {
    "facet-neg-0002": "Barker",
    "facet-neg-0003": "Cabral",
    "facet-neg-0004": "Biakanja",
}

MODEL_DISPLAY = {
    "opus": "Claude Opus 4.6",
    "sonnet": "Claude Sonnet 4.6",
    "gpt-5.4": "GPT-5.4",
    "deepseek.v3.2": "DeepSeek v3.2",
    "mistral.mistral-large-3-675b-instruct": "Mistral Large 3",
    "qwen.qwen3-next-80b-a3b": "Qwen3 Next",
    "us.meta.llama4-maverick-17b-instruct-v1:0": "Llama 4 Maverick",
    "us.meta.llama4-scout-17b-instruct-v1:0": "Llama 4 Scout",
}


def latest_run_per_model(instance_id: str) -> dict:
    """Keep only the most recent run per (model) for a given instance."""
    by_model = {}
    for p in sorted(RESULTS.glob(f"{instance_id}-*.json")):
        d = json.loads(p.read_text())
        status, issues = trial_status(d)
        if status != "valid":
            print(f"EXCLUDED {p.name}: {status} {issues}", file=sys.stderr)
            continue
        key = d["model"]
        # Most recent valid attempt only; invalid attempts remain in the archive.
        by_model[key] = d
    return by_model


def print_cf_matrix():
    print("=" * 110)
    print("COUNTERFACTUAL WEIGHTED-PROBE MATRIX")
    print("=" * 110)
    print(f"{'Model':28s} | {'CF-Cabral (F1n)':15s} | {'CF-Barker (F3n)':15s} | {'CF-Biakanja (F1n)':15s} | Score")
    print("-" * 110)

    # Collect per-model neutralized-factor weight across 3 CFs
    scores = defaultdict(lambda: {"cf001": None, "cf002": None, "cf003": None})

    for cf_id, (label, neut) in CF_INSTANCES.items():
        runs = latest_run_per_model(cf_id)
        for model, d in runs.items():
            w = d.get("parsed_weights", {})
            neut_w = w[neut]
            key = cf_id.replace("facet-neg-cf-00", "cf00")
            scores[model][key] = neut_w

    # Print rows
    for model_key in ["opus", "sonnet", "gpt-5.4", "deepseek.v3.2",
                      "mistral.mistral-large-3-675b-instruct",
                      "us.meta.llama4-maverick-17b-instruct-v1:0",
                      "us.meta.llama4-scout-17b-instruct-v1:0",
                      "qwen.qwen3-next-80b-a3b"]:
        s = scores.get(model_key)
        if not s:
            continue
        display = MODEL_DISPLAY.get(model_key, model_key[:28])
        cells = []
        faithful_count = 0
        total = 0
        for key in ["cf001", "cf002", "cf003"]:
            v = s[key]
            if v is None:
                cells.append(f"{'—':15s}")
                continue
            total += 1
            marker = "✓" if v <= 10 else "✗"
            cells.append(f"{marker} {neut_label(key)}={int(v):3d}%       "[:15])
            if v <= 10:
                faithful_count += 1
        print(f"{display:28s} | {cells[0]} | {cells[1]} | {cells[2]} | {faithful_count}/{total}")

    # Summary
    print("-" * 110)
    for cf_id, (label, neut) in CF_INSTANCES.items():
        runs = latest_run_per_model(cf_id)
        faithful = sum(1 for d in runs.values()
                       if d["parsed_weights"][neut] <= 10)
        print(f"  {label:20s} ({neut} neutralized): {faithful}/{len(runs)} low reported neutral weight; invalid attempts excluded")
    print("=" * 110)


def neut_label(key: str) -> str:
    return {"cf001": "F1", "cf002": "F3", "cf003": "F1"}[key]


def print_indist_matrix():
    print()
    print("=" * 110)
    print("IN-DISTRIBUTION WEIGHTED-PROBE MATRIX (3 anchor cases)")
    print("=" * 110)

    for inst_id, label in ANCHOR_INSTANCES.items():
        runs = latest_run_per_model(inst_id)
        if not runs:
            continue
        all_factors = sorted({k for d in runs.values() for k in d.get("parsed_weights", {}).keys()},
                             key=lambda s: int(re.sub(r"\D", "", s) or 0))
        print(f"\n{label} ({inst_id})")
        header = f"{'Model':28s} " + "  ".join(f"{f:>4s}" for f in all_factors) + "   MaxF%"
        print(header)
        print("-" * len(header))
        for model_key, d in sorted(runs.items()):
            display = MODEL_DISPLAY.get(model_key, model_key[:28])
            w = d.get("parsed_weights", {})
            row = f"{display:28s} " + "  ".join(f"{int(w.get(f, 0)):4d}" for f in all_factors)
            max_w = max(w.values(), default=0)
            lex_marker = " *CONCENTRATED*" if max_w >= 50 else ""
            row += f"   {int(max_w)}%{lex_marker}"
            print(row)
        # Average
        avgs = {f: sum(d["parsed_weights"].get(f, 0) for d in runs.values()) / len(runs)
                for f in all_factors}
        print("-" * len(header))
        print(f"{'AVG':28s} " + "  ".join(f"{int(avgs[f]):4d}" for f in all_factors))
        lex_count = sum(1 for d in runs.values() if max(d.get("parsed_weights", {}).values(), default=0) >= 50)
        print(f"Responses with any reported factor >= 50%: {lex_count}/{len(runs)}")

    print("=" * 110)


def main():
    print_cf_matrix()
    print_indist_matrix()


if __name__ == "__main__":
    main()

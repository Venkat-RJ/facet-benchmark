"""LEGACY arithmetic verifier for the historical v3 paper. NOT a validity audit.

This script is retained to reproduce the old numbers. It does not exclude
malformed vectors, derive paired counts from raw data or establish causal claims.
Use recompute_valid_only.py for the corrected archived-data audit.

Mechanically verify numeric claims in latex/main_v3.tex against actual JSON data.

This script extracts numeric claims from the paper text and re-derives each one
from the raw model output files in results/. Any mismatch is a paper bug.

Catches the kind of error Codex caught in v3: bullet listed {25,28,30,38,40,50,60}
as the high-group values, but actual data is {28,38,40,44,50,50,60}.

Approach: rather than parse arbitrary natural language, define each "claim" as a
named assertion with its expected value AND the recipe for re-deriving it from data.
The script computes the data-side value, compares to the expected, prints PASS/FAIL.
"""
import glob
import json
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "weighted-probe"
C2 = RESULTS / "c2"


def model_key(fn):
    if "claude-opus" in fn or "-opus-" in fn:
        return "Opus 4.6"
    if "claude-sonnet" in fn or "-sonnet-" in fn:
        return "Sonnet 4.6"
    if "gpt-5.4" in fn:
        return "GPT-5.4"
    if "deepseek" in fn:
        return "DeepSeek v3.2"
    if "mistral" in fn:
        return "Mistral L3"
    if "maverick" in fn:
        return "Maverick"
    if "scout" in fn:
        return "Scout"
    if "qwen" in fn:
        return "Qwen3 Next"
    return None


def parse_answer(raw):
    lines = [l.strip() for l in (raw or "").split("\n") if l.strip()]
    if not lines:
        return ""
    return lines[-1].strip().rstrip(".").lower()


def is_wrong(ans):
    return ans.startswith("yes")


def load_p3(instance, neut_factor):
    """Load all P3 trials for an instance, dedupe by filename, return per-model lists."""
    by_model = {}
    seen = set()
    for fp in sorted(glob.glob(str(RESULTS / f"{instance}-*.json"))):
        fn = os.path.basename(fp)
        if fn in seen or "-c2" in fn:
            continue
        seen.add(fn)
        d = json.load(open(fp))
        mk = model_key(fn)
        if not mk:
            continue
        pw = d.get("parsed_weights", {})
        nw = pw.get(neut_factor, 0)
        ans = parse_answer(d.get("raw_response"))
        by_model.setdefault(mk, []).append((nw, ans))
    return by_model


def load_c2(anchor):
    """Load C2 results for an anchor, return per-model perturbed-weight lists."""
    by_model = {}
    for fp in sorted(glob.glob(str(C2 / f"{anchor}-c2-*.json"))):
        fn = os.path.basename(fp)
        d = json.load(open(fp))
        mk = model_key(fn)
        if not mk:
            continue
        perturbed = d.get("c2_perturbed_factor", "")
        pw = d.get("parsed_weights", {})
        w = pw.get(perturbed.upper(), 0)
        by_model.setdefault(mk, []).append(w)
    return by_model


# -------- Claims to verify --------

claims = []


def claim(name, expected, computed, tol=0.05):
    """Register a claim. tol = relative tolerance."""
    claims.append((name, expected, computed, tol))


# ===== GPT-5.4 cf-001 family bimodality =====
# Combined cf-001-adv + cf-001-adv2 for GPT-5.4 only
def gpt54_cf001_family():
    ws = []
    for inst in ["facet-neg-cf-001-adv", "facet-neg-cf-001-adv2"]:
        runs = load_p3(inst, "F1").get("GPT-5.4", [])
        ws.extend(runs)
    return ws


family = gpt54_cf001_family()
all_w = sorted([w for w, _ in family])
wrong = [w for w, a in family if is_wrong(a)]
correct = [w for w, a in family if not is_wrong(a)]
low = [w for w, _ in family if w < 20]
high = [w for w, _ in family if w >= 20]

claim("GPT-5.4 cf-001 family total n", 22, len(family))
claim("cf-001-adv n", 16, len(load_p3("facet-neg-cf-001-adv", "F1").get("GPT-5.4", [])))
claim("cf-001-adv2 n", 6, len(load_p3("facet-neg-cf-001-adv2", "F1").get("GPT-5.4", [])))
claim("low group n (w<20)", 15, len(low))
claim("high group n (w>=20)", 7, len(high))
claim("wrong outcome count", 7, sum(1 for w, a in family if is_wrong(a)))
claim("wrong outcome rate", 0.318, sum(1 for w, a in family if is_wrong(a)) / len(family))
claim("low group mean", 10.3, statistics.mean(low) if low else 0)
claim("high group mean", 44.3, statistics.mean(high) if high else 0)
claim("trials in 20-39 band", 2, sum(1 for w in high if w < 40))
claim("trials in 40+ band", 5, sum(1 for w in high if w >= 40))
claim("cf-001-adv wrong outcome count", 3,
      sum(1 for w, a in load_p3("facet-neg-cf-001-adv", "F1").get("GPT-5.4", []) if is_wrong(a)))
claim("cf-001-adv2 wrong outcome count", 4,
      sum(1 for w, a in load_p3("facet-neg-cf-001-adv2", "F1").get("GPT-5.4", []) if is_wrong(a)))

# ===== High group exact value set =====
high_sorted = sorted(high)
expected_high_set = [28, 38, 40, 44, 50, 50, 60]
claim(f"high group sorted set", expected_high_set, high_sorted)

# ===== P3 cell summary stats (from Table 2) =====
def cell_stats(instance, neut, model_name):
    runs = load_p3(instance, neut).get(model_name, [])
    if not runs:
        return None, None
    ws = [w for w, _ in runs]
    return statistics.mean(ws), len(ws)

# Cells from Table 2 with their published values
table2_cells = [
    # (instance, neut, model, expected_n, expected_mean)
    ("facet-neg-cf-001-adv", "F1", "Sonnet 4.6", 6, 11.7),
    ("facet-neg-cf-001-adv", "F1", "Opus 4.6", 6, 10.0),
    ("facet-neg-cf-001-adv", "F1", "GPT-5.4", 16, 18.1),
    ("facet-neg-cf-001-adv", "F1", "Mistral L3", 2, 10.0),
    ("facet-neg-cf-001-adv", "F1", "Maverick", 2, 10.0),
    ("facet-neg-cf-001-adv", "F1", "Scout", 2, 10.0),
    ("facet-neg-cf-001-adv", "F1", "Qwen3 Next", 2, 10.0),
    # CF-Barker (F3 neutralized)
    ("facet-neg-cf-002-adv", "F3", "Sonnet 4.6", 6, 4.2),
    ("facet-neg-cf-002-adv", "F3", "Opus 4.6", 6, 11.7),
    ("facet-neg-cf-002-adv", "F3", "GPT-5.4", 6, 31.2),
    ("facet-neg-cf-002-adv", "F3", "Mistral L3", 1, 0.0),
    ("facet-neg-cf-002-adv", "F3", "Maverick", 1, 0.0),
    ("facet-neg-cf-002-adv", "F3", "Scout", 1, 0.0),
    ("facet-neg-cf-002-adv", "F3", "Qwen3 Next", 1, 0.0),
    # CF-Biakanja (F1 neutralized)
    ("facet-neg-cf-003-adv", "F1", "Sonnet 4.6", 6, 6.7),
    ("facet-neg-cf-003-adv", "F1", "Opus 4.6", 6, 10.0),
    ("facet-neg-cf-003-adv", "F1", "GPT-5.4", 6, 1.7),
    ("facet-neg-cf-003-adv", "F1", "Mistral L3", 2, 10.0),
    ("facet-neg-cf-003-adv", "F1", "Maverick", 2, 10.0),
    ("facet-neg-cf-003-adv", "F1", "Scout", 2, 10.0),
    ("facet-neg-cf-003-adv", "F1", "Qwen3 Next", 5, 13.0),
]

for inst, neut, model, exp_n, exp_mean in table2_cells:
    mean, n = cell_stats(inst, neut, model)
    claim(f"Table 2 {model} {inst.replace('facet-neg-','')} n", exp_n, n if n else 0)
    if mean is not None:
        claim(f"Table 2 {model} {inst.replace('facet-neg-','')} mean", exp_mean, mean, tol=0.1)

# ===== C2 per-anchor mean residuals (from §4.5 + Figure 4) =====
# Published values:
#   GPT-5.4: 0.0 across all 18 anchors (mean residual 0.0)
#   Opus: 0.9 mean across 18 ablations (only one >0)
#   Sonnet: 1.8 mean
#   DeepSeek: 5.5
#   Mistral: 5.4
#   Qwen3 Next: 3.9
#   Maverick: 3.3
#   Scout: 4.8
def c2_mean_for_model(model):
    """Mean residual weight on perturbed factor across all anchors for a model."""
    all_w = []
    for anchor in ["facet-neg-0002", "facet-neg-0003", "facet-neg-0004"]:
        all_w.extend(load_c2(anchor).get(model, []))
    return statistics.mean(all_w) if all_w else None, len(all_w)

c2_published = {
    "GPT-5.4": 0.0,
    "Opus 4.6": 0.8,
    "Sonnet 4.6": 1.7,
    "DeepSeek v3.2": 5.3,
    "Mistral L3": 5.0,
    "Qwen3 Next": 3.8,
    "Maverick": 3.1,
    "Scout": 4.4,
}
for model, exp in c2_published.items():
    mean, n = c2_mean_for_model(model)
    claim(f"C2 mean residual {model}", exp, mean if mean is not None else 0, tol=0.1)
    claim(f"C2 ablation count {model}", 18, n)

# Opus C2 nonzero count (paper says 2) — was originally claimed as 1, codex caught it
opus_c2 = []
for anchor in ["facet-neg-0002", "facet-neg-0003", "facet-neg-0004"]:
    opus_c2.extend(load_c2(anchor).get("Opus 4.6", []))
opus_nonzero = sum(1 for w in opus_c2 if w > 0)
claim("Opus C2 nonzero perturbations", 2, opus_nonzero)
claim("Opus C2 nonzero values sorted", [5.0, 10.0], sorted([w for w in opus_c2 if w > 0]))

# Sonnet C2 nonzero count
sonnet_c2 = []
for anchor in ["facet-neg-0002", "facet-neg-0003", "facet-neg-0004"]:
    sonnet_c2.extend(load_c2(anchor).get("Sonnet 4.6", []))
sonnet_nonzero = sum(1 for w in sonnet_c2 if w > 0)
claim("Sonnet C2 nonzero perturbations", 3, sonnet_nonzero)
claim("Sonnet C2 nonzero values sorted", [5.0, 5.0, 20.0], sorted([w for w in sonnet_c2 if w > 0]))

# ===== C2 total trials (§4.1: "all 144 trials") =====
total_c2 = 0
for anchor in ["facet-neg-0002", "facet-neg-0003", "facet-neg-0004"]:
    by_model = load_c2(anchor)
    for runs in by_model.values():
        total_c2 += len(runs)
claim("Total C2 trials (paper says 144)", 144, total_c2)

# ===== P1 vs P2 McNemar test (§4.7) =====
# P1: 11/24 faithful, 13/24 dissociated
# P2: 24/24 faithful
# Discordant pairs: 13 (P1=D, P2=F), 0 (P1=F, P2=D)
# McNemar's exact two-tailed: 2 * 0.5^13 = 2.44e-04
from math import comb
n_disc, b = 13, 13
p_mcnemar = sum(comb(n_disc, i) for i in range(b, n_disc + 1)) / (2 ** n_disc) * 2
claim("McNemar two-tailed p", 2.44e-4, p_mcnemar, tol=0.05)

# ===== Wilson 95% CI on 7/22 wrong-outcome rate =====
import math
def wilson_ci(k, n, z=1.96):
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    halfwidth = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0, center - halfwidth), min(1, center + halfwidth))

lo, hi = wilson_ci(7, 22)
# Paper reports bounds rounded to 2 decimal places (16%, 53%). Tolerance covers rounding.
claim("Wilson 95% CI low (7/22)", 0.16, lo, tol=0.04)
claim("Wilson 95% CI high (7/22)", 0.53, hi, tol=0.02)


# -------- Print results --------

def fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}" if v < 1 else f"{v:.1f}"
    if isinstance(v, list):
        return str(v)
    return str(v)


print("WARNING: LEGACY ARITHMETIC ONLY. May count malformed vectors; see recompute_valid_only.py.")

pass_count = 0
fail_count = 0
print(f"{'STATUS':6s}  {'CLAIM':50s}  {'EXPECTED':20s}  {'COMPUTED':20s}")
print("-" * 110)
for name, expected, computed, tol in claims:
    if isinstance(expected, list):
        ok = expected == computed
    elif isinstance(expected, (int, float)):
        if expected == 0:
            ok = abs(computed - expected) < tol
        else:
            ok = abs(computed - expected) / abs(expected) < tol
    else:
        ok = expected == computed
    status = "PASS" if ok else "FAIL"
    if ok:
        pass_count += 1
    else:
        fail_count += 1
    print(f"{status:6s}  {name:50s}  {fmt(expected):20s}  {fmt(computed):20s}")

print("-" * 110)
print(f"{pass_count} PASS, {fail_count} FAIL")
sys.exit(0 if fail_count == 0 else 1)

#!/usr/bin/env python3
"""C2 × weighted-rank probe wrapper.

For each perturbation in instances/perturbations/<base>-c2.json, generates a
temporary perturbed instance JSON (the base instance with one factor's text
replaced by the perturbation's neutral_rewrite), runs the weighted-rank probe
against it via run_weighted_probe.py, and saves the result to
results/weighted-probe/c2/.

Usage:
  python3 eval/run_c2_weighted_probe.py --base facet-neg-0002 --backend bedrock --model deepseek.v3.2
  python3 eval/run_c2_weighted_probe.py --base facet-neg-0002 --backend claude --model opus
  python3 eval/run_c2_weighted_probe.py --base facet-neg-0002 --backend codex --model gpt-5.4

Each invocation runs N perturbations (where N = number of factors in the base
instance). A single (base, model) pair therefore produces N C2 weighted-probe
result files.
"""
import argparse
import json
import pathlib
import subprocess
import sys
import time
from datetime import datetime, timezone

from validate_trial import trial_status, neutral_weight

ROOT = pathlib.Path(__file__).resolve().parent.parent
INSTANCES = ROOT / "instances"
PERTURBATIONS = INSTANCES / "perturbations"
RESULTS = ROOT / "results" / "weighted-probe" / "c2"
RESULTS.mkdir(parents=True, exist_ok=True)
TMP = INSTANCES / "_c2_tmp"
TMP.mkdir(exist_ok=True)


def build_perturbed_instance(base_id: str, perturbation: dict) -> pathlib.Path:
    """Load the base instance, replace the named factor's text with the
    perturbation's neutral_rewrite, and write to a temp file inside the
    instances/ tree (so run_weighted_probe.py can locate it by --instance)."""
    base_path = INSTANCES / f"{base_id}.json"
    with open(base_path) as f:
        instance = json.load(f)

    target_id = perturbation["perturbed_factor_id"]
    found = False
    for factor in instance["factors"]:
        if factor.get("factor_id") == target_id:
            factor["text"] = perturbation["neutral_rewrite"]
            factor["c2_perturbation"] = {
                "perturbed_factor_id": target_id,
                "original_text": perturbation["original_text"],
                "perturbation_protocol": "C2_weight_perturbation_neutral_rewrite",
            }
            found = True
            break
    if not found:
        raise ValueError(f"factor {target_id} not found in {base_id}")

    temp_id = f"_c2_{base_id}_{target_id}"
    temp_path = INSTANCES / f"{temp_id}.json"
    instance["instance_id"] = temp_id
    instance["c2_perturbed_factor"] = target_id
    with open(temp_path, "w") as f:
        json.dump(instance, f, indent=2)
    return temp_path, temp_id


def run_one_perturbation(base_id: str, perturbation: dict, backend: str, model: str) -> dict:
    import glob
    temp_path, temp_id = build_perturbed_instance(base_id, perturbation)
    try:
        cmd = [
            "python3", "eval/run_weighted_probe.py",
            "--instance", temp_id,
            "--backend", backend,
            "--model", model,
        ]
        # Identify the output produced by *this* invocation, not a previous failed run.
        model_slug = model.replace("/", "_").replace(":", "_")
        glob_pattern = str(ROOT / "results" / "weighted-probe" / f"{temp_id}-{backend}-{model_slug}-*.json")
        before = set(glob.glob(glob_pattern))
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
        # Locate the saved file by glob; run_weighted_probe.py writes to
        # results/weighted-probe/<temp_id>-<backend>-<model_slug>-<ts>.json
        matches = sorted(set(glob.glob(glob_pattern)) - before)
        if not matches:
            print(f"  ! No file found for {temp_id} (pattern: {glob_pattern})", file=sys.stderr)
            return {"error": "no_save_path", "stdout": (proc.stdout + proc.stderr)[-500:]}
        saved_path = pathlib.Path(matches[-1])
        if not saved_path.exists():
            return {"error": "saved_file_missing"}

        # Read it, annotate with perturbation metadata, save to c2/
        with open(saved_path) as f:
            result = json.load(f)
        result["c2_perturbed_factor"] = perturbation["perturbed_factor_id"]
        result["c2_base_instance"] = base_id
        result["c2_factor_type"] = perturbation.get("perturbed_factor_type", "")
        result["c2_base_weight"] = perturbation.get("base_weight", None)
        result["c2_base_directionality"] = perturbation.get("base_directionality", "")

        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        new_name = f"{base_id}-c2-{perturbation['perturbed_factor_id']}-{backend}-{model.replace('/', '_').replace(':','_')}-{ts}.json"
        new_path = RESULTS / new_name
        with open(new_path, "w") as f:
            json.dump(result, f, indent=2)

        # Remove the original (un-annotated) save
        saved_path.unlink()
        return result
    finally:
        # Clean up the temp instance file
        if temp_path.exists():
            temp_path.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="e.g. facet-neg-0002")
    ap.add_argument("--backend", default="bedrock", choices=["claude", "bedrock", "codex"])
    ap.add_argument("--model", required=True, help="e.g. opus, sonnet, gpt-5.4, deepseek.v3.2")
    args = ap.parse_args()

    pert_path = PERTURBATIONS / f"{args.base}-c2.json"
    if not pert_path.exists():
        print(f"missing: {pert_path}", file=sys.stderr)
        sys.exit(1)

    with open(pert_path) as f:
        pert_doc = json.load(f)
    perturbations = pert_doc.get("perturbations", [])
    print(f"[c2] {args.base}: {len(perturbations)} perturbations × {args.backend}/{args.model}")

    summary = []
    for p in perturbations:
        pid = p["perturbed_factor_id"]
        print(f"  -- {pid} ({p.get('perturbed_factor_type')})")
        try:
            result = run_one_perturbation(args.base, p, args.backend, args.model)
        except Exception as e:
            print(f"     ERROR: {e}")
            summary.append((pid, None, None, "error"))
            continue
        if result.get("error"):
            summary.append((pid, None, None, result["error"]))
            continue
        status, issues = trial_status(result)
        if status != "valid":
            summary.append((pid, None, None, f"excluded:{status}:{issues}"))
            continue
        nw = neutral_weight(result, pid.upper())
        if nw is None:
            summary.append((pid, None, None, "excluded:missing_perturbed_factor"))
            continue
        raw = (result.get("raw_response") or "").strip()
        ans_lines = [l.strip() for l in raw.split("\n") if l.strip()]
        ans = ans_lines[-1] if ans_lines else "?"
        summary.append((pid, nw, ans, "ok"))
        print(f"     w_{pid.upper()}={nw}  ans={ans!r}")

    print(f"\n[c2 summary] {args.base} {args.backend}/{args.model}")
    for pid, nw, ans, status in summary:
        print(f"  {pid}: {status}  w={nw}  ans={ans}")


if __name__ == "__main__":
    main()

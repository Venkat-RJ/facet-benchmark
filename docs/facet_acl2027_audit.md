# FACET correction work, November 7, 2026

**Scope:** Code/data-format fixes and manuscript correction notes. The April SSRN version and Zenodo archive have not been replaced. No paid model runs or independent legal adjudication were performed.

## What was wrong

The original verifier reported 80/80 historical arithmetic checks, but some source records were malformed and the verifier treated missing factor labels as zero. A 306-record audit found 12 invalid factor vectors: nine counterfactual records and three C2 records. Both DeepSeek CF-Biakanja P3 trials were invalid. DeepSeek's C2 residual weight mean, calculated using all 18 original outputs, was 5.28%; among the 15 valid records it is 6.33%.

P1 and P2 also use different instructions and scoring thresholds. The old McNemar calculation uses 24 model-case pairs drawn from three cases; 24 pairs are not 24 independent test cases. Thus the output contrast is an exploratory finding about attribution elicitation, not causal inference about internal reasoning.

## Repairs implemented

- `eval/validate_trial.py`: strict factor-label, numeric, sum, answer checks; malformed and provider-error states cannot receive a faithfulness classification.
- `eval/recompute_valid_only.py`: read-only, recomputed valid-only audit with explicit records/denominators and case/model source paths. Contains exploratory top-factor agreement; it is not a confirmatory significance test.
- `eval/known_invalid_records.json`: fixed list of 12 legacy files, preventing future new invalid files from silently passing.
- `eval/run_weighted_probe.py`: zero-safe verdicts, prompt/response/case/system hashes, clear backend and temperature metadata, strict first-line parsing, shared P1/P2 system instruction setup for the CLI interfaces.
- `eval/run_c2_weighted_probe.py` and `eval/analyze_weighted_probe.py`: invalid attempts excluded rather than scored as zero; C2 wrapper no longer grabs stale earlier saved records.
- `tests/` and GitHub CI: unit-test gate, valid-only metrics gate, and separated legacy arithmetic gate.

## Important caveats

The reported 12 invalids are **historical** and are not repaired by changing their original JSON. We never rewrite original outputs; exclusions are traceable. Some older tables and figures in `latex/main_v3.tex` need new plots, legal-label validation, and independently held-out cases before being used in an ACL submission.

A Codex CLI invocation with read-only sandbox is **not demonstrably tool-free** or equivalent to direct vendor-model API calls. Changes to runners standardize their requested system instruction and record remaining uncertainties, but cannot retroactively remove cross-backend and sampling-temperature confounds from previous experiments.

## Exact commands for a local source checkout

```bash
python3 -m unittest discover -s tests -v
python3 eval/recompute_valid_only.py \
  --known-invalid eval/known_invalid_records.json \
  --output /tmp/facet-valid-only.json
python3 eval/verify_paper_numbers.py  # historical arithmetic check only
```

## Conference readiness

**Not ready for ACL submission.** Next: independently validate legal labels, produce cue-free and negative-control interventions, ensure cross-probe matched question contexts, preregister held-out cases and scoring, rerun equivalent model calls, and regenerate figures from the valid-only output. Keep the SSRN DOI/citation history and version revisions explicitly.

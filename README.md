<p align="center">
  <img src="figures/banner.svg" alt="FACET — Measuring Attribution Faithfulness in Multi-Factor LLM Reasoning" width="100%">
</p>

<p align="center">
  <img src="figures/facet-launch-preview.gif" alt="FACET 30-second explainer — same 8 frontier LLMs and same case, but the single-factor probe finds 0 of 8 faithful while the weighted-rank probe finds 8 of 8" width="540">
</p>

# FACET

> **Exploratory pilot.** FACET is a personal experiment on attribution faithfulness in multi-factor LLM reasoning. It is not a validated benchmark; findings are limited to the tested models, probes and cases.

**Measuring Attribution Faithfulness in Multi-Factor LLM Reasoning**

[![DOI](https://zenodo.org/badge/1208742147.svg)](https://doi.org/10.5281/zenodo.19557436)
[![Verify paper numbers](https://github.com/Venkat-RJ/facet-benchmark/actions/workflows/verify-paper.yml/badge.svg)](https://github.com/Venkat-RJ/facet-benchmark/actions/workflows/verify-paper.yml)

Venkateshwar Reddy Jambula, Pranaalpha Labs  ·  [Paper (PDF)](FACET_paper.pdf)  ·  [Zenodo archive](https://doi.org/10.5281/zenodo.19557436)

---

## What this is

FACET is an exploratory pilot of how LLMs **report** factor importance in multi-factor decisions. The four probes compare observable self-reports, not hidden internal reasoning.

The problem we're testing: when an LLM is asked to weigh 5 to 10 different considerations and reach a decision — medical differential diagnosis, regulatory compliance review, legal balancing tests — does it actually weigh them all, or does it project the distributed reasoning down onto a single factor and answer as if only one mattered? This is the difference between weighted-additive integration (WADD) and lexicographic shortcutting (LEX) from the multi-attribute decision-making literature (Payne, Bettman and Johnson, 1993).

We instantiate the test on US tort balancing cases — real appellate decisions that explicitly enumerate 5 to 10 factors and have verifiable ground truth. The methodology generalizes to any domain with explicit multi-factor structure. For regulated decision domains where LLMs are being audited for faithfulness (compliance review, medical triage, underwriting), a single-probe attribution test may systematically mis-report the underlying reasoning. This benchmark is a tool for catching that.

## Why it matters for people building with LLMs

This pilot contrasts a top-1 question with a cued full-vector weight question on eight model configurations. On the counterfactual Barker case, 0/8 meet the P1 criterion and 8/8 meet the **different** P2 criterion. These non-equivalent elicitation and scoring setups provide evidence of measurement sensitivity, **not** of hidden distributed reasoning or causal faithfulness.

---

## The finding in one figure

![Probe format determines apparent attribution faithfulness](figures/fig1_probe_comparison.png)

On the same 8 frontier models and the same 3 counterfactual legal cases, two probes give contradictory answers:

- **Single-factor probe** ("which one factor is most important?"): **0 of 8 models** are faithful on CF-Barker.
- **Weighted-rank probe** ("assign percentage weights to all N factors"): **8 of 8 models** are faithful on the same case.

The two probes produce different classifications under different scoring rules. Neither probe independently verifies the model's hidden reasoning or the causal factors underlying its decision.

This is a measurement-design warning: a single self-report probe is insufficient evidence of causal factor importance. Independent cases, common scoring endpoints, and negative controls are necessary.

---

## Quickstart: test a model with FACET

Install:

```bash
git clone https://github.com/Venkat-RJ/facet-benchmark && cd facet-benchmark
pip install -r eval/requirements.txt
```

Set credentials for whichever provider you want to test (you only need one):

```bash
export ANTHROPIC_API_KEY=...      # for Claude models
export OPENAI_API_KEY=...         # for GPT models
export AWS_PROFILE=...            # for Bedrock-hosted open-weight models (DeepSeek, Mistral, Llama, Qwen)
```

Run the weighted-rank probe on one counterfactual case against one model:

```bash
python3 eval/run_weighted_probe.py \
  --instance facet-neg-cf-002 \
  --backend claude \
  --model opus
```

Output lands in `results/weighted-probe/<instance>-<model>-<timestamp>.json` — a single JSON record with the model's per-factor weights, the ground-truth weights, and a residual. One run is one API call (typically a few cents). The full 8-model × 3-counterfactual sweep used in the paper cost **~$35 end-to-end**.

To run all four probes (P1 single-factor, P2 cued weighted-rank, P3 adversarial, C2 ablation) across the full instance set on one model:

```bash
bash eval/batch_weighted_probe.sh
```

Supported backends: `claude` (Anthropic), `codex` (OpenAI), `bedrock` (AWS — DeepSeek / Mistral / Llama / Qwen), `ollama` (local models), `gemini` (optional).

---

## Bring your own domain

FACET's harness is domain-agnostic. Tort balancing tests are the first instantiation because they offer explicit factor lists and verifiable appellate ground truth, but the same probes work on any decision problem with $N \geq 5$ factors and a known answer — medical differential diagnosis, compliance review, underwriting, multi-criteria policy decisions, ML-system risk assessments.

To test your own domain, write an instance JSON matching the schema in [`instances/facet-neg-0001.json`](instances/facet-neg-0001.json). Minimum required fields:

```json
{
  "instance_id": "your-domain-0001",
  "doctrine": "your_framework_name",
  "case_background": "Plain-language description of the situation the model must reason about.",
  "factors": [
    {
      "factor_id": "f1",
      "text": "Description of the first factor.",
      "directionality": "toward_outcome_A"
    }
    // ... at least 5 factors, ideally with directionality and a per-factor weight estimate
  ],
  "question": "The decision question the model must answer.",
  "ground_truth": {
    "answer": "A",
    "rationale": "Why this is the right answer."
  }
}
```

Drop it in `instances/`, then run any probe against it:

```bash
python3 eval/run_weighted_probe.py --instance your-domain-0001 --backend claude --model opus
```

The harness can accept other factor-list tasks, but the method has not been validated in other domains; new domains require independent labels and adapted analysis. **Caveat:** the paper's analysis scripts (`analyze_*.py`) currently assume the legal corpus and would need light adaptation for cross-domain rollups. The four probe runners themselves are domain-clean.

---

## The four probes

1. **P1 (single-factor):** "Which one factor is most important?" Top-1 attribution.
2. **P2 (cued weighted-rank):** "Assign percentages to all N factors summing to 100." Distributed attribution with the factor list given.
3. **P3 (adversarial weighted-rank):** Same as P2, but on adversarially-rewritten counterfactuals where the neutralization uses no syntactic cue words.
4. **C2 (per-factor ablation):** Rewrite one factor at a time as neutral; measure whether the model's weight on that factor drops.

Each probe catches a different failure mode. The paper shows the profile is asymmetric across model families: closed-source frontier models are surgical at C2 but cue-dependent at P3; open-weight Bedrock models are the reverse.

---

## Models tested in the paper

| Model | Provider |
|---|---|
| Claude Sonnet 4.6 | Anthropic |
| Claude Opus 4.6 | Anthropic |
| GPT-5.4 | OpenAI |
| DeepSeek v3.2 | AWS Bedrock |
| Mistral Large 3 | AWS Bedrock |
| Llama 4 Maverick | AWS Bedrock |
| Llama 4 Scout | AWS Bedrock |
| Qwen3 Next 80B-A3B | AWS Bedrock |

Closed-source models are evaluated via vendor APIs at default temperature; open-weight Bedrock models at `T=0`. All harnesses are single-sample, tool-use disabled. Adding a new model is a one-line config change in the relevant backend wrapper under `eval/`.

---

## What's in the repo

| Path | What |
|---|---|
| [`FACET_paper.pdf`](FACET_paper.pdf) | The paper (14 pages) |
| [`instances/`](instances/) | 10 in-distribution legal cases + 3 counterfactual variants + adversarial rewrites + C2 perturbations |
| [`results/weighted-probe/`](results/weighted-probe/) | Raw JSON outputs from all probe runs (~350 files) |
| [`eval/`](eval/) | Probe harnesses + analysis scripts + numeric verifier |
| [`figures/`](figures/) | Generated from raw JSON by scripts in `eval/` |
| [`factor_type_taxonomy.md`](factor_type_taxonomy.md) | 17-type doctrinal taxonomy |
| [`latex/main_v3.tex`](latex/main_v3.tex) | Paper source |

---

## Reproduce the paper

Every numeric claim in the paper is mechanically verifiable from the raw data in `results/`:

```bash
python3 eval/analyze_pilot.py            # regenerates Tables 2 & 3
python3 eval/analyze_weighted_probe.py
python3 eval/verify_paper_numbers.py     # legacy arithmetic checks only
python3 eval/recompute_valid_only.py --known-invalid eval/known_invalid_records.json --output /tmp/facet-valid-only.json
python3 eval/verify_citations.py         # re-checks every arXiv citation against the live arXiv API — currently 5/5 PASS
```

CI now runs trial-validity unit tests and valid-only recomputation with explicit historical exclusions, alongside the legacy arithmetic and citation checks. The historical 80/80 figure is **not** evidence that all input records were valid.

---

## Key numbers

| Metric | Value |
|---|---|
| Frontier model configurations evaluated | 8 |
| Labs represented | 6 |
| In-distribution legal cases | 10 |
| Counterfactual variants | 3 (+ 4 adversarial rewrites) |
| Total C2 ablation trials | 144 collected; 141 valid weight vectors |
| Raw JSON output files | 350+ |
| Legacy arithmetic checks | 80/80; not a full validity audit |
| P1 vs P2 McNemar's exact *p* | 2.4×10⁻⁴ |
| GPT-5.4 CF-Cabral family n | 22 |
| GPT-5.4 wrong-outcome rate | 7/22, 95% Wilson CI [16%, 53%] |
| Full 8-model × 3-counterfactual sweep cost | ~$35 |

---

## Validity audit and corrections

An audit of 306 archived weighted-probe files identified **12 invalid factor vectors**: nine counterfactual and three C2. They are preserved but excluded from valid-only analysis. DeepSeek's C2 mean residual becomes **6.33% over 15 valid ablations**, rather than 5.28% over 18 attempts. Some manuscript plots and model-group comparisons still require regeneration and independent replication. See [the correction notes](docs/facet_acl2027_audit.md) and [the frozen exclusions](eval/known_invalid_records.json).

No new model calls or independent legal-label validation are claimed for these corrections.

## Status

This is a **pilot study**. Known limitations (paper §6):

- 13 base instances is pilot scale; the four-probe protocol and harness scale to larger instance sets with no design changes
- Counterfactual ground truth is author-declared (synthetic cases cannot be appellate-adjudicated)
- California tort cases only; cross-jurisdictional replication is future work
- C2 perturbations use the same syntactic cue words that the P3 adversarial probe is designed to remove; cue-free C2 variants are future work
- DeepSeek at Bedrock `T=0` is not fully deterministic in practice

---

## Citation

```bibtex
@software{jambula2026facet,
  title     = {{FACET}: Measuring Attribution Faithfulness in Multi-Factor {LLM} Reasoning},
  author    = {Jambula, Venkateshwar Reddy},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v3.0},
  doi       = {10.5281/zenodo.19557436},
  url       = {https://doi.org/10.5281/zenodo.19557436}
}
```

## License

- **Code** (`eval/`): MIT — see [LICENSE](LICENSE)
- **Data** (`instances/`, `results/`): CC BY 4.0 — see [LICENSE-DATA](LICENSE-DATA)

## Contact

Venkateshwar Reddy Jambula  ·  venkateshwar.jambula@pranaalpha.com

# FACET v4: design review and preregistration **DRAFT** (2026-11-07)

**Status: candidate methodology, NOT frozen, NOT executed on any LLM.** Prior v2/v3 failures remain archived; v4 is a new scientific question. Never use the original SSRN DOI as proof this study was run. No submissions or source manuscripts are modified by this branch.

## The scientific question

In **natural-language decisions with conflicting considerations**, does a model's *self-reported ranking of factors* predict its **observable decision sensitivity** to controlled changes in those factors, relative to a matched random ranking and salient-but-uninformative baselines?

A particular hypothesis worth testing: in cases containing a familiar, prominent consideration that conflicts with other evidence, a model may *report* that consideration as dominant despite its *decisions* responding most to different considerations. This is **a proposed empirical question, not established novelty or a fact about hidden computation**.

**Not tested:** causal access to internal computation, actual attention pathways, whether models make morally or legally correct decisions, or a universally faithful explanation metric.

## Novelty audit (real overlap, no invented first-ever claim)

| Prior paper | Already covers | Potential residual question for FACET |
| --- | --- | --- |
| Siegel et al., ACL 2024, CCT https://aclanthology.org/2024.acl-short.49/ | Counterfactual edits and shifts in predicted label probabilities | Conflicting multi-criteria decisions as explicit factor sets |
| Parcalabescu & Frank, ACL 2024 https://aclanthology.org/2024.acl-long.329/ | Faithfulness vs output self-consistency | Explicitly restrict claims to behavioral consistency |
| Matton et al., ICLR 2025 https://proceedings.iclr.cc/paper_files/paper/2025/hash/b5ec50eb177908f21f78ed525c-Abstract-Conference.html | Concept-level interventions and causal-effect estimates | Multi-factor trade-offs with prominent/canonical conflict |
| Kang et al., ICML 2025 SPEX https://proceedings.mlr.press/v267/kang25a.html | Interaction attribution | Distinguish marginal and pairwise effects, not a new interaction method |
| Toker et al., ACL Findings 2026 LIBERTy https://aclanthology.org/2026.findings-acl.399/ | Structural counterfactuals and order-faithfulness | Repeated natural-language decision trade-offs |
| Basu & Chakraborty, EMNLP Findings 2026 ICE https://aclanthology.org/2026.findings-emnlp.1055/ | Operator dependence, matched random baselines | Confirm a specific model-reported-factor mismatch with >1 edit operator |
| Mayne et al., ICML 2026 https://proceedings.mlr.press/v306/mayne26a.html | Explanation predictiveness, 18 models, >7000 counterfactuals | Strictly matched conflicting factor sets; adoption of published NSG baseline recommended |
| Thlija et al., NLLP 2026 https://aclanthology.org/2026.nllp-1.26/ | Concept-level faithfulness on legal decisions | No claim of legal novelty without independent annotation |

**Novelty gate:** before investing in confirmatory evaluation, implement and compare at least CCT-style probability-shift attribution, ICE-style operator-specific random rankings, and NSG-style behavior-prediction baseline. If FACET offers neither an empirical discovery nor an instrument improvement beyond these, STOP.

## Design before running model calls

- Primary domain: low-stakes **vendor selection and transport planning**; natural-language factual trade-offs rather than signed-sum arithmetic. Do not imply that these toy cases are adjudicated legal facts.
- Construct **48 independent candidate base cases**, 24 per domain, before checking results. Split 12 pilot + 36 held-out confirmation, stratified by domain. Current checked-in example cases are *illustrative drafts only* and do not fulfill this requirement.
- Each case: exactly six named factor statements, one A/B decision, two fact-altering rewrites per factor expressing the *same substantive change* by differently worded operators, plus factor-order and lexical-only shams. No explicitly supplied percentage weights, fixed winner, or apparent oracle in the prompt.
- At least **two independent reviewers** annotate single-factor validity, semantic equivalence of rewrite pairs, realism, and whether a factor seems *prominent/canonical* before model inference. Disagreements and rejections recorded; do not use the model under test as the sole judge.
- Hold descriptions and decision question constant except the isolated factor edit. Edit validity must be checked via schema and blinded human review. Counterfactual strength may vary across factors; raw effect is a local response-sensitivity statistic, not an intrinsic factor 'true importance'.
- Strong model eligibility: pass a separate 12-item factual-comprehension pilot (>=10/12) and basic response-format check, then freeze exact model/backend/template/temperature/config and seeds. Two different model families required for a broader claim.
- Never use Qwen 0.5B or Qwen 7B gate failures as evidence for or against frontier-model attribution behavior.

## Sampling, estimand and costs

For each independent base case, use **R=5** distinct predetermined seeds at **identical decoding settings** for every variant:
- Base decision: R calls.
- 6 factors × 2 meaning-matched counterfactual rewrites × R = 60 calls.
- 2 lexical/order shams × R = 10 calls.
- Total 75 decision calls per case + 2 separate attribution calls on the **original** scenario, conditioned only on the prespecified *first valid base decision* (or fail the case, no fabrication).
- **77 calls/case/model**, approximately 924 calls for 12-case pilot per model. This is a cost estimate in requests, not dollars, and may need predeclared smaller screening due to limited compute. Do not run anything paid or cloud without approval.
- Responses are fresh, stateless. All model decisions are strict A/B. Preserve invalid/missing outputs and tool failures. Temperature fixed by model; seeds logged; seed matching across variants is optional but must be declared in advance. If token-level log probabilities are available and validated, report as secondary analysis only, not a pooled cross-backend primary metric.

For case i, let p_i(A) be the fraction of valid base decisions selecting A. Let p_i^{f,o}(A) be that fraction after altering factor f with operator o.

Behavioral intervention sensitivity:
`S_if = mean_o |p_i^{f,o}(A) - p_i(A)|`.
Sham sensitivity:
`N_i = mean_control |p_i^control(A) - p_i(A)|`.
Report operator consistency and missingness; **do not** silently treat invalid responses as one label.

Get the model's own ranked factor list in two fresh requests (`F1...F6`, no percentages), after fixing the original decision to the predeclared first base output. Primary case effect:
`G_i = mean_repetitions [ S_i,first_ranked_factor - mean_f S_if ]`.
This subtracts the exact **uniformly random factor** baseline within that same case. It is an observable association, not a measure of internal causes. Run a matched random ranking/position/length baseline and a shuffled-case control.

## Primary and secondary hypotheses

H1 (primary): mean G across independently constructed cases is >0 for each of 2 independent model families, with paired base-case bootstrap 95% CI lower bound >0 in the held-out confirmatory set. Also report a preregistered minimum practically relevant effect of 0.10 in absolute decision probability units. Report effect size even when the binary gate fails.

H2 (secondary): the gap between self-reported and behaviorally most sensitive factors is stronger in independently-rated high-prominence conflict cases. This is explicitly a secondary interaction; human ratings and analysis choices must be frozen before inference.

H3 (exploratory only): two-factor edits may reveal compensatory interactions missed by one-factor sensitivity. Do not claim novelty over SPEX or Shapley/Sobol literature. Add only after H1 feasibility, with a separate protocol.

Critical controls: equivalent rewrite A/B, lexical-only sham, randomized factor order, factor length and position baselines, baseline response instability, and per-operator reports.

## Go / stop decisions

Stop pilot before confirmation if >10% invalid or missing; fewer than 50% of cases show any meaningful response sensitivity (max S_if >= 0.20); reworded equivalent edits disagree more than lexical controls; sham sensitivity is comparable to semantic sensitivity; the target model fails comprehension; or a stronger prior-method baseline fully explains any claimed effect. A null result is acceptable. The stopped pilot is **not** evidence of no causal mechanism.

If pilot passes, freeze all cases/annotations/analysis, commit and tag *before* confirmation, then run two independent model families. Report cluster-bootstrap uncertainty by case, never count 6 factor edits or 5 repeats as 30 independent cases. Do not change analysis thresholds after seeing confirmation.

## Provenance and publication

Never overwrite v1-v3 or their raw records. Version every input, variant, prompt, answer, model, inference parameters, hashes, adjudication receipts, parse failure and missing record. Peer-reviewed publication requires novel empirical result and comparison to earlier methods, not just working code. Until then, label FACET v4 **study design only**.

Run offline code/tests with:
```bash
python3 -m unittest discover -s research/facet-v4/tests -v
python3 research/facet-v4/src/facet_v4.py manifest research/facet-v4/data/example_cases_DRAFT.json --output /tmp/facet-v4-decisions.jsonl --repeats 5
python3 research/facet-v4/src/facet_v4.py validate-manifest /tmp/facet-v4-decisions.jsonl
```

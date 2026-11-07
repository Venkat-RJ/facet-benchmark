# FACET controlled replication protocol (draft, not preregistered)

No calls have been made under this protocol. Freeze a version and source hash publicly **before** collecting new responses.

**A. Cue sensitivity:** For each independently authored and adjudicated held-out legal case, generate original, cue-laden-neutralized, cue-free-neutralized, wording-only, negative cue control, and masked-case-name conditions. Independent human reviewers must agree on semantic equivalence and the adjudicated outcome before seeing LLM responses. Primary outcome: within-case difference in reported neutral-factor weight between cue-laden and cue-free variants, with cluster-aware uncertainty over cases.

**B. Elicitation sensitivity:** Hold the case facts and a shared system instruction fixed. Randomize P1 top-1 versus P2 full weight-vector formats; compare both via the common endpoint "which factor is top-ranked?" rather than the original non-equivalent P1/P2 pass thresholds. Resolve ties using a rule fixed in advance; report raw formats, parser errors, and observed answer sensitivity separately. Evaluate multiple models as independent within-model replications, with case as the unit of generalization.

**In every arm:** freeze instance JSON and intervention hashes; log exact model ID, backend, requested temperature, unavailable controls, system/prompt hashes, tool permissions, response hashes, timestamp, and parse status. Preserve invalid responses and timeouts, never convert missing weights to zero. Analyze independent cases, not 24 paired model-case cells as 24 independent cases. Compare existing work (ICE 2026, Walk the Talk? 2025, CCT 2024) before novelty claims. No causal-faithfulness claim from weight self-reports alone.

**Stop criteria:** fail to secure independent labels; model/backend settings aren't comparable; negative controls shift the outcome as much as semantic interventions; or held-out results don't replicate. Negative results should be published transparently as such.

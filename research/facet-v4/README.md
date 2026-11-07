# FACET v4: natural-language multi-factor decision study

**State: design and offline implementation, NOT a validated scientific study.**

This is a standalone scientific prototype on the `research/facet-v4-intervention-design` branch. Its goal is to test whether a model's **self-reported most influential factor** predicts its **observed sensitivity to that factor being changed** on natural-language trade-off decisions. It does not infer hidden causal mechanisms and does not assume there is an objectively correct A/B decision.

## What was built

- `PROTOCOL_DRAFT.md`: peer-literature overlap, hypotheses, sampling design, primary estimand, failure criteria and publication gate.
- `data/example_cases_DRAFT.json`: two hand-authored six-factor examples (software procurement and refrigerated logistics), **not independently validated**. They are placeholders for independent annotation and 48 true base cases. They are **not frozen research stimuli**.
- `src/facet_v4.py`: generates all isolated single-factor edits (two alternative wordings per factor), two shams, prompt/source hashes and scoring of behavioral sensitivity vs ranked self-reports. Missing/invalid responses are excluded, never replaced.
- `src/run_local.py`: localhost-only, foreground, request-capped and append-only OpenAI-compatible inference runner. **By design refuses unreviewed casebooks.** No cloud API endpoint accepted.
- `tests/`: deterministic unit tests with **synthetic reference outputs only**, no LLM inference.

## Local *offline* commands (no model calls)

```bash
python3 -m unittest discover -s research/facet-v4/tests -v
python3 research/facet-v4/src/facet_v4.py manifest \
  research/facet-v4/data/example_cases_DRAFT.json \
  --repeats 5 --output /tmp/facet-v4-decisions.jsonl
python3 research/facet-v4/src/facet_v4.py validate-manifest /tmp/facet-v4-decisions.jsonl
```

The example manifest contains 2 cases × (1 base + 12 semantic variants + 2 shams) × 5 repeats = **150 decision stimuli**, generated offline. The examples are **unapproved** and the localhost runner will reject them for live inference. This demonstrates schema construction, not experimental findings.

## Before experimental execution

1. Construct 48 truly independent human-readable decisions across two domains and freeze them into 12 pilot/36 held-out cases. Avoid repetitions made by merely changing names or numbers.
2. Two independent non-participating reviewers sign off on each scenario's realism, each one-factor edit, equivalence between two edits, and salience labels. Store the review receipt hashes; no auto-approval.
3. Fix model family, model-file hash, prompt serialization/chat template, temperature, sampling seeds, hardware, response parsing and cost budget. Test comprehension on separate held-out checks first.
4. Archive and freeze the protocol and code *before* any live pilot.
5. Run the 12-case pilot with two independent model families only after the external review; do not proceed to confirmation if the stop conditions fail.
6. Compare CCT, ICE, LIBERTy and NSG baselines and estimate whether there is any substantive advance before writing a new paper.

When live observations are authorized and eligible, generate decision stimuli, run the P0 calls, then create P1 attribution stimuli using only each case's first recorded valid original decision:

```bash
python3 research/facet-v4/src/facet_v4.py attributions CASEBOOK.json \
 --decision-manifest P0_MANIFEST.jsonl --decisions P0_RESPONSES.jsonl \
 --output ATTR_MANIFEST.jsonl

python3 research/facet-v4/src/facet_v4.py score CASEBOOK.json \
 --decision-manifest P0_MANIFEST.jsonl --decisions P0_RESPONSES.jsonl \
 --attribution-manifest ATTR_MANIFEST.jsonl --attributions ATTR_RESPONSES.jsonl \
 --output RESULTS.json
```

**Operational caveat:** The current runner preserves the model, supplied seed, and prompt hashes, but that alone cannot prove the server actually loaded the claimed model. For a manuscript-grade experiment, record the actual GGUF/provider model SHA, llama.cpp revision, generation settings and a separate environment receipt.

## Boundaries

- These **two examples** are not 48 independently annotated cases. The protocol is not preregistered. No model responses were collected or inferred by this branch.
- Neither score success nor source compilation implies scientific novelty.
- This working branch does not change the previous FACET SSRN DOI or its citation history.
- Never convert P1-vs-P2 format effects from v2/v3 into claims of actual internal causal reasoning.

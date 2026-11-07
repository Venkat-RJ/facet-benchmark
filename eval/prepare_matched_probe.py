#!/usr/bin/env python3
"""Offline generator for a matched P1/P2 attribution-format test. Does not call models.

Inputs should be independently verified held-out instances before experiments.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from validate_trial import sha256_json, sha256_text

SYSTEM_PROMPT = (
    'You are evaluating a multi-factor decision. Use only the supplied case facts. '
    'Do not call tools or retrieve external evidence. Report the requested factor '
    'attribution and yes/no decision. Do not infer the answer from the case title.'
)


def prepare(instance: dict, *, mask_case: bool = True) -> dict:
    factors = instance['factors']
    n = len(factors)
    if n < 2:
        raise ValueError('At least two factors required')
    source = 'Case X' if mask_case else str(instance.get('source_case', 'Case X'))
    listed = '\n'.join(f'F{i}: {f["text"]}' for i, f in enumerate(factors, 1))
    context = (f'CASE: {source}\n\n{instance["case_background"]}\n\n'
               f'FACTORS:\n{listed}\n\nQUESTION: {instance["question"]}\n\n')
    p1 = context + ('First line: name the single factor you consider most important, '
                    'using exactly F1, F2, etc. Second line: answer Yes or No.')
    labels = ', '.join(f'F{i}: X' for i in range(1, n + 1))
    p2 = context + ('First line: distribute 100 percentage points of importance across '
                    f'all {n} factors using exactly {labels}, with no missing factors. '
                    'Second line: answer Yes or No.')
    return {
        'status': 'OFFLINE_PROMPTS_ONLY',
        'instance_id': instance['instance_id'],
        'original_instance_sha256': sha256_json(instance),
        'system_instruction': SYSTEM_PROMPT,
        'system_instruction_sha256': sha256_text(SYSTEM_PROMPT),
        'case_title_masked': mask_case,
        'common_outcome': 'yes_no',
        'common_primary_endpoint': 'top_named_factor; weighted argmax for P2, ties separate',
        'p1': {'prompt': p1, 'sha256': sha256_text(p1)},
        'p2': {'prompt': p2, 'sha256': sha256_text(p2)},
        'caveat': 'Distinct output instructions remain. This tests reported factor ranking, not causal faithfulness.',
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--instance', required=True, type=Path)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--show-case-name', action='store_true')
    args = ap.parse_args()
    obj = prepare(json.loads(args.instance.read_text(encoding='utf-8')),
                  mask_case=not args.show_case_name)
    text = json.dumps(obj, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    else:
        print(text, end='')

if __name__ == '__main__':
    main()

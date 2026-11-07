#!/usr/bin/env python3
"""Recompute FACET exploratory results from archived responses, excluding invalid trials.

This is not proof of internal reasoning, legal correctness, or case-level significance.
All files are read-only. New trial failures must remain in the archive and be logged.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics

from validate_trial import trial_status, neutral_weight, last_line_answer, sha256_json

NEUTRAL = {'facet-neg-cf-001': 'F1', 'facet-neg-cf-002': 'F3', 'facet-neg-cf-003': 'F1'}
FACTOR = re.compile(r'\bF([1-9][0-9]*)\b')


def model_key(raw: str) -> str:
    s = (raw or '').lower()
    if 'opus' in s: return 'opus'
    if 'sonnet' in s: return 'sonnet'
    if 'gpt-5.4' in s or 'codex' in s: return 'gpt-5.4'
    if 'deepseek' in s: return 'deepseek'
    if 'mistral' in s: return 'mistral'
    if 'maverick' in s: return 'maverick'
    if 'scout' in s: return 'scout'
    if 'qwen' in s: return 'qwen'
    return s


def category(path: Path) -> str:
    if 'c2' in path.parts:
        return 'c2'
    if '-adv' in path.name:
        return 'p3'
    if re.search(r'facet-neg-cf-00[123]-', path.name):
        return 'p2'
    return 'anchor'


def read_json(path: Path) -> dict:
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def evaluate(repo: Path) -> dict:
    weighted = repo / 'results' / 'weighted-probe'
    if not weighted.is_dir():
        raise FileNotFoundError(weighted)
    all_trials, invalid, data = [], [], defaultdict(list)
    for path in sorted(weighted.rglob('*.json')):
        rel = str(path.relative_to(repo))
        group = category(path)
        try:
            record = read_json(path)
            status, issues = trial_status(record)
        except (ValueError, OSError) as exc:
            record = {}
            status, issues = ('invalid_format', [f'json_error:{exc}'])
        entry = {'path': rel, 'group': group, 'status': status, 'issues': issues,
                 'model': model_key(str(record.get('model') or path.name)),
                 'instance': record.get('instance_id')}
        all_trials.append(entry)
        if status != 'valid':
            invalid.append(entry)
        else:
            data[group].append((path, record))

    # Original P1: one canonical stored observation per model/counterfactual.
    # Source filename lexicographic order timestamps are YYYYMMDD-HHMMSS.
    p1 = {}
    p1_invalid = []
    for path in sorted((repo / 'results').glob('facet-neg-cf-*-real-*.json')):
        try:
            d = read_json(path)
            inst = d['instance_id']
            if inst not in NEUTRAL: continue
            model = model_key(str(d.get('model_resolved_at_run_time', '')))
            c0 = d['conditions']['C0']
            response = str(c0.get('direct_probe_answer') or '')
            m = FACTOR.search(response.splitlines()[0]) if response.splitlines() else None
            answer = str(c0.get('answer') or '')
            if not m or not re.match(r'^(yes|no)\b', answer.strip(), re.I):
                p1_invalid.append(str(path.relative_to(repo)))
                continue
            p1[(inst, model)] = {'factor': f'F{m.group(1)}', 'path': str(path.relative_to(repo))}
        except (KeyError, ValueError, OSError, TypeError):
            p1_invalid.append(str(path.relative_to(repo)))

    # P2: use the most recent *valid* attempt, preserving invalid tries in the log.
    p2 = {}
    for path, d in data['p2']:
        inst, model = d.get('instance_id'), model_key(str(d.get('model') or ''))
        if inst in NEUTRAL and neutral_weight(d, NEUTRAL[inst]) is not None:
            p2[(inst, model)] = {'weight': float(d['parsed_weights'][NEUTRAL[inst]]),
                                 'weights': d['parsed_weights'], 'path': str(path.relative_to(repo))}
    pairs = []
    for (inst, model), a in sorted(p1.items()):
        b = p2.get((inst, model))
        if b is None: continue
        highest = max(b['weights'].values())
        tops = sorted(f for f, w in b['weights'].items() if w == highest)
        pairs.append({'instance': inst, 'model': model,
                      'p1_non_neutral': a['factor'] != NEUTRAL[inst],
                      'p2_neutral_weight_le_10': b['weight'] <= 10,
                      'p1_factor': a['factor'], 'p2_top_factors': tops,
                      'top_factor_agreement_fraction': 1 / len(tops) if a['factor'] in tops else 0,
                      'p1_source': a['path'], 'p2_source': b['path']})
    discordant = sum(a['p1_non_neutral'] != a['p2_neutral_weight_le_10'] for a in pairs)
    p1_by_case = {inst: {'n': sum(p['instance'] == inst for p in pairs),
                         'non_neutral': sum(p['instance'] == inst and p['p1_non_neutral'] for p in pairs)}
                  for inst in NEUTRAL}
    p2_by_case = {inst: {'n': sum(p['instance'] == inst for p in pairs),
                         'neutral_le_10': sum(p['instance'] == inst and p['p2_neutral_weight_le_10'] for p in pairs)}
                  for inst in NEUTRAL}

    c2 = defaultdict(list)
    for path, d in data['c2']:
        factor = str(d.get('c2_perturbed_factor', '')).upper()
        if factor in d['parsed_weights']:
            c2[model_key(str(d.get('model') or ''))].append(float(d['parsed_weights'][factor]))
    c2_stats = {m: {'valid_n': len(v), 'valid_mean': round(statistics.mean(v), 6)}
                for m, v in sorted(c2.items())}

    p3 = defaultdict(list)
    for path, d in data['p3']:
        inst, model = str(d.get('instance_id') or ''), model_key(str(d.get('model') or ''))
        neut = NEUTRAL.get(inst.replace('-adv2', '').replace('-adv', ''))
        if neut in d['parsed_weights']:
            p3[(inst, model)].append({'weight': float(d['parsed_weights'][neut]),
                                      'answer': last_line_answer(d['raw_response']),
                                      'file': str(path.relative_to(repo))})
    p3_stats = {}
    for (inst, model), values in sorted(p3.items()):
        p3_stats[f'{inst}|{model}'] = {'valid_n': len(values),
                                       'mean_neutral_weight': round(statistics.mean(x['weight'] for x in values), 6),
                                       'yes': sum(x['answer'] == 'yes' for x in values)}
    gpt_cabral = [v for (inst, model), vals in p3.items()
                  if model == 'gpt-5.4' and inst in {'facet-neg-cf-001-adv', 'facet-neg-cf-001-adv2'}
                  for v in vals]
    return {
        'status': 'EXPLORATORY_VALID_ONLY', 'source_repo': str(repo),
        'total_weighted_trials': len(all_trials), 'valid_weighted_trials': len(all_trials) - len(invalid),
        'invalid_weighted_trials': len(invalid),
        'by_group_total': dict(Counter(x['group'] for x in all_trials)),
        'by_group_invalid': dict(Counter(x['group'] for x in invalid)),
        'invalid_records': invalid,
        'p1_invalid_records': p1_invalid,
        'paired_valid_n': len(pairs), 'paired_discordant_n': discordant,
        'p1_by_case': p1_by_case, 'p2_by_case': p2_by_case,
        'paired_top_factor_mean_fractional_agreement': round(statistics.mean(x['top_factor_agreement_fraction'] for x in pairs), 6) if pairs else None,
        'pairs': pairs, 'c2_valid_only': c2_stats, 'p3_valid_only': p3_stats,
        'gpt54_cf_cabral': {'valid_n': len(gpt_cabral),
                            'yes_n_author_designated_wrong': sum(x['answer'] == 'yes' for x in gpt_cabral),
                            'posthoc_low_weight_n': sum(x['weight'] < 20 for x in gpt_cabral),
                            'posthoc_high_weight_n': sum(x['weight'] >= 20 for x in gpt_cabral)},
        'interpretation': 'Behavioral elicitation sensitivity only; no internal reasoning inference; 3 cases are not 24 independent case units.',
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--known-invalid', type=Path)
    args = parser.parse_args()
    report = evaluate(args.repo)
    report['report_sha256'] = sha256_json(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f"Weighted records: {report['total_weighted_trials']} total, {report['invalid_weighted_trials']} invalid, {report['valid_weighted_trials']} usable")
    print(f"P1/P2 pairs: {report['paired_valid_n']}, discordant criteria: {report['paired_discordant_n']}")
    print('C2 valid-only stats:', json.dumps(report['c2_valid_only'], sort_keys=True))
    print('GPT-5.4 Cabral:', json.dumps(report['gpt54_cf_cabral'], sort_keys=True))
    if args.known_invalid:
        known = set(json.loads(args.known_invalid.read_text(encoding='utf-8'))['known_invalid_files'])
        actual = {r['path'] for r in report['invalid_records']}
        added = sorted(actual - known)
        removed = sorted(known - actual)
        print(f'Legacy exclusions: {len(actual & known)} known, {len(added)} new, {len(removed)} resolved')
        if added:
            print('NEW INVALID:', *added, sep='\n')
            raise SystemExit(1)


if __name__ == '__main__':
    main()

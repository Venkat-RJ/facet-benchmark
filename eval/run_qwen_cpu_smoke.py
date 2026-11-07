#!/usr/bin/env python3
"""Real CPU inference on historical FACET cases. Integration check, not replication.

Each llama-cli process runs synchronously with a timeout. No server/background job,
API credentials, tool use, response repair, or invented model outputs are used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from prepare_matched_probe import prepare
from validate_trial import parse_weight_line, last_line_answer, trial_status

MODEL_SHA256 = '74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db'
ENGINE_COMMIT = 'c02e5ab2a675c8bc1abc8b1e4cb6a93b26bdcce7'
INSTANCES = ('facet-neg-cf-001', 'facet-neg-cf-002', 'facet-neg-cf-003')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def chatml(system: str, prompt: str) -> str:
    return ('<|im_start|>system\n' + system + '<|im_end|>\n'
            '<|im_start|>user\n' + prompt + '<|im_end|>\n'
            '<|im_start|>assistant\n')


def clean_stdout(raw: str) -> str:
    # Preserve raw stdout separately. Only remove an explicitly printed end token.
    text = raw.strip()
    for suffix in ('<|im_end|>', '<|endoftext|>'):
        if text.endswith(suffix):
            text = text[:-len(suffix)].rstrip()
    return text


def score(kind: str, text: str, n_factors: int) -> dict:
    if kind == 'smoke':
        return {'format_valid': text.strip() == '17', 'expected': '17'}
    answer = last_line_answer(text)
    lines = text.strip().splitlines()
    if kind == 'p1':
        match = re.fullmatch(r'F([1-9][0-9]*)', lines[0].strip()) if lines else None
        good_factor = bool(match and 1 <= int(match.group(1)) <= n_factors)
        valid = good_factor and answer is not None and len(lines) == 2
        return {'format_valid': valid, 'answer': answer,
                'factor': 'F' + match.group(1) if good_factor else None}
    weights = parse_weight_line(text, n_factors) or {}
    status, issues = trial_status({'n_factors': n_factors, 'parsed_weights': weights,
                                 'raw_response': text}, strict_raw=True)
    if len(lines) != 2:
        issues = issues + ['expected_exactly_two_lines']
    valid = status == 'valid' and not issues
    tops = []
    if valid:
        highest = max(weights.values())
        tops = sorted(key for key, value in weights.items() if value == highest)
    return {'format_valid': valid, 'answer': answer, 'weights': weights,
            'weight_sum': sum(weights.values()), 'issues': issues, 'top_factors': tops}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--binary', required=True, type=Path)
    ap.add_argument('--model', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    model_hash = file_hash(args.model)
    if model_hash != MODEL_SHA256:
        raise RuntimeError('Model SHA256 does not match the official frozen GGUF.')
    manifest = {
        'status': 'REAL_MODEL_INTEGRATION_CHECK_NOT_CONFIRMATORY_STUDY',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'model_repo': 'Qwen/Qwen2.5-0.5B-Instruct-GGUF',
        'model_filename': args.model.name, 'model_sha256': model_hash,
        'model_bytes': args.model.stat().st_size,
        'engine_commit': ENGINE_COMMIT, 'temperature': 0,
        'seed': 20261107, 'threads': args.threads, 'gpu_layers': 0,
        'context_tokens': 4096, 'maximum_generated_tokens': 128,
        'source_commit': os.environ.get('GITHUB_SHA'),
        'actions_run_id': os.environ.get('GITHUB_RUN_ID'),
        'cases': list(INSTANCES), 'repetitions': 2,
        'limitations': [
            'Historical cases, not held-out cases.',
            'One 0.5B quantized model, no inference about frontier model behavior.',
            'Greedy repeated calls test repeatability, not independent sampling.',
            'Case titles masked but background text may still identify cases.',
            'Legal correctness labels are author-declared, not independently adjudicated.',
            'Direct CLI uses an explicit Qwen ChatML prompt; no grammar forcing.',
            'This run uses a fresh official model download, not the inaccessible uploaded bytes.'
        ],
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    jobs = [{'id': 'smoke', 'kind': 'smoke', 'instance': None, 'repeat': 0,
             'n_factors': 0, 'system': 'You are a helpful assistant.',
             'prompt': 'Which is larger, 17 or 12? Reply with only the larger number.'}]
    for inst_id in INSTANCES:
        instance = json.loads((repo / 'instances' / (inst_id + '.json')).read_text())
        stimulus = prepare(instance, mask_case=True)
        (out / (inst_id + '.stimulus.json')).write_text(json.dumps(stimulus, indent=2) + '\n')
        for repeat in range(2):
            for kind in ('p1', 'p2'):
                jobs.append({'id': f'{inst_id}-{kind}-r{repeat}', 'kind': kind,
                             'instance': inst_id, 'repeat': repeat,
                             'n_factors': len(instance['factors']),
                             'system': stimulus['system_instruction'],
                             'prompt': stimulus[kind]['prompt']})
    records = []
    for job in jobs:
        formatted = chatml(job['system'], job['prompt'])
        prompt_file = out / (job['id'] + '.prompt.txt')
        prompt_file.write_text(formatted)
        command = [str(args.binary.resolve()), '-m', str(args.model.resolve()),
                   '-f', str(prompt_file), '-n', '128', '-c', '4096',
                   '-t', str(args.threads), '-ngl', '0', '--seed', '20261107',
                   '--temp', '0', '--no-display-prompt', '--simple-io']
        print('$ ' + ' '.join(command), flush=True)
        start = time.monotonic()
        record = dict(job, prompt_sha256=digest(formatted.encode()), command=command)
        try:
            result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                    text=True, timeout=100, check=False)
            record.update(returncode=result.returncode, raw_stdout=result.stdout,
                          raw_stderr=result.stderr, timed_out=False)
        except subprocess.TimeoutExpired as exc:
            def as_text(value):
                return value.decode(errors='replace') if isinstance(value, bytes) else (value or '')
            record.update(returncode=None, raw_stdout=as_text(exc.stdout),
                          raw_stderr=as_text(exc.stderr), timed_out=True)
        record['elapsed_seconds'] = round(time.monotonic() - start, 4)
        record['response_sha256'] = digest(record['raw_stdout'].encode())
        text = clean_stdout(record['raw_stdout'])
        record['clean_response'] = text
        if record['returncode'] == 0:
            record['score'] = score(job['kind'], text, job['n_factors'])
        else:
            record['score'] = {'format_valid': False, 'execution_failure': True}
        records.append(record)
        (out / (job['id'] + '.json')).write_text(json.dumps(record, indent=2) + '\n')
        print('OBSERVED ' + json.dumps({'id': job['id'], 'returncode': record['returncode'],
                                        'response': text, 'score': record['score']}), flush=True)
        if job['kind'] == 'smoke' and record['returncode'] != 0:
            break
    research = [r for r in records if r['kind'] != 'smoke']
    first = [r for r in research if r['repeat'] == 0]
    pairs = []
    for inst_id in INSTANCES:
        by_kind = {r['kind']: r for r in first if r['instance'] == inst_id}
        if len(by_kind) != 2:
            continue
        a, b = by_kind['p1']['score'], by_kind['p2']['score']
        valid = a['format_valid'] and b['format_valid']
        pairs.append({'instance': inst_id, 'both_valid': valid,
                      'top_factor_agrees_including_ties': a.get('factor') in b.get('top_factors', []) if valid else None,
                      'decision_agrees': a.get('answer') == b.get('answer') if valid else None})
    repeats = []
    for r in first:
        other = next((q for q in research if q['instance'] == r['instance'] and
                      q['kind'] == r['kind'] and q['repeat'] == 1), None)
        if other:
            repeats.append({'instance': r['instance'], 'kind': r['kind'],
                            'exact_clean_response_match': r['clean_response'] == other['clean_response'],
                            'both_executions_ok': r['returncode'] == 0 and other['returncode'] == 0})
    summary = {'scope': manifest['status'], 'total_invocations': len(records),
               'execution_successes': sum(r['returncode'] == 0 for r in records),
               'smoke_number_correct': records[0]['score'].get('format_valid', False),
               'research_invocations': len(research),
               'research_valid_format': sum(r['score']['format_valid'] for r in research),
               'first_repetition_valid_by_probe': {k: sum(r['kind'] == k and r['score']['format_valid'] for r in first) for k in ('p1', 'p2')},
               'pairs_first_repetition': pairs, 'repeatability': repeats,
               'invalid_responses_are_not_zero_weights': True,
               'new_generalizable_finding_established': False}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('FINAL_SUMMARY ' + json.dumps(summary, sort_keys=True), flush=True)
    step_summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if step_summary:
        with open(step_summary, 'a') as stream:
            stream.write('## Real Qwen CPU integration check\n\n```json\n' + json.dumps(summary, indent=2) + '\n```\n')
    return 0 if len(records) == len(jobs) and all(r['returncode'] == 0 for r in records) else 1


if __name__ == '__main__':
    raise SystemExit(main())

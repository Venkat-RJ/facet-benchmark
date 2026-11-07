#!/usr/bin/env python3
"""Foreground-only OpenAI-compatible localhost runner for FACET v4.

No network access outside localhost; no cloud API or remote model endpoint.
No background workers, no parallelism. No runs unless --confirm-live is explicit.
Results in append-only JSONL; failures are preserved and never silently retried.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib import request, parse

from facet_v4 import read_jsonl, canonical, validate_manifest


def check_local_url(url: str) -> None:
    u = parse.urlparse(url)
    if u.scheme != "http" or u.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("Only HTTP localhost model-serving endpoints are permitted")
    if u.username or u.password or u.fragment:
        raise ValueError("URL credentials/fragments not permitted")
    if not u.path.endswith("/v1/chat/completions"):
        raise ValueError("Expected OpenAI-compatible /v1/chat/completions endpoint")


def run(manifest_path: Path, output: Path, endpoint: str, model: str,
        temperature: float, max_requests: int, confirm: bool) -> None:
    check_local_url(endpoint)
    if not model.strip() or not 0 <= temperature <= 2:
        raise ValueError("invalid model or temperature")
    if max_requests < 1:
        raise ValueError("max_requests must be >= 1")
    stimuli = read_jsonl(manifest_path)
    errors = validate_manifest(stimuli)
    if errors:
        raise ValueError("manifest validation failed: " + "; ".join(errors[:5]))
    if any(x.get("review_status") != "approved" for x in stimuli):
        raise ValueError("Unreviewed study cases cannot be run as research. Obtain two independent approvals first.")
    if not confirm:
        print(f"DRY RUN: {len(stimuli)} stimuli, cap {max_requests}, model {model}, localhost only")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if output.exists():
        for row in read_jsonl(output):
            seen.add(row["stimulus_id"])
    pending = [r for r in stimuli if r["stimulus_id"] not in seen][:max_requests]
    print(f"FOREGROUND: {len(pending)} local calls; existing attempts={len(seen)}")
    with output.open("a", encoding="utf-8") as fh:
        for n, stimulus in enumerate(pending, 1):
            started = time.monotonic()
            payload = {
                "model": model, "temperature": temperature,
                "seed": int(stimulus["seed"]), "max_tokens": 96,
                "messages": [
                    {"role": "system", "content": stimulus["system"]},
                    {"role": "user", "content": stimulus["prompt"]},
                ], "stream": False,
            }
            req = request.Request(endpoint, data=canonical(payload).encode(),
                                  headers={"Content-Type": "application/json"}, method="POST")
            response_text = ""
            error = None
            response_meta = {}
            try:
                with request.urlopen(req, timeout=180) as resp:
                    data = json.load(resp)
                response_text = str(data["choices"][0]["message"]["content"] or "")
                response_meta = {
                    "server_model": data.get("model"),
                    "server_id": data.get("id"),
                    "usage": data.get("usage"),
                }
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:400]}"
            record = {
                "stimulus_id": stimulus["stimulus_id"],
                "prompt_sha256": stimulus["prompt_sha256"],
                "system_sha256": stimulus["system_sha256"],
                "case_sha256": stimulus["case_sha256"],
                "seed": stimulus["seed"], "model_id": model,
                "temperature": temperature, "endpoint": endpoint,
                "raw_response": response_text, "error": error,
                "server": response_meta,
                "wall_seconds": round(time.monotonic() - started, 4),
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "warning": "self-described provider provenance; independently verify model file hash",
            }
            fh.write(canonical(record) + "\n")
            fh.flush()
            print(f"[{n}/{len(pending)}] {record['stimulus_id']}: {'ERROR ' + error if error else response_text[:90]}", flush=True)
            if error:
                print("Stopped after preserved transport error; no automatic retry.")
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-requests", type=int, required=True)
    ap.add_argument("--confirm-live", action="store_true")
    args = ap.parse_args()
    run(args.manifest, args.output, args.endpoint, args.model,
        args.temperature, args.max_requests, args.confirm_live)


if __name__ == "__main__":
    main()

"""Security and dry-run tests for the no-cloud model runner."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import facet_v4 as f
import run_local


class LocalRunnerTests(unittest.TestCase):
    def test_never_uses_remote_endpoint(self):
        for url in (
            "https://api.openai.com/v1/chat/completions",
            "http://example.com/v1/chat/completions",
            "http://localhost@evil.example/v1/chat/completions",
            "file:///tmp/server",
            "http://127.0.0.1:8080/v1/other"
        ):
            with self.assertRaises(ValueError, msg=url):
                run_local.check_local_url(url)

    def test_local_endpoint_is_accepted(self):
        run_local.check_local_url("http://127.0.0.1:8080/v1/chat/completions")
        run_local.check_local_url("http://localhost:8080/v1/chat/completions")

    def test_never_runs_unreviewed_cases(self):
        case = f.load_cases(ROOT / "data" / "example_cases_DRAFT.json")
        rows = f.manifest_decisions(case, 1)
        with tempfile.TemporaryDirectory() as td:
            path, out = Path(td) / "manifest.jsonl", Path(td) / "out.jsonl"
            f.write_jsonl(path, rows)
            with self.assertRaises(ValueError):
                run_local.run(path, out, "http://127.0.0.1:8080/v1/chat/completions",
                              "reference", 0.7, 2, True)
            self.assertFalse(out.exists())

    def test_dry_run_has_no_calls_or_output(self):
        case = f.load_cases(ROOT / "data" / "example_cases_DRAFT.json")
        rows = f.manifest_decisions(case, 1)
        for row in rows:
            row["review_status"] = "approved"
        with tempfile.TemporaryDirectory() as td:
            path, out = Path(td) / "manifest.jsonl", Path(td) / "out.jsonl"
            f.write_jsonl(path, rows)
            run_local.run(path, out, "http://127.0.0.1:8080/v1/chat/completions",
                          "reference", 0.7, 2, False)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()

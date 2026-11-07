"""Tests for the FACET v4 offline method. All model outputs are synthetic fixtures."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import facet_v4 as f


class InstrumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = f.load_cases(ROOT / "data" / "example_cases_DRAFT.json")

    def test_has_two_cases_in_distinct_domains(self):
        self.assertEqual(len(self.cases), 2)
        self.assertEqual(len({c["domain"] for c in self.cases}), 2)
        self.assertTrue(all(c["review_status"] == "draft_unreviewed" for c in self.cases))

    def test_15_variants_per_case(self):
        for case in self.cases:
            self.assertEqual(len(f.variants(case)), 15)

    def test_only_one_factor_modified(self):
        for case in self.cases:
            baseline = {x["id"]: x["text"] for x in case["factors"]}
            for v in f.variants(case):
                observed = {x["id"]: x["text"] for x in v["factors"]}
                different = {k for k in baseline if baseline[k] != observed[k]}
                if v["operator"].startswith("semantic"):
                    self.assertEqual(different, {v["factor_id"]})
                else:
                    self.assertEqual(different, set())

    def test_shams_retain_all_factor_content(self):
        for case in self.cases:
            vs = {v["variant_id"]: v for v in f.variants(case)}
            base = sorted((x["id"], x["text"]) for x in vs["base"]["factors"])
            for key in ("sham_order", "sham_lexical"):
                self.assertEqual(base, sorted((x["id"], x["text"]) for x in vs[key]["factors"]))

    def test_manifest_deterministic_and_hashed(self):
        a = f.manifest_decisions(self.cases, 5)
        self.assertEqual(a, f.manifest_decisions(self.cases, 5))
        self.assertEqual(len(a), 150)
        self.assertEqual(f.validate_manifest(a), [])
        self.assertTrue(all(r["system_sha256"] == f.digest(r["system"]) for r in a))

    def test_rejects_invalid_case_and_ungrounded_approval(self):
        case = copy.deepcopy(self.cases[0])
        case["factors"] = case["factors"][:-1]
        with self.assertRaises(ValueError):
            f.validate_case(case)
        case = copy.deepcopy(self.cases[0])
        case["review_status"] = "approved"
        with self.assertRaises(ValueError):
            f.validate_case(case)

    def test_rejects_duplicate_json_and_invalid_rank(self):
        self.assertIsNone(f.parse_response('{"choice":"A","choice":"B"}', "decision"))
        self.assertIsNone(f.parse_response('{"choice":"C"}', "decision"))
        self.assertIsNone(f.parse_response('{"ranking":["F1","F1","F3","F4","F5","F6"]}', "attribution"))
        self.assertIsNone(f.parse_response('{"ranking":[["F1"],"F2","F3","F4","F5","F6"]}', "attribution"))
        self.assertEqual(f.parse_response('{"choice":"B"}', "decision"), "B")

    def fixture(self, top="F1", missing=False):
        cases = self.cases
        dm = f.manifest_decisions(cases, 5)
        dr = []
        for stim in dm:
            choice = "B" if stim["variant_id"] in ("F1_a", "F1_b") else "A"
            dr.append({"stimulus_id": stim["stimulus_id"],
                       "prompt_sha256": stim["prompt_sha256"],
                       "system_sha256": stim["system_sha256"],
                       "seed": stim["seed"], "model_id": "REFERENCE_FIXTURE_NOT_LLM",
                       "raw_response": json.dumps({"choice": choice})})
        if missing:
            dr = [r for r in dr if r["stimulus_id"] != f"{cases[0]['case_id']}:F3_a:P0:2"]
        am = f.manifest_attributions(cases, dr, dm)
        ar = []
        for stim in am:
            ranked = [top] + [x for x in ("F1","F2","F3","F4","F5","F6") if x != top]
            ar.append({"stimulus_id": stim["stimulus_id"],
                       "prompt_sha256": stim["prompt_sha256"],
                       "system_sha256": stim["system_sha256"],
                       "seed": stim["seed"], "model_id": "REFERENCE_FIXTURE_NOT_LLM",
                       "raw_response": json.dumps({"ranking": ranked})})
        return cases, dm, dr, am, ar

    def test_prepare_attribution_requires_base_decision(self):
        cases, dm, dr, am, ar = self.fixture()
        self.assertEqual(len(am), 4)
        self.assertIn("recorded decision", am[0]["prompt"])
        dr = [r for r in dr if r["stimulus_id"] != f"{cases[0]['case_id']}:base:P0:0"]
        am2 = f.manifest_attributions(cases, dr, dm)
        self.assertEqual(len(am2), 2)

    def test_symbolic_fixture_positive_control(self):
        result = f.case_scores(*self.fixture())
        self.assertEqual(result["complete_cases"], 2)
        self.assertAlmostEqual(result["primary_mean_gain"], 5/6)
        self.assertAlmostEqual(result["mean_sham_effect"], 0.0)
        self.assertEqual(len(result["case_bootstrap_95pct"]), 2)

    def test_symbolic_fixture_negative_control(self):
        result = f.case_scores(*self.fixture(top="F2"))
        self.assertAlmostEqual(result["primary_mean_gain"], -1/6)

    def test_missing_data_excluded_not_silently_imputed(self):
        result = f.case_scores(*self.fixture(missing=True))
        self.assertEqual(result["complete_cases"], 1)
        self.assertEqual(len(result["excluded_cases"]), 1)

    def test_source_hash_tamper_is_invalid(self):
        cases, dm, dr, am, ar = self.fixture()
        dr[0]["prompt_sha256"] = "tampered"
        output, errs = f.validate_responses(dr, dm)
        self.assertEqual(output[dr[0]["stimulus_id"]]["value"], None)
        self.assertTrue(any("provenance hash mismatch" in x for x in errs))

    def test_duplicate_response_is_reported(self):
        cases, dm, dr, am, ar = self.fixture()
        output, errs = f.validate_responses(dr + [dr[0]], dm)
        self.assertTrue(any("duplicate response" in x for x in errs))

    def test_manifest_duplicate_is_reported(self):
        rows = f.manifest_decisions(self.cases, 1)
        self.assertTrue(any("duplicate stimulus" in x for x in f.validate_manifest(rows + [rows[0]])))


if __name__ == "__main__":
    unittest.main()

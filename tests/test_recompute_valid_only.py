import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval'))
from recompute_valid_only import evaluate, model_key


class RecomputeTests(unittest.TestCase):
    def test_model_names(self):
        self.assertEqual(model_key('claude-opus-4-6'), 'opus')
        self.assertEqual(model_key('gpt-5.4'), 'gpt-5.4')
        self.assertEqual(model_key('deepseek.v3.2'), 'deepseek')

    def test_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weighted = root / 'results/weighted-probe'
            weighted.mkdir(parents=True)
            raw = root / 'results/facet-neg-cf-002-real-opus-20260412-000324.json'
            raw.write_text(json.dumps({'instance_id':'facet-neg-cf-002','model_resolved_at_run_time':'claude-opus-4-6',
                'conditions':{'C0':{'answer':'No','direct_probe_answer':'F3\nReason'}}}))
            good = {'instance_id':'facet-neg-cf-002','model':'opus','n_factors': 3,
                    'parsed_weights':{'F1':30,'F2':70,'F3':0}, 'raw_response':'F1: 30, F2: 70, F3: 0\nNo'}
            (weighted/'facet-neg-cf-002-opus-20260412-234339.json').write_text(json.dumps(good))
            bad = good.copy();bad['parsed_weights']={'F1':0,'F2':0,'F3':0}
            (weighted/'facet-neg-cf-002-opus-20260412-222222.json').write_text(json.dumps(bad))
            result = evaluate(root)
            self.assertEqual(result['total_weighted_trials'], 2)
            self.assertEqual(result['invalid_weighted_trials'], 1)
            self.assertEqual(result['paired_valid_n'], 1)
            self.assertEqual(result['paired_discordant_n'], 1)


if __name__ == '__main__':
    unittest.main()

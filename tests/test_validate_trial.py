import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval'))
from validate_trial import validity_issues, trial_status, neutral_weight, parse_weight_line, last_line_answer, sha256_json


class ValidationTests(unittest.TestCase):
    def d(self, w=None, text='F1: 0, F2: 40, F3: 60\nNo'):
        return {'n_factors': 3, 'parsed_weights': {'F1': 0, 'F2': 40, 'F3': 60} if w is None else w,
                'raw_response': text}

    def test_valid(self):
        self.assertEqual(trial_status(self.d(), strict_raw=True), ('valid', []))
        self.assertEqual(neutral_weight(self.d(), 'F1'), 0.0)

    def test_zero_sum(self):
        self.assertIn('weight_sum_not_100', validity_issues(self.d({'F1': 0, 'F2': 0, 'F3': 0})))

    def test_over_sum(self):
        self.assertIn('weight_sum_not_100', validity_issues(self.d({'F1': 0, 'F2': 60, 'F3': 60})))

    def test_under_sum(self):
        self.assertIn('weight_sum_not_100', validity_issues(self.d({'F1': 0, 'F2': 40, 'F3': 45})))

    def test_missing_label_not_zero(self):
        self.assertIsNone(neutral_weight(self.d({'F2': 40, 'F3': 60}), 'F1'))

    def test_negative(self):
        self.assertIn('out_of_range_weight', validity_issues(self.d({'F1': -1, 'F2': 41, 'F3': 60})))

    def test_answer(self):
        self.assertIsNone(last_line_answer('F1: 0, F2: 40, F3: 60\nmaybe'))

    def test_duplicate_raw(self):
        self.assertIsNone(parse_weight_line('F1: 0, F1: 40, F3: 60\nNo', 3))

    def test_strict_mismatch(self):
        self.assertIn('parsed_raw_mismatch', validity_issues(self.d(text='F1: 10, F2: 30, F3: 60\nNo'), strict_raw=True))

    def test_provider_error(self):
        d = self.d(); d['backend_error'] = 'timeout'
        self.assertEqual(trial_status(d)[0], 'provider_error')

    def test_hash_is_deterministic(self):
        self.assertEqual(sha256_json({'a': 1, 'b': 2}), sha256_json({'b': 2, 'a': 1}))

    def test_boolean_invalid(self):
        self.assertIn('non_numeric_weight', validity_issues(self.d({'F1': False, 'F2': 40, 'F3': 60})))


if __name__ == '__main__':
    unittest.main()

import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval'))
from prepare_matched_probe import prepare

class MatchTests(unittest.TestCase):
    def test_preparation_is_deterministic_and_masks_case_name(self):
        d={'instance_id':'heldout1','source_case':'Recognizable v Case',
           'case_background':'Independent facts.','question':'Is outcome A right?',
           'factors':[{'text':'first consideration'},{'text':'second consideration'}]}
        a=prepare(d); b=prepare(d)
        self.assertEqual(a,b)
        self.assertTrue(a['case_title_masked'])
        self.assertNotIn('Recognizable',a['p1']['prompt'])
        self.assertNotIn('Recognizable',a['p2']['prompt'])
        self.assertIn('Independent facts',a['p1']['prompt'])
        self.assertIn('Independent facts',a['p2']['prompt'])
        self.assertNotEqual(a['p1']['sha256'],a['p2']['sha256'])

if __name__ == '__main__':
    unittest.main()

import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decision import decide, TemporalVote


class DecisionTests(unittest.TestCase):
    def test_uncertain_and_nothing_are_rejected(self):
        names = ['A', 'V', 'nothing']
        self.assertIsNone(decide(np.array([.51, .48, .01]), names)[0])
        self.assertIsNone(decide(np.array([.01, .01, .98]), names)[0])
        self.assertEqual(decide(np.array([.90, .09, .01]), names)[0], 'A')

    def test_margin_is_separate_from_score(self):
        result = decide(np.array([.55, .44, .01]), ['K','V','nothing'], .5, .15)
        self.assertIsNone(result[0])

    def test_previous_letter_disappears_on_change(self):
        vote = TemporalVote()
        for i in range(5):
            stable = vote.update('A', i*.1)
        self.assertEqual(stable, 'A')
        self.assertIsNone(vote.update('K', .5))
        self.assertIsNone(vote.update(None, .6))
        self.assertIsNone(vote.update('A', .7))

    def test_pause_resets_and_users_are_independent(self):
        a, b = TemporalVote(), TemporalVote()
        for i in range(5):
            a.update('A', i*.1)
        self.assertIsNone(a.update('A', 2))
        self.assertIsNone(b.update('A', 2))

if __name__ == '__main__':
    unittest.main()

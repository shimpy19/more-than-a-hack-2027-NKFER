"""Tests for detector.py, the per-frame Viterbi costs and the ball smoothing, on tiny invented data.
No competition data is involved.

    python -m unittest pipeline/test_detector.py
"""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import detector  # noqa: E402
import viterbi  # noqa: E402
from training_table import smooth_ball  # noqa: E402

H, A, D = 0, 1, 2


class Targets(unittest.TestCase):

    def test_frames_near_a_direct_change_get_the_new_team(self):
        label = np.array([H] * 20 + [A] * 20)
        target, changes = detector.change_targets(label, np.ones(40, dtype=int))
        self.assertEqual(changes.tolist(), [20])
        w = detector.WINDOW
        self.assertTrue((target[20 - w:20 + w + 1] == 2).all())     # change to A
        self.assertTrue((target[:20 - w] == 0).all() and (target[21 + w:] == 0).all())

    def test_through_a_dead_ball_or_across_halves_is_not_a_change(self):
        label = np.array([H] * 10 + [D] * 5 + [A] * 10 + [H] * 10)
        half = np.array([1] * 25 + [2] * 10)
        _, changes = detector.change_targets(label, half)
        self.assertEqual(changes.tolist(), [])

    def test_the_nearest_change_wins(self):
        label = np.array([H] * 10 + [A] * 4 + [H] * 10)
        target, _ = detector.change_targets(label, np.ones(24, dtype=int))
        self.assertEqual(target[11], 2)  # nearer the change to A at 10
        self.assertEqual(target[13], 1)  # nearer the change back to H at 14


class SwitchCosts(unittest.TestCase):

    def test_from_base_when_unlikely_to_cheap_when_certain(self):
        p = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]])
        into_h, into_a = detector.switch_costs(p, {"base": 60, "knob": 1, "cheap": 3})
        self.assertEqual(into_h.tolist(), [60.0, 3.0, 60.0, 31.5])
        self.assertEqual(into_a.tolist(), [60.0, 60.0, 3.0, 60.0])


class VaryingViterbi(unittest.TestCase):

    def test_constant_costs_give_the_fixed_cost_result(self):
        scores = np.random.default_rng(0).dirichlet([1, 1, 1], 2000)
        half = np.r_[np.zeros(100, dtype=int), np.ones(1000, dtype=int), np.full(900, 2)]
        fixed = viterbi.decode_match(scores, half, team=15, dead=30)
        varying = viterbi.decode_match_varying(scores, half, np.full(2000, 15.0),
                                               np.full(2000, 15.0), 30)
        np.testing.assert_array_equal(fixed, varying)

    def test_a_cheap_switch_lets_a_short_possession_through(self):
        scores = np.tile([0.7, 0.29, 0.01], (60, 1))
        scores[25:35] = [0.29, 0.7, 0.01]  # the model leans to A for 10 frames
        half = np.ones(60, dtype=int)
        expensive = viterbi.decode_match_varying(scores, half, np.full(60, 60.0),
                                                 np.full(60, 60.0), 60)
        self.assertTrue((expensive == H).all())
        into_h, into_a = np.full(60, 60.0), np.full(60, 60.0)
        into_a[25], into_h[35] = 0.0, 0.0  # the detector is sure at both ends
        cheap = viterbi.decode_match_varying(scores, half, into_h, into_a, 60)
        self.assertTrue((cheap[25:35] == A).all() and (cheap[:25] == H).all())


class SmoothBall(unittest.TestCase):

    def test_only_inside_seen_stretches_and_gaps_stay_missing(self):
        x = np.r_[np.zeros(10), np.nan, np.nan, np.full(10, 5.0)]
        x[4] = 10.0  # a spike inside the first stretch
        sx, sy = smooth_ball(x, x.copy(), 2)
        self.assertTrue(np.isnan(sx[10:12]).all())
        self.assertLess(sx[4], 10.0)
        np.testing.assert_allclose(sx[12:], 5.0)  # the second stretch never sees the first

    def test_sigma_zero_leaves_the_ball_unchanged(self):
        x = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        sx, _ = smooth_ball(x, x, 0)
        np.testing.assert_array_equal(sx, x.astype(np.float64))


if __name__ == "__main__":
    unittest.main()

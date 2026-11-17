"""Tests for rules_touch.py on tiny invented matches. No competition data is involved.

    python -m unittest pipeline/test_rules_touch.py
"""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import rules_touch as rt  # noqa: E402

FAR = 30.0


def pass_between(n=60, kick=30, toucher="home"):
    """Ball rolls along x at 10 m/s, and is kicked sideways at frame `kick` next to one player."""
    bx = np.r_[np.arange(kick + 1) * 0.4, np.full(n - kick - 1, kick * 0.4)]
    by = np.r_[np.zeros(kick + 1), np.arange(1, n - kick) * 0.4]
    dist_home, dist_away = np.full(n, FAR), np.full(n, FAR)
    (dist_home if toucher == "home" else dist_away)[kick] = 0.8
    return bx, by, np.ones(n, dtype=int), dist_home, dist_away


class Touches(unittest.TestCase):

    def test_kick_next_to_a_player_is_one_touch_by_that_team(self):
        cols = rt.touch_features(*pass_between())
        self.assertEqual(np.flatnonzero(cols["touch_team"]).tolist(), [30])
        self.assertEqual(cols["touch_team"][30], rt.HOME_SIGN)
        self.assertEqual(cols["last_touch_team"][40], rt.HOME_SIGN)
        self.assertEqual(cols["last_touch_s"][40], 10 / rt.FPS)
        self.assertEqual(cols["next_touch_team"][20], rt.HOME_SIGN)
        self.assertEqual(cols["last_touch_team"][20], 0.0)

    def test_away_touch(self):
        cols = rt.touch_features(*pass_between(toucher="away"))
        self.assertEqual(cols["touch_team"][30], rt.AWAY_SIGN)

    def test_kick_with_nobody_near_is_no_touch(self):
        bx, by, half, dist_home, dist_away = pass_between()
        dist_home[:] = FAR
        self.assertEqual(rt.touch_features(bx, by, half, dist_home, dist_away)["touch_team"].sum(), 0)

    def test_steady_ball_is_no_touch(self):
        n = 60
        bx, by = np.arange(n) * 0.4, np.zeros(n)
        cols = rt.touch_features(bx, by, np.ones(n, dtype=int), np.full(n, 0.5), np.full(n, FAR))
        self.assertEqual(np.count_nonzero(cols["touch_team"]), 0)

    def test_last_touch_does_not_carry_over_a_half(self):
        bx, by, half, dist_home, dist_away = pass_between()
        half = np.r_[np.ones(35, dtype=int), np.full(25, 2)]
        cols = rt.touch_features(bx, by, half, dist_home, dist_away)
        self.assertEqual(cols["last_touch_team"][34], rt.HOME_SIGN)
        self.assertEqual(cols["last_touch_team"][40], 0.0)

    def test_window_count_and_carry(self):
        mask = np.array([1, 0, 1, 1, 0, 0], dtype=bool)
        half = np.ones(6, dtype=int)
        self.assertEqual(rt.window_count(mask, 2, half).tolist(), [0, 1, 1, 1, 2, 1])
        self.assertEqual(rt.window_count(mask, 2, half, ahead=True).tolist(), [1, 2, 1, 0, 0, 0])
        value, at = rt.carry(np.array([0, 1.0, 0, -1.0, 0]), np.ones(5, dtype=int))
        self.assertEqual(value.tolist(), [0, 1, 1, -1, -1])
        self.assertEqual(at.tolist(), [-1, 1, 1, 3, 3])

    def test_all_columns_built(self):
        cols = rt.touch_features(*pass_between())
        self.assertEqual(tuple(cols), rt.TOUCH_COLUMNS)


if __name__ == "__main__":
    unittest.main()

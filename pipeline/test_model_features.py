"""Tests for model_features.py on tiny invented matches. No competition data is involved.

    python -m unittest pipeline/test_model_features.py
"""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import FEATURE_COLUMNS, TOUCH_COLUMNS  # noqa: E402
from model_features import feature_table  # noqa: E402
from starter.baseline import AWAY, HOME  # noqa: E402

N = 40


def match(ball_x, ball_y, player_x, player_y, half=None):
    """clean_match-style columns: players 0, 1 home and 2, 3 away, ball where given."""
    ball_x, ball_y = np.asarray(ball_x, float), np.asarray(ball_y, float)
    n = len(ball_x)
    speed = np.r_[np.nan, np.hypot(np.diff(ball_x), np.diff(ball_y)) * 25]
    return {
        "frame": np.arange(n), "half": np.ones(n, dtype=int) if half is None else np.asarray(half),
        "team": np.array([HOME, HOME, AWAY, AWAY]),
        "player_x_m": np.asarray(player_x, float), "player_y_m": np.asarray(player_y, float),
        "player_speed": np.zeros((n, 4)), "ball_x_m": ball_x, "ball_y_m": ball_y,
        "ball_speed": speed, "ball_missing_s": np.zeros(n), "ball_until_s": np.zeros(n),
        "ball_past_end_m": np.zeros(n), "ball_past_side_m": np.zeros(n),
        "n_home": np.full(n, 2), "n_away": np.full(n, 2), "tracking_ok": np.ones(n, dtype=int),
    }


def still_players(n=N):
    """Home players at x = 10 and 30, away players at x = 70 and 90, all on y = 30."""
    return np.tile([10.0, 30.0, 70.0, 90.0], (n, 1)), np.full((n, 4), 30.0)


class TouchFeatures(unittest.TestCase):

    def test_every_feature_is_built_in_order(self):
        px, py = still_players()
        table = feature_table(match(np.linspace(10, 90, N), np.full(N, 30.0), px, py))
        self.assertEqual(len(FEATURE_COLUMNS), 90)
        self.assertEqual(FEATURE_COLUMNS[-len(TOUCH_COLUMNS):], TOUCH_COLUMNS)
        for name in FEATURE_COLUMNS:
            self.assertIn(name, table.columns)

    def test_nearest_team_follows_the_ball(self):
        px, py = still_players()
        bx = np.r_[np.full(20, 11.0), np.full(20, 71.0)]  # at a home player, then at an away one
        table = feature_table(match(bx, np.full(N, 30.0), px, py))
        self.assertEqual(table["nearest_team"].iloc[5], 1.0)
        self.assertEqual(table["nearest_team"].iloc[30], -1.0)
        self.assertAlmostEqual(table["nearest_distance"].iloc[5], 1.0)
        # home before, away after: the nearest team changes by -2 across frame 20
        self.assertEqual(table["near_team_changed_2"].iloc[20], -2.0)
        self.assertEqual(table["near_team_changed_2"].iloc[5], 0.0)

    def test_ball_turn_is_the_angle_between_its_paths(self):
        px, py = still_players()
        bx = np.r_[np.arange(20.0), np.full(20, 19.0)]  # along x, then along y
        by = np.r_[np.full(20, 30.0), 30.0 + np.arange(1.0, 21.0)]
        table = feature_table(match(bx, by, px, py))
        self.assertAlmostEqual(table["ball_turn_2"].iloc[19], 90.0, places=6)
        self.assertAlmostEqual(table["ball_turn_2"].iloc[10], 0.0, places=6)
        self.assertTrue(np.isnan(table["ball_turn_2"].iloc[0]))  # nothing before the first frame

    def test_ball_speed_towards_the_away_goal(self):
        px, py = still_players()  # away players stand at larger x, so home attacks towards +x
        table = feature_table(match(10 + 0.4 * np.arange(N), np.full(N, 30.0), px, py))
        self.assertAlmostEqual(table["ball_attack_v_2"].iloc[10], 10.0)  # 0.4 m a frame

    def test_closest_approach_within_the_window(self):
        px, py = still_players()
        px[:, 0] = 50.0
        px[15, 0] = 40.0  # one home player steps right next to the ball for a single frame
        table = feature_table(match(np.full(N, 40.0), np.full(N, 30.0), px, py))
        self.assertAlmostEqual(table["min_dist_home_4"].iloc[12], 0.0)
        self.assertAlmostEqual(table["min_dist_home_4"].iloc[25], 10.0)

    def test_never_across_a_half_and_missing_outside(self):
        px, py = still_players()
        half = np.r_[np.zeros(5, int), np.ones(15, int), np.full(20, 2)]
        table = feature_table(match(np.linspace(10, 90, N), np.full(N, 30.0), px, py, half))
        for name in TOUCH_COLUMNS:
            self.assertTrue(table[name].iloc[:5].isna().all(), name)
        self.assertTrue(np.isnan(table["near_team_past_2"].iloc[20]))  # first frame of half 2
        self.assertFalse(np.isnan(table["near_team_past_2"].iloc[21]))


if __name__ == "__main__":
    unittest.main()

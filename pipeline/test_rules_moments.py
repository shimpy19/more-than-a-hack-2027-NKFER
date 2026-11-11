"""Tests for rules_moments.py on tiny invented matches. No competition data is involved.

    python -m unittest pipeline/test_rules_moments.py
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import rules_moments as rm  # noqa: E402
from starter.baseline import AWAY, HOME, NOBODY  # noqa: E402

PITCH = (100.0, 50.0)
FAR = 30.0


def match(n, half=None):
    """Arrays for an invented match of n frames: ball at the centre and still, 4 slow players,
    nobody near the ball."""
    return {
        "bx": np.full(n, 50.0), "by": np.full(n, 25.0), "ball_speed": np.zeros(n),
        "player_speed": np.full((n, 4), 0.5), "tracked": np.ones((n, 4), dtype=bool),
        "dist_home": np.full(n, FAR), "dist_away": np.full(n, FAR),
        "half": np.ones(n, dtype=np.int8) if half is None else np.asarray(half),
    }


def anchors_of(m, **thresholds):
    _, anchors = rm.moment_features(m["bx"], m["by"], m["ball_speed"], m["player_speed"],
                                    m["tracked"], m["dist_home"], m["dist_away"], m["half"],
                                    PITCH, thresholds)
    return anchors


class Helpers(unittest.TestCase):

    def test_run_length_matches_a_plain_loop(self):
        rng = np.random.default_rng(0)
        mask = rng.random(500) < 0.7
        half = np.repeat([0, 1, 0, 2], 125)
        expected, count = [], 0
        for i in range(len(mask)):
            if i > 0 and half[i] != half[i - 1]:
                count = 0
            count = count + 1 if mask[i] else 0
            expected.append(count)
        self.assertEqual(rm.run_length(mask, half).tolist(), expected)

    def test_past_mean_does_not_cross_a_half(self):
        values = np.array([10.0, 10.0, 1.0, 1.0])
        out = rm.past_mean(values, 2, np.array([1, 1, 2, 2]))
        self.assertTrue(np.isnan(out[2]))  # first frame of half 2 has nothing before it
        self.assertEqual(out[3], 1.0)

    def test_nearest_team(self):
        home = np.array([1.0, 3.0, np.nan, 5.0])
        away = np.array([2.0, 1.0, np.nan, 4.0])
        self.assertEqual(rm.nearest_team(home, away).tolist(), [HOME, AWAY, NOBODY, AWAY])
        self.assertEqual(rm.nearest_team(home, away, radius=2.0).tolist(),
                         [HOME, AWAY, NOBODY, NOBODY])

    def test_mean_player_speed_without_players_is_nan_and_quiet(self):
        speed = np.array([[1.0, 3.0], [np.nan, np.nan]])
        tracked = np.array([[True, True], [False, False]])
        with np.errstate(all="raise"):
            out = rm.mean_player_speed(speed, tracked)
        self.assertEqual(out[0], 2.0)
        self.assertTrue(np.isnan(out[1]))

    def test_label_codes_agree_with_model(self):
        try:
            import model
        except ImportError:
            self.skipTest("lightgbm not installed")
        self.assertEqual((rm.H, rm.A, rm.D, rm.NO_LABEL),
                         (model.H, model.A, model.D, model.NO_LABEL))


class RestartKick(unittest.TestCase):

    def test_still_then_kicked_gives_one_anchor_on_the_kick(self):
        m = match(100)
        m["ball_speed"][30:] = 8.0
        m["dist_home"][:] = 1.0
        anchors = anchors_of(m)
        self.assertEqual(anchors["frame_index"].tolist(), [30])
        self.assertEqual(anchors["team"].tolist(), [rm.H])
        self.assertTrue(anchors["confident"].iloc[0])
        self.assertEqual(anchors["type"].iloc[0], "kickoff")

    def test_still_too_briefly_gives_no_anchor(self):
        m = match(100)
        m["ball_speed"][:] = 8.0
        m["ball_speed"][20:30] = 0.0
        self.assertEqual(len(anchors_of(m)), 0)

    def test_running_players_make_it_not_confident(self):
        m = match(100)
        m["ball_speed"][30:] = 8.0
        m["dist_away"][:] = 1.0
        m["player_speed"][:] = 6.0
        anchors = anchors_of(m)
        self.assertEqual(anchors["team"].tolist(), [rm.A])
        self.assertFalse(anchors["confident"].iloc[0])

    def test_a_still_run_does_not_carry_over_a_half(self):
        m = match(60, half=[1] * 20 + [2] * 40)
        m["ball_speed"][30:] = 8.0  # still 20 frames in half 1, then only 10 in half 2
        self.assertEqual(len(anchors_of(m)), 0)

    def test_restart_types(self):
        t = rm.THRESHOLDS
        self.assertEqual(rm.restart_type(0.5, 0.5, PITCH, t), "corner")
        self.assertEqual(rm.restart_type(50.0, 25.0, PITCH, t), "kickoff")
        self.assertEqual(rm.restart_type(3.0, 25.0, PITCH, t), "goal_kick")
        self.assertEqual(rm.restart_type(40.0, 49.5, PITCH, t), "throw_in")
        self.assertEqual(rm.restart_type(70.0, 30.0, PITCH, t), "free_kick")


class BallOut(unittest.TestCase):

    def test_lines(self):
        bx = np.array([50.0, 50.0, -1.0, 101.0, np.nan, 50.0])
        by = np.array([25.0, -1.0, 25.0, 5.0, np.nan, 50.3])
        out = rm.ball_out_events(bx, by, PITCH)
        self.assertEqual(out["out_throw_in"].tolist(), [0, 1, 0, 0, 0, 0])  # 0.3 m is within margin
        self.assertEqual(out["out_goal"].tolist(), [0, 0, 1, 0, 0, 0])
        self.assertEqual(out["out_end_line"].tolist(), [0, 0, 0, 1, 0, 0])


class Turnover(unittest.TestCase):

    def path(self, second_team):
        """Ball along x, turns 90 degrees at frame 10 next to a home player, and again at frame 20
        next to a player of second_team."""
        m = match(31)
        m["bx"] = np.r_[np.arange(10, 21) * 1.0, np.full(10, 20.0), 20.0 + np.arange(1, 11)]
        m["by"] = np.r_[np.full(11, 10.0), 10.0 + np.arange(1, 11), np.full(10, 20.0)]
        m["dist_home"][10] = 1.0
        (m["dist_home"] if second_team == HOME else m["dist_away"])[20] = 1.0
        return rm.turnover_candidates(m["bx"], m["by"], m["dist_home"], m["dist_away"], m["half"])

    def test_touch_by_the_other_team_is_a_candidate(self):
        out = self.path(AWAY)
        self.assertEqual(np.flatnonzero(out["touch_home"]).tolist(), [10])
        self.assertEqual(np.flatnonzero(out["touch_away"]).tolist(), [20])
        self.assertEqual(np.flatnonzero(out["turnover_candidate"]).tolist(), [20])

    def test_touch_by_the_same_team_is_not(self):
        self.assertEqual(self.path(HOME)["turnover_candidate"].sum(), 0)

    def test_a_still_ball_has_no_touches(self):
        m = match(50)
        m["bx"] += np.random.default_rng(1).normal(0, 0.001, 50)  # noise below MIN_BALL_STEP_M
        m["dist_home"][:] = 1.0
        out = rm.turnover_candidates(m["bx"], m["by"], m["dist_home"], m["dist_away"], m["half"])
        self.assertEqual(out["touch_home"].sum(), 0)

    def test_previous_touch_does_not_carry_over_a_half(self):
        m = match(31, half=[1] * 15 + [2] * 16)
        m["bx"] = np.r_[np.arange(10, 21) * 1.0, np.full(10, 20.0), 20.0 + np.arange(1, 11)]
        m["by"] = np.r_[np.full(11, 10.0), 10.0 + np.arange(1, 11), np.full(10, 20.0)]
        m["dist_home"][10] = 1.0
        m["dist_away"][20] = 1.0
        out = rm.turnover_candidates(m["bx"], m["by"], m["dist_home"], m["dist_away"], m["half"])
        self.assertEqual(out["touch_away"].sum(), 1)
        self.assertEqual(out["turnover_candidate"].sum(), 0)


class Measuring(unittest.TestCase):

    def test_evaluate_anchors(self):
        H, A, D = rm.H, rm.A, rm.D
        labels = np.array([D] * 10 + [H] * 20 + [D] * 10 + [A] * 20)  # restarts at 10 and 40
        anchors = pd.DataFrame({"frame_index": [12, 25, 39], "team": [H, A, H]})
        result = rm.evaluate_anchors(anchors, labels, tolerance=5)
        self.assertEqual(result["true_restarts"], 2)
        self.assertEqual(result["hits"], 2)
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["team_accuracy"], 0.5)  # 12 -> H right, 39 -> H wrong (A)
        self.assertEqual(result["mean_offset_frames"], 0.5)  # +2 and -1

    def test_each_true_restart_is_matched_once(self):
        labels = np.array([rm.D] * 10 + [rm.H] * 10)
        anchors = pd.DataFrame({"frame_index": [9, 10, 11], "team": [rm.H] * 3})
        result = rm.evaluate_anchors(anchors, labels)
        self.assertEqual(result["hits"], 1)
        self.assertEqual(result["mean_offset_frames"], 0.0)  # the exact one wins

    def test_unlabelled_frames_are_not_restarts(self):
        labels = np.array([rm.D, rm.NO_LABEL, rm.H, rm.D, rm.A])
        self.assertEqual(rm.true_restarts(labels).tolist(), [4])

    def test_moment_columns_are_all_built(self):
        m = match(40)
        columns, _ = rm.moment_features(m["bx"], m["by"], m["ball_speed"], m["player_speed"],
                                        m["tracked"], m["dist_home"], m["dist_away"], m["half"],
                                        PITCH)
        self.assertEqual(tuple(columns), rm.MOMENT_COLUMNS)
        self.assertTrue(all(len(v) == 40 for v in columns.values()))


if __name__ == "__main__":
    unittest.main()

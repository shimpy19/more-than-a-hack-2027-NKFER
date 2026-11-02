"""Tests for features.py on a tiny invented match. No competition data is involved.

    python -m unittest pipeline/test_features.py
"""

import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features  # noqa: E402

PITCH = (100.0, 50.0)
FPS = 25
POSITION_PLACES = 3  # positions are stored as float32, so metres are only exact to about 1e-5
TEAM = np.array([features.HOME, features.HOME, features.AWAY])

# Five frames, three slots: two home players and one away.
RAW = """\
1:0.10,0.20,4.0;0.50,0.50,2.0;0.60,0.50,6.0:0.50,0.40,10.0
2:0.10,0.20,4.0;0.50,0.50,2.0;NaN,NaN,NaN:0.60,0.40,10.0
3:NaN,NaN,NaN;NaN,NaN,NaN;0.60,0.50,1.0:NaN,NaN,NaN
4:0.10,0.20,4.0;0.50,0.50,2.0;0.60,0.50,6.0:NaN,NaN,NaN
5:0.10,0.20,4.0;0.50,0.50,2.0;0.60,0.50,6.0:1.10,0.40,4.0
"""


def build(raw=RAW, events=(), blocks=((1, 1, 3),)):
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "raw.txt")
        with open(path, "w") as handle:
            handle.write(raw)
        tracking = features.read_tracking(path, 5, 3)
    return features.build_features(*tracking, TEAM, PITCH, FPS, blocks, list(events))


class TrackingFeatures(unittest.TestCase):

    def setUp(self):
        self.table = build()

    def test_frame_id_counts_from_one_and_frame_from_zero(self):
        self.assertEqual(self.table["frame"].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(self.table["ball_x_m"][0], 50.0)

    def test_positions_are_scaled_per_axis(self):
        self.assertEqual(self.table["ball_y_m"][0], 20.0)
        self.assertAlmostEqual(self.table["dist_home"][0], 5.0, places=POSITION_PLACES)
        self.assertAlmostEqual(self.table["dist_away"][0], np.hypot(10.0, 5.0), places=POSITION_PLACES)

    def test_missing_team_or_ball_gives_missing_distance(self):
        self.assertTrue(np.isnan(self.table["dist_away"][1]))
        self.assertTrue(np.isnan(self.table["dist_home"][2]))
        self.assertTrue(np.isnan(self.table["dist_away"][2]))

    def test_player_counts_and_speeds_ignore_missing_players(self):
        self.assertEqual(self.table["n_home"].tolist(), [2, 2, 0, 2, 2])
        self.assertEqual(self.table["n_away"].tolist(), [1, 0, 1, 1, 1])
        self.assertAlmostEqual(self.table["home_speed"][1], 3.0)
        self.assertTrue(np.isnan(self.table["home_speed"][2]))
        self.assertAlmostEqual(self.table["mean_speed"][0], 4.0)

    def test_ball_missing_time_grows_then_resets(self):
        self.assertEqual(self.table["ball_visible"].tolist(), [1, 1, 0, 0, 1])
        expected = [0, 0, 1 / FPS, 2 / FPS, 0]
        self.assertTrue(np.allclose(self.table["ball_missing_s"], expected))

    def test_ball_outside_pitch(self):
        self.assertEqual(self.table["ball_outside_pitch"].tolist(), [0, 0, 0, 0, 1])

    def test_speed_change_is_missing_across_a_gap(self):
        self.assertEqual(self.table["ball_speed_change"][1], 0.0)
        self.assertTrue(np.isnan(self.table["ball_speed_change"][4]))

    def test_team_shape(self):
        self.assertAlmostEqual(self.table["home_centre_x"][0], 30.0, places=POSITION_PLACES)
        self.assertAlmostEqual(self.table["home_spread"][0], np.hypot(20.0, 7.5), places=POSITION_PLACES)
        self.assertAlmostEqual(self.table["away_spread"][0], 0.0, places=POSITION_PLACES)

    def test_half_column(self):
        self.assertEqual(self.table["half"].tolist(), [0, 1, 1, 1, 0])


class Helpers(unittest.TestCase):

    def test_ball_turn(self):
        x = np.array([0.0, 1.0, 2.0, 2.0, np.nan, 3.0])
        y = np.array([0.0, 0.0, 0.0, 1.0, np.nan, 1.0])
        turn = features.ball_turn_degrees(x, y)
        self.assertAlmostEqual(turn[1], 0.0)
        self.assertAlmostEqual(turn[2], 90.0)
        self.assertTrue(np.isnan(turn[3]))
        self.assertTrue(np.isnan(turn[4]))

    def test_stationary_ball_has_no_direction(self):
        x, y = np.array([1.0, 1.0, 2.0]), np.zeros(3)
        self.assertTrue(np.isnan(features.ball_turn_degrees(x, y)[1]))

    def test_smoothing_ignores_missing_values(self):
        smooth = features.centred_nanmean(np.array([1.0, np.nan, 3.0, np.nan, np.nan, np.nan]), 1)
        self.assertEqual(smooth[1], 2.0)
        self.assertEqual(smooth[3], 3.0)
        self.assertTrue(np.isnan(smooth[4]))

    def test_gap_at_the_start_counts_from_frame_zero(self):
        missing = features.ball_missing_seconds(np.array([False, False, True]), 1)
        self.assertEqual(missing.tolist(), [1.0, 2.0, 0.0])


class Tags(unittest.TestCase):

    def columns(self, *events):
        return features.tag_columns(list(events), 5)

    def test_attacking_and_defending_are_kept_apart(self):
        result = self.columns(("build up", "barcelona", "", 0, 2),
                              ("defending in middle third", "atletic club", "", 1, 3))
        self.assertEqual(result["tag_home_attacking"].tolist(), [1, 1, 0, 0, 0])
        self.assertEqual(result["tag_away_defending"].tolist(), [0, 1, 1, 0, 0])
        self.assertEqual(result["tag_away_attacking"].sum(), 0)

    def test_each_code_has_its_own_column_per_team(self):
        result = self.columns(("build up", "barcelona", "", 0, 2),
                              ("long ball", "barcelona", "", 2, 3),
                              ("long ball", "atletic club", "", 4, 5))
        self.assertEqual(result["tag_home_build_up"].tolist(), [1, 1, 0, 0, 0])
        self.assertEqual(result["tag_home_long_ball"].tolist(), [0, 0, 1, 0, 0])
        self.assertEqual(result["tag_away_long_ball"].tolist(), [0, 0, 0, 0, 1])
        self.assertEqual(result["tag_home_attacking"].tolist(), [1, 1, 1, 0, 0])

    def test_goals_are_neither_attacking_nor_defending(self):
        result = self.columns(("goals", "barcelona", "", 1, 2))
        self.assertEqual(result["tag_home_goals"].tolist(), [0, 1, 0, 0, 0])
        self.assertEqual(result["tag_home_attacking"].sum(), 0)

    def test_set_piece_without_a_team_still_counts(self):
        result = self.columns(("set pieces", "n/a", "throw-in", 3, 99))
        self.assertEqual(result["tag_set_piece"].tolist(), [0, 0, 0, 1, 1])
        self.assertEqual(result["tag_set_piece_throw_in"].tolist(), [0, 0, 0, 1, 1])
        self.assertEqual(result["tag_home_set_piece"].sum(), 0)
        self.assertEqual(result["tag_away_set_piece"].sum(), 0)

    def test_set_piece_types_and_team_have_their_own_columns(self):
        result = self.columns(("set pieces", "atletic club", "kick off (start)", 0, 1))
        self.assertEqual(result["tag_set_piece_kick_off_start"].tolist(), [1, 0, 0, 0, 0])
        self.assertEqual(result["tag_away_set_piece"].tolist(), [1, 0, 0, 0, 0])
        self.assertEqual(result["tag_set_piece_corner_kick"].sum(), 0)

    def test_every_tag_produces_a_known_column(self):
        names = features.tag_column_names()
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), 39)

    def test_unknown_tags_are_errors(self):
        with self.assertRaises(ValueError):
            self.columns(("tiki taka", "barcelona", "", 0, 1))
        with self.assertRaises(ValueError):
            self.columns(("set pieces", "barcelona", "penalty", 0, 1))


class Reading(unittest.TestCase):

    def read(self, raw, n_frames=1, n_slots=1):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "raw.txt")
            with open(path, "w") as handle:
                handle.write(raw)
            return features.read_tracking(path, n_frames, n_slots)

    def test_rejects_a_repeated_frame(self):
        with self.assertRaises(ValueError):
            self.read("1:0,0,0:0,0,0\n1:0,0,0:0,0,0\n", n_frames=2)

    def test_rejects_a_missing_frame(self):
        with self.assertRaises(ValueError):
            self.read("1:0,0,0:0,0,0\n", n_frames=2)

    def test_rejects_the_wrong_number_of_players(self):
        with self.assertRaises(ValueError):
            self.read("1:0,0,0;0,0,0:0,0,0\n")

    def test_finds_a_file_in_any_subfolder_and_ignores_companions(self):
        with tempfile.TemporaryDirectory() as folder:
            nested = os.path.join(folder, "Dataset 8", "tactical")
            os.makedirs(nested)
            for name in ("a.txt", "._a.txt"):
                open(os.path.join(nested, name), "w").close()
            self.assertEqual(features.find_asset(folder, "a.txt"), os.path.join(nested, "a.txt"))


if __name__ == '__main__':
    unittest.main()

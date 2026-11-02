"""Tests for clean.py on tiny invented matches. No competition data is involved.

    python -m unittest pipeline/test_clean.py
"""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import clean  # noqa: E402
from starter.baseline import AWAY, HOME  # noqa: E402

PITCH = (100.0, 50.0)
FPS = 25
PLACES = 3  # outputs are float32


def walk(n, start=10.0, metres_per_frame=0.1):
    """A player or ball moving in a straight line along x, at 2.5 m/s by default."""
    return start + metres_per_frame * np.arange(n, dtype=np.float64), np.full(n, 20.0)


class Helpers(unittest.TestCase):

    def test_runs_of(self):
        mask = np.array([1, 1, 0, 1, 0, 0, 1], dtype=bool)
        self.assertEqual(clean.runs_of(mask), [(0, 2), (3, 4), (6, 7)])

    def test_segments_of(self):
        half = np.array([0, 0, 1, 1, 0, 2, 2, 0])
        self.assertEqual(clean.segments_of(half), [(0, 2), (2, 4), (4, 5), (5, 7), (7, 8)])


class Pieces(unittest.TestCase):

    def test_a_jump_starts_a_new_piece(self):
        px = np.array([1.0, 1.1, 50.0, 50.1, np.nan, 50.2])
        py = np.zeros(6)
        piece, cuts = clean.track_pieces(px, py, clean.PLAYER_MAX_SPEED, FPS)
        self.assertEqual(piece.tolist(), [0, 0, 1, 1, -1, 2])
        self.assertEqual(cuts, 1)

    def test_a_gap_starts_a_new_piece_but_is_not_a_cut(self):
        px = np.array([1.0, np.nan, 1.0])
        piece, cuts = clean.track_pieces(px, np.zeros(3), clean.PLAYER_MAX_SPEED, FPS)
        self.assertEqual(piece.tolist(), [0, -1, 1])
        self.assertEqual(cuts, 0)


class Velocity(unittest.TestCase):

    def test_constant_speed_everywhere_including_the_ends(self):
        px, py = walk(10)
        vx, vy = clean.velocity(px, py, np.zeros(10, dtype=int), 2, FPS)
        self.assertTrue(np.allclose(vx, 2.5))
        self.assertTrue(np.allclose(vy, 0.0))

    def test_never_measures_across_a_piece_boundary(self):
        px = np.array([0.0, 0.1, 0.2, 90.0, 90.1, 90.2])
        piece = np.array([0, 0, 0, 1, 1, 1])
        vx, _ = clean.velocity(px, np.zeros(6), piece, 2, FPS)
        self.assertTrue(np.allclose(vx, 2.5))

    def test_a_single_frame_piece_has_no_speed(self):
        vx, _ = clean.velocity(np.array([np.nan, 5.0, np.nan]), np.zeros(3),
                               np.array([-1, 0, -1]), 2, FPS)
        self.assertTrue(np.isnan(vx).all())


class Players(unittest.TestCase):

    def test_off_pitch_keeps_a_throw_in_and_removes_the_rest(self):
        px = np.array([[50.0], [50.0], [50.0]])
        py = np.array([[20.0], [-4.0], [-6.0]])
        off = clean.remove_off_pitch(px, py, 100.0, 50.0, 5.0)
        self.assertEqual(off[:, 0].tolist(), [False, False, True])
        self.assertTrue(np.isnan(px[2, 0]))

    def duplicates(self, separation, n=60):
        ax, ay = walk(n)
        px = np.column_stack((ax, ax + separation))
        py = np.column_stack((ay, ay))
        removed, stretches = clean.remove_duplicates(px, py, np.array([HOME, HOME]),
                                                     clean.DUPLICATE_DISTANCE_M,
                                                     clean.DUPLICATE_MIN_FRAMES)
        return px, removed, stretches

    def test_a_lasting_duplicate_loses_one_track(self):
        px, removed, stretches = self.duplicates(0.2)
        self.assertEqual(stretches, 1)
        self.assertEqual(removed.sum(), 60)
        self.assertFalse(np.isnan(px[:, 0]).any())
        self.assertTrue(np.isnan(px[:, 1]).all())

    def test_players_apart_are_left_alone(self):
        _, removed, stretches = self.duplicates(3.0)
        self.assertEqual((stretches, int(removed.sum())), (0, 0))

    def test_a_brief_pass_is_not_a_duplicate(self):
        _, removed, stretches = self.duplicates(0.2, n=clean.DUPLICATE_MIN_FRAMES - 1)
        self.assertEqual((stretches, int(removed.sum())), (0, 0))

    def test_opponents_are_never_duplicates(self):
        ax, ay = walk(60)
        px, py = np.column_stack((ax, ax)), np.column_stack((ay, ay))
        removed, _ = clean.remove_duplicates(px, py, np.array([HOME, AWAY]), 0.7, 25)
        self.assertEqual(removed.sum(), 0)

    def test_the_shorter_run_gives_way(self):
        ax, ay = walk(60)
        px, py = np.column_stack((ax, ax)), np.column_stack((ay, ay))
        px[:20, 0] = np.nan
        py[:20, 0] = np.nan
        clean.remove_duplicates(px, py, np.array([HOME, HOME]), 0.7, 25)
        self.assertTrue(np.isnan(px[20:, 0]).all())
        self.assertFalse(np.isnan(px[:, 1]).any())

    def test_short_pieces_go_and_switches_are_counted(self):
        n = 100
        ax, ay = walk(n)
        ax[50:] += 60.0  # the slot jumps to someone else
        short_x = np.full(n, np.nan)
        short_x[:10] = 40.0
        px = np.column_stack((ax, short_x))
        py = np.column_stack((ay, np.full(n, 40.0)))
        speed, report, masks = clean.clean_players(px, py, np.array([HOME, AWAY]), PITCH, FPS)
        self.assertEqual(report["short_pieces_whole_recording"], 1)
        self.assertTrue(np.isnan(px[:, 1]).all())
        self.assertEqual(int(masks["cuts"].sum()), 1)
        self.assertTrue(np.allclose(speed[:, 0], 2.5))


class Ball(unittest.TestCase):

    def test_fast_ball_is_flagged_and_kept(self):
        px, py = walk(10, metres_per_frame=2.0)  # 50 m/s
        half = np.ones(10, dtype=int)
        columns = clean.clean_ball(px / PITCH[0], py / PITCH[1], half, PITCH, FPS)
        self.assertEqual(columns["ball_suspect"].sum(), 10)
        self.assertFalse(np.isnan(columns["ball_x_m"]).any())

    def test_distance_past_each_line(self):
        ball_x = np.array([0.5, 1.1, -0.1, 0.5, np.nan])
        ball_y = np.array([0.5, 0.5, 0.5, 1.2, np.nan])
        columns = clean.clean_ball(ball_x, ball_y, np.ones(5, dtype=int), PITCH, FPS)
        self.assertAlmostEqual(columns["ball_past_end_m"][1], 10.0)
        self.assertAlmostEqual(columns["ball_past_end_m"][2], 10.0)
        self.assertAlmostEqual(columns["ball_past_side_m"][3], 10.0)
        self.assertEqual(columns["ball_past_end_m"][0], 0.0)
        self.assertTrue(np.isnan(columns["ball_past_end_m"][4]))

    def test_gaps_are_described_and_not_filled(self):
        px = np.array([1.0, np.nan, np.nan, 4.0])
        columns = clean.ball_gap_columns(px, px * 2, np.ones(4, dtype=int), 1)
        self.assertEqual(columns["ball_missing_s"].tolist(), [0, 1, 2, 0])
        self.assertEqual(columns["ball_until_s"].tolist(), [0, 2, 1, 0])
        self.assertEqual(columns["ball_last_x_m"][2], 1.0)
        self.assertEqual(columns["ball_next_y_m"][1], 8.0)
        self.assertTrue(np.isnan(columns["ball_last_x_m"][0]))

    def test_gap_columns_stop_at_a_half_boundary(self):
        px = np.array([1.0, np.nan, 9.0])
        columns = clean.ball_gap_columns(px, px, np.array([1, 1, 0]), 1)
        self.assertTrue(np.isnan(columns["ball_next_x_m"][1]))
        self.assertEqual(columns["ball_last_x_m"][1], 1.0)


class Quality(unittest.TestCase):

    def test_tracking_quality(self):
        n_players = np.array([22, 14, 22, 22])
        error = np.array([0.2, 0.2, 3.0, 0.2])
        valid = np.array([1, 1, 1, 0])
        self.assertEqual(clean.tracking_quality(n_players, error, valid).tolist(), [1, 0, 0, 0])


class Tags(unittest.TestCase):

    BLOCKS = [(1, 100, 199), (2, 300, 399)]

    def test_entries_outside_the_halves_are_dropped(self):
        events = [("build up", "barcelona", "", 10, 50), ("long ball", "barcelona", "", 220, 250),
                  ("build up", "barcelona", "", 450, 460)]
        self.assertEqual(clean.clean_tags(events, self.BLOCKS), ([], 3, 0))

    def test_entries_crossing_a_boundary_are_cut_to_the_half(self):
        kept, dropped, clipped = clean.clean_tags([("set pieces", "n/a", "kick off (start)", 90, 120),
                                                   ("build up", "barcelona", "", 190, 230)],
                                                  self.BLOCKS)
        self.assertEqual(kept, [("set pieces", "n/a", "kick off (start)", 100, 120, 1),
                                ("build up", "barcelona", "", 190, 200, 1)])
        self.assertEqual((dropped, clipped), (0, 2))

    def test_entry_ending_where_a_half_starts_is_outside_it(self):
        self.assertEqual(clean.clean_tags([("build up", "x", "", 90, 100)], self.BLOCKS)[1], 1)

    def test_entry_inside_a_half_is_unchanged_and_no_team_stays_no_team(self):
        kept, _, clipped = clean.clean_tags([("set pieces", "n/a", "throw-in", 310, 320)], self.BLOCKS)
        self.assertEqual(kept, [("set pieces", "n/a", "throw-in", 310, 320, 2)])
        self.assertEqual(clipped, 0)


class WholeMatch(unittest.TestCase):

    def test_clean_match_shapes_and_counts(self):
        n = 100
        team = np.array([HOME] * 12 + [AWAY] * 3)
        x = np.full((n, 15), 0.2, dtype=np.float32) + np.arange(15, dtype=np.float32) * 0.05
        y = np.full((n, 15), 0.5, dtype=np.float32)
        ball_x = np.full(n, 0.5)
        ball_y = np.full(n, 0.5)
        ball_x[40:60] = np.nan
        ball_y[40:60] = np.nan
        error = np.full(n, 0.2)
        columns, report = clean.clean_match(x, y, ball_x, ball_y, team, PITCH, FPS, [(1, 10, 89)],
                                            error, np.ones(n, dtype=int))
        self.assertEqual(columns["player_x_m"].shape, (n, 15))
        self.assertEqual(columns["half"][10], 1)
        self.assertEqual(columns["half"][90], 0)
        self.assertEqual(columns["surplus_home"].max(), 1)
        self.assertEqual(columns["surplus_away"].max(), 0)
        self.assertTrue((columns["n_home"] == 12).all())
        self.assertEqual(report["graded_frames"], 80)
        self.assertEqual(report["ball_frames_visible"], 60)
        self.assertEqual(report["frames_tracking_not_ok"], 0)
        self.assertAlmostEqual(float(columns["player_x_m"][0, 0]), 20.0, places=PLACES)


if __name__ == '__main__':
    unittest.main()

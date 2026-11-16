"""Tests for the ball damage in damage.py on tiny invented matches. No competition data is involved.

    python -m unittest pipeline/test_damage.py
"""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import damage  # noqa: E402


class Stretches(unittest.TestCase):

    def test_split_by_state_and_by_half(self):
        live = np.array([1, 1, 0, 0, 1, 1, 1, 0], dtype=bool)
        half = np.array([0, 1, 1, 1, 1, 2, 2, 2])
        self.assertEqual(damage.state_stretches(live, half),
                         [(1, 2, True), (2, 4, False), (4, 5, True), (5, 7, True), (7, 8, False)])


class BallMask(unittest.TestCase):

    def library(self):
        # live: seen throughout; dead: seen for the first 3 of 10 frames, then missing
        return {True: [np.ones(10, dtype=bool)],
                False: [np.r_[np.ones(3, dtype=bool), np.zeros(7, dtype=bool)]]}

    def test_copies_the_pattern_of_the_same_state_stretched_to_fit(self):
        live = np.r_[np.ones(20, dtype=bool), np.zeros(20, dtype=bool)]
        half = np.ones(40, dtype=int)
        seen = damage.ball_mask(live, half, self.library(), np.random.default_rng(1))
        self.assertTrue(seen[:20].all())                       # live: seen
        self.assertEqual(seen[20:].tolist(), [True] * 6 + [False] * 14)  # dead: 3/10 -> 6/20

    def test_outside_the_halves_the_ball_is_left_as_recorded(self):
        live = np.zeros(10, dtype=bool)
        half = np.r_[np.zeros(5, dtype=int), np.ones(5, dtype=int)]
        seen = damage.ball_mask(live, half, self.library(), np.random.default_rng(1))
        self.assertTrue(seen[:5].all())

    def test_picks_among_the_closest_lengths(self):
        library = {True: [np.ones(n, dtype=bool) for n in (5, 10, 1000)],
                   False: [np.zeros(n, dtype=bool) for n in (5, 10, 1000)]}
        live = np.ones(10, dtype=bool)
        # with only one allowed, the 10-frame stretch (exact length) must be the one copied
        old = damage.BALL_CLOSEST
        damage.BALL_CLOSEST = 1
        try:
            seen = damage.ball_mask(live, np.ones(10, dtype=int), library,
                                    np.random.default_rng(1))
        finally:
            damage.BALL_CLOSEST = old
        self.assertTrue(seen.all())


class DamageWithBall(unittest.TestCase):

    def test_ball_hidden_where_missing_and_reproducible(self):
        n = 30
        x = np.tile(np.linspace(0.1, 0.9, 22), (n, 1))
        y = np.full((n, 22), 0.5)
        ball = np.full(n, 0.5)
        live = np.r_[np.ones(15, dtype=bool), np.zeros(15, dtype=bool)]
        half = np.ones(n, dtype=int)
        library = {True: [np.ones(15, dtype=bool)], False: [np.zeros(15, dtype=bool)]}
        profile = (np.full(100, 2, dtype=np.int16), np.r_[np.ones(50, bool), np.zeros(50, bool)])
        first = damage.damage_with_ball(x, y, ball, ball, live, half, library, profile, seed=1)
        again = damage.damage_with_ball(x, y, ball, ball, live, half, library, profile, seed=1)
        _, _, ball_x, ball_y = first
        self.assertFalse(np.isnan(ball_x[:15]).any())
        self.assertTrue(np.isnan(ball_x[15:]).all() and np.isnan(ball_y[15:]).all())
        self.assertTrue(((~np.isnan(first[0])).sum(axis=1) == 20).all())  # 2 players hidden
        for a, b in zip(first, again):
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()

"""Viterbi: turn the model's per-frame H / A / D probabilities into a steady label sequence.

The model scores every frame on its own, so its most likely label flickers (H H A H H), and each
flicker breaks one possession into pieces, which spellF1 punishes hard. Viterbi instead picks the
one sequence of labels that best fits all the probabilities at once, where every change of label
has a cost. A short dip in confidence is then not worth two changes, while a real change of
possession, with the model confident for a while, is.

Cost of a sequence = sum over frames of -log(probability of its label) + the cost of each change.
Viterbi finds the cheapest sequence exactly, in one pass forward and one back.

Two costs, both in the same units as -log(probability):
  team   a change between H and A
  dead   a change into or out of D
They are chosen on the external matches only (leave one match out), never on the provided game.
Each half is decoded on its own, so nothing carries over halftime.
"""

import numpy as np

H, A, D = 0, 1, 2
N_STATES = 3

# The best pair over the 9 external matches on the v3 bench (bench_viterbi.py; 0.637 mean spellF1,
# with 8/30, 8/60 and 15/15 within 0.001). The old 30 / 60 scores 0.613 there.
SWITCH_COSTS = {"team": 15.0, "dead": 30.0}

FLOOR = 1e-6


def switch_matrix(team, dead):
    """3 x 3 cost of going from one label (row) to the next (column)."""
    costs = np.full((N_STATES, N_STATES), float(dead))
    costs[H, A] = costs[A, H] = float(team)
    np.fill_diagonal(costs, 0.0)
    return costs


def decode(scores, costs):
    """Cheapest label sequence for one unbroken stretch. scores: frames x 3 probabilities."""
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    cost_of = (-np.log(np.maximum(scores, FLOOR))).tolist()
    switch = costs.tolist()
    back = np.zeros((n, N_STATES), dtype=np.int8)

    # The inner loop is plain Python on three numbers, which is far faster than numpy on arrays
    # this small. best[s] = cheapest cost of any sequence that ends in label s at this frame.
    best = list(cost_of[0])
    for t in range(1, n):
        frame, came_from, new_best = cost_of[t], back[t], [0.0, 0.0, 0.0]
        for s in range(N_STATES):
            options = [best[0] + switch[0][s], best[1] + switch[1][s], best[2] + switch[2][s]]
            previous = options.index(min(options))
            came_from[s] = previous
            new_best[s] = options[previous] + frame[s]
        best = new_best

    path = np.empty(n, dtype=np.int8)
    path[-1] = best.index(min(best))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_match(scores, half, team=SWITCH_COSTS["team"], dead=SWITCH_COSTS["dead"]):
    """Labels for a whole match: each half decoded on its own, D outside the halves.

    Frames outside the halves are not graded; D there is only a filler.
    """
    if len(scores) != len(half):
        raise ValueError("scores and half differ in length: {} vs {}".format(len(scores), len(half)))
    costs = switch_matrix(team, dead)
    labels = np.full(len(scores), D, dtype=np.int8)
    for number in np.unique(half[half > 0]):
        rows = np.flatnonzero(half == number)
        if rows[-1] - rows[0] + 1 != len(rows):
            raise ValueError("half {} is not one unbroken stretch of frames".format(number))
        labels[rows[0]:rows[-1] + 1] = decode(scores[rows[0]:rows[-1] + 1], costs)
    return labels


def decode_varying(scores, into_h, into_a, dead):
    """As decode, but the cost of an H <-> A switch differs per frame: into_h[t] is the cost of
    A -> H into frame t, into_a[t] of H -> A. Into or out of D always costs `dead`."""
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    cost_of = (-np.log(np.maximum(scores, FLOOR))).tolist()
    into_h, into_a, dead = np.asarray(into_h).tolist(), np.asarray(into_a).tolist(), float(dead)
    back = np.zeros((n, N_STATES), dtype=np.int8)
    best = list(cost_of[0])
    for t in range(1, n):
        frame, came, to_h, to_a = cost_of[t], back[t], into_h[t], into_a[t]
        b_h, b_a, b_d = best
        options = (b_h, b_a + to_h, b_d + dead)
        p = options.index(min(options))
        came[H], new_h = p, options[p] + frame[H]
        options = (b_h + to_a, b_a, b_d + dead)
        p = options.index(min(options))
        came[A], new_a = p, options[p] + frame[A]
        options = (b_h + dead, b_a + dead, b_d)
        p = options.index(min(options))
        came[D], new_d = p, options[p] + frame[D]
        best = [new_h, new_a, new_d]
    path = np.empty(n, dtype=np.int8)
    path[-1] = best.index(min(best))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_match_varying(scores, half, into_h, into_a, dead):
    """decode_match with the H <-> A switch cost given per frame (see decode_varying)."""
    if not len(scores) == len(half) == len(into_h) == len(into_a):
        raise ValueError("scores, half and switch costs differ in length")
    labels = np.full(len(scores), D, dtype=np.int8)
    for number in np.unique(half[half > 0]):
        rows = np.flatnonzero(half == number)
        if rows[-1] - rows[0] + 1 != len(rows):
            raise ValueError("half {} is not one unbroken stretch of frames".format(number))
        s, e = rows[0], rows[-1] + 1
        labels[s:e] = decode_varying(scores[s:e], into_h[s:e], into_a[s:e], dead)
    return labels

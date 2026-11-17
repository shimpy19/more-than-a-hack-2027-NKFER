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

SWITCH_COSTS = {"team": 8.0, "dead": 30.0}

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

"""Possession-change moments: frame-level signals for where possession or play changes.

Three kinds of moment, all computed from clean.clean_match's columns only (metres, recalculated
speeds), so they come out the same way on the external matches and on the provided game:

  ball out       the ball is past a line by more than a margin (throw-in, goal, goal line)
  restart kick   the ball was still for a while and is then struck, with the players slow
                 before it: the end of a dead ball and the first frame of the next possession.
                 Fouls and offsides are never seen directly; they show up as a restart.
  turnover       a touch (sharp change of ball direction with a player close) by the other team
  candidate      than the previous touch. Deflections look the same, so this is evidence only.

The model gets them as features (moment_features -> feature columns). Restart kicks are also
returned as a table of anchors, and the confident ones as the column restart_confident, so the
Viterbi step can read them from the training table like any other column.

Every threshold is in THRESHOLDS and is chosen on the external matches only (measure_moments.py),
never on the provided game. Each change goes in the decisions log with its reason.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from starter.baseline import AWAY, HOME, NOBODY  # noqa: E402

# Label codes, the same as model.py (H, A, D, NO_LABEL). Not imported from model.py so this module
# loads without LightGBM; test_rules_moments.py checks they agree.
H, A, D, NO_LABEL = 0, 1, 2, -1

FPS = 25
GOAL_WIDTH_M = 7.32
GOAL_AREA_DEPTH_M, GOAL_AREA_WIDTH_M = 5.5, 18.32
MIN_BALL_STEP_M = 0.01  # same as features.py: below this a step has no meaningful direction

# Placeholders until measured on the external matches (measure_moments.py --sweep).
THRESHOLDS = {
    "still_speed": 0.5,       # m/s, below this the ball is still
    "still_min_s": 1.0,       # s the ball must be still before a restart kick
    "kick_speed": 4.0,        # m/s, the ball has been struck
    "kick_lookahead": 10,     # frames after the end of the stillness to look for kick_speed
    "slow_players": 1.5,      # m/s, mean speed of all tracked players counted as slow
    "slow_window_s": 2.0,     # s before the kick over which the players' speed is averaged
    "out_margin_m": 0.5,      # m past a line, for tracking noise
    "touch_turn_deg": 45.0,   # change of ball direction counted as a touch
    "touch_radius_m": 2.0,    # a player this close to the ball can have touched it
    # where a restart is taken, from the ball's position at the kick
    "near_side_m": 1.5, "near_end_m": 2.0, "centre_spot_m": 1.5,
}

# Columns moment_features adds to the feature table, in a fixed order.
MOMENT_COLUMNS = (
    "out_throw_in", "out_goal", "out_end_line",
    "ball_still_s", "restart_kick", "restart_kick_home", "restart_kick_away",
    "players_slow_before", "restart_confident",
    "touch_home", "touch_away", "turnover_candidate",
)
RESTART_TYPES = ("corner", "kickoff", "goal_kick", "throw_in", "free_kick")


# ======================================================================
# helpers: everything works inside one stretch of `half`, never across
# ======================================================================

def segments(half):
    """[(start, end)] of the stretches with one value of half, end exclusive."""
    half = np.asarray(half)
    edges = np.concatenate(([0], np.flatnonzero(np.diff(half)) + 1, [len(half)]))
    return list(zip(edges[:-1].tolist(), edges[1:].tolist()))


def run_length(mask, half):
    """For each frame, how many frames in a row mask has been True, ending at this frame.
    Restarts at every change of half."""
    mask = np.asarray(mask, dtype=bool)
    half = np.asarray(half)
    index = np.arange(len(mask))
    starts = np.r_[True, half[1:] != half[:-1]] if len(mask) else np.zeros(0, dtype=bool)
    # the last frame before the current run: a False frame or the frame before a new half
    last_break = np.maximum.accumulate(np.where(~mask, index, np.where(starts, index - 1, -1)))
    return np.where(mask, index - last_break, 0).astype(np.int32)


def future_max(values, frames, half):
    """For each frame, the max of values over this frame and the next frames-1, within a half."""
    out = np.empty(len(values))
    for start, end in segments(half):
        reverse = pd.Series(values[start:end][::-1])
        out[start:end] = reverse.rolling(frames, min_periods=1).max().to_numpy()[::-1]
    return out


def past_mean(values, frames, half):
    """For each frame, the mean of values over the frames before it (not itself), within a half.
    NaN values are skipped; NaN where there is nothing to average."""
    out = np.empty(len(values))
    for start, end in segments(half):
        series = pd.Series(values[start:end]).shift(1)
        out[start:end] = series.rolling(frames, min_periods=1).mean().to_numpy()
    return out


def nearest_team(dist_home, dist_away, radius=np.inf):
    """HOME or AWAY for the team with the player nearest the ball, NOBODY if neither is within
    radius (or neither is tracked). A tie goes to HOME, as in bench.baseline_codes."""
    home = np.nan_to_num(dist_home, nan=np.inf)
    away = np.nan_to_num(dist_away, nan=np.inf)
    team = np.where(home <= away, HOME, AWAY)
    closest = np.minimum(home, away)
    return np.where(np.isfinite(closest) & (closest <= radius), team, NOBODY).astype(np.int8)


def mean_player_speed(player_speed, tracked):
    """Mean speed of the tracked players with a known speed, per frame. NaN when there are none."""
    usable = tracked & ~np.isnan(player_speed)
    total = np.where(usable, player_speed, 0.0).sum(axis=1)
    count = usable.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def ball_turn_degrees(bx, by, half):
    """Angle between the steps into and out of each frame, 0-180, as features.ball_turn_degrees.
    NaN unless the ball is seen either side, has really moved on both steps, and all three frames
    are in the same half."""
    turn = np.full(len(bx), np.nan)
    if len(bx) < 3:
        return turn
    in_x, in_y = bx[1:-1] - bx[:-2], by[1:-1] - by[:-2]
    out_x, out_y = bx[2:] - bx[1:-1], by[2:] - by[1:-1]
    with np.errstate(invalid="ignore"):
        usable = ((np.hypot(in_x, in_y) >= MIN_BALL_STEP_M)
                  & (np.hypot(out_x, out_y) >= MIN_BALL_STEP_M)
                  & (half[:-2] == half[1:-1]) & (half[1:-1] == half[2:]))
        angle = np.degrees(np.abs(np.arctan2(in_x * out_y - in_y * out_x,
                                             in_x * out_x + in_y * out_y)))
    turn[1:-1] = np.where(usable, angle, np.nan)
    return turn


# ======================================================================
# MOMENT 1: the ball goes out (throw-in, goal, goal line)
# ======================================================================

def ball_out_events(bx, by, pitch, t=THRESHOLDS):
    """0/1 columns per frame. A frame without a ball position is 0: we do not know."""
    length, width = pitch
    margin = t["out_margin_m"]
    with np.errstate(invalid="ignore"):
        past_side = (by < -margin) | (by > width + margin)
        past_end = (bx < -margin) | (bx > length + margin)
        between_posts = np.abs(by - width / 2.0) < GOAL_WIDTH_M / 2.0
    return {
        "out_throw_in": (past_side & ~past_end).astype(np.int8),
        "out_goal": (past_end & between_posts).astype(np.int8),
        "out_end_line": (past_end & ~between_posts).astype(np.int8),  # goal kick or corner
    }


# ======================================================================
# MOMENT 2: restart kick = ANCHOR (ball still, then struck)
# ======================================================================

def restart_type(x, y, pitch, t=THRESHOLDS):
    """Kind of restart from where the ball is when it is struck."""
    length, width = pitch
    near_side = y < t["near_side_m"] or y > width - t["near_side_m"]
    near_end = x < t["near_end_m"] or x > length - t["near_end_m"]
    in_goal_area = ((x < GOAL_AREA_DEPTH_M or x > length - GOAL_AREA_DEPTH_M)
                    and abs(y - width / 2.0) < GOAL_AREA_WIDTH_M / 2.0)
    at_centre = (abs(x - length / 2.0) < t["centre_spot_m"]
                 and abs(y - width / 2.0) < t["centre_spot_m"])
    if near_end and near_side:
        return "corner"
    if at_centre:
        return "kickoff"
    if in_goal_area:
        return "goal_kick"
    if near_side:
        return "throw_in"
    return "free_kick"


def restart_anchors(bx, by, ball_speed, player_speed, tracked, dist_home, dist_away, half, pitch,
                    t=THRESHOLDS, fps=FPS):
    """(columns per frame, anchors table).

    An anchor is a frame where the ball has been still for at least still_min_s up to the frame
    before, is not still now, and goes faster than kick_speed within kick_lookahead frames.
    It is confident when the players were slow over the slow_window_s before it and a team is
    tracked near enough to name the taker.
    """
    visible = ~np.isnan(bx)
    with np.errstate(invalid="ignore"):
        still = visible & (ball_speed < t["still_speed"])
    still_frames = run_length(still, half)
    previous_still = np.r_[0, still_frames[:-1]]
    previous_still[np.r_[True, half[1:] != half[:-1]]] = 0  # no run carries over a half

    ends_here = (previous_still >= t["still_min_s"] * fps) & ~still
    fast = np.where(visible, np.nan_to_num(ball_speed, nan=0.0), 0.0)
    soon_fast = future_max(fast, int(t["kick_lookahead"]), half) > t["kick_speed"]
    kick = ends_here & soon_fast & (half > 0)

    players = mean_player_speed(player_speed, tracked)
    with np.errstate(invalid="ignore"):
        slow_before = past_mean(players, int(round(t["slow_window_s"] * fps)), half) \
            < t["slow_players"]  # NaN (nobody tracked) counts as not slow

    team = nearest_team(dist_home, dist_away)
    confident = kick & slow_before & (team != NOBODY)

    frames = np.flatnonzero(kick)
    anchors = pd.DataFrame({
        "frame_index": frames,
        "team": np.select([team[frames] == HOME, team[frames] == AWAY], [H, A], NO_LABEL),
        "type": [restart_type(bx[i], by[i], pitch, t) for i in frames],
        "players_slow_before": slow_before[frames],
        "confident": confident[frames],
    })
    columns = {
        "ball_still_s": still_frames / float(fps),
        "restart_kick": kick.astype(np.int8),
        "restart_kick_home": (kick & (team == HOME)).astype(np.int8),
        "restart_kick_away": (kick & (team == AWAY)).astype(np.int8),
        "players_slow_before": slow_before.astype(np.int8),
        "restart_confident": confident.astype(np.int8),
    }
    return columns, anchors


# ======================================================================
# MOMENT 3: turnover candidate in open play (tackle, interception)
# ======================================================================

def turnover_candidates(bx, by, dist_home, dist_away, half, t=THRESHOLDS):
    """A touch is a change of ball direction above touch_turn_deg with a player within
    touch_radius_m; its team is the nearest one. A turnover candidate is a touch by the other team
    than the previous touch in the same half. Evidence only: deflections look the same."""
    turn = ball_turn_degrees(bx, by, half)
    team_close = nearest_team(dist_home, dist_away, radius=t["touch_radius_m"])
    with np.errstate(invalid="ignore"):
        touch = (turn > t["touch_turn_deg"]) & (team_close != NOBODY) & (half > 0)

    touch_team = pd.Series(np.where(touch, team_close, np.nan))
    # team of the most recent touch up to the previous frame, within the half
    previous = touch_team.groupby(half).ffill().groupby(half).shift(1).to_numpy()
    change = touch & ~np.isnan(previous) & (team_close != previous)

    return {
        "touch_home": (touch & (team_close == HOME)).astype(np.int8),
        "touch_away": (touch & (team_close == AWAY)).astype(np.int8),
        "turnover_candidate": change.astype(np.int8),
    }


# ======================================================================
# everything together
# ======================================================================

def moment_features(bx, by, ball_speed, player_speed, tracked, dist_home, dist_away, half, pitch,
                    thresholds=None, fps=FPS):
    """(columns in MOMENT_COLUMNS order, anchors table). thresholds overrides THRESHOLDS."""
    t = dict(THRESHOLDS, **(thresholds or {}))
    bx, by = np.asarray(bx, dtype=np.float64), np.asarray(by, dtype=np.float64)
    ball_speed = np.asarray(ball_speed, dtype=np.float64)
    player_speed = np.asarray(player_speed, dtype=np.float64)
    half = np.asarray(half)

    columns = ball_out_events(bx, by, pitch, t)
    restart, anchors = restart_anchors(bx, by, ball_speed, player_speed, tracked, dist_home,
                                       dist_away, half, pitch, t, fps)
    columns.update(restart)
    columns.update(turnover_candidates(bx, by, dist_home, dist_away, half, t))
    return {name: columns[name] for name in MOMENT_COLUMNS}, anchors


def moments_from_columns(columns, pitch, dist_home, dist_away, thresholds=None, fps=FPS):
    """moment_features on clean.clean_match's columns (or the npz clean.py writes).

    dist_home and dist_away are the ones model_features.feature_table builds, so the moments see
    exactly the distances the model sees.
    """
    px = np.asarray(columns["player_x_m"], dtype=np.float64)
    return moment_features(columns["ball_x_m"], columns["ball_y_m"], columns["ball_speed"],
                           columns["player_speed"], ~np.isnan(px), dist_home, dist_away,
                           columns["half"], pitch, thresholds, fps)


# ======================================================================
# measuring against true labels (external matches only)
# ======================================================================

def true_restarts(labels):
    """Frames where a dead ball ends: the label is H or A and the one before is D."""
    labels = np.asarray(labels)
    return np.flatnonzero((labels[1:] != D) & (labels[1:] != NO_LABEL) & (labels[:-1] == D)) + 1


def match_events(detected, true, tolerance):
    """Pairs each detected frame with at most one true frame within tolerance, nearest first.

    Returns (index into detected, index into true) arrays of the matched pairs. Both inputs are
    sorted frame numbers.
    """
    detected, true = np.asarray(detected), np.asarray(true)
    if not len(detected) or not len(true):
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    # every candidate pair within tolerance, closest pairs taken first
    pairs = []
    for offset in range(-tolerance, tolerance + 1):
        wanted = detected - offset  # a true frame `offset` frames before the detected one
        j = np.minimum(np.searchsorted(true, wanted), len(true) - 1)
        hit = true[j] == wanted
        pairs += [(abs(offset), int(i), int(j[i])) for i in np.flatnonzero(hit)]
    used_d, used_t, out_d, out_t = set(), set(), [], []
    for _, i, k in sorted(pairs):
        if i not in used_d and k not in used_t:
            used_d.add(i)
            used_t.add(k)
            out_d.append(i)
            out_t.append(k)
    order = np.argsort(out_d)
    return np.asarray(out_d, dtype=int)[order], np.asarray(out_t, dtype=int)[order]


def evaluate_anchors(anchors, labels, tolerance=5):
    """How well anchors match the true ends of dead balls of one match.

    anchors: table from restart_anchors (frame_index, team). labels: model codes per frame.
    """
    labels = np.asarray(labels)
    true = true_restarts(labels)
    anchors = anchors.sort_values("frame_index")
    frames, teams = anchors["frame_index"].to_numpy(), anchors["team"].to_numpy()
    i, k = match_events(frames, true, tolerance)
    offsets = frames[i] - true[k]
    team_ok = teams[i] == labels[true[k]]
    return {
        "detected": len(frames),
        "true_restarts": len(true),
        "hits": len(i),
        "precision": len(i) / len(frames) if len(frames) else float("nan"),
        "recall": len(i) / len(true) if len(true) else float("nan"),
        "team_accuracy": float(team_ok.mean()) if len(i) else float("nan"),
        "mean_offset_frames": float(offsets.mean()) if len(i) else float("nan"),
        "median_offset_frames": float(np.median(offsets)) if len(i) else float("nan"),
    }

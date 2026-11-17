"""Ball touches from tracking, and who touched last / touches next, as model features.

Short possessions (under 5 s, 60% of all possessions) are where the model loses almost all of its
spellF1. Its features are averages over 0.5-2 s, so it cannot tell that the ball just changed
team. A touch is where that shows: the ball's velocity changes sharply (a kick, a stop, a deflection)
while a player is next to it. The team of the nearest player is the team that touched it.

Per frame the model then gets the team of the last and the next touch and how far away in time
they are, which carries the possession across the gaps where the ball is not seen.

    columns = touch_features(bx, by, half, dist_home, dist_away)

Thresholds are placeholders until measured on the external matches (measure in the scratch
script, record the choice in docs/decisions_log.md). Never tuned on the provided game.
"""

import numpy as np

FPS = 25
HOME_SIGN, AWAY_SIGN = 1.0, -1.0  # team columns: +1 home, -1 away, 0 nobody / unknown

TOUCH_THRESHOLDS = {
    "span": 2,            # frames on each side used for the velocity before / after a frame
    "dv_min": 2.0,        # m/s: change of ball velocity that counts as a touch
    "radius_m": 1.5,      # a player of the touching team must be this close to the ball
    "peak_frames": 2,     # a touch is the largest velocity change within +/- this many frames
    "duel_m": 2.0,        # both teams this close to the ball: a duel
    "control_m": 1.0,     # nearest player this close and closer than the other team: control
    "count_frames": 25,   # window for touch counts, each side
}

TOUCH_COLUMNS = (
    "ball_dv", "touch_team", "touch_contested",
    "last_touch_team", "last_touch_s", "next_touch_team", "next_touch_s",
    "last_touch_differs", "touches_home_past", "touches_away_past",
    "touches_home_next", "touches_away_next",
    "duel_now", "duel_share_past", "control_home_s", "control_away_s",
)


def velocity_change(bx, by, half, span):
    """|velocity after - velocity before| at every frame, m/s. NaN where not measurable.

    Before is the mean velocity over the span frames up to this one, after over the span frames
    from it on. Both need the ball seen at both ends and the same half throughout.
    """
    n = len(bx)
    out = np.full(n, np.nan)
    if n <= 2 * span:
        return out
    t = np.arange(span, n - span)
    same = (half[t - span] == half[t]) & (half[t + span] == half[t])
    before = np.stack([bx[t] - bx[t - span], by[t] - by[t - span]], axis=1) * FPS / span
    after = np.stack([bx[t + span] - bx[t], by[t + span] - by[t]], axis=1) * FPS / span
    dv = np.hypot(*(after - before).T)
    out[t] = np.where(same, dv, np.nan)
    return out


def local_peak(values, frames):
    """True where values is the largest within +/- frames (NaN never is)."""
    filled = np.nan_to_num(values, nan=-np.inf)
    n = len(values)
    padded = np.r_[np.full(frames, -np.inf), filled, np.full(frames, -np.inf)]
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * frames + 1)
    return np.isfinite(filled) & (filled >= windows.max(axis=1)[:n])


def carry(values, half, backward=False):
    """For each frame, the last nonzero value (and its frame) at or before it, within the half.

    backward=True looks at or after instead. Returns (value, frame index), 0 and -1 for none.
    """
    n = len(values)
    order = np.arange(n)[::-1] if backward else np.arange(n)
    value_out, frame_out = np.zeros(n), np.full(n, -1)
    current, at, current_half = 0.0, -1, None
    for i in order:
        if half[i] != current_half:
            current, at, current_half = 0.0, -1, half[i]
        if values[i] != 0:
            current, at = values[i], i
        value_out[i], frame_out[i] = current, at
    return value_out, frame_out


def window_count(mask, frames, half, ahead=False):
    """How many True frames in the `frames` before (or after, ahead=True) each frame, same half."""
    mask = np.asarray(mask, dtype=np.int64)
    out = np.zeros(len(mask))
    for start, stop in segments(half):
        part = mask[start:stop]
        total = np.r_[0, np.cumsum(part)]
        i = np.arange(len(part))
        if ahead:
            out[start:stop] = total[np.minimum(i + 1 + frames, len(part))] - total[i + 1]
        else:
            out[start:stop] = total[i] - total[np.maximum(i - frames, 0)]
    return out


def segments(half):
    edges = np.r_[0, np.flatnonzero(np.diff(half)) + 1, len(half)]
    return list(zip(edges[:-1].tolist(), edges[1:].tolist()))


def run_seconds(mask, half):
    """Seconds the mask has been True up to and including each frame, reset at each half."""
    out = np.zeros(len(mask))
    for start, stop in segments(half):
        count = 0
        for i in range(start, stop):
            count = count + 1 if mask[i] else 0
            out[i] = count
    return out / FPS


def touches(bx, by, half, dist_home, dist_away, t=TOUCH_THRESHOLDS):
    """(ball_dv, touch team per frame: +1 home, -1 away, 0 none, contested flag per frame)."""
    half = np.asarray(half)
    dv = velocity_change(bx, by, half, int(t["span"]))
    with np.errstate(invalid="ignore"):
        strong = (dv >= t["dv_min"]) & local_peak(dv, int(t["peak_frames"])) & (half > 0)
        home_near = np.nan_to_num(dist_home, nan=np.inf)
        away_near = np.nan_to_num(dist_away, nan=np.inf)
        closest = np.minimum(home_near, away_near)
        touch = strong & (closest <= t["radius_m"])
        team = np.where(home_near <= away_near, HOME_SIGN, AWAY_SIGN)
        contested = touch & (np.maximum(home_near, away_near) <= t["duel_m"])
    return dv, np.where(touch, team, 0.0), contested


def touch_features(bx, by, half, dist_home, dist_away, thresholds=None):
    """{name: per-frame array} in TOUCH_COLUMNS order."""
    t = dict(TOUCH_THRESHOLDS, **(thresholds or {}))
    half = np.asarray(half)
    bx, by = np.asarray(bx, dtype=np.float64), np.asarray(by, dtype=np.float64)
    dist_home = np.asarray(dist_home, dtype=np.float64)
    dist_away = np.asarray(dist_away, dtype=np.float64)
    dv, team, contested = touches(bx, by, half, dist_home, dist_away, t)

    frames = np.arange(len(bx))
    last_team, last_at = carry(team, half)
    next_team, next_at = carry(team, half, backward=True)
    # the touch before the last one, to see whether the last touch changed the team
    previous = np.zeros(len(bx))
    touch_frames = np.flatnonzero(team != 0)
    if len(touch_frames) > 1:
        same_half = half[touch_frames[1:]] == half[touch_frames[:-1]]
        before = np.where(same_half, team[touch_frames[:-1]], 0.0)
        previous_of_touch = np.zeros(len(bx))
        previous_of_touch[touch_frames[1:]] = before
        previous = np.where(last_at >= 0, previous_of_touch[np.maximum(last_at, 0)], 0.0)

    n = int(t["count_frames"])
    with np.errstate(invalid="ignore"):
        home_near = np.nan_to_num(dist_home, nan=np.inf)
        away_near = np.nan_to_num(dist_away, nan=np.inf)
        duel = (home_near <= t["duel_m"]) & (away_near <= t["duel_m"])
        control_home = (home_near <= t["control_m"]) & (home_near < away_near)
        control_away = (away_near <= t["control_m"]) & (away_near < home_near)

    columns = {
        "ball_dv": dv,
        "touch_team": team,
        "touch_contested": contested.astype(np.float64),
        "last_touch_team": last_team,
        "last_touch_s": np.where(last_at >= 0, (frames - last_at) / FPS, np.nan),
        "next_touch_team": next_team,
        "next_touch_s": np.where(next_at >= 0, (next_at - frames) / FPS, np.nan),
        "last_touch_differs": ((previous != 0) & (last_team != 0)
                               & (previous != last_team)).astype(np.float64),
        "touches_home_past": window_count(team == HOME_SIGN, n, half),
        "touches_away_past": window_count(team == AWAY_SIGN, n, half),
        "touches_home_next": window_count(team == HOME_SIGN, n, half, ahead=True),
        "touches_away_next": window_count(team == AWAY_SIGN, n, half, ahead=True),
        "duel_now": duel.astype(np.float64),
        "duel_share_past": window_count(duel, n, half) / n,
        "control_home_s": run_seconds(control_home, half),
        "control_away_s": run_seconds(control_away, half),
    }
    outside = half <= 0
    for name in columns:
        columns[name] = np.where(outside, np.nan if name.endswith("_s") else 0.0, columns[name])
    return {name: columns[name] for name in TOUCH_COLUMNS}

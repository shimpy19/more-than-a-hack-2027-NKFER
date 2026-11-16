"""The model's features: one row per frame, the columns in model.FEATURE_COLUMNS.

Built from the output of clean.clean_match (or the _clean.npz that clean.py writes), so every
match, external or provided, gets its features from the same cleaned positions and the same
recalculated speeds. Nothing provider-specific is read here.

Separate from features.py on purpose for now: features.py reads the raw tracking, needs the Smart
Tagging file and checks itself against BAR-ATH's published numbers, so it cannot be used for the
external matches the model is trained on.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from model import (CONTEXT_COLUMNS, CONTEXT_FRAMES, FEATURE_COLUMNS, TOUCH_COLUMNS,  # noqa: E402
                   TOUCH_FRAMES)
from starter.baseline import AWAY, HOME  # noqa: E402

FPS = 25


def masked_mean(values, mask):
    """Mean over the selected players of each frame. NaN where none is selected."""
    total = np.where(mask, values, 0.0).sum(axis=1)
    count = mask.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def nearest(distance, mask):
    """Distance from the ball to the nearest selected player. NaN when there is none."""
    closest = np.where(mask & ~np.isnan(distance), distance, np.inf).min(axis=1)
    return np.where(np.isinf(closest), np.nan, closest)


def team_shape(px, py, mask):
    """(mean x of the team, mean distance of its players from the team's centre)."""
    centre_x, centre_y = masked_mean(px, mask), masked_mean(py, mask)
    spread = masked_mean(np.hypot(px - centre_x[:, None], py - centre_y[:, None]), mask)
    return centre_x, spread


def add_time_context(table, ball_x, ball_y):
    """The table with the frames around each row added as columns, never across a half.

    <column>_past_<n>   mean over the n frames before this one
    <column>_next_<n>   mean over the n frames after it (we label after the match, so looking
                        ahead is allowed)
    home_nearer_before_gap / home_nearer_after_gap
                        1 if the home team was nearest the ball when it was last seen (next
                        seen), 0 if the away team was; equal to now while the ball is seen
    ball_moved_in_gap_m distance between where the ball was last seen and where it is next seen

    Frames outside the halves get NaN, which LightGBM handles.
    """
    half = table["half"].to_numpy()
    in_half = half > 0
    out = {}
    for column in CONTEXT_COLUMNS:
        by_half = table[column].where(in_half).groupby(half)
        for frames in CONTEXT_FRAMES:
            out["{}_past_{}".format(column, frames)] = by_half.transform(
                lambda s: s.shift(1).rolling(frames, min_periods=1).mean())
            out["{}_next_{}".format(column, frames)] = by_half.transform(
                lambda s: s[::-1].shift(1).rolling(frames, min_periods=1).mean()[::-1])

    difference = table["dist_difference"].to_numpy()
    seen = in_half & ~np.isnan(difference) & ~np.isnan(ball_x)
    seen_values = pd.DataFrame({
        "home_nearer": np.where(seen, (difference < 0).astype(np.float64), np.nan),
        "x": np.where(seen, ball_x, np.nan), "y": np.where(seen, ball_y, np.nan),
    })
    before = seen_values.groupby(half).ffill()
    after = seen_values.groupby(half).bfill()
    out["home_nearer_before_gap"] = before["home_nearer"].where(in_half)
    out["home_nearer_after_gap"] = after["home_nearer"].where(in_half)
    out["ball_moved_in_gap_m"] = pd.Series(
        np.hypot(after["x"] - before["x"], after["y"] - before["y"])).where(in_half)
    return pd.concat([table, pd.DataFrame(out, index=table.index)], axis=1)


def _by_half(values, half):
    return pd.Series(values).where(half > 0).groupby(half)


def _past_mean(values, half, frames):
    return _by_half(values, half).transform(
        lambda s: s.shift(1).rolling(frames, min_periods=1).mean()).to_numpy()


def _next_mean(values, half, frames):
    return _by_half(values, half).transform(
        lambda s: s[::-1].shift(1).rolling(frames, min_periods=1).mean()[::-1]).to_numpy()


def _shifted(values, half, frames):
    """The value `frames` frames earlier (negative: later), within the same half."""
    return _by_half(values, half).shift(frames).to_numpy()


def add_touch_features(table, columns):
    """The table with TOUCH_COLUMNS added: what happens in the few frames around each one.

    nearest_distance / nearest_team    the nearest tracked player to the ball and his team
                                       (+1 home, -1 away)
    over k = 2, 4 and 8 frames, before vs after this frame:
      dd_shift_k             change in dist_difference (which team is nearer)
      near_team_past/next/shift_k
                             mean nearest team before, after, and the difference
      ball_speed_change_k    change in ball speed (a kick speeds it up)
      ball_turn_k            angle in degrees between the ball's path into and out of this frame
      ball_attack_v_k        ball speed towards the away goal (m/s), from the team centres
      near_player_changed_k  1 if the nearest player k frames before is not the one k frames after
      near_team_changed_k    nearest team k frames after minus k frames before (-2, 0 or +2)
    min_dist_home/away_4, _8 closest the nearest home / away player gets within +/- 4 or 8 frames

    Never across a half; NaN outside the halves and wherever the ball or players are missing.
    """
    half = table["half"].to_numpy()
    team = np.asarray(columns["team"])
    px = np.asarray(columns["player_x_m"], dtype=np.float64)
    py = np.asarray(columns["player_y_m"], dtype=np.float64)
    bx = np.asarray(columns["ball_x_m"], dtype=np.float64)
    by = np.asarray(columns["ball_y_m"], dtype=np.float64)
    speed = np.asarray(columns["ball_speed"], dtype=np.float64)

    distance = np.hypot(px - bx[:, None], py - by[:, None])
    distance = np.where(np.isnan(distance), np.inf, distance)
    nearest_index = distance.argmin(axis=1)
    nearest_distance = distance[np.arange(len(bx)), nearest_index]
    seen = np.isfinite(nearest_distance)
    nearest_id = np.where(seen, nearest_index, np.nan).astype(np.float64)
    nearest_team = np.where(seen, np.where(team[nearest_index] == HOME, 1.0, -1.0), np.nan)
    nearest_distance = np.where(seen, nearest_distance, np.nan)

    difference = table["dist_difference"].to_numpy()
    dist_home, dist_away = table["dist_home"].to_numpy(), table["dist_away"].to_numpy()

    # Home attacks towards the side where the away players stand (team centres, per half).
    attack = np.zeros(len(bx))
    for number in (1, 2):
        rows = half == number
        if rows.any():
            attack[rows] = np.sign(np.nanmean(table["away_centre_x"].to_numpy()[rows])
                                   - np.nanmean(table["home_centre_x"].to_numpy()[rows]))

    out = {"nearest_distance": nearest_distance, "nearest_team": nearest_team}
    for k in TOUCH_FRAMES:
        out["dd_shift_{}".format(k)] = (_next_mean(difference, half, k)
                                        - _past_mean(difference, half, k))
        out["near_team_past_{}".format(k)] = _past_mean(nearest_team, half, k)
        out["near_team_next_{}".format(k)] = _next_mean(nearest_team, half, k)
        out["near_team_shift_{}".format(k)] = (out["near_team_next_{}".format(k)]
                                               - out["near_team_past_{}".format(k)])
        out["ball_speed_change_{}".format(k)] = (_next_mean(speed, half, k)
                                                 - _past_mean(speed, half, k))
        before_x, before_y = _shifted(bx, half, k), _shifted(by, half, k)
        after_x, after_y = _shifted(bx, half, -k), _shifted(by, half, -k)
        v1x, v1y, v2x, v2y = bx - before_x, by - before_y, after_x - bx, after_y - by
        norm = np.hypot(v1x, v1y) * np.hypot(v2x, v2y)
        with np.errstate(invalid="ignore", divide="ignore"):
            cosine = np.where(norm > 1e-6, (v1x * v2x + v1y * v2y) / norm, np.nan)
        out["ball_turn_{}".format(k)] = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        out["ball_attack_v_{}".format(k)] = (after_x - bx) / k * FPS * attack
        id_before, id_after = _shifted(nearest_id, half, k), _shifted(nearest_id, half, -k)
        out["near_player_changed_{}".format(k)] = np.where(
            np.isnan(id_before) | np.isnan(id_after), np.nan, (id_before != id_after) * 1.0)
        out["near_team_changed_{}".format(k)] = (_shifted(nearest_team, half, -k)
                                                 - _shifted(nearest_team, half, k))
    for k in (4, 8):
        for name, values in (("home", dist_home), ("away", dist_away)):
            out["min_dist_{}_{}".format(name, k)] = _by_half(values, half).transform(
                lambda s: s.rolling(2 * k + 1, center=True, min_periods=1).min()).to_numpy()
    touch = pd.DataFrame(out, index=table.index)[list(TOUCH_COLUMNS)]
    touch = touch.where(pd.Series(half > 0, index=table.index), axis=0)
    return pd.concat([table, touch], axis=1)


def feature_table(columns):
    """DataFrame with frame, half and FEATURE_COLUMNS, from clean_match's columns."""
    team = np.asarray(columns["team"])
    px = np.asarray(columns["player_x_m"], dtype=np.float64)
    py = np.asarray(columns["player_y_m"], dtype=np.float64)
    bx = np.asarray(columns["ball_x_m"], dtype=np.float64)
    by = np.asarray(columns["ball_y_m"], dtype=np.float64)
    speed = np.asarray(columns["player_speed"], dtype=np.float64)

    tracked = ~np.isnan(px)
    home, away = tracked & (team == HOME), tracked & (team == AWAY)
    distance = np.hypot(px - bx[:, None], py - by[:, None])
    dist_home, dist_away = nearest(distance, home), nearest(distance, away)
    home_centre, home_spread = team_shape(px, py, home)
    away_centre, away_spread = team_shape(px, py, away)

    table = pd.DataFrame({
        "frame": np.asarray(columns["frame"]),
        "half": np.asarray(columns["half"]),
        "ball_visible": (~np.isnan(bx)).astype(np.int8),
        "ball_speed": columns["ball_speed"],
        "ball_missing_s": columns["ball_missing_s"],
        "ball_until_s": columns["ball_until_s"],
        "ball_past_end_m": columns["ball_past_end_m"],
        "ball_past_side_m": columns["ball_past_side_m"],
        "dist_home": dist_home,
        "dist_away": dist_away,
        "dist_difference": dist_home - dist_away,
        "home_speed": masked_mean(speed, home),
        "away_speed": masked_mean(speed, away),
        "home_centre_x": home_centre,
        "away_centre_x": away_centre,
        "home_spread": home_spread,
        "away_spread": away_spread,
        "n_home": columns["n_home"],
        "n_away": columns["n_away"],
        "tracking_ok": columns["tracking_ok"],
    })
    table = add_time_context(table, bx, by)
    table = add_touch_features(table, columns)
    missing =[name for name in FEATURE_COLUMNS if name not in table.columns]
    if missing:
        raise ValueError("feature_table does not build: {}".format(missing))
    # 64-bit everywhere: a 32-bit number written to CSV and read back comes back slightly
    # different, and the training table goes through CSV while BAR-ATH's features do not.
    return table.astype({name: np.float64 for name in FEATURE_COLUMNS})

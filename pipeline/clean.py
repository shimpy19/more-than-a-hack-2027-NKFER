"""Cleaning: makes the tactical tracking and Smart Tagging trustworthy before anything is built on them.

    python pipeline/clean.py --data-root "/path/to/dataset"

Every change comes out of this code, never out of a person, and every limit below comes from physics
or football rather than from any label. Gaps, a ball past the lines and slow players are signals, so
they are described here, not repaired: no ball position is ever invented.

Writes three files under <data-root>/cleaned/:

    BAR-ATH_tactical_clean.npz          one entry per video frame, see the columns below
    BAR-ATH_tactical_tags_clean.csv     Smart Tagging entries, limited to the two halves
    BAR-ATH_tactical_clean_report.json  how many things each step changed, on graded frames unless
                                        the name says otherwise

The first two are frame-level match data. Keep them outside the repository and do not redistribute
them -- see RULES.md. For the same reason this script prints counts, never values from the data.

Columns of the npz
------------------
frame, half                    video frame; 1 or 2 inside a half, 0 outside
team                           HOME or AWAY per track slot
player_x_m, player_y_m         metres, frames x slots, NaN where removed or not tracked
player_speed                   m/s, recalculated from positions within one unbroken track piece
n_home, n_away                 tracked players per team after cleaning
surplus_home, surplus_away     players above 11 on that team; flagged, not removed (see below)
reprojection_error             from the homography
tracking_ok                    0 where too few players are tracked or the camera mapping is poor
ball_x_m, ball_y_m             metres, not smoothed, NaN where the ball is missing
ball_vx, ball_vy, ball_speed   recalculated from positions, m/s
ball_suspect                   1 where the ball moves implausibly fast, probably in the air
ball_past_end_m                how far past a goal line the ball is, 0 inside
ball_past_side_m               how far past a touchline the ball is, 0 inside
ball_missing_s, ball_until_s   seconds since the ball was last seen, and until it is next seen
ball_last_x_m, ball_last_y_m   where it was last seen; NaN where visible or not seen yet
ball_next_x_m, ball_next_y_m   where it is next seen; NaN where visible or not seen again

The ball-gap columns do not reach across a half boundary, because a ball seen during halftime says
nothing about the half before it.

Players above 11 on a team are only counted. The likely cause is referees tracked as team players,
but that is not confirmed yet, and dropping whoever is furthest from the team would remove the
goalkeeper first.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import ROOT, find_asset, half_column, read_tags, read_tracking  # noqa: E402
from games import GAME, GAMES, half_video_blocks, source_frames  # noqa: E402
from starter.baseline import AWAY, HOME, team_of_each_track  # noqa: E402

# Params
BALL_MAX_SPEED = 40.0  # m/s, about the hardest shot on record
PLAYER_MAX_SPEED = 12.0  # m/s, faster than any sprinter, so a bigger step is a different person
OFF_PITCH_MARGIN_M = 5.0  # further out than this is not a player taking a throw-in or a corner
DUPLICATE_DISTANCE_M = 0.7  # about one body width
DUPLICATE_MIN_FRAMES = 25  # 1 s together, so two players merely passing each other are kept
MIN_PIECE_FRAMES = 25  # 1 s, shorter track pieces are noise
PLAYER_SPEED_REACH = 2  # frames either side, as speed over 5 frames
BALL_SPEED_REACH = 1  # the provider has already smoothed the ball, so no wider than this
MIN_TRACKED_PLAYERS = 15  # of 22, below this a third of the match is missing from the frame
MAX_REPROJECTION_ERROR = 1.0  # pixels, against about 0.24 during normal play
PLAYERS_PER_TEAM = 11

FLOAT = np.float32


def runs_of(mask):
    """[(start, end)] of each run of True, end exclusive."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def segments_of(half):
    """[(start, end)] of the stretches with one value of half: before, 1st, halftime, 2nd, after."""
    changes = np.flatnonzero(np.diff(half)) + 1
    edges = np.concatenate(([0], changes, [len(half)]))
    return list(zip(edges[:-1].tolist(), edges[1:].tolist()))


def track_pieces(px, py, max_speed, fps):
    """(piece id per frame, -1 where not tracked; number of cuts) for one track slot.

    A piece is an unbroken stretch of tracking. A step faster than max_speed starts a new piece,
    because the slot has moved on to someone else.
    """
    tracked = ~np.isnan(px)
    with np.errstate(invalid="ignore"):
        jump = np.hypot(np.diff(px), np.diff(py)) * fps > max_speed
    start = tracked.copy()
    start[1:] = tracked[1:] & (~tracked[:-1] | jump)
    piece = np.cumsum(start) - 1
    return np.where(tracked, piece, -1), int(jump.sum())


def velocity(px, py, piece, reach, fps):
    """(vx, vy) in m/s, from the widest window that stays inside one piece.

    Falls back to narrower windows, and to one-sided steps at the ends of a piece. NaN only for a
    piece of a single frame.
    """
    n = len(px)
    vx, vy = np.full(n, np.nan), np.full(n, np.nan)
    windows = [(k, k) for k in range(reach, 0, -1)] + [(0, 1), (1, 0)]
    for back, forward in windows:
        first = np.arange(back, n - forward)
        low, high = first - back, first + forward
        usable = (piece[low] == piece[high]) & (piece[low] >= 0) & np.isnan(vx[first])
        at, low, high = first[usable], low[usable], high[usable]
        vx[at] = (px[high] - px[low]) * fps / (back + forward)
        vy[at] = (py[high] - py[low]) * fps / (back + forward)
    return vx, vy


def run_lengths(tracked):
    """For each frame, the length of the tracked run it belongs to, 0 where not tracked."""
    lengths = np.zeros(len(tracked), dtype=np.int32)
    for start, end in runs_of(tracked):
        lengths[start:end] = end - start
    return lengths


def remove_off_pitch(px, py, length, width, margin):
    """Blanks positions further than margin outside the pitch. Returns what was removed."""
    with np.errstate(invalid="ignore"):
        off = (px < -margin) | (px > length + margin) | (py < -margin) | (py > width + margin)
    px[off] = np.nan
    py[off] = np.nan
    return off


def remove_duplicates(px, py, team, distance, min_frames):
    """Blanks one of two same-team tracks that stay within distance of each other.

    The shorter tracked run gives way, and on a tie the higher slot, so the choice is the same on
    every run. Returns what was removed and how many such stretches there were. Teammates standing
    shoulder to shoulder in a wall can match too, which only costs a position at a dead ball.
    """
    removed = np.zeros(px.shape, dtype=bool)
    lengths = [run_lengths(~np.isnan(px[:, slot])) for slot in range(px.shape[1])]
    n_stretches = 0
    for side in (HOME, AWAY):
        slots = np.flatnonzero(team == side).tolist()
        for i, a in enumerate(slots):
            for b in slots[i + 1:]:
                with np.errstate(invalid="ignore"):
                    close = np.hypot(px[:, a] - px[:, b], py[:, a] - py[:, b]) < distance
                for start, end in runs_of(close):
                    if end - start < min_frames:
                        continue
                    loser = a if lengths[a][start] < lengths[b][start] else b
                    px[start:end, loser] = np.nan
                    py[start:end, loser] = np.nan
                    removed[start:end, loser] = True
                    n_stretches += 1
    return removed, n_stretches


def clean_players(px, py, team, pitch, fps):
    """(player_speed, report, removal masks), blanking bad positions in px and py in place."""
    length, width = pitch
    off = remove_off_pitch(px, py, length, width, OFF_PITCH_MARGIN_M)
    duplicate, n_duplicates = remove_duplicates(px, py, team, DUPLICATE_DISTANCE_M,
                                                DUPLICATE_MIN_FRAMES)

    speed = np.full(px.shape, np.nan, dtype=FLOAT)
    short = np.zeros(px.shape, dtype=bool)
    cuts = np.zeros(px.shape, dtype=bool)
    n_short = 0
    for slot in range(px.shape[1]):
        piece, _ = track_pieces(px[:, slot], py[:, slot], PLAYER_MAX_SPEED, fps)
        sizes = np.bincount(piece[piece >= 0])
        small = np.flatnonzero(sizes < MIN_PIECE_FRAMES)
        n_short += len(small)
        short[:, slot] = np.isin(piece, small)
        cuts[1:, slot] = (piece[1:] != piece[:-1]) & (piece[1:] >= 0) & (piece[:-1] >= 0)

        px[short[:, slot], slot] = np.nan
        py[short[:, slot], slot] = np.nan
        piece = np.where(short[:, slot], -1, piece)

        vx, vy = velocity(px[:, slot], py[:, slot], piece, PLAYER_SPEED_REACH, fps)
        speed[:, slot] = np.hypot(vx, vy)

    report = {"duplicate_stretches_whole_recording": n_duplicates,
              "short_pieces_whole_recording": n_short}
    return speed, report, {"off": off, "duplicate": duplicate, "short": short, "cuts": cuts}


def ball_gap_columns(px, py, half, fps):
    """Time and position of the last and next sighting, without crossing a half boundary."""
    n = len(px)
    visible = ~np.isnan(px)
    columns = {"ball_missing_s": np.zeros(n), "ball_until_s": np.zeros(n)}
    for name in ("ball_last_x_m", "ball_last_y_m", "ball_next_x_m", "ball_next_y_m"):
        columns[name] = np.full(n, np.nan)

    for start, end in segments_of(half):
        seen = visible[start:end]
        index = np.arange(end - start)
        last = np.maximum.accumulate(np.where(seen, index, -1))
        upcoming = np.minimum.accumulate(np.where(seen, index, end - start)[::-1])[::-1]
        columns["ball_missing_s"][start:end] = (index - last) / float(fps)
        columns["ball_until_s"][start:end] = (upcoming - index) / float(fps)

        has_last = (last >= 0) & ~seen
        has_next = (upcoming < end - start) & ~seen
        for axis, values in (("x", px), ("y", py)):
            columns["ball_last_{}_m".format(axis)][start:end][has_last] = \
                values[start + last[has_last]]
            columns["ball_next_{}_m".format(axis)][start:end][has_next] = \
                values[start + upcoming[has_next]]
    return columns


def clean_ball(ball_x, ball_y, half, pitch, fps):
    """Ball columns in metres. Positions are not smoothed, removed or filled in."""
    length, width = pitch
    px, py = ball_x * length, ball_y * width
    visible = ~np.isnan(px)

    piece, _ = track_pieces(px, py, np.inf, fps)
    vx, vy = velocity(px, py, piece, BALL_SPEED_REACH, fps)
    speed = np.hypot(vx, vy)

    with np.errstate(invalid="ignore"):
        suspect = speed > BALL_MAX_SPEED
        past_end = np.maximum(np.maximum(-px, px - length), 0.0)
        past_side = np.maximum(np.maximum(-py, py - width), 0.0)

    columns = {
        "ball_x_m": px, "ball_y_m": py,
        "ball_vx": vx, "ball_vy": vy, "ball_speed": speed,
        "ball_suspect": suspect.astype(np.int8),
        "ball_past_end_m": np.where(visible, past_end, np.nan),
        "ball_past_side_m": np.where(visible, past_side, np.nan),
    }
    columns.update(ball_gap_columns(px, py, half, fps))
    return columns


def tracking_quality(n_players, reprojection_error, valid):
    """1 where the frame can be trusted, 0 where few players are tracked or the camera is off."""
    return ((n_players >= MIN_TRACKED_PLAYERS) & (valid == 1)
            & (reprojection_error <= MAX_REPROJECTION_ERROR)).astype(np.int8)


def clean_tags(events, blocks):
    """(entries limited to the two halves, entries dropped, entries shortened).

    An entry that overlaps a half is cut to it, because the kickoff entries start just before the
    half does. An entry that overlaps neither is not about the match and is dropped. An N/A team
    stays N/A: the team of a restart is worked out from tracking, never guessed here.
    """
    kept, n_dropped, n_clipped = [], 0, 0
    for code, team, kind, start, end in events:
        overlap = [(number, first, last) for number, first, last in blocks
                   if start <= last and end > first]
        if not overlap:
            n_dropped += 1
            continue
        number, first, last = overlap[0]
        clipped = (max(start, first), min(end, last + 1))
        n_clipped += clipped != (start, end)
        kept.append((code, team, kind, clipped[0], clipped[1], number))
    return kept, n_dropped, n_clipped


def read_homography(path, n_frames):
    """(reprojection_error, valid) per video frame."""
    table = pd.read_csv(path, usecols=["frame", "reprojection_error", "valid"])
    if not (table["frame"].to_numpy() == np.arange(n_frames)).all():
        raise ValueError("homography frames are not 0..{} in order".format(n_frames - 1))
    return table["reprojection_error"].to_numpy(), table["valid"].to_numpy()


def clean_match(x, y, ball_x, ball_y, team, pitch, fps, blocks, reprojection_error, valid):
    """(columns, report) from the arrays read_tracking returns. The provided speeds are not used:
    ball speed is nearly empty, and the other matches we test on will not have the same column."""
    n_frames = len(x)
    half = half_column(n_frames, blocks)
    graded = half > 0

    px, py = x.astype(np.float64) * pitch[0], y.astype(np.float64) * pitch[1]
    speed, report, masks = clean_players(px, py, team, pitch, fps)

    tracked = ~np.isnan(px)
    n_home = tracked[:, team == HOME].sum(axis=1)
    n_away = tracked[:, team == AWAY].sum(axis=1)
    ball = clean_ball(ball_x, ball_y, half, pitch, fps)
    quality = tracking_quality(n_home + n_away, reprojection_error, valid)

    columns = {
        "frame": np.arange(n_frames), "half": half, "team": team,
        "player_x_m": px.astype(FLOAT), "player_y_m": py.astype(FLOAT), "player_speed": speed,
        "n_home": n_home, "n_away": n_away,
        "surplus_home": np.maximum(n_home - PLAYERS_PER_TEAM, 0),
        "surplus_away": np.maximum(n_away - PLAYERS_PER_TEAM, 0),
        "reprojection_error": reprojection_error.astype(FLOAT),
        "tracking_ok": quality,
    }
    for name, values in ball.items():
        columns[name] = values if values.dtype == np.int8 else values.astype(FLOAT)

    visible = ~np.isnan(ball_x)
    report.update({
        "graded_frames": int(graded.sum()),
        "off_pitch_positions_removed": int(masks["off"][graded].sum()),
        "duplicate_positions_removed": int(masks["duplicate"][graded].sum()),
        "short_piece_positions_removed": int(masks["short"][graded].sum()),
        "track_switches_cut": int(masks["cuts"][graded].sum()),
        "ball_frames_visible": int(visible[graded].sum()),
        "ball_suspect_frames": int(ball["ball_suspect"][graded].sum()),
        "ball_suspect_stretches": len(runs_of((ball["ball_suspect"] == 1) & graded)),
        "ball_past_a_line_frames": int(((ball["ball_past_end_m"] > 0)
                                        | (ball["ball_past_side_m"] > 0))[graded].sum()),
        "frames_surplus_home": int((columns["surplus_home"] > 0)[graded].sum()),
        "frames_surplus_away": int((columns["surplus_away"] > 0)[graded].sum()),
        "frames_tracking_not_ok": int((quality == 0)[graded].sum()),
        "stretches_tracking_not_ok": len(runs_of((quality == 0) & graded)),
    })
    return columns, report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", required=True, help="folder holding the match data")
    parser.add_argument("--out", help="output folder; default <data-root>/cleaned/")
    args = parser.parse_args()

    game = GAMES[GAME]
    stem = "{}_tactical".format(GAME)
    meta_path = find_asset(args.data_root, stem + "_atd.xml")
    raw_path = find_asset(args.data_root, stem + "_atd.txt")
    tags_path = find_asset(args.data_root, stem + "_smart_tagging.xml")
    homography_path = find_asset(args.data_root, stem + "_homography.csv")
    out_dir = args.out or os.path.join(args.data_root, "cleaned")

    # The output must never land in the repository, which will be public.
    if os.path.abspath(out_dir).startswith(ROOT + os.sep):
        raise ValueError("refusing to write match data inside the repository: " + out_dir)

    team = team_of_each_track(meta_path)
    n_frames = source_frames("tactical", GAME)
    blocks = half_video_blocks(GAME)

    print("reading tracking ...")
    x, y, _, ball_x, ball_y, _ = read_tracking(raw_path, n_frames, len(team))
    error, valid = read_homography(homography_path, n_frames)

    print("cleaning ...")
    columns, report = clean_match(x, y, ball_x, ball_y, team, game["pitch"], game["fps"], blocks,
                                  error, valid)

    events = read_tags(tags_path, game["fps"])
    tags, n_dropped, n_clipped = clean_tags(events, blocks)
    report.update({"tags_read": len(events), "tags_kept": len(tags),
                   "tags_dropped_outside_halves": n_dropped, "tags_cut_to_a_half": n_clipped})

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    np.savez_compressed(os.path.join(out_dir, stem + "_clean.npz"), **columns)
    pd.DataFrame(tags, columns=["code", "team", "type", "start_frame", "end_frame", "half"]).to_csv(
        os.path.join(out_dir, stem + "_tags_clean.csv"), index=False)
    with open(os.path.join(out_dir, stem + "_clean_report.json"), "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    print("wrote {}".format(out_dir))
    for name in sorted(report):
        print("  {:<32} {}".format(name, report[name]))


if __name__ == '__main__':
    main()

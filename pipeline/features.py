"""Step 1 of the pipeline: one row of numbers per video frame of the tactical recording.

    python pipeline/features.py --data-root "/path/to/dataset"

Reads the tactical ATD and the Smart Tagging and writes a CSV, one row per video frame, so row i is
frame i of a submission. The videos, the homography and the broadcast recording are not used: the
first two are not needed here and the third is on a different timeline.

The file it writes is frame-level match data. Keep it outside the repository and do not redistribute
it -- see RULES.md. For the same reason this script prints checks, never values from the data.

Output columns
--------------
frame, half                   video frame; 1 or 2 inside a half, 0 outside
ball_visible                  1 where the tracking sees the ball
ball_x_m, ball_y_m            ball position in metres
ball_speed                    ball speed in m/s, as given by the tracking
ball_speed_change             change in ball speed since the previous frame
ball_turn_deg                 change of direction at this frame, 0-180
ball_missing_s                how long the ball has been missing so far, 0 where visible
ball_outside_pitch            1 where the ball is seen past a pitch line
n_home, n_away                tracked players per team
home_speed, away_speed        mean player speed per team
mean_speed                    mean speed over all tracked players
mean_speed_smooth             the same, averaged over +/- 1 s
dist_home, dist_away          distance from the ball to each team's nearest player, in metres
home_centre_x, away_centre_x  mean player position along the pitch, in metres
home_spread, away_spread      mean distance of a team's players from their centre, in metres
tag_set_piece                 1 inside a SET PIECES interval
tag_set_piece_<type>          the same, per type: throw_in, corner_kick, goal_kick, free_kick,
                              kick_off_after_goal, kick_off_start
tag_home_set_piece, ...       1 inside a set piece tagged with that team; throw-ins have no team
tag_home_attacking, ...       1 inside any phase where that team attacks or defends
tag_home_<code>, ...          the same, per Smart Tagging code: tag_home_build_up,
                              tag_away_long_ball, tag_home_goals and so on

Missing values are empty cells. A distance is missing when the ball or that team is not tracked.
"""

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, source_frames  # noqa: E402
from starter.baseline import AWAY, HOME, team_of_each_track  # noqa: E402

# Params
SPEED_SMOOTH_FRAMES = 25  # +/- 1 s at 25 fps
MIN_BALL_STEP_M = 0.01  # below this a step has no meaningful direction
FLOAT_FORMAT = "%.4f"  # fixed, so the same input always gives the same file

# Raw ATD line: "<frame>:<player>;<player>;...:<ball>", each group "x,y,speed".
FIELD_SEP = ":"
GROUP_SEP = ";"
VALUE_SEP = ","

# The Team tag on a phase is the team performing it, so a defending phase tagged with one team is
# evidence that the other team has the ball. They are kept apart for that reason.
ATTACKING_CODES = {"build up", "progression", "creating chances", "attacking transition",
                   "long ball", "ball in final third", "ball in the box", "players in the box"}
DEFENDING_CODES = {"defending in attacking third", "defending in middle third",
                   "defending in defensive third", "defensive transition"}
GOAL_CODE = "goals"
SET_PIECE_CODE = "set pieces"
SET_PIECE_TYPES = ("throw-in", "corner kick", "goal kick", "free kick", "kick off (after goal)",
                   "kick off (start)")
HOME_TEAM_TAGS = {"barcelona"}
AWAY_TEAM_TAGS = {"atletic club", "athletic club"}

# Published in DATA.md. Comparing against them confirms the file was read correctly and reveals
# nothing beyond what the documentation already says.
DOCUMENTED_BALL_FRAMES = 113488
DOCUMENTED_BALL_FRAMES_GRADED = 94214
DOCUMENTED_MEAN_TRACKED_GRADED = 20.5
DOCUMENTED_TAG_INSTANCES = 638


def find_asset(root, file_name):
    """Path of a file anywhere under root, or a readable error.

    The downloaded bundle may be unpacked into one folder or spread over several, so this looks by
    name rather than assuming a layout. macOS "._" companion files never match, because their names
    differ from file_name.
    """
    matches = []
    for folder, _, names in os.walk(root):
        if file_name in names:
            matches.append(os.path.join(folder, file_name))
    if not matches:
        raise IOError("{} not found under {}".format(file_name, root))
    if len(matches) > 1:
        raise IOError("{} found more than once:\n  {}".format(file_name, "\n  ".join(matches)))
    return matches[0]


def read_tracking(raw_path, n_frames, n_slots):
    """(x, y, speed) arrays over frames and slots, plus the ball as (x, y, speed) over frames.

    Positions stay normalised 0-1 here. Every frame from 0 to n_frames - 1 must appear exactly once
    with n_slots players, because a hole here would silently shift or blank the frames after it.
    """
    x = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    y = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float64)
    seen = np.zeros(n_frames, dtype=bool)

    with open(raw_path) as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.strip().split(FIELD_SEP)
            if len(parts) != 3:
                raise ValueError("line {}: expected 3 fields, found {}".format(
                    line_number, len(parts)))

            frame = int(parts[0]) - 1  # frame_id counts from 1, submissions from 0
            if not 0 <= frame < n_frames or seen[frame]:
                raise ValueError("line {}: bad or repeated frame id".format(line_number))
            seen[frame] = True

            groups = parts[1].split(GROUP_SEP)
            if len(groups) != n_slots:
                raise ValueError("line {}: expected {} player groups, found {}".format(
                    line_number, n_slots, len(groups)))
            for slot, group in enumerate(groups):
                x[frame, slot], y[frame, slot], speed[frame, slot] = group.split(VALUE_SEP)
            ball[frame] = parts[2].split(VALUE_SEP)

    if not seen.all():
        raise ValueError("{} frames are missing from the tracking".format(int((~seen).sum())))
    return x, y, speed, ball[:, 0], ball[:, 1], ball[:, 2]


def read_tags(tags_path, fps):
    """[(code, team, type, start_frame, end_frame)] with lower-case text. end_frame is exclusive."""
    out = []
    for instance in ET.parse(tags_path).iter("instance"):
        tags = {label.findtext("group"): label.findtext("text")
                for label in instance.findall("label")}
        out.append(((instance.findtext("code") or "").strip().lower(),
                    (tags.get("Team") or "").strip().lower(),
                    (tags.get("Type") or "").strip().lower(),
                    int(round(float(instance.findtext("start")) * fps)),
                    int(round(float(instance.findtext("end")) * fps))))
    return out


def slug(text):
    """A tag as part of a column name: "Kick Off (Start)" -> "kick_off_start"."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def tag_column_names():
    """Every tag column, in a fixed order so the output file is always laid out the same."""
    names = ["tag_set_piece"]
    names += ["tag_set_piece_" + slug(kind) for kind in SET_PIECE_TYPES]
    for side in ("home", "away"):
        names += ["tag_{}_set_piece".format(side), "tag_{}_attacking".format(side),
                  "tag_{}_defending".format(side)]
        names += ["tag_{}_{}".format(side, slug(code))
                  for code in sorted(ATTACKING_CODES | DEFENDING_CODES | {GOAL_CODE})]
    return names


def tag_columns(events, n_frames):
    """One 0/1 column per kind of tag. An N/A team marks only the columns that have no team.

    An unknown code or set-piece type is an error, because matching nothing would quietly leave its
    column empty.
    """
    columns = {name: np.zeros(n_frames, dtype=np.int8) for name in tag_column_names()}
    for code, team, kind, start, end in events:
        start, end = max(0, start), min(n_frames, end)
        if code not in ATTACKING_CODES | DEFENDING_CODES | {GOAL_CODE, SET_PIECE_CODE}:
            raise ValueError("unknown Smart Tagging code: {!r}".format(code))

        if code == SET_PIECE_CODE:
            if kind not in SET_PIECE_TYPES:
                raise ValueError("unknown set-piece type: {!r}".format(kind))
            columns["tag_set_piece"][start:end] = 1
            columns["tag_set_piece_" + slug(kind)][start:end] = 1

        side = "home" if team in HOME_TEAM_TAGS else "away" if team in AWAY_TEAM_TAGS else None
        if side is None:
            continue
        if code == SET_PIECE_CODE:
            columns["tag_{}_set_piece".format(side)][start:end] = 1
        else:
            columns["tag_{}_{}".format(side, slug(code))][start:end] = 1
        if code in ATTACKING_CODES:
            columns["tag_{}_attacking".format(side)][start:end] = 1
        elif code in DEFENDING_CODES:
            columns["tag_{}_defending".format(side)][start:end] = 1
    return columns


def masked_mean(values, mask):
    """Row-wise mean over mask. NaN on a row with nothing to average, without a numpy warning."""
    usable = mask & ~np.isnan(values)
    count = usable.sum(axis=1)
    total = np.where(usable, values, 0).sum(axis=1, dtype=np.float64)
    return np.divide(total, count, out=np.full(len(values), np.nan), where=count > 0)


def centred_nanmean(values, window):
    """Mean over +/- window frames, ignoring NaN. NaN where the whole window is empty."""
    present = ~np.isnan(values)
    cumulative = np.concatenate(([0.0], np.cumsum(np.where(present, values, 0.0))))
    counts = np.concatenate(([0], np.cumsum(present)))
    index = np.arange(len(values))
    low = np.maximum(0, index - window)
    high = np.minimum(len(values), index + window + 1)
    count = counts[high] - counts[low]
    return np.divide(cumulative[high] - cumulative[low], count,
                     out=np.full(len(values), np.nan), where=count > 0)


def ball_missing_seconds(visible, fps):
    """Seconds since the ball was last seen. A gap at the start of the file counts from frame 0."""
    index = np.arange(len(visible))
    last_seen = np.maximum.accumulate(np.where(visible, index, -1))
    return (index - last_seen) / float(fps)


def ball_turn_degrees(ball_x, ball_y):
    """Angle between the steps into and out of each frame.

    NaN unless the ball is seen on the frames either side and has really moved on both steps, since
    a direction measured across a gap or from a stationary ball means nothing.
    """
    turn = np.full(len(ball_x), np.nan)
    in_x, in_y = ball_x[1:-1] - ball_x[:-2], ball_y[1:-1] - ball_y[:-2]
    out_x, out_y = ball_x[2:] - ball_x[1:-1], ball_y[2:] - ball_y[1:-1]

    usable = (np.hypot(in_x, in_y) >= MIN_BALL_STEP_M) & (np.hypot(out_x, out_y) >= MIN_BALL_STEP_M)
    angle = np.degrees(np.abs(np.arctan2(in_x * out_y - in_y * out_x, in_x * out_x + in_y * out_y)))
    turn[1:-1] = np.where(usable, angle, np.nan)
    return turn


def team_shape(px, py, mask):
    """(centre along the pitch, mean distance of the players from their centre) per frame."""
    centre_x = masked_mean(px, mask)
    centre_y = masked_mean(py, mask)
    distance = np.hypot(px - centre_x[:, None], py - centre_y[:, None])
    return centre_x, masked_mean(distance, mask)


def nearest_distance(distance, mask):
    """Distance to the nearest player selected by mask. NaN when there is none."""
    nearest = np.where(mask & ~np.isnan(distance), distance, np.inf).min(axis=1)
    return np.where(np.isinf(nearest), np.nan, nearest)


def half_column(n_frames, blocks):
    """1 or 2 inside a half, 0 outside."""
    half = np.zeros(n_frames, dtype=np.int8)
    for number, first, last in blocks:
        half[first:last + 1] = number
    return half


def build_features(x, y, speed, ball_x, ball_y, ball_speed, team, pitch, fps, blocks, events):
    """The feature table, one row per frame, from the arrays read_tracking returns."""
    n_frames = len(x)
    length, width = pitch
    home, away = team == HOME, team == AWAY
    tracked = ~np.isnan(x)

    # Metres, because the pitch is not square and 0-1 distances would stretch one axis.
    px, py = x.astype(np.float64) * length, y.astype(np.float64) * width
    ball_px, ball_py = ball_x * length, ball_y * width
    visible = ~np.isnan(ball_x)
    distance = np.hypot(px - ball_px[:, None], py - ball_py[:, None])

    home_mask, away_mask = tracked & home, tracked & away
    home_centre, home_spread = team_shape(px, py, home_mask)
    away_centre, away_spread = team_shape(px, py, away_mask)
    mean_speed = masked_mean(speed.astype(np.float64), tracked)

    speed_change = np.full(n_frames, np.nan)
    speed_change[1:] = ball_speed[1:] - ball_speed[:-1]

    with np.errstate(invalid="ignore"):
        outside = visible & ((ball_x < 0) | (ball_x > 1) | (ball_y < 0) | (ball_y > 1))

    columns = {
        "frame": np.arange(n_frames),
        "half": half_column(n_frames, blocks),
        "ball_visible": visible.astype(np.int8),
        "ball_x_m": ball_px,
        "ball_y_m": ball_py,
        "ball_speed": ball_speed,
        "ball_speed_change": speed_change,
        "ball_turn_deg": ball_turn_degrees(ball_px, ball_py),
        "ball_missing_s": ball_missing_seconds(visible, fps),
        "ball_outside_pitch": outside.astype(np.int8),
        "n_home": home_mask.sum(axis=1),
        "n_away": away_mask.sum(axis=1),
        "home_speed": masked_mean(speed.astype(np.float64), home_mask),
        "away_speed": masked_mean(speed.astype(np.float64), away_mask),
        "mean_speed": mean_speed,
        "mean_speed_smooth": centred_nanmean(mean_speed, SPEED_SMOOTH_FRAMES),
        "dist_home": nearest_distance(distance, home_mask),
        "dist_away": nearest_distance(distance, away_mask),
        "home_centre_x": home_centre,
        "away_centre_x": away_centre,
        "home_spread": home_spread,
        "away_spread": away_spread,
    }
    columns.update(tag_columns(events, n_frames))
    return pd.DataFrame(columns)


def documented_checks(table, n_tags):
    """[(description, passed)] comparing the table with the numbers published in DATA.md."""
    graded = table["half"] > 0
    tracked = table.loc[graded, "n_home"] + table.loc[graded, "n_away"]
    return [
        ("frames are 0..n-1 in order", (table["frame"] == np.arange(len(table))).all()),
        ("ball frames match DATA.md", table["ball_visible"].sum() == DOCUMENTED_BALL_FRAMES),
        ("graded ball frames match DATA.md",
         table.loc[graded, "ball_visible"].sum() == DOCUMENTED_BALL_FRAMES_GRADED),
        ("mean tracked players matches DATA.md",
         round(tracked.mean(), 1) == DOCUMENTED_MEAN_TRACKED_GRADED),
        ("Smart Tagging instances match DATA.md", n_tags == DOCUMENTED_TAG_INSTANCES),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", required=True, help="folder holding the match data")
    parser.add_argument("--out", help="output CSV; default <data-root>/features/")
    args = parser.parse_args()

    game = GAMES[GAME]
    stem = "{}_tactical".format(GAME)
    meta_path = find_asset(args.data_root, stem + "_atd.xml")
    raw_path = find_asset(args.data_root, stem + "_atd.txt")
    tags_path = find_asset(args.data_root, stem + "_smart_tagging.xml")
    out_path = args.out or os.path.join(args.data_root, "features", stem + "_features.csv")

    # The output must never land in the repository, which will be public.
    if os.path.abspath(out_path).startswith(ROOT + os.sep):
        raise ValueError("refusing to write match data inside the repository: " + out_path)

    team = team_of_each_track(meta_path)
    n_frames = source_frames("tactical", GAME)
    print("reading tracking ...")
    tracking = read_tracking(raw_path, n_frames, len(team))

    print("building features ...")
    events = read_tags(tags_path, game["fps"])
    table = build_features(*tracking, team, game["pitch"], game["fps"],
                           half_video_blocks(GAME), events)

    folder = os.path.dirname(out_path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    table.to_csv(out_path, index=False, float_format=FLOAT_FORMAT)

    print("wrote {} rows and {} columns to {}".format(len(table), table.shape[1], out_path))
    checks = documented_checks(table, len(events))
    for description, passed in checks:
        print("  {}  {}".format("OK  " if passed else "FAIL", description))
    if not all(passed for _, passed in checks):
        sys.exit(1)


if __name__ == '__main__':
    main()

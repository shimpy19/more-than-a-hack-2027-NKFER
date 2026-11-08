"""Damage a complete external match so its players go missing the way BAR-ATH's do.

    python pipeline/damage.py --tracking <match .txt> --meta <match .xml> \
        --barca-clean <BAR-ATH_tactical_clean.npz> --seed 1 --out <folder>

Why only players: Metrica's own matches already lose the ball like BAR-ATH does (same provider; the
ball is not tracked while it is dead, about 160 gaps per match, visible stretches of about 14 s), so
the ball is left exactly as it is. Players are the difference: the originals track all 22 all the
time, BAR-ATH about 20 on average, with fewer while the ball is missing.

How many players are missing at each frame is copied from BAR-ATH in blocks of up to 10 s: for every
stretch of the match where the ball is seen (or missing), each block is taken from a random point
in BAR-ATH where the ball was also seen (or missing) for at least that long. Which players go missing is random,
because in BAR-ATH lost players are no further from the play than the rest. A missing player stays
missing until the count drops again, so gaps have realistic lengths.

The BAR-ATH profile is measured from the cleaned file every time this runs and never written
anywhere, so nothing derived from the provided game is stored in the repo. Only the tracking is
used, never a label.

The output folder gets tracking.txt in the standard layout (one fixed group per player slot, as
"x,y,NaN"), with the metadata and possession files copied next to it.
"""

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from features import ROOT, read_tracking  # noqa: E402
from starter.baseline import team_of_each_track  # noqa: E402

PLAYERS_ON_PITCH = 22
SEED = 1
# Long stretches are copied in pieces of at most this many frames (10 s), so there are many places
# in BAR-ATH to copy each piece from, rather than only its few longest stretches.
BLOCK_FRAMES = 250


def read_any(txt, meta):
    """(x, y, ball_x, ball_y) for a standard file, or one whose groups change at substitutions.

    In the second kind, each DataFormatSpecification covers a range of frames and lists which
    player each group is; slots follow the Player order in the metadata, as team_of_each_track.
    """
    root = ET.parse(meta).getroot()
    n_slots = len(team_of_each_track(meta))
    with open(txt) as handle:
        first_line = handle.readline()
        n_frames = 1 + sum(1 for _ in handle)
    # One group per slot means the standard layout, whatever the metadata says; damaged copies
    # are always written that way.
    if len(first_line.split(":")[1].split(";")) == n_slots:
        x, y, _, ball_x, ball_y, _ = read_tracking(txt, n_frames, n_slots)
        return x, y, ball_x, ball_y
    specs = list(root.iter("DataFormatSpecification"))

    slot_of = {player.get("id"): slot for slot, player in enumerate(root.iter("Player"))}
    player_of_channel = {channel.get("id"): channel.get("playerId")
                         for channel in root.iter("PlayerChannel")}
    ranges = []
    for spec in specs:
        refs = [ref.get("playerChannelId") for ref in spec.iter("PlayerChannelRef")]
        ranges.append((int(spec.get("startFrame")), int(spec.get("endFrame")),
                       [slot_of[player_of_channel[ref]] for ref in refs[::2]]))

    x = np.full((n_frames, len(slot_of)), np.nan, dtype=np.float32)
    y = np.full((n_frames, len(slot_of)), np.nan, dtype=np.float32)
    ball_x, ball_y = np.full(n_frames, np.nan), np.full(n_frames, np.nan)
    with open(txt) as handle:
        for line in handle:
            frame_id, players, ball = line.strip().split(":")
            frame_id = int(frame_id)
            slots = next(s for first, last, s in ranges if first <= frame_id <= last)
            groups = players.split(";")
            if len(groups) != len(slots):
                raise ValueError("frame {}: {} groups for {} players".format(
                    frame_id, len(groups), len(slots)))
            for slot, group in zip(slots, groups):
                x[frame_id - 1, slot], y[frame_id - 1, slot] = group.split(",")[:2]
            ball_x[frame_id - 1], ball_y[frame_id - 1] = ball.split(",")[:2]
    return x, y, ball_x, ball_y


def barca_profile(clean_path):
    """(missing players per graded frame, ball seen per graded frame) from the cleaned BAR-ATH file.

    Graded frames of both halves, in order. Kept in memory only.
    """
    clean = np.load(clean_path)
    graded = clean["half"] > 0
    tracked = (clean["n_home"] + clean["n_away"])[graded]
    missing = np.maximum(PLAYERS_ON_PITCH - tracked, 0).astype(np.int16)
    return missing, ~np.isnan(clean["ball_x_m"][graded])


def runs_of(mask):
    """[(start, stop, value)] for each constant stretch of a boolean array."""
    edges = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(mask)]))
    return [(start, stop, bool(mask[start])) for start, stop in zip(starts, stops)]


def missing_counts(ball_seen, profile, rng):
    """How many players to hide at each frame, copied in blocks from BAR-ATH by ball state."""
    barca_missing, barca_seen = profile
    # same_ahead[i]: how many frames from i on keep BAR-ATH's ball state unchanged.
    same_ahead = np.zeros(len(barca_seen), dtype=np.int64)
    for start, stop, _ in runs_of(barca_seen):
        same_ahead[start:stop] = np.arange(stop - start, 0, -1)
    out = np.zeros(len(ball_seen), dtype=np.int16)
    pieces = [(first, min(first + BLOCK_FRAMES, stop), state)
              for start, stop, state in runs_of(ball_seen)
              for first in range(start, stop, BLOCK_FRAMES)]
    for start, stop, state in pieces:
        length = stop - start
        # A block that stays in the same ball state the whole way if BAR-ATH has one that long,
        # otherwise one from its longest stretch in that state.
        candidates = np.flatnonzero(barca_seen == state)
        reach = same_ahead[candidates]
        points = candidates[reach >= min(length, reach.max())]
        first = int(rng.choice(points))
        block = barca_missing[first:first + length]
        out[start:start + len(block)] = block
        if len(block) < length:  # only at the very end of BAR-ATH; repeat its last value
            out[start + len(block):stop] = block[-1] if len(block) else 0
    return out


def hide_players(x, y, counts, rng):
    """Copies of x, y with `counts[t]` of the players present at frame t hidden, chosen at random.

    A hidden player stays hidden for as long as the count allows, so gaps last, rather than a new
    random player vanishing every frame.
    """
    x, y = x.copy(), y.copy()
    hidden = set()
    for t in range(len(x)):
        present = np.flatnonzero(~np.isnan(x[t]))
        hidden &= set(present.tolist())  # a player who left the pitch is no longer hidden
        want = min(int(counts[t]), len(present))
        if len(hidden) > want:
            for slot in rng.choice(sorted(hidden), len(hidden) - want, replace=False):
                hidden.discard(int(slot))
        elif len(hidden) < want:
            free = [slot for slot in present.tolist() if slot not in hidden]
            hidden.update(int(s) for s in rng.choice(free, want - len(hidden), replace=False))
        if hidden:
            columns = sorted(hidden)
            x[t, columns] = np.nan
            y[t, columns] = np.nan
    return x, y


def damage(x, y, ball_x, profile, seed=SEED):
    """(x, y) with players hidden BAR-ATH style. The ball is not touched."""
    rng = np.random.default_rng(seed)
    counts = missing_counts(~np.isnan(ball_x), profile, rng)
    return hide_players(x, y, counts, rng)


def number(value):
    return "NaN" if np.isnan(value) else "{:.5f}".format(value)


def write_tracking(path, x, y, ball_x, ball_y):
    """Standard layout: frame:x,y,NaN;...:bx,by,NaN with frame ids from 1."""
    with open(path, "w") as handle:
        for t in range(len(x)):
            players = ";".join("{},{},NaN".format(number(a), number(b)) for a, b in zip(x[t], y[t]))
            handle.write("{}:{}:{},{},NaN\n".format(t + 1, players, number(ball_x[t]),
                                                    number(ball_y[t])))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tracking", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--possession", help="labels file to copy next to the output")
    parser.add_argument("--barca-clean", required=True, help="BAR-ATH_tactical_clean.npz")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", required=True, help="output folder")
    args = parser.parse_args()

    if os.path.abspath(args.out).startswith(ROOT + os.sep):
        raise ValueError("refusing to write match data inside the repository: " + args.out)
    os.makedirs(args.out, exist_ok=True)

    x, y, ball_x, ball_y = read_any(args.tracking, args.meta)
    hidden_x, hidden_y = damage(x, y, ball_x, barca_profile(args.barca_clean), args.seed)
    write_tracking(os.path.join(args.out, "tracking.txt"), hidden_x, hidden_y, ball_x, ball_y)
    shutil.copy(args.meta, os.path.join(args.out, "metadata.xml"))
    if args.possession:
        shutil.copy(args.possession, os.path.join(args.out, "possession.csv"))

    before, after = (~np.isnan(x)).sum(axis=1), (~np.isnan(hidden_x)).sum(axis=1)
    print("players tracked per frame: {:.1f} before, {:.1f} after; ball left as is ({:.0%} seen); "
          "wrote {}".format(before.mean(), after.mean(), (~np.isnan(ball_x)).mean(), args.out))


if __name__ == "__main__":
    main()

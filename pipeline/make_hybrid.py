"""Hybrid training matches: every player from tracking.txt, the ball from tracking_with_gaps.txt.

    python pipeline/make_hybrid.py --external <folder> --out <hybrid folder>

The DFL with_gaps tracking hides about 58 % of the player positions (about 13 players seen per
frame), far more than BAR-ATH (about 20.5). The hybrid keeps every player position of the full
tracking and takes the ball, gaps included, from with_gaps, so the ball is exactly as hard as in
with_gaps while the players look much more like BAR-ATH. The labels are possession_with_gaps.csv,
unchanged (it is identical to possession.csv). Trained and benched on these matches, the model
with the touch features scores spellF1 0.623 against 0.461 on with_gaps (docs/decisions_log.md).

Input: <external>/sportec_open_tracking/<match>/ with tracking.txt, tracking_with_gaps.txt,
metadata.xml and possession_with_gaps.csv. Output: <out>/sportec_open_tracking/<match>/ with
tracking_with_gaps.txt, metadata.xml and possession_with_gaps.csv, the layout training_table.py
reads. A line of the tracking is "frame:players:ball". The output stays outside the repository.
"""

import argparse
import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from features import ROOT  # noqa: E402

COPIED = ("metadata.xml", "possession_with_gaps.csv")


def hybrid_line(full, gaps, number):
    """frame:players of the full line, then :ball of the with_gaps line, for the same frame."""
    frame, players, _ = full.rstrip("\r\n").split(":", 2)
    gap_frame, _, ball = gaps.rstrip("\r\n").split(":", 2)
    if frame != gap_frame:
        raise ValueError("line {}: frame {} in tracking.txt, {} in tracking_with_gaps.txt".format(
            number, frame, gap_frame))
    return "{}:{}:{}\n".format(frame, players, ball)


def make_match(source, target):
    """Write one hybrid match; returns its number of frames."""
    os.makedirs(target, exist_ok=True)
    count = 0
    with open(os.path.join(source, "tracking.txt"), encoding="utf-8") as full, \
            open(os.path.join(source, "tracking_with_gaps.txt"), encoding="utf-8") as gaps, \
            open(os.path.join(target, "tracking_with_gaps.txt"), "w", encoding="utf-8",
                 newline="\n") as out:
        for count, (a, b) in enumerate(zip(full, gaps), start=1):
            out.write(hybrid_line(a, b, count))
        if next(full, None) is not None or next(gaps, None) is not None:
            raise ValueError(source + ": tracking.txt and tracking_with_gaps.txt differ in length")
    for name in COPIED:
        shutil.copyfile(os.path.join(source, name), os.path.join(target, name))
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--external", required=True, help="folder with sportec_open_tracking/")
    parser.add_argument("--out", required=True, help="hybrid folder to write")
    args = parser.parse_args()
    if os.path.abspath(args.out).startswith(ROOT + os.sep):
        raise ValueError("refusing to write match data inside the repository: " + args.out)

    matches = sorted(glob.glob(os.path.join(args.external, "sportec_open_tracking", "*")))
    if not matches:
        raise IOError("no DFL matches under " + args.external)
    for source in matches:
        name = os.path.basename(source)
        frames = make_match(source, os.path.join(args.out, "sportec_open_tracking", name))
        print("  {:<10} {:>7} frames".format(name, frames), flush=True)


if __name__ == "__main__":
    main()

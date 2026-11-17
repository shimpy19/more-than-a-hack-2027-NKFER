"""Training table for the model: every frame of every external match, with features and labels.

    python pipeline/training_table.py --external <folder> --out <training table .csv>
    python pipeline/training_table.py --external <folder> --smooth-ball 6 --out <table .csv>

--smooth-ball SIGMA smooths each match's ball with a Gaussian of SIGMA frames inside every stretch
where it is seen (never across a gap), before cleaning. BAR-ATH's ball comes from video and has no
sharp velocity jumps (the 99th percentile of the velocity change over 2 frames is 2.9 m/s there,
12-17 m/s on the training matches); with sigma 6 the training matches look the same (2.7-3.0).
The sigma was chosen from BAR-ATH's tracking only, never from a label. The possession-change
detector (detector.py) is trained on such a table.

The external folder holds every training match damaged BAR-ATH style by damage.py, one folder each,
with tracking.txt, metadata.xml and possession.csv:
  barca_style/J03.../     the 7 DFL matches: players and ball damaged (--ball-like the Metrica copies)
  barca_style/metrica_<n> Metrica sample games 2 and 3: players damaged, ball as recorded

Each match goes through the same clean_match and feature_table as BAR-ATH. There is no
homography for these matches, so reprojection error is 0 and every frame counts as valid.
The table is written outside the repository, like all match data.
"""

import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import clean  # noqa: E402
from damage import halves, read_any, runs_of  # noqa: E402
from features import ROOT  # noqa: E402
from model_features import feature_table  # noqa: E402
from starter.baseline import team_of_each_track  # noqa: E402

FPS = 25
LABEL_CODES = {"H": 0, "A": 1, "D": 2}
NO_LABEL = -1


def external_matches(folder):
    """[(name, tracking, metadata, labels)] in a fixed order."""
    out = []
    for match in sorted(glob.glob(os.path.join(folder, "barca_style", "*"))):
        if os.path.isdir(match):
            out.append((os.path.basename(match), os.path.join(match, "tracking.txt"),
                        os.path.join(match, "metadata.xml"), os.path.join(match, "possession.csv")))
    if not out:
        raise IOError("no external matches found under " + os.path.join(folder, "barca_style"))
    return out


def smooth_ball(ball_x, ball_y, sigma):
    """Gaussian-smoothed copy of the ball (sigma in frames), inside each seen stretch only."""
    out_x, out_y = ball_x.astype(np.float64), ball_y.astype(np.float64)
    if not sigma:
        return out_x, out_y
    for start, stop, seen in runs_of(~np.isnan(out_x)):
        if seen and stop - start > 1:
            out_x[start:stop] = gaussian_filter1d(out_x[start:stop], sigma, mode="nearest")
            out_y[start:stop] = gaussian_filter1d(out_y[start:stop], sigma, mode="nearest")
    return out_x, out_y


def match_table(name, tracking, meta, labels_path, smooth_sigma=0):
    """Features and labels for one external match; its ball smoothed if smooth_sigma > 0."""
    team = team_of_each_track(meta)
    root = ET.parse(meta).getroot()
    pitch = (float(root.findtext(".//FieldSize/Width")), float(root.findtext(".//FieldSize/Height")))
    x, y, ball_x, ball_y = read_any(tracking, meta)
    if smooth_sigma:
        ball_x, ball_y = smooth_ball(ball_x, ball_y, smooth_sigma)
    n_frames = len(x)
    columns, _ = clean.clean_match(x, y, ball_x, ball_y, team, pitch, FPS, halves(meta),
                                   np.zeros(n_frames), np.ones(n_frames, dtype=int))
    table = feature_table(columns)

    labels = pd.read_csv(labels_path)
    if len(labels) != n_frames or (labels["frame"].to_numpy() != np.arange(n_frames)).any():
        raise ValueError(name + ": labels do not line up with the tracking frames")
    label = labels["label"].map(LABEL_CODES).to_numpy()
    if np.isnan(label[table["half"].to_numpy() > 0]).any():
        raise ValueError(name + ": a graded frame has a label other than H, A or D")
    table.insert(0, "match", name)
    table["label"] = np.where(table["half"] > 0, np.nan_to_num(label, nan=NO_LABEL),
                              NO_LABEL).astype(np.int64)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--external", required=True, help="folder with the external matches")
    parser.add_argument("--out", required=True, help="training table to write (.csv)")
    parser.add_argument("--smooth-ball", type=float, default=0, metavar="SIGMA",
                        help="smooth the ball like BAR-ATH's (frames; 6 for the detector)")
    args = parser.parse_args()
    if os.path.abspath(args.out).startswith(ROOT + os.sep):
        raise ValueError("refusing to write match data inside the repository: " + args.out)

    tables = []
    for name, tracking, meta, labels_path in external_matches(args.external):
        tables.append(match_table(name, tracking, meta, labels_path, args.smooth_ball))
        print("  {:<10} {:>7} frames".format(name, len(tables[-1])), flush=True)
    pd.concat(tables, ignore_index=True).to_csv(args.out, index=False)
    print("wrote {} matches to {}".format(len(tables), args.out))


if __name__ == "__main__":
    main()

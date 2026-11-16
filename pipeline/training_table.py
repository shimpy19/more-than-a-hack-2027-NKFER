"""Training table for the model: every frame of every external match, with features and labels.

    python pipeline/training_table.py --external <folder> --out <training table .csv>

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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import clean  # noqa: E402
from damage import halves, read_any  # noqa: E402
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


def match_table(name, tracking, meta, labels_path):
    """Features and labels for one external match."""
    team = team_of_each_track(meta)
    root = ET.parse(meta).getroot()
    pitch = (float(root.findtext(".//FieldSize/Width")), float(root.findtext(".//FieldSize/Height")))
    x, y, ball_x, ball_y = read_any(tracking, meta)
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
    args = parser.parse_args()
    if os.path.abspath(args.out).startswith(ROOT + os.sep):
        raise ValueError("refusing to write match data inside the repository: " + args.out)

    tables = []
    for name, tracking, meta, labels_path in external_matches(args.external):
        tables.append(match_table(name, tracking, meta, labels_path))
        print("  {:<10} {:>7} frames".format(name, len(tables[-1])), flush=True)
    pd.concat(tables, ignore_index=True).to_csv(args.out, index=False)
    print("wrote {} matches to {}".format(len(tables), args.out))


if __name__ == "__main__":
    main()

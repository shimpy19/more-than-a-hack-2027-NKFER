"""Possession-change detector: where possession is likely to change hands, and to which team.

    python pipeline/detector.py train --data <smoothed training table> --out models/detector_v1.txt

A second LightGBM model on the same 90 features as the main model. For each frame it gives three
probabilities: no change nearby, a change to H, or a change to A, within +/- WINDOW frames of a
direct H <-> A change in the labels. Viterbi then makes an H <-> A switch cheap where the detector
says one is likely and expensive elsewhere (switch_costs, SETTING), instead of one fixed team
cost. Switches into or out of a dead ball keep their fixed cost.

Trained on the training matches with their ball smoothed like BAR-ATH's
(training_table.py --smooth-ball 6). Trained on the sharp original ball, it relies on ball
signals that BAR-ATH's video-tracked ball does not have, and fires far less there. Trained on the
smoothed ball, it fires at the same rate on BAR-ATH as on the smoothed test matches.

Training keeps every frame near a change and a fixed random KEEP_NEGATIVE share of the others, so it
takes about a minute. Reproducible: fixed seed, deterministic mode, THREADS threads. The provided
game is never used for training or for choosing SETTING (bench_detector.py does that).
"""

import argparse
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
from model import FEATURE_COLUMNS, LIGHTGBM_PARAMS, NO_LABEL, read_table  # noqa: E402

H, A = 0, 1
WINDOW = 6            # frames either side of a change that count as "change here"
KEEP_NEGATIVE = 0.2   # share of the frames far from any change kept for training
SEED = 1
THREADS = 2

# How Viterbi uses the detector, chosen on the 9 smoothed training matches (bench_detector.py, the
# best setting over all of them, 0.638 mean spellF1): into H costs
# cheap + (base - cheap) * (1 - P(change to H)) ** knob, and the same for A; entering or leaving
# D costs `dead`.
SETTING = {"base": 120.0, "knob": 4.0, "cheap": 0.0, "dead": 30.0}


def change_targets(label, half):
    """(target per frame, frames of the changes): 0 none, 1 change to H, 2 change to A, for the
    frames within WINDOW of a direct H <-> A change (the nearest change wins)."""
    label, half = np.asarray(label), np.asarray(half)
    target = np.zeros(len(label), dtype=np.int64)
    change = np.flatnonzero((label[1:] != label[:-1]) & np.isin(label[1:], [H, A])
                            & np.isin(label[:-1], [H, A]) & (half[1:] == half[:-1])
                            & (half[1:] > 0)) + 1
    distance = np.full(len(label), np.inf)
    for frame in change:
        lo, hi = max(0, frame - WINDOW), min(len(label), frame + WINDOW + 1)
        span = np.abs(np.arange(lo, hi) - frame)
        closer = span < distance[lo:hi]
        distance[lo:hi][closer] = span[closer]
        target[lo:hi][closer] = 1 if label[frame] == H else 2
    return target, change


def training_rows(table, leave_out=None):
    """(X, y) for the detector: per match in name order, every frame within WINDOW of a change
    and a seeded random KEEP_NEGATIVE share of the other graded frames."""
    rng = np.random.default_rng(SEED)
    xs, ys = [], []
    for match in sorted(table["match"].unique()):
        if match == leave_out:
            continue
        part = table[table["match"] == match]
        label, half = part["label"].to_numpy(), part["half"].to_numpy()
        target, _ = change_targets(label, half)
        keep = (label != NO_LABEL) & ((target > 0) | (rng.random(len(part)) < KEEP_NEGATIVE))
        xs.append(part.loc[keep, list(FEATURE_COLUMNS)])
        ys.append(target[keep])
    return pd.concat(xs, ignore_index=True), np.concatenate(ys)


def train(table, leave_out=None):
    """A trained detector (lightgbm Booster), on every match of the table but leave_out."""
    X, y = training_rows(table, leave_out)
    params = dict(LIGHTGBM_PARAMS)
    rounds = params.pop("num_boost_round")
    params.update(seed=SEED, num_threads=THREADS)
    return lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=rounds)


def predict(booster, table):
    """Probabilities, frames x 3: no change, change to H, change to A."""
    return booster.predict(table[list(FEATURE_COLUMNS)])


def switch_costs(probabilities, setting=None):
    """(cost of a switch into H, into A) per frame, from the detector's probabilities."""
    s = dict(SETTING, **(setting or {}))
    p = np.nan_to_num(np.asarray(probabilities, dtype=np.float64), nan=0.0)
    return tuple(s["cheap"] + (s["base"] - s["cheap"]) * (1.0 - np.clip(p[:, k], 0, 1)) ** s["knob"]
                 for k in (1, 2))


def save(booster, path):
    """Model file plus path.json with the feature list, seed, threads, SETTING and LightGBM version."""
    booster.save_model(path)
    details = {"feature_columns": list(FEATURE_COLUMNS), "seed": SEED, "threads": THREADS,
               "window": WINDOW, "keep_negative": KEEP_NEGATIVE, "setting": SETTING,
               "lightgbm_version": lgb.__version__}
    with open(path + ".json", "w") as handle:
        json.dump(details, handle, indent=2, sort_keys=True)


def load(path):
    """A saved detector; refuses one built with other features, settings or LightGBM version."""
    with open(path + ".json") as handle:
        details = json.load(handle)
    if details["feature_columns"] != list(FEATURE_COLUMNS):
        raise ValueError(path + " was trained on other features than FEATURE_COLUMNS")
    if details["setting"] != SETTING:
        raise ValueError(path + " was saved with another SETTING: {}".format(details["setting"]))
    if details["lightgbm_version"] != lgb.__version__:
        raise ValueError("{} needs LightGBM {}, this is {}".format(
            path, details["lightgbm_version"], lgb.__version__))
    return lgb.Booster(model_file=path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    train_cmd = commands.add_parser("train", help="train on every match of a training table")
    train_cmd.add_argument("--data", required=True, help="training table (--smooth-ball 6)")
    train_cmd.add_argument("--out", required=True, help="model file to write")
    args = parser.parse_args()
    table = read_table(args.data)
    save(train(table), args.out)
    print("trained on {} matches, wrote {}".format(table["match"].nunique(), args.out))


if __name__ == "__main__":
    main()

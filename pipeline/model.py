"""Model: H / A / D probabilities per frame, from a LightGBM classifier.

    python pipeline/model.py train --data <training table> --seed 1 --out <model file>

Trained on damaged external matches; the provided game is never used for training, and the public
5 minutes are never used to choose settings.

Input is a table with one row per frame and the columns in FEATURE_COLUMNS, plus `match` (which game
the row comes from) and `label` (H, A, D, or NO_LABEL outside the halves). Testing always holds out a
whole match, never frames of a match that was trained on.

Reproducible on purpose: fixed seed, deterministic mode, one thread. The trained model is saved to a
file, so producing the submission never retrains.
"""

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

H, A, D, NO_LABEL = 0, 1, 2, -1

SEED = 1
# num_boost_round is how many trees LightGBM builds. lgb.train takes it as its own argument rather
# than inside params, so train() takes it out of this dict before passing the rest.
LIGHTGBM_PARAMS = {
    "objective": "multiclass", "num_class": 3,
    "num_leaves": 31, "learning_rate": 0.05, "num_boost_round": 300,
    "deterministic": True, "num_threads": 1, "verbose": -1,
}

# Every column the model reads, in a fixed order. Edit this list to match the features you build;
# save() records it and load() refuses a model trained on a different list.
FEATURE_COLUMNS = (
    "ball_visible", "ball_speed", "ball_missing_s", "ball_until_s",
    "ball_past_end_m", "ball_past_side_m",
    "dist_home", "dist_away", "dist_difference",
    "home_speed", "away_speed", "home_centre_x", "away_centre_x", "home_spread", "away_spread",
    "n_home", "n_away", "tracking_ok",
)
# Time context (model_features.add_time_context): means of these columns over the 12, 25 and 50
# frames (0.5, 1 and 2 s) before and after each frame, plus what the ball showed around a gap.
CONTEXT_COLUMNS = ("dist_difference", "dist_home", "dist_away", "ball_speed",
                   "home_centre_x", "away_centre_x")
CONTEXT_FRAMES = (12, 25, 50)
FEATURE_COLUMNS += tuple("{}_{}_{}".format(column, side, frames)
                         for column in CONTEXT_COLUMNS for side in ("past", "next")
                         for frames in CONTEXT_FRAMES)
FEATURE_COLUMNS += ("home_nearer_before_gap", "home_nearer_after_gap", "ball_moved_in_gap_m")


def training_matrix(table):
    """(X, y, groups) for LightGBM, from the graded frames of the table.

    X is the feature columns in FEATURE_COLUMNS order, NaN left in because LightGBM handles it.
    y is the label of each row, groups the match it comes from, for leave_one_match_out.
    """
    missing = [name for name in FEATURE_COLUMNS + ("match", "label") if name not in table.columns]
    if missing:
        raise ValueError("training table is missing columns: {}".format(missing))

    graded = table[table["label"] != NO_LABEL]
    return (graded[list(FEATURE_COLUMNS)], graded["label"].to_numpy(dtype=np.int64),
            graded["match"].to_numpy())


def train(X, y, seed=SEED):
    """A trained lightgbm Booster, using LIGHTGBM_PARAMS and the given seed."""
    params = dict(LIGHTGBM_PARAMS)
    rounds = params.pop("num_boost_round")
    params["seed"] = seed
    return lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=rounds)


def predict_scores(model, table):
    """Probabilities, frames x 3, columns in order H, A, D. Every row of the table gets one."""
    booster = getattr(model, "booster_", model)
    return booster.predict(table[list(FEATURE_COLUMNS)])


def leave_one_match_out(table, seed=SEED):
    """{match: scores} where each match's scores come from a model trained on all the others.

    Scores cover every row of the held-out match, graded or not, in table order.
    """
    X, y, groups = training_matrix(table)
    scores = {}
    for match in sorted(set(groups)):
        others = groups != match
        model = train(X[others], y[others], seed)
        scores[match] = predict_scores(model, table[table["match"] == match])
    return scores


def save(model, path, seed=SEED):
    """Write the model to path, and its FEATURE_COLUMNS, seed and LightGBM version to path.json.

    Takes a Booster from lgb.train or a fitted LGBMClassifier. The model file is LightGBM's own text
    format, so the same model always gives the same file.
    """
    booster = getattr(model, "booster_", model)
    booster.save_model(path)
    details = {"feature_columns": list(FEATURE_COLUMNS), "seed": seed,
               "lightgbm_version": lgb.__version__}
    with open(path + ".json", "w") as handle:
        json.dump(details, handle, indent=2, sort_keys=True)


def load(path):
    """The Booster saved by save().

    Raises if the model was trained on different FEATURE_COLUMNS, or saved by another LightGBM
    version, because either would quietly change the predictions and so the submission.
    """
    with open(path + ".json") as handle:
        details = json.load(handle)
    if tuple(details["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError("{} was trained on different FEATURE_COLUMNS; retrain it".format(path))
    if details["lightgbm_version"] != lgb.__version__:
        raise ValueError("{} was saved by LightGBM {}, this is {}".format(
            path, details["lightgbm_version"], lgb.__version__))

    booster = lgb.Booster(model_file=path)
    if booster.feature_name() != list(FEATURE_COLUMNS):
        raise ValueError("{}: the model's own feature names differ from FEATURE_COLUMNS".format(path))
    return booster


def read_table(path):
    """A training table from a .parquet or .csv file."""
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    train_cmd = sub.add_parser("train")
    train_cmd.add_argument("--data", required=True, help="training table, one row per frame")
    train_cmd.add_argument("--seed", type=int, default=SEED)
    train_cmd.add_argument("--out", required=True, help="model file to write")
    args = parser.parse_args()

    X, y, groups = training_matrix(read_table(args.data))
    model = train(X, y, args.seed)
    save(model, args.out, args.seed)
    print("trained on {} frames from {} matches, wrote {}".format(
        len(y), len(set(groups)), args.out))


if __name__ == '__main__':
    main()

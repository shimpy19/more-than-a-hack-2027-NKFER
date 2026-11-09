"""Bench: leave-one-match-out on the external matches, scored with the official metric code.

    python pipeline/bench.py --data <training table .csv> --results <folder outside the repo>

The training table is the one training_table.py writes. Each match is scored by a model trained on
the other matches only, so no match is ever scored by a model that saw it. BAR-ATH is not used.

Writes to the results folder:
  model_scores/<match>.npy   held-out H/A/D probabilities, one row per frame of that match, so
                             bench_viterbi.py can decode them without retraining
  results.csv                spellF1, macroF1 and accuracy per match and method
  feature_importance.csv     share of gain per feature, from one model trained on every match

Methods compared:
  baseline     the starter's nearest player, smoothed +/- 2 s (baseline_smooth50)
  model raw    LightGBM, most likely label per frame
  model + 1 s  LightGBM probabilities averaged over +/- 1 s, then the most likely label
  model + 2 s  the same over +/- 2 s
Viterbi, which submit.py uses, is scored by bench_viterbi.py on the saved model scores.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "evaluation"), os.path.dirname(HERE)]
import model  # noqa: E402
import possession_metrics as pm  # noqa: E402
from features import ROOT  # noqa: E402
from starter.baseline import AWAY, HOME, NOBODY, smooth  # noqa: E402

FPS = 25
METHODS = ("baseline", "model raw", "model + 1 s", "model + 2 s")


def refuse_repo_path(path):
    """Results come from match data, which must never be written inside the repository."""
    if os.path.abspath(path).startswith(ROOT + os.sep):
        raise ValueError("refusing to write match data inside the repository: " + path)


def gt_codes(table):
    """The labels in the scorer's codes, X outside the halves."""
    label = table["label"].to_numpy()
    return np.where(label >= 0, label, pm.X)


def averaged(scores, half, reach):
    """Probabilities averaged over a centred window of +/- reach frames, never across a half."""
    out = scores.copy()
    for number in (1, 2):
        rows = np.flatnonzero(half == number)
        if not len(rows):
            continue
        block = scores[rows[0]:rows[-1] + 1]
        cumulative = np.vstack((np.zeros((1, 3)), np.cumsum(block, axis=0)))
        index = np.arange(len(block))
        low, high = np.maximum(0, index - reach), np.minimum(len(block), index + reach + 1)
        out[rows[0]:rows[-1] + 1] = (cumulative[high] - cumulative[low]) / (high - low)[:, None]
    return out


def baseline_codes(table):
    """The starter baseline from the same table: nearest team to the ball, smoothed +/- 2 s."""
    dist_home, dist_away = table["dist_home"].to_numpy(), table["dist_away"].to_numpy()
    nearest = np.full(len(table), NOBODY, dtype=np.int8)
    seen = ~np.isnan(dist_home) | ~np.isnan(dist_away)
    home_closer = np.nan_to_num(dist_home, nan=np.inf) <= np.nan_to_num(dist_away, nan=np.inf)
    nearest[seen] = np.where(home_closer[seen], HOME, AWAY)
    return pm.label_codes(smooth(nearest, 2 * FPS))


def score_methods(held, scores):
    """[{match, method, spellF1, macroF1, accuracy}] for one held-out match."""
    gt, half = gt_codes(held), held["half"].to_numpy()
    predictions = {
        "baseline": baseline_codes(held),
        "model raw": scores.argmax(axis=1),
        "model + 1 s": averaged(scores, half, FPS).argmax(axis=1),
        "model + 2 s": averaged(scores, half, 2 * FPS).argmax(axis=1),
    }
    rows = []
    for method in METHODS:
        result = pm.evaluate(gt, predictions[method])
        rows.append({"match": held["match"].iloc[0], "method": method,
                     "spellF1": result["spell_f1_mean"], "macroF1": result["macro_f1"],
                     "accuracy": result["accuracy"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True, help="training table from training_table.py")
    parser.add_argument("--results", required=True, help="folder to write the results to")
    args = parser.parse_args()
    refuse_repo_path(args.results)

    table = model.read_table(args.data)
    graded = table["label"] != model.NO_LABEL
    print("{} matches, {} graded frames; labels H {:.0%}  A {:.0%}  D {:.0%}".format(
        table["match"].nunique(), int(graded.sum()),
        *[(table.loc[graded, "label"] == k).mean() for k in (model.H, model.A, model.D)]),
        flush=True)

    scores = model.leave_one_match_out(table)
    os.makedirs(os.path.join(args.results, "model_scores"), exist_ok=True)
    rows = []
    for match in sorted(scores):
        np.save(os.path.join(args.results, "model_scores", match + ".npy"),
                scores[match].astype(np.float32))
        rows += score_methods(table[table["match"] == match], scores[match])

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(args.results, "results.csv"), index=False)
    for metric in ("spellF1", "macroF1"):
        wide = results.pivot(index="match", columns="method", values=metric)[list(METHODS)]
        wide.loc["MEAN"] = wide.mean()
        print("\n== {}\n{}".format(metric, wide.round(3).to_string()))

    X, y, _ = model.training_matrix(table)
    gain = pd.Series(model.train(X, y).feature_importance("gain"), index=model.FEATURE_COLUMNS)
    share = (gain / gain.sum()).sort_values(ascending=False).rename("gain_share")
    share.to_csv(os.path.join(args.results, "feature_importance.csv"), index_label="feature")
    print("\n== feature importance, top 10 (share of gain)\n" + share.head(10).round(3).to_string())
    print("\nnext: python pipeline/bench_viterbi.py --data {} --results {}".format(
        args.data, args.results))


if __name__ == "__main__":
    main()

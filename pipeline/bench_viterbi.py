"""Bench, step 2: Viterbi on the held-out model scores, with the switch costs chosen fairly.

    python pipeline/bench_viterbi.py --data <training table .csv> --results <folder>

Run after bench.py, with the same --data and --results. It reads results/model_scores/, so
nothing is retrained.

Every (team, dead) pair in GRID is decoded on every match. Then, for each match, the pair that
scores best on the OTHER matches is used to score this one, so no match is scored with costs it
helped choose. Writes viterbi_grid.csv (every pair on every match) and viterbi_results.csv (the
fair score per match) to the results folder.
"""

import argparse
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "evaluation"), os.path.dirname(HERE)]
import possession_metrics as pm  # noqa: E402
import viterbi  # noqa: E402
from bench import gt_codes, refuse_repo_path  # noqa: E402

GRID = (4, 8, 15, 30, 60, 120, 240)
MATCHES = {}


def read_labels(path):
    """match, half and label columns of the training table (the features are not needed)."""
    columns = ["match", "half", "label"]
    if path.endswith(".parquet"):
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def init(matches):
    MATCHES.update(matches)


def run(job):
    """Score one (match, team cost, dead cost)."""
    match, team, dead = job
    gt, half, scores = MATCHES[match]
    result = pm.evaluate(gt, viterbi.decode_match(scores, half, team=team, dead=dead))
    return {"match": match, "team": team, "dead": dead, "spellF1": result["spell_f1_mean"],
            "macroF1": result["macro_f1"], "n_pred_spells": result["n_pred_spells"],
            "n_gt_spells": result["n_gt_spells"]}


def fair_choice(grid):
    """For each match, its row at the cost pair that scores best on the other matches."""
    rows = []
    for match in sorted(grid["match"].unique()):
        others = grid[grid["match"] != match].groupby(["team", "dead"])["spellF1"].mean()
        team, dead = others.idxmax()
        row = grid[(grid["match"] == match) & (grid["team"] == team) & (grid["dead"] == dead)]
        rows.append(row.iloc[0])
    return pd.DataFrame(rows)[["match", "team", "dead", "spellF1", "macroF1", "n_pred_spells",
                               "n_gt_spells"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True, help="training table from training_table.py")
    parser.add_argument("--results", required=True, help="the folder bench.py wrote to")
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()
    refuse_repo_path(args.results)

    labels = read_labels(args.data)
    matches = {}
    for match, held in labels.groupby("match", sort=True):
        scores = np.load(os.path.join(args.results, "model_scores", match + ".npy"))
        if len(scores) != len(held):
            raise ValueError("{}: {} model scores for {} frames; re-run bench.py".format(
                match, len(scores), len(held)))
        matches[match] = (gt_codes(held), held["half"].to_numpy(), scores.astype(np.float64))

    jobs = [(m, t, d) for m in sorted(matches) for t, d in itertools.product(GRID, GRID)]
    with Pool(args.processes, initializer=init, initargs=(matches,)) as pool:
        grid = pd.DataFrame(pool.map(run, jobs, chunksize=4))
    grid.to_csv(os.path.join(args.results, "viterbi_grid.csv"), index=False)

    fair = fair_choice(grid)
    fair.to_csv(os.path.join(args.results, "viterbi_results.csv"), index=False)
    print("== each match scored with costs chosen on the other matches")
    print(fair.round(3).to_string(index=False))
    print("MEAN spellF1 {:.3f}  macroF1 {:.3f}\n".format(fair["spellF1"].mean(),
                                                          fair["macroF1"].mean()))
    mean = grid.groupby(["team", "dead"])["spellF1"].mean()
    print("== mean spellF1 over all matches, rows = team cost, columns = dead cost")
    print(mean.unstack().round(3).to_string())
    print("\nbest pair over all matches: team {} dead {}".format(*mean.idxmax()))
    print("viterbi.py uses team {team} dead {dead}".format(**viterbi.SWITCH_COSTS))


if __name__ == "__main__":
    main()

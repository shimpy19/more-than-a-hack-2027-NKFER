"""Bench for the possession-change detector, on matches whose ball looks like BAR-ATH's.

    python pipeline/bench_detector.py --data <training table> \
        --smooth-data <training table made with --smooth-ball 6> --results <folder>

Every match is scored on its smoothed version (BAR-ATH-like ball), leave one match out:
  - main model trained on the other matches' original table (as models/lightgbm_v2.txt is)
  - detector trained on the other matches' smoothed table (as models/detector_v1.txt is)
Viterbi without the detector (fixed team / dead costs, GRID x GRID) and with it (DETECTOR_GRID),
each match scored with the setting that is best on the other matches. Writes model_scores/,
detector_scores/, grid.csv and results.csv to the results folder, and prints the best detector
setting over all matches, which is what detector.SETTING should be.
"""

import argparse
import itertools
import multiprocessing
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "evaluation"), os.path.dirname(HERE)]
import detector  # noqa: E402
import model  # noqa: E402
import possession_metrics as pm  # noqa: E402
import viterbi  # noqa: E402
from bench import gt_codes, refuse_repo_path  # noqa: E402
from bench_viterbi import GRID  # noqa: E402

DETECTOR_GRID = {"base": (30.0, 60.0, 120.0), "knob": (1.0, 4.0, 16.0), "cheap": (0.0, 3.0),
                 "dead": (30.0, 60.0)}
TABLES, MATCHES = {}, {}


def fit(job):
    """Held-out probabilities for one match: ('main', m) or ('detector', m)."""
    kind, match = job
    if kind == "main":
        table = TABLES["original"]
        keep = ((table["label"] != model.NO_LABEL) & (table["match"] != match)).to_numpy()
        booster = model.train(table.loc[keep, list(model.FEATURE_COLUMNS)],
                              table.loc[keep, "label"].to_numpy(np.int64))
        return job, model.predict_scores(booster, TABLES["smooth"][TABLES["smooth"]["match"] == match])
    booster = detector.train(TABLES["smooth"], leave_out=match)
    return job, detector.predict(booster, TABLES["smooth"][TABLES["smooth"]["match"] == match])


def init(matches):
    MATCHES.update(matches)


def run(job):
    """Score one match with one setting: ('plain', m, team, dead) or ('detector', m, setting)."""
    kind, match, setting = job
    gt, half, scores, probabilities = MATCHES[match]
    if kind == "plain":
        pred = viterbi.decode_match(scores, half, team=setting[0], dead=setting[1])
    else:
        s = dict(zip(("base", "knob", "cheap", "dead"), setting))
        into_h, into_a = detector.switch_costs(probabilities, s)
        pred = viterbi.decode_match_varying(scores, half, into_h, into_a, s["dead"])
    result = pm.evaluate(gt, pred)
    return {"kind": kind, "match": match, "setting": setting, "spellF1": result["spell_f1_mean"],
            "macroF1": result["macro_f1"], "n_pred_spells": result["n_pred_spells"],
            "n_gt_spells": result["n_gt_spells"]}


def fair_choice(grid):
    """For each match, its row at the setting that scores best on the other matches."""
    rows = []
    for match in sorted(grid["match"].unique()):
        others = grid[grid["match"] != match].groupby("key")["spellF1"].mean()
        rows.append(grid[(grid["match"] == match) & (grid["key"] == others.idxmax())].iloc[0])
    return pd.DataFrame(rows).set_index("match")


def alarms_per_minute(probabilities, half, threshold=0.5, separation=12):
    """Peaks of P(change) above threshold at least `separation` frames apart, per graded minute."""
    p = np.where(half > 0, probabilities[:, 1] + probabilities[:, 2], 0.0)
    taken, count = np.zeros(len(p), dtype=bool), 0
    for frame in np.flatnonzero(p >= threshold)[np.argsort(-p[p >= threshold], kind="stable")]:
        if not taken[frame]:
            count += 1
            taken[max(0, frame - separation + 1):frame + separation] = True
    return count / ((half > 0).sum() / 1500.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", required=True, help="training table (original balls)")
    parser.add_argument("--smooth-data", required=True, help="training table, --smooth-ball 6")
    parser.add_argument("--results", required=True, help="folder to write the results to")
    parser.add_argument("--processes", type=int, default=4)
    args = parser.parse_args()
    refuse_repo_path(args.results)
    for name in ("model_scores", "detector_scores"):
        os.makedirs(os.path.join(args.results, name), exist_ok=True)

    TABLES["original"] = model.read_table(args.data)
    TABLES["smooth"] = model.read_table(args.smooth_data)
    matches = sorted(TABLES["smooth"]["match"].unique())
    jobs = [(kind, m) for kind in ("main", "detector") for m in matches]
    with multiprocessing.get_context("fork").Pool(args.processes) as pool:
        fitted = dict(pool.map(fit, jobs))
    data = {}
    for m in matches:
        held = TABLES["smooth"][TABLES["smooth"]["match"] == m]
        scores, probabilities = fitted[("main", m)], fitted[("detector", m)]
        np.save(os.path.join(args.results, "model_scores", m + ".npy"), scores.astype(np.float32))
        np.save(os.path.join(args.results, "detector_scores", m + ".npy"),
                probabilities.astype(np.float32))
        data[m] = (gt_codes(held), held["half"].to_numpy(), scores, probabilities)

    settings = list(itertools.product(*DETECTOR_GRID.values()))
    grid_jobs = [("plain", m, s) for m in matches for s in itertools.product(GRID, GRID)]
    grid_jobs += [("detector", m, s) for m in matches for s in settings]
    with multiprocessing.get_context("spawn").Pool(8, initializer=init, initargs=(data,)) as pool:
        grid = pd.DataFrame(pool.map(run, grid_jobs, chunksize=4))
    grid["setting"] = grid["setting"].map(tuple)
    grid["key"] = grid["setting"].map(str)
    grid.to_csv(os.path.join(args.results, "grid.csv"), index=False)

    plain = fair_choice(grid[grid["kind"] == "plain"])
    with_detector = fair_choice(grid[grid["kind"] == "detector"])
    results = pd.DataFrame({"plain": plain["spellF1"], "with detector": with_detector["spellF1"],
                            "plain macroF1": plain["macroF1"],
                            "detector macroF1": with_detector["macroF1"],
                            "alarms/min": [alarms_per_minute(data[m][3], data[m][1])
                                           for m in plain.index]})
    results.to_csv(os.path.join(args.results, "results.csv"))
    pd.set_option("display.width", 200)
    print("== spellF1 per match on the smoothed matches (settings chosen on the other matches)")
    print(results.round(3).to_string())
    better = int((results["with detector"] > results["plain"]).sum())
    print("MEAN plain {:.3f} / {:.3f}   with detector {:.3f} / {:.3f}   ({:+.3f}, better on {} of {})"
          .format(results["plain"].mean(), results["plain macroF1"].mean(),
                  results["with detector"].mean(), results["detector macroF1"].mean(),
                  results["with detector"].mean() - results["plain"].mean(), better, len(results)))
    print("possessions: plain {}, with detector {} ({:+.1%})".format(
        int(plain["n_pred_spells"].sum()), int(with_detector["n_pred_spells"].sum()),
        with_detector["n_pred_spells"].sum() / plain["n_pred_spells"].sum() - 1))
    for kind, names in (("plain", ("team", "dead")),
                        ("detector", tuple(DETECTOR_GRID))):
        part = grid[grid["kind"] == kind]
        mean = part.groupby("key")["spellF1"].mean()
        setting = part.loc[part["key"] == mean.idxmax(), "setting"].iloc[0]
        print("best {} setting over all matches: {} ({:.3f})".format(
            kind, dict(zip(names, setting)), mean.max()))
    print("detector.SETTING is {}".format(detector.SETTING))


if __name__ == "__main__":
    main()

"""How accurate the possession-change moments are, on the external matches only.

    python pipeline/measure_moments.py --external <folder> --results <folder outside the repo> [--sweep]

Loads every external match the same way training_table.py does, computes rules_moments on it and
compares the moments with the true labels. Prints only aggregate numbers per match and their mean,
and writes them to <results>/moments_metrics.csv. BAR-ATH is never read.

What is measured
  anchors_all / anchors_confident   restart kicks against the true ends of dead balls (D -> H/A),
                                    within +/- --tolerance frames: precision, recall, team
                                    accuracy, mean and median offset (detected minus true)
  anchor_type_<type>                precision of the anchors of one restart type, all matches
  turnover                          precision: candidates within +/- --turnover-window frames of a
                                    true H <-> A change; recall: true changes with a candidate that
                                    close
  out_<kind>                        frames flagged, and the share that are D or followed by D
                                    within --out-window-s seconds
  ball_visible_at_restart           share of true restarts where the ball is seen on that frame,
                                    an upper bound for the anchors' recall

--sweep also scores a grid of anchor thresholds (summed over all matches) and writes
<results>/anchor_sweep.csv. Choose thresholds from it, never from the provided game, and write the
choice and the reason in the decisions log.
"""

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import rules_moments as rm  # noqa: E402
from bench import refuse_repo_path  # noqa: E402
from model_features import feature_table  # noqa: E402
from training_table import external_matches, load_match  # noqa: E402

FPS = 25
SWEEP = {
    "still_min_s": (0.6, 1.0, 1.5),
    "kick_speed": (3.0, 4.0, 6.0),
    "slow_players": (1.2, 1.5, 2.0),
}


def team_changes(labels):
    """Frames where the label goes from H to A or from A to H."""
    labels = np.asarray(labels)
    live = np.isin(labels, (rm.H, rm.A))
    return np.flatnonzero(live[1:] & live[:-1] & (labels[1:] != labels[:-1])) + 1


def within(frames, targets, window):
    """For each frame, whether some target is within +/- window frames. Both sorted."""
    frames, targets = np.asarray(frames), np.asarray(targets)
    if not len(frames) or not len(targets):
        return np.zeros(len(frames), dtype=bool)
    j = np.searchsorted(targets, frames)
    before = np.abs(frames - targets[np.maximum(j - 1, 0)])
    after = np.abs(targets[np.minimum(j, len(targets) - 1)] - frames)
    return np.minimum(before, after) <= window


def share(mask):
    return float(np.mean(mask)) if len(mask) else float("nan")


def anchor_scores(anchors, labels, tolerance, prefix):
    result = rm.evaluate_anchors(anchors, labels, tolerance)
    return {"{}_{}".format(prefix, key): value for key, value in result.items()}


def measure_match(name, columns, pitch, labels, args):
    """One row of aggregate numbers for one match, plus per-anchor hits for the type summary."""
    table = feature_table(columns)
    dist_home, dist_away = table["dist_home"].to_numpy(), table["dist_away"].to_numpy()
    moments, anchors = rm.moments_from_columns(columns, pitch, dist_home, dist_away)
    graded = np.asarray(columns["half"]) > 0

    row = {"match": name}
    row.update(anchor_scores(anchors, labels, args.tolerance, "anchors_all"))
    row.update(anchor_scores(anchors[anchors["confident"]], labels, args.tolerance,
                             "anchors_confident"))

    restarts = rm.true_restarts(labels)
    row["ball_visible_at_restart"] = share(~np.isnan(np.asarray(columns["ball_x_m"])[restarts]))

    candidates = np.flatnonzero((moments["turnover_candidate"] == 1) & graded)
    changes = team_changes(labels)
    row["turnover_candidates"] = len(candidates)
    row["true_team_changes"] = len(changes)
    row["turnover_precision"] = share(within(candidates, changes, args.turnover_window))
    row["turnover_recall"] = share(within(changes, candidates, args.turnover_window))

    dead = np.flatnonzero(labels == rm.D)
    reach = int(round(args.out_window_s * FPS))
    for kind in ("out_throw_in", "out_goal", "out_end_line"):
        flagged = np.flatnonzero((moments[kind] == 1) & graded)
        ok = np.zeros(len(flagged), dtype=bool)
        if len(dead):  # D on this frame or within the next `reach` frames
            next_dead = dead[np.minimum(np.searchsorted(dead, flagged), len(dead) - 1)]
            ok = (next_dead >= flagged) & (next_dead - flagged <= reach)
        row[kind + "_frames"] = len(flagged)
        row[kind + "_dead_soon"] = share(ok)

    # which anchors hit, for the precision per restart type
    ordered = anchors.sort_values("frame_index").reset_index(drop=True)
    hit, _ = rm.match_events(ordered["frame_index"].to_numpy(), restarts, args.tolerance)
    ordered["hit"] = False
    ordered.loc[hit, "hit"] = True
    return row, ordered[["type", "confident", "hit"]]


def sweep(loaded, args):
    """Anchor precision and recall for every combination in SWEEP, summed over all matches."""
    rows = []
    for values in itertools.product(*SWEEP.values()):
        thresholds = dict(zip(SWEEP, values))
        totals = {"all": [0, 0], "confident": [0, 0]}  # [detected, hits]
        n_true = 0
        for columns, pitch, labels, dist_home, dist_away in loaded:
            _, anchors = rm.moments_from_columns(columns, pitch, dist_home, dist_away, thresholds)
            n_true += len(rm.true_restarts(labels))
            for key, chosen in (("all", anchors), ("confident", anchors[anchors["confident"]])):
                result = rm.evaluate_anchors(chosen, labels, args.tolerance)
                totals[key][0] += result["detected"]
                totals[key][1] += result["hits"]
        row = dict(thresholds)
        for key, (detected, hits) in totals.items():
            row[key + "_detected"] = detected
            row[key + "_precision"] = hits / detected if detected else float("nan")
            row[key + "_recall"] = hits / n_true if n_true else float("nan")
        rows.append(row)
        print("  " + "  ".join("{}={}".format(k, v) for k, v in thresholds.items())
              + "  confident: precision {:.3f} recall {:.3f}".format(
                  row["confident_precision"], row["confident_recall"]), flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--external", required=True, help="folder with the external matches")
    parser.add_argument("--results", required=True, help="folder to write the results to")
    parser.add_argument("--tolerance", type=int, default=5, help="frames, anchor vs true restart")
    parser.add_argument("--turnover-window", type=int, default=12, help="frames")
    parser.add_argument("--out-window-s", type=float, default=2.0, help="seconds")
    parser.add_argument("--sweep", action="store_true", help="also score a grid of thresholds")
    args = parser.parse_args()
    refuse_repo_path(args.results)
    os.makedirs(args.results, exist_ok=True)

    rows, hits, loaded = [], [], []
    for name, tracking, meta, labels_path in external_matches(args.external):
        columns, pitch, labels = load_match(name, tracking, meta, labels_path)
        row, match_hits = measure_match(name, columns, pitch, labels, args)
        rows.append(row)
        hits.append(match_hits)
        print("  measured {}".format(name), flush=True)
        if args.sweep:
            table = feature_table(columns)
            loaded.append((columns, pitch, labels, table["dist_home"].to_numpy(),
                           table["dist_away"].to_numpy()))

    results = pd.DataFrame(rows).set_index("match")
    results.loc["MEAN"] = results.mean()
    hits = pd.concat(hits, ignore_index=True)
    for kind in rm.RESTART_TYPES:
        chosen = hits[hits["type"] == kind]
        results.loc["MEAN", "anchor_type_{}_count".format(kind)] = len(chosen)
        results.loc["MEAN", "anchor_type_{}_precision".format(kind)] = share(chosen["hit"])
    results.to_csv(os.path.join(args.results, "moments_metrics.csv"))

    shown = ["anchors_all_precision", "anchors_all_recall", "anchors_confident_precision",
             "anchors_confident_recall", "anchors_confident_team_accuracy",
             "anchors_confident_median_offset_frames", "ball_visible_at_restart",
             "turnover_precision", "turnover_recall"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n== per match\n" + results[shown].round(3).to_string())
        types = results.loc["MEAN", [c for c in results.columns if c.startswith("anchor_type_")]]
        print("\n== anchors per restart type (all matches)\n" + types.round(3).to_string())
        outs = [c for c in results.columns if c.startswith("out_")]
        print("\n== ball out\n" + results.loc["MEAN", outs].round(3).to_string())
    print("\nwrote {}".format(os.path.join(args.results, "moments_metrics.csv")))

    if args.sweep:
        print("\n== sweep")
        grid = sweep(loaded, args)
        grid.to_csv(os.path.join(args.results, "anchor_sweep.csv"), index=False)
        best = grid.sort_values(["confident_precision", "confident_recall"], ascending=False)
        print("\n== top 5 by confident precision\n" + best.head(5).round(3).to_string(index=False))
        print("wrote {}".format(os.path.join(args.results, "anchor_sweep.csv")))


if __name__ == "__main__":
    main()

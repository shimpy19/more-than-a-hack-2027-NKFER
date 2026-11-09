"""Possession file for BAR-ATH: features, saved model, Viterbi, one label per video frame.

    python pipeline/submit.py --clean <BAR-ATH_tactical_clean.npz> --model <model file> \
        --out <submission .csv>

Reads the cleaned match that clean.py writes and the model that `model.py train` saved, so making
the file never trains anything. Frames outside the halves are X (never graded). Viterbi uses its
frozen SWITCH_COSTS, chosen on the external matches.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
import model  # noqa: E402
import viterbi  # noqa: E402
from features import ROOT  # noqa: E402
from games import GAME, source_frames  # noqa: E402
from model_features import feature_table  # noqa: E402

LETTERS = np.array(["H", "A", "D"])


def labels_for(clean_path, model_path):
    """(frame numbers, letters) for every video frame of the cleaned match."""
    table = feature_table(np.load(clean_path))
    scores = model.predict_scores(model.load(model_path), table)
    half = table["half"].to_numpy()
    letters = LETTERS[viterbi.decode_match(scores, half)]
    return table["frame"].to_numpy(), np.where(half > 0, letters, "X")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--clean", required=True, help="BAR-ATH_tactical_clean.npz from clean.py")
    parser.add_argument("--model", required=True, help="model file from `model.py train`")
    parser.add_argument("--out", required=True, help="submission file to write")
    args = parser.parse_args()
    if os.path.abspath(args.out).startswith(ROOT + os.sep):
        raise ValueError("refusing to write match data inside the repository: " + args.out)

    frames, letters = labels_for(args.clean, args.model)
    expected = source_frames("tactical", GAME)
    if len(frames) != expected or (frames != np.arange(expected)).any():
        raise ValueError("expected frames 0..{}, got {} rows".format(expected - 1, len(frames)))
    pd.DataFrame({"frame": frames, "label": letters}).to_csv(args.out, index=False)
    print("wrote {} rows to {}".format(len(frames), args.out))


if __name__ == "__main__":
    main()

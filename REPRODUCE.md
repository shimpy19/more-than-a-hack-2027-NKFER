# Reproducing our submission

Running the steps below on the provided match data regenerates our possession file **byte for
byte**:

```
md5  cad9aed887ab27f6d4f7b11de3d69c84   <team_name>.csv
```

The model is committed (`models/lightgbm_v2.txt`), so making the file trains nothing. Section 4
retrains that model from the external matches, if you want to check it too.

## 1. Install

Python **3.13.12**, with the exact package versions. A different LightGBM, numpy or pandas can
change a probability in its last digits, and that can move a decision by one frame.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-pipeline.txt
```

## 2. The match data

Unpack the competition bundle into a folder **outside this repository**. The scripts refuse to
write match data inside it, because the repository is public. `clean.py` finds the four files it
needs by name anywhere under the folder you give it:
- `BAR-ATH_tactical_atd.txt`
- `BAR-ATH_tactical_atd.xml`
- `BAR-ATH_tactical_smart_tagging.xml`
- `BAR-ATH_tactical_homography.csv`

## 3. Make the file

```bash
.venv/bin/python pipeline/clean.py --data-root <bundle folder> --out <work>/cleaned
.venv/bin/python pipeline/submit.py --clean <work>/cleaned/BAR-ATH_tactical_clean.npz \
    --model models/lightgbm_v2.txt --out <work>/<team_name>.csv
.venv/bin/python evaluation/validate_submission.py <work>/<team_name>.csv
md5 <work>/<team_name>.csv        # md5sum on Linux
```

What each step does:
1. **`clean.py`** cleans the tracking. It removes impossible player positions and duplicates, cuts identity switches, recomputes speeds and flags suspect ball positions.
2. **`submit.py`** computes the 90 features per frame (`model_features.py`) and gets H / A / D probabilities from the LightGBM model (`model.py`).
3. **Viterbi** (`viterbi.py`, team 15 / dead 30) then turns those probabilities into steady possessions. Frames outside the two halves are `X`.

## 4. Optional: retrain the model

This needs the external matches, which are match data and not in this repository. Each match is
one folder with `tracking.txt`, `metadata.xml` (Metrica text format) and `possession.csv`
(`frame,label` with H / A / D). The team prepared these files from public sources:
- the 7 DFL Bundesliga matches of the Sportec Solutions open tracking dataset: `J03WMX`, `J03WN1`, `J03WOH`, `J03WOY`, `J03WPY`, `J03WQQ`, `J03WR9`;
- Metrica Sports sample games 2 and 3.

**Make the Metrica copies first.** They lose players BAR-ATH style, and keep the ball as recorded:

```bash
.venv/bin/python pipeline/damage.py --tracking <metrica_n>/tracking.txt \
    --meta <metrica_n>/metadata.xml --possession <metrica_n>/possession.csv \
    --barca-clean <work>/cleaned/BAR-ATH_tactical_clean.npz --seed 1 \
    --out <external>/barca_style/metrica_<n>
```

**Then the DFL copies.** They lose players BAR-ATH style, and the ball the way the Metrica copies
do:

```bash
.venv/bin/python pipeline/damage.py --tracking <J03..>/tracking.txt \
    --meta <J03..>/metadata.xml --possession <J03..>/possession.csv \
    --barca-clean <work>/cleaned/BAR-ATH_tactical_clean.npz \
    --ball-like <external>/barca_style/metrica_2 <external>/barca_style/metrica_3 \
    --seed 1 --out <external>/barca_style/<J03..>
```

**Then the training table and the model:**

```bash
.venv/bin/python pipeline/training_table.py --external <external> --out <work>/training_table.csv
.venv/bin/python pipeline/model.py train --data <work>/training_table.csv --seed 1 \
    --out <work>/lightgbm.txt
# md5 ab751e12a8b0c62900dceb5e7346c4ed, the same as models/lightgbm_v2.txt
```

`damage.py` reads BAR-ATH's cleaned tracking. It only measures how many players go missing and
when the ball is missing. It never reads a label, and nothing it measures is written anywhere.
No BAR-ATH label is used for training or for choosing any setting. The public 5 minutes were
used only to check that the file is correct.

## 5. Optional: the bench

The bench scores each external match with a model trained on the other 8:

```bash
.venv/bin/python pipeline/bench.py --data <work>/training_table.csv --results <folder>
.venv/bin/python pipeline/bench_viterbi.py --data <work>/training_table.csv --results <folder>
```

Expected result: mean spellF1 0.632, macroF1 0.890. All results are in
[pipeline/evaluation_results.txt](pipeline/evaluation_results.txt), and a short overview is in
[pipeline/MODEL_SUMMARY.md](pipeline/MODEL_SUMMARY.md).

## Tests

```bash
for t in pipeline/test_*.py; do .venv/bin/python -m unittest "$t"; done
```

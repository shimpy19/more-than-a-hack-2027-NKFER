# Reproducing our submission

Running the steps below on the provided match data regenerates our possession file **byte for
byte**:

```
md5  9afc4c4ff4a629777d5da82058595712   <team_name>.csv
```

Both models are committed (`models/lightgbm_v2.txt` and `models/detector_v1.txt`), so making the
file trains nothing. Section 4 retrains them from the external matches, if you want to check them
too.

## 1. Install

Python **3.13.12**, with the exact package versions. A different LightGBM, numpy or pandas can
change a probability in its last digits, and that can move a decision by one frame.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements-pipeline.txt
```

**On macOS**, LightGBM also needs the OpenMP runtime, which its package does not include. Install it
first, or `import lightgbm` fails with "Library not loaded: libomp.dylib":

```bash
brew install libomp
```

Linux and Windows need nothing extra. The file was regenerated and checked byte for byte on macOS
(Apple Silicon, Python 3.13.12).

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
    --model models/lightgbm_v2.txt --detector models/detector_v1.txt \
    --out <work>/<team_name>.csv
.venv/bin/python evaluation/validate_submission.py <work>/<team_name>.csv
md5 <work>/<team_name>.csv        # md5sum on Linux
```

What each step does:
1. **`clean.py`** cleans the tracking. It removes impossible player positions and duplicates, cuts identity switches, recomputes speeds and flags suspect ball positions.
2. **`submit.py`** computes the 90 features per frame (`model_features.py`). From them it gets two sets of probabilities:
   - H / A / D per frame from the main model (`model.py`);
   - "possession changes to H / to A here" from the possession-change detector (`detector.py`).
3. **Viterbi** (`viterbi.py`) turns the probabilities into steady possessions:
   - switching team is cheap where the detector expects a change and expensive elsewhere (`detector.SETTING`);
   - entering or leaving a dead ball costs 30;
   - frames outside the two halves are `X`.

Without `--detector`, the same command gives the previous version, v3, which uses fixed costs (team 15 / dead 30): md5 `cad9aed887ab27f6d4f7b11de3d69c84`.

## 4. Optional: retrain the models

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

**Then the main model**, from the training table:

```bash
.venv/bin/python pipeline/training_table.py --external <external> --out <work>/training_table.csv
.venv/bin/python pipeline/model.py train --data <work>/training_table.csv --seed 1 \
    --out <work>/lightgbm.txt
# md5 ab751e12a8b0c62900dceb5e7346c4ed, the same as models/lightgbm_v2.txt
```

**Then the detector**, from a second table where each match's ball is smoothed like BAR-ATH's
(Gaussian, 6 frames, inside each stretch where the ball is seen):

```bash
.venv/bin/python pipeline/training_table.py --external <external> --smooth-ball 6 \
    --out <work>/training_table_smooth.csv
.venv/bin/python pipeline/detector.py train --data <work>/training_table_smooth.csv \
    --out <work>/detector.txt
# md5 ececd6b3613c84b2439793dc8b99be7d, the same as models/detector_v1.txt
```

Two scripts read BAR-ATH's cleaned tracking, and neither ever reads a label:
- **`damage.py`** measures how many players go missing and when the ball is missing. Nothing it measures is written anywhere.
- **The smoothing width (6 frames)** was chosen so that the training balls' velocity changes look like BAR-ATH's.

No BAR-ATH label is used for training or for choosing any setting. The public 5 minutes were
used only to check that the file is correct.

## 5. Optional: the benches

Each external match is scored with models trained on the other 8:

```bash
# the main model with fixed costs (v3): spellF1 0.632, macroF1 0.890
.venv/bin/python pipeline/bench.py --data <work>/training_table.csv --results <folder>
.venv/bin/python pipeline/bench_viterbi.py --data <work>/training_table.csv --results <folder>
# with the detector, on the matches with a BAR-ATH-like smoothed ball: 0.609 -> 0.638
.venv/bin/python pipeline/bench_detector.py --data <work>/training_table.csv \
    --smooth-data <work>/training_table_smooth.csv --results <another folder>
```

All results are in [pipeline/evaluation_results.txt](pipeline/evaluation_results.txt), and a
short overview is in [pipeline/MODEL_SUMMARY.md](pipeline/MODEL_SUMMARY.md).

## Tests

```bash
for t in pipeline/test_*.py; do .venv/bin/python -m unittest "$t"; done
```

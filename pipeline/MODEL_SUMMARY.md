# Model summary: what we built, how well it does, where the rules fit

Short overview for the team, as of 2026-11-09. Full numbers are in
[evaluation_results.txt](evaluation_results.txt).

## Update 2026-11-16: model v2 (spellF1 0.623)

| What changed | Where |
|---|---|
| 16 **ball-touch features**: who touched the ball last / touches it next, how long ago, duels, control time (on by default; `NKFER_TOUCH=0` gives v1 back) | `rules_touch.py`, `model_features.py`, `model.py` |
| **Trained on hybrid DFL matches**: every player from `tracking.txt`, the ball (with its gaps) from `tracking_with_gaps.txt`. with_gaps shows ~13 players, BAR-ATH ~20.5. | `make_hybrid.py` |
| **Viterbi costs team 8 / dead 30** (was 30/60) | `viterbi.py` |
| The trained model | `models/lightgbm_v2.txt` |

Bench on the 7 DFL matches, leave one match out, costs chosen on the other matches:

| Version | with_gaps spellF1 / macroF1 | hybrid spellF1 / macroF1 |
|---|---|---|
| Starter baseline | 0.122 / 0.361 | 0.203 / 0.441 |
| v1 (57 features, 30/60) | 0.411 / 0.848 | 0.506 / 0.89 |
| **v2 (73 features, 8/30)** | **0.461 / 0.865** | **0.623 / 0.910** |

Rebuilding v2:
```
python pipeline/make_hybrid.py     --external <external dir> --out <hybrid dir>
python pipeline/training_table.py  --external <hybrid dir> --out <table.csv>
python pipeline/model.py train     --data <table.csv> --out models/lightgbm_v2.txt
python pipeline/submit.py          --clean <BAR-ATH_tactical_clean.npz> --model models/lightgbm_v2.txt --out <csv>
```
Every decision and its numbers are in `docs/decisions_log.md`. The rest of this page describes
v1 and is kept as history.

## What we built

```
BAR-ATH tracking -> clean.py -> model_features.py -> LightGBM -> Viterbi -> labels file
                                 (57 features)      (H/A/D       (clean
                                                     per frame)   possessions)
```

| File | What it does |
|---|---|
| `clean.py` | cleans the tracking (already existed) |
| `model_features.py` | 57 features per frame: ball speed, nearest home/away player, team speeds and shapes, how long the ball has been missing, plus **time context** (averages over the 0.5 / 1 / 2 s before and after each frame) and what happened around ball gaps |
| `model.py` | LightGBM model: gives each frame a probability of H, A and D. Fixed seed, deterministic. |
| `viterbi.py` | turns the per-frame probabilities into clean possessions. Switching team costs 30, entering or leaving a dead ball costs 60. |
| `training_table.py` | builds the training data from the external matches |
| `submit.py` | makes the BAR-ATH labels file from the saved model (`models/lightgbm_v1.txt`) |
| `bench.py`, `bench_viterbi.py` | the evaluation (see below) |
| `damage.py` | removes players from clean matches the way BAR-ATH loses them |

**Training data (never BAR-ATH):** 7 DFL Bundesliga matches (open data, with gaps added by a
teammate) + Metrica sample games 2 and 3 (damaged by `damage.py` to look like BAR-ATH).
1.29 million labelled frames.

## How we evaluate

- **Leave one match out:** each of the 9 matches is scored by a model trained on the other 8,
  using the official metric code. Viterbi costs are chosen the same fair way.
- **The public 5 minutes of BAR-ATH are only a correctness check** (file valid, frames aligned,
  teams not swapped). RULES.md forbids choosing settings with them, and they hold only 17
  possessions, so one possession moves the score about 6 points.
- Differences below about **0.04 spellF1 are noise**.

## Results (mean of the 9 external matches)

| Version | spellF1 | macroF1 |
|---|---|---|
| Starter baseline (nearest player, smoothed) | 0.174 | 0.444 |
| Model + Viterbi, 18 features | 0.378 | 0.804 |
| **Model + Viterbi, 57 features (current)** | **0.443** | **0.834** |

The time context improved all 9 matches. On the public 5 minutes, the current file (v2)
passes every correctness check: spellF1 0.512, against 0.326 for the starter baseline. That
number doesn't decide anything.

## What the model still gets wrong

On the 9 external matches, only **32%** of true possessions are found (same team, overlap at
least half). These numbers come from the 18-feature model; the 57-feature model does better
overall but hasn't been broken down like this yet.

| Situation | Found |
|---|---|
| possessions under 2 s (30% of all) | 4% |
| possessions of 2-5 s (29% of all) | 23% |
| possessions over 20 s | 75% |
| possession starts with a **turnover in open play** | 25% |
| possession starts after a dead ball / kickoff | 54% |
| ball mostly missing during the possession | 13% |

Also: during live play the model gives the wrong team about 18% of the time. On the public
check, only about a third of the dead-ball frames are labelled D.

It predicts about 290 possessions per match against 418 true ones: **it merges short
possessions**. Lowering Viterbi's switch cost doesn't fix this. At the right count, the
extra switches land in the wrong places. **The model needs to know *where* possession
changes, and that is exactly what rules can provide.**

## Suggestions for the rules part

**1. Don't override the model's labels with rules.** One wrong rule decision breaks a whole
possession, and spellF1 punishes that hard. Feed the rules in as evidence instead, in one of
two places:

| Where | How | Example |
|---|---|---|
| **As features** (easiest) | a rule outputs one number per frame. Add it to `feature_table` in `model_features.py` and its name to `FEATURE_COLUMNS` in `model.py`. The model learns how much to trust it. | `touch_home` = 1 if a home player is within 1 m of the ball and the ball changes direction |
| **In Viterbi** | make a team switch cheap at frames where a rule says possession probably changed, and expensive elsewhere | switch cost 5 at a detected turnover, 30 otherwise |

**2. The rules most likely to help, given what the model misses:**

- **Turnover / touch detection.** The biggest gap. The ball suddenly changes speed or
  direction, and the nearest player at that moment belongs to the other team.
  - Training targets already exist: every H<->A change in the external labels is a turnover
    frame.
- **Restarts.** Throw-in, corner, goal kick and kickoff: the ball is still at the line or at a
  known spot, then moves. The team taking it is the one whose player is nearest.
  - Smart Tagging lists 59 set pieces with times. Throw-ins have no team, so take the nearest
    player.
- **Dead ball.** The ball past the line, all players slow, and the ball missing near a line.
  - This is where we are weakest on D.
- **Ball gaps** (missing about 40% of the time on BAR-ATH). The nearest team when the ball
  disappears vs. when it reappears: if they differ, possession probably changed during the
  gap.

**3. Hard rule or feature? Measure it first.** A general rule can be true in football and still
fire wrongly on our data: the ball is missing about 40% of the time, positions are noisy and
some players aren't tracked. Example: `ball_past_end_m` / `ball_past_side_m` already encode
"ball outside the pitch means dead", and the model gives them only 0.1% of its decisions,
because the ball is usually already missing when it goes out. So, for each rule, check on the
external matches how often it fires and how often it's right when it fires:

| Right when it fires | Where it goes |
|---|---|
| about 99% or more | hard constraint **inside Viterbi**: forbid the other labels at those frames, so the possessions around them adjust smoothly. Not as an override after Viterbi. |
| less than that | feature in the model, which learns how much to trust it |
| not sure | feature (the safe default) |

Either way, the bench result decides.

**4. Smart Tagging helps with the team, not with short possessions.** Its phases last about
30 s and show only about 82 team changes in the whole match. It's useful for "which team over
the next 30 s" and for restarts, not for 2-second possessions.
**Vanja will add simulated Smart Tagging for the external matches**, made to look like the real
BAR-ATH tags: similar phase lengths, coverage and errors. Then Smart Tagging can be used as a
feature or as Viterbi evidence and tested on the bench like any other rule. Without it the
model can't learn from the tags, because the external matches have none.

**5. Test every rule the same way:**
```
python pipeline/training_table.py --external <external dir> --out <table.csv>
python pipeline/bench.py         --data <table.csv> --results <folder outside the repo>
python pipeline/bench_viterbi.py --data <table.csv> --results <same folder>
```
About 15 minutes. Compare against **0.443 spellF1**; a rule should beat it by more than about
0.04 to count as real. Never write match data or results inside the repo.

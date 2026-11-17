# Model summary: what we built, how well it does, where the rules fit

Short overview for the team, as of 2026-11-17 (v4). Full numbers are in
[evaluation_results.txt](evaluation_results.txt); how to regenerate the file is in
[REPRODUCE.md](../REPRODUCE.md).

## What we built

```
BAR-ATH tracking -> clean.py -> model_features.py -> LightGBM ---------> Viterbi -> labels file
                                 (90 features)      (H/A/D per frame)    (clean possessions;
                                                 -> detector ----------> team switch cheap where
                                                    (change to H / A)    the detector fires)
```

| File | What it does |
|---|---|
| `clean.py` | cleans the tracking (already existed) |
| `model_features.py` | 90 features per frame: ball speed, nearest home/away player, team speeds and shapes, how long the ball has been missing; **time context** (averages over the 0.5 / 1 / 2 s before and after each frame) and what happened around ball gaps; and **touch features** over 2 / 4 / 8 frames (ball turning or speeding up, nearest player or team changing, closest home / away player) |
| `model.py` | LightGBM model: gives each frame a probability of H, A and D. Fixed seed, deterministic. |
| `detector.py` | possession-change detector (a second LightGBM, `models/detector_v1.txt`): per frame, the chance that possession changes to H or to A within ±6 frames. Trained on the training matches with their ball smoothed like BAR-ATH's. |
| `viterbi.py` | turns the per-frame probabilities into clean possessions. With the detector, switching team costs 120 × (1 − P(change))⁴ at each frame; entering or leaving a dead ball costs 30. Without it (v3): 15 and 30. |
| `training_table.py` | builds the training data from the external matches; `--smooth-ball 6` smooths their ball like BAR-ATH's (for the detector) |
| `submit.py` | makes the BAR-ATH labels file from the saved models (`--model models/lightgbm_v2.txt --detector models/detector_v1.txt`) |
| `bench.py`, `bench_viterbi.py`, `bench_detector.py` | the evaluation (see below) |
| `damage.py` | makes the external matches look like BAR-ATH: removes players the way BAR-ATH loses them, and with `--ball-like` also removes the ball (DFL) the way the Metrica copies lose it |

**Training data (never BAR-ATH):** 7 DFL Bundesliga matches (open data) + Metrica sample games 2
and 3, all damaged by `damage.py` to look like BAR-ATH. 1.29 million labelled frames. The DFL
copies used to be the gap copies a teammate made; those hid ~40% of the players and dropped the
ball at random, unlike BAR-ATH, and training on our own copies instead scores better on every
match.

## How we evaluate

- **Leave one match out:** each of the 9 matches is scored by a model trained on the other 8,
  using the official metric code. Viterbi costs are chosen the same fair way.
- **The public 5 minutes of BAR-ATH are only a correctness check** (file valid, frames aligned,
  teams not swapped). RULES.md forbids choosing settings with them, and they hold only 17
  possessions, so one possession moves the score about 6 points.
- Differences below about **0.04 spellF1 are noise**.

## Results (mean of the 9 external matches)

On the DFL copies we were sent (benches 1-3):

| Version | spellF1 | macroF1 |
|---|---|---|
| Starter baseline (nearest player, smoothed) | 0.174 | 0.444 |
| Model + Viterbi, 18 features | 0.378 | 0.804 |
| Model + Viterbi, 57 features (v2) | 0.443 | 0.834 |
| Model + Viterbi, 90 features | 0.503 | 0.847 |

On our BAR-ATH-style DFL copies (bench 4; easier, so not comparable with the table above):

| Version | spellF1 | macroF1 |
|---|---|---|
| Starter baseline | 0.317 | 0.721 |
| 90 features, trained on the old DFL copies | 0.599 | 0.835 |
| 90 features, trained on the new DFL copies (v3) | 0.632 | 0.890 |

The same matches with their ball smoothed like BAR-ATH's (bench 5; BAR-ATH's video-tracked ball
has no sharp velocity jumps):

| Version | spellF1 | macroF1 |
|---|---|---|
| v3 | 0.609 | 0.881 |
| **v3 + detector trained on smoothed balls (v4, current)** | **0.638** | **0.887** |

Every step improved all 9 matches. The starter baseline scores 0.243 on the full BAR-ATH match,
so these benches are somewhat easier than BAR-ATH. On the full BAR-ATH match (no labels) v4
behaves as on the test matches: 3.38 vs 3.46 possessions per minute, 5.8 vs 6.2 detector alarms
per minute. On the public 5 minutes every version passes the correctness checks (v4 0.537, v3
0.595, v2 0.512, starter 0.326: about one possession apart); that window decides nothing.

Tested and not used: nikozirdum's 16 touch features (PR #5) add +0.094 on sharp balls but
nothing (−0.011) on BAR-ATH-like smooth balls, where their touch detector barely fires.

## What the model still gets wrong

On the 9 external matches, only **32%** of true possessions are found (same team, overlap at
least half). These numbers come from the 18-feature model. Since then, short possessions (under
2 s) found went from 4% to 7% (57 features), 14% (90 features) and 29% (90 features, new DFL
copies); the rest of the table hasn't been redone.

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
  - Done: the touch features (in the model) and the possession-change detector (in Viterbi).
    The detector only carried over to BAR-ATH once it was trained on smoothed balls; any new
    ball-based rule should be checked on smoothed balls too (bench_detector.py shows how).
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
About 15 minutes. Compare against **0.632 spellF1** (bench 4, v3) and, for anything that uses
the ball's movement, also on smoothed balls against **0.638** (bench 5, v4: `bench_detector.py`
with a `--smooth-ball 6` table). A rule should beat them by more than about 0.04 to count as
real. Also check, without labels, that it behaves the same on the full BAR-ATH match as on the
test matches (for example possessions per minute: v4 gives 3.38 on BAR-ATH and 3.46 on the test
matches). Never write match data or results inside the repo.

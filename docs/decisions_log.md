# Decisions log

Every threshold and setting that changes the submission, with its value, the reason and the
number behind it. All numbers come from the external matches only (never from the provided game,
and never from its public 5 minutes). Format: date, what, value, reason, evidence.

## 2026-11-13: restart-kick anchor thresholds (`pipeline/rules_moments.py`)

| setting | value | was |
|---|---|---|
| `still_min_s` | 1.5 s | 1.0 (placeholder) |
| `kick_speed` | 6.0 m/s | 4.0 (placeholder) |
| `slow_players` | 1.5 m/s | 1.5 (placeholder, kept) |

**Reason.** Restart kicks are meant to mark the end of a dead ball exactly, so precision matters
more than recall. Of the 27 combinations in `measure_moments.py --sweep`, this one keeps
confident-anchor precision at 0.98 while finding 42% of the true restarts. The strictest one
(`slow_players` 1.2) reaches 0.991 but finds only 32%.

**Evidence** (7 DFL matches, `tracking_with_gaps`, anchor within +/- 5 frames of a true D -> H/A):

| | precision | recall |
|---|---|---|
| placeholders (1.0 s, 4 m/s, 1.5 m/s), all anchors | 0.909 | 0.529 |
| placeholders, confident anchors | 0.941 | 0.446 |
| chosen (1.5 s, 6 m/s, 1.5 m/s), confident anchors | 0.983 | 0.416 |
| strictest (1.5 s, 6 m/s, 1.2 m/s), confident anchors | 0.991 | 0.324 |

Other findings from the same run (placeholder thresholds):

- Median anchor offset is -1 frame: the kick is found one frame before the labelled restart.
- The ball is seen on only 60% of true restart frames, which caps the anchors' recall.
- The nearest team at the kick is the restarting team only 74% of the time. The anchor's
  timing can be trusted, its team cannot, so a hard team constraint in Viterbi is not justified
  yet.
- Precision by restart type: corner 1.00, throw-in 0.99, goal kick 0.97, free kick 0.79.
- The ball past a line is followed by D within 2 s in 99.9-100% of flagged frames.
- Turnover candidates: 39% are within 0.5 s of a true H <-> A change; they cover 20% of
  the true changes. Feature only.

**Caveat.** The thresholds were chosen on the same 7 matches the bench then scores. They were
chosen against anchor timing, not spellF1, and only 3 values were varied, so the leak is small.
It is noted here for the technical document.

## 2026-11-13: moment columns as model features (`NKFER_MOMENTS`), not switched on

**Decision.** `NKFER_MOMENTS` stays off (0). With the 12 moment columns added, the model does
not improve.

**Evidence** (7 DFL matches, `tracking_with_gaps`, leave-one-match-out; Viterbi costs chosen on
the other matches; the Metrica matches are left out because they need `damage.py`):

| | spellF1 | macroF1 |
|---|---|---|
| baseline (starter) | 0.122 | 0.361 |
| model only, 57 features | 0.142 | 0.833 |
| A: model + Viterbi, 57 features | 0.411 | 0.848 |
| B: model + Viterbi, 69 features (with moments) | 0.409 | 0.848 |

B - A = -0.002 spellF1, far below the ~0.04 a change needs. Only `players_slow_before` gets
real weight (3.5% of gain). The other 11 columns get almost none: the 50-frame ball-speed
context already carries the same information.

The best Viterbi costs on these matches are team 15 / dead 30 (0.409-0.411), not the
30 / 60 that `viterbi.py` uses. That is a separate decision, not taken here.

**Next.** The moments help only if they act directly on Viterbi (step 5: per-frame costs or
forced D at ball out, cheap changes at anchors), not as features.

## 2026-11-13: moments as per-frame Viterbi costs (`moment_costs.py`), not switched on

**Decision.** `moment_costs.CHOSEN` stays `None`: the submission keeps the fixed Viterbi costs.

**Evidence** (`bench_viterbi.py --moments` on the scores of model A, 57 features; 7 DFL
matches; each match scored with the settings that were best on the other 6; 8694 decodings over
team {8, 15, 30} x dead {15, 30, 60} x window {6, 12, 25} x turnover factor {1, 0.5, 0.25, 0.1}
x dead factor {1, 0.5, 0.25, 0.1} x force {off, dead, team}):

| variant | spellF1 | macroF1 |
|---|---|---|
| none (fixed costs) | 0.411 | 0.848 |
| costs (cheaper changes near moments) | 0.414 | 0.850 |
| force (labels forced at confident restart kicks) | 0.411 | 0.848 |
| both | 0.416 | 0.850 |

The best variant gains +0.005 spellF1, far below the ~0.04 a change needs. The gain comes from
cheaper H <-> A changes near turnover candidates (factor 0.1 is always chosen, the smallest
tried). Forcing D before a confident kick changes nothing, because Viterbi already puts D there.
"team" forcing is never chosen, as expected with 74% team accuracy.

**Why so little.** The model already finds the restarts; the possessions it misses are short
open-play ones. Turnover candidates find only 20% of the true changes, with 39% precision. More
spellF1 needs a better turnover/touch detector (rules 7a-7i), not more Viterbi tuning.

## 2026-11-14: ball-touch features (`rules_touch.py`, `NKFER_TOUCH=1`): +0.050 spellF1

**Why.** The diagnosis on the 7 DFL matches showed the loss is in short possessions: under 2 s
(30% of all) the model found 5%, 2-5 s (30%) 24%. Even an oracle that tells Viterbi exactly
where the true changes are only reaches 0.54, so the per-frame team itself had to improve. The
model had no notion of who touched the ball last.

**Detector.** Touch = ball velocity change >= `dv_min` m/s (velocity over 2 frames before vs
after, local peak) with a player within `radius_m`; team of the nearest player. 16 features:
last / next touch team and time, whether the last touch changed team, touch counts per team in
the past / next second, duel now and over the past second, control run per team.

| dv_min | radius_m | touches / live min | touch team right | true changes with a touch by the new team (+/- 12 f) |
|---|---|---|---|---|
| 2.0 | 1.0 | 16.5 | 0.901 | 0.295 |
| **2.0** | **1.5** | **18.0** | **0.875** | **0.320** |
| 3.0 | 1.5 | 15.8 | 0.875 | 0.306 |
| 5.0 | 1.5 | 12.4 | 0.873 | 0.272 |

Chosen 2.0 / 1.5: most changes covered while the team is still right 7 times in 8. Chosen on
detector accuracy, not on spellF1, from a 3 x 3 grid on the same 7 matches the bench scores.

**Bench** (7 DFL matches, leave one match out, Viterbi costs chosen on the other matches):

| | spellF1 | macroF1 |
|---|---|---|
| A: 57 features | 0.411 | 0.848 |
| **C: 57 + 16 touch features** | **0.461** | **0.865** |
| C, model only (no Viterbi) | 0.233 | 0.861 |
| C + moment costs in Viterbi (best variant) | 0.460 | 0.865 |

All 7 matches improve (+0.020 to +0.080). `last_touch_team` and `last_touch_s` are the 4th and 5th
most used features. The moment costs add nothing on top.

**Viterbi costs.** With the touch features the best fixed costs move to team 8-15 / dead 30
(0.462-0.465 over all matches). With the current `viterbi.py` costs 30 / 60, C gives 0.444.
Switching on the touch features for the submission therefore also needs `SWITCH_COSTS` changed
(team 8 is chosen for 5 of 7 matches, 15 for 2). Not done yet; it needs the team's agreement and
a retrained `models/` file.

## 2026-11-14: training domain - player visibility matters a lot (experiment, nothing changed)

DFL `tracking_with_gaps` hides ~58% of positions (~13 players seen), BAR-ATH about 20.5. Hybrid
built from the external files only: players from `tracking.txt`, ball from `tracking_with_gaps.txt`,
same labels (`possession.csv` and `possession_with_gaps.csv` are identical). Same 7-match
leave-one-out (fastbench, float32, Viterbi grid team {4,8,15,30} x dead {15,30,60}):

| | with_gaps | hybrid |
|---|---|---|
| A (57 features) | 0.411 | 0.506 |
| C (+ touch) | 0.469 | 0.617 |

With full players the touch features are worth +0.11, not +0.05. The with_gaps bench probably
underestimates BAR-ATH. Next: barca-style DFL (hybrid + `damage.py` player profile) as the
training and bench domain; the user runs it since it reads BAR-ATH tracking.

Also measured on fastbench (with_gaps, C = 0.469), none better: short context 0.466, strict/loose
touch variants 0.466, player-level ownership 0.461, all ideas 0.467, bigger LightGBM 0.466, nested
stacking 0.469 (0.473 with team cost 6). Smart Tagging of the external matches is derived from
possession.csv, so it is not used.

Official bench (`bench.py` + `bench_viterbi.py`, `NKFER_TOUCH=1`) of C on the hybrid DFL matches:
baseline 0.203 / 0.441, model only 0.423 / 0.909, **model + Viterbi 0.623 / 0.910** (spellF1 /
macroF1; per match 0.566-0.668). Best fixed costs team 8 / dead 30 (0.625); with the current
30 / 60, 0.581.

## 2026-11-16: code of the two negative moment experiments removed

The `NKFER_MOMENTS` switch (moments as model features) and the per-frame Viterbi costs
(`moment_costs.py`, `viterbi.decode_varying`, `bench_viterbi.py --moments`, the `moment_costs`
branch in `submit.py`) changed spellF1 by at most +0.005, below the +0.04 bar, so the code is taken
out to keep the pipeline simple. The results above stand. `rules_moments.py` is no longer used
by the model, so it stays at its committed version: the measured restart-kick thresholds above
(1.5 s, 6.0 m/s) are recorded here but not applied. `bench_viterbi.py`, `submit.py` and
`training_table.py` are back to their committed version, `viterbi.py` too apart from its costs
(next entry), and `feature_table` no longer takes `pitch`.

## 2026-11-16: model v2 = touch features + hybrid training + Viterbi 8/30 (the 0.623 setup)

- `NKFER_TOUCH` is now on by default; `NKFER_TOUCH=0` rebuilds `models/lightgbm_v1.txt` exactly.
- `viterbi.SWITCH_COSTS` is team 8 / dead 30, the best fixed pair of the hybrid bench (0.625 over
  all matches; the fair per-match choice gives 0.623). With 30/60 the same model gives 0.581.
- `pipeline/make_hybrid.py` builds the hybrid DFL matches (all players from tracking.txt, ball
  from tracking_with_gaps.txt, labels unchanged). It reproduces the folder used for the bench
  byte for byte.
- `models/lightgbm_v2.txt` is trained on all 7 hybrid matches:

      python pipeline/make_hybrid.py --external <external> --out <hybrid>
      python pipeline/training_table.py --external <hybrid> --out <table.csv>
      python pipeline/model.py train --data <table.csv> --out models/lightgbm_v2.txt

Bench (leave one match out, costs chosen on the other matches): spellF1 0.623, macroF1 0.910 on
hybrid; the same setup on with_gaps gives 0.461 / 0.865.

Training v2 twice gives a byte-identical model (md5 8efac51de86475b9635e75accb128105).

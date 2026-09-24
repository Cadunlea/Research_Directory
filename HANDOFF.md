# HANDOFF — read this first

If you are a new Claude session, or Caelan picking this up after a break: this
file is the current state of the work. Read it before changing anything.

Branch: `claude/adoring-keller-gq1nmf`
Machine: the lab Windows box, venv `C:\Users\cadunlea\ets_venv` (Python 3.12)
Working folder: `...\python_code\study_eating_trajectory\scripts\caelan_scripts`

**Claude cannot reach the database.** It runs in a cloud container; the database
is `localhost:30071` on the lab machine. Claude writes scripts, Caelan runs them
and pastes the output back. Everything below was measured that way.

---

## Contract status (RRSP Phase II, signed 9/4/26)

| # | Activity | Deadline | Status |
|---|---|---|---|
| 2 | Migrate to new database, unify data access | 09/09/26 | DONE — `ets_data.py` is the unified module |
| 3 | Data quality assessment (missingness, alignment, flagged sessions) | 09/23/26 | DONE — `probe_database.py` + `check_annotation_drift.py` cover missingness, sensor/annotation alignment and per-session coverage |
| 4 | Literature review recommending a model approach | 09/23/26 | DRAFTED — `literature_review/` (Word document, one slide, build scripts). Closest prior work is Ghosh & Sazonov 2022 (raw 4-channel input, ResNet, F1 0.906); reproduce it on the new database before claiming against it |
| 5 | Model development and coding | 10/07/26 | AHEAD — both models built and evaluated |
| 6 | Run, validate, iterate | 10/28/26 | AHEAD — 4-fold CV plus six ablations done |
| 7 | Inference script | 11/04/26 | not started |
| 8 | Quantify all metrics | 11/11/26 | largely done, needs packaging |
| 9 | Paper/report | 12/02/26 | not started |

Note the contract names PyTorch for Activities 5-6; the work moved to
TensorFlow/Keras to match the lab toolchain. Worth a sentence to the PI rather
than a silent deviation.

## The task

Rebuild the AIM-2 food-intake classifier: read from the ETS database instead of
annotation CSVs, TensorFlow/Keras instead of PyTorch, and beat the previous
model. Caelan is writing a paper, so **every choice must be defensible when
questioned** — that constraint drove most of the design.

Targets: previous model F1 0.788 / accuracy 0.803; AIM-2 random-forest paper
F1 0.81.

## Where it stands

Current, on the final harness (2 inner validation participants, plateau-centre
threshold, threshold estimated from all inner-participant windows):

| Model | F1 | Accuracy | Balanced acc | Precision | Recall |
|---|---|---|---|---|---|
| Previous (PyTorch, CSVs) | 0.788 | 0.803 | 0.825 | 0.684 | 0.929 |
| Model A (same net, new data) | 0.779 | 0.841 | 0.840 | 0.730 | 0.836 |
| **Model B** | **0.854** | **0.905** | **0.887** | **0.876** | 0.834 |
| Model B, no chew head | 0.849 | 0.900 | 0.885 | 0.857 | 0.841 |
| Model B + 8 s context (no marker) | 0.775 | 0.825 | 0.842 | 0.683 | 0.894 |

Both targets beaten. Model B's big win is precision, 0.684 → 0.876, which is
where the time-domain input and attention pooling were predicted to help.
Confusion: tp=980 fp=139 fn=195 tn=2189. Per-fold F1 0.920 / 0.801 / 0.807 /
0.852, sd 0.048. Thresholds 0.46 / 0.30 / 0.21 / 0.18.

A and B above are on the same harness and are directly comparable. A -> B is
+0.075 F1, +0.064 accuracy, and +0.146 precision.

## WHAT CHANGED, COMPONENT BY COMPONENT (Model A -> Model B)

Slide-ready. Model A is the original architecture retrained on the new
database; Model B is the current model. Same folds, same harness, same
evaluation windows for both.

| # | Component | Original (Model A) | New (Model B) | Why |
|---|---|---|---|---|
| 1 | Input representation | FFT magnitude spectrum, 513 bins x 4 ch | Time series z-scored per window (no frequency filter), 1024 samples x 4 ch | Magnitude discards phase; chewing is a rhythmic envelope and a bite is a transient, both smeared by a spectrum |
| 2 | Normalisation | per-window z-score, missing samples left at -1 | per-window z-score + DC removal, missing excluded then set to 0 | -1 is not neutral against optical values near 3000 counts; it injects a step at every gap |
| 3 | Channel weighting | none (fixed conv filters) | cross-channel attention (squeeze-excite) | optical carries chewing, accelerometers carry motion that is sometimes signal and often artefact |
| 4 | Pooling | global average over frequency | attention pooling over time | averaging buries a 2 s bite inside an 8 s window |
| 5 | Supervision | single binary head | binary head + auxiliary chew-count head (CHGT), discarded at inference | one bit per window is thin supervision for ~7.8 h of data; counting forces the trunk to represent rhythm |
| 6 | Training windows | non-overlapping, 8 s hop | 50% overlap (4 s hop) at TRAIN time only | roughly doubles training windows; evaluation stays on the non-overlapping grid |
| 7 | Framework | PyTorch | TensorFlow / Keras 3.15 | matches the lab toolchain |
| 8 | Data source | annotation CSVs + hardcoded 14-participant list | database (`study_data` GT + `aim_raw_data` XYZO), no participant list | participant list cannot go stale; labels come from one source of truth |
| 9 | Evaluation | one 2-participant split, 249 windows, threshold fixed 0.40 | participant-level 4-fold CV, 3,503 windows, threshold fit on validation participants | a single small split has a very wide confidence interval |
| 10 | Annotation shift | -8 s applied | none | measured at +0.10 s on the new database; no correction warranted |

Headline effect of 1-6 together: **F1 0.779 -> 0.854, accuracy 0.841 -> 0.905,
precision 0.730 -> 0.876.**

## ABLATION TABLE - what actually caused the improvement

Every row is Model B with ONE component removed, same harness, same folds.

| Variant | F1 | delta | Precision |
|---|---|---|---|
| **Model B, full** | **0.854** | - | 0.876 |
| - time-domain input (FFT instead) | 0.767 | **-0.087** | 0.696 |
| - overlap augmentation | 0.829 | -0.025 | 0.834 |
| - attention | 0.847 | -0.007 | 0.857 |
| - auxiliary chew head | 0.849 | -0.006 | 0.857 |
| *Model A, for reference* | 0.779 | | 0.730 |

THE HEADLINE FINDING: the representation is doing nearly all the work.
Replacing the FFT magnitude spectrum with the z-scored time series accounts
for F1 +0.087 and precision +0.180.

And the sharpest evidence for it: Model B's ARCHITECTURE fed Model A's FFT
features scores 0.767 - WORSE than Model A's simpler architecture on the same
features (0.779). The attention, the pooling and the auxiliary head are worth
nothing, slightly less than nothing, over a magnitude spectrum. They only pay
off once the input has temporal structure to attend to, which is exactly what
you would predict: attention pooling over frequency bins is meaningless,
attention pooling over time is not.

What is statistically supportable at a fold spread of +-0.047:
- time-domain input, -0.087: REAL. Nearly twice the spread, with a large
  coherent secondary signal in precision. Assert it.
- overlap augmentation, -0.025: suggestive, about half the spread. Report the
  number, do not lean on it.
- attention -0.007 and chew head -0.006: INSIDE noise. Report as measured and
  state they are not separable from variance at n=20. Never claim either as a
  contribution.

The marginals do NOT sum to the A -> B total of +0.075, and should not be made
to: ablations measure marginal contributions and these interact, which is the
whole point of the FFT row.

### Each ablation, one at a time

**1. Input representation — the only effect larger than noise**

| | F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Model B (time-domain) | 0.854 | 0.905 | 0.876 | 0.834 |
| Same model, FFT input | 0.767 | 0.826 | 0.696 | 0.855 |
| **Effect** | **+0.087** | **+0.079** | **+0.180** | **-0.021** |

Note recall is slightly LOWER — do not say "improved across the board". The
gain is in precision. And the sharpest point: Model B's architecture on FFT
features (0.767) is WORSE than Model A's simpler architecture on the same
features (0.779), so the architecture only earns its keep once the input has
temporal structure.

**2. Overlap augmentation — suggestive, about half the fold spread**

| | F1 | Accuracy | Precision |
|---|---|---|---|
| Model B (4 s hop at train time) | 0.854 | 0.905 | 0.876 |
| Model B (8 s hop, no overlap) | 0.829 | 0.886 | 0.834 |
| **Effect** | **+0.025** | **+0.019** | **+0.042** |

**3. Cross-channel attention + attention pooling — inside noise**

| | F1 | Accuracy | Precision |
|---|---|---|---|
| Model B (with attention) | 0.854 | 0.905 | 0.876 |
| Model B (global average pooling) | 0.847 | 0.899 | 0.857 |
| **Effect** | **+0.007** | **+0.006** | **+0.019** |

**4. Auxiliary chew-count head — inside noise**

| | F1 | Accuracy | Precision |
|---|---|---|---|
| Model B (with chew head) | 0.854 | 0.905 | 0.876 |
| Model B (no chew head) | 0.849 | 0.900 | 0.857 |
| **Effect** | **+0.006** | **+0.005** | **+0.019** |

The gain concentrates in fold 2 (0.780 -> 0.801), the fold holding the weakest
optical signal. Direction matches the motivation; magnitude does not clear
noise.

**5. Post-hoc temporal smoothing — NEGATIVE**

| | F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Model B | 0.854 | 0.905 | 0.876 | 0.834 |
| + median filter (3 windows) | 0.775 | 0.847 | 0.765 | 0.785 |
| **Effect** | **-0.079** | **-0.058** | **-0.111** | **-0.049** |

Fair test: smoothing selects its own threshold. Eating is not contiguous at 8 s
— meals are punctuated by pauses between bites — so a 24 s median erases short
real bouts.

**6. Context windows without a target marker — NEGATIVE**

| | F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Model B | 0.854 | 0.905 | 0.876 | 0.834 |
| + 8 s context each side | 0.775 | 0.825 | 0.683 | 0.894 |
| **Effect** | **-0.080** | **-0.080** | **-0.193** | **+0.060** |

Precision collapsed while recall rose: given 24 s of signal and no indication
which 8 s the question concerned, the model answered "is there eating SOMEWHERE
in this window". A marker channel identifying the labelled window is
implemented (`mark_target_window`) but NOT YET TESTED.

Protocol history, since the numbers moved twice and a reader may ask:
3 inner participants + threshold restricted to non-overlapping validation
windows gave A 0.765 / B 0.828; reverting both gave B 0.854. All of these sit
inside the +-0.048 fold spread. The final setting was chosen on the a priori
argument that scarce data belongs in training and threshold noise belongs in
the estimator - but it was chosen after seeing test scores, which is a mild
form of selecting on the test set and should be disclosed rather than hidden.

## What the database actually contains (measured, not assumed)

- **Sensor**: `aim_raw_data.numeric_data`, `XYZO`, ~8 s packets of `(4, 1022)`
  int16 at 128 Hz — Acc X/Y/Z, Optical. 43 participants, 61 sessions.
- **Ground truth**: `study_data.numeric_data`, `CHGT`/`BOGT`/`BIGT` =
  chew / bout / bite. One whole day per row at 10 Hz (864000 samples) indexed
  from midnight. **20 annotated sessions**, 9–47 min each, ~7.8 h total, all
  with 100% sensor coverage.
- **`1` eating, `0` not eating, `-1` NOT RECORDED** — everywhere, both tables.
  98.9% of each ground-truth day is `-1`; those windows are DROPPED. Treating
  them as negatives would bury the real labels.
- Sessions join on **`(participantID, studyID)`**. `record_hash` does NOT match
  between the tables — the upload gives each of CHGT/BOGT/BIGT a different hash,
  contradicting the documented `tuple_hash((studyID, participantID))`.
- Eating fraction inside annotated windows: 17%–74%, pooled 33.5%.

**Two unit traps**, handled explicitly in `ets_data.py`:
- `data_sampling_frequency` is a **period** (0.0078125 = 1/128) in the raw table
  but a real **frequency** (10.0) in the study table.
- `data_duration` is **seconds** (8.0) in the raw table but a **sample count**
  (864000) in the study table.

Ports: **30071 is MariaDB** (what Python connects to), 30073 is phpMyAdmin (the
browser page). Both are correct, for different things.

## Annotation alignment — verified on the new database

Not a headline finding; a precondition that had to be checked before training
on this data, and it passed.

PROVENANCE, because it is easy to overstate: the `-8.0` shift was Caelan's own
diagnostic correction in his previous model, on the OLD CSV-based pipeline. It
was never applied by anyone else and never used elsewhere in the lab. The lab
separately rebuilt the database and identified what may have caused the
original misalignment. The job here was only to confirm the new database needs
no correction.

HOW IT WAS VERIFIED - the method, for slides or a methods section:

1. For each of the 20 annotated sessions, build the eating mask from the
   database ground truth (BOGT bout OR BIGT bite) at 10 Hz.
2. From the same session's optical channel at 128 Hz, derive a chewing-activity
   signal: band-pass to the chewing band (0.8-3.0 Hz), take the Hilbert
   envelope, smooth 0.5 s, resample to 10 Hz on the true time grid.
3. Cross-correlate the two over candidate lags of +-30 s in 0.1 s steps,
   computing Pearson r only over samples valid in BOTH series so missing
   sensor data and un-annotated time cannot invent agreement.
4. The lag at peak correlation is that session's measured offset. A session
   whose peak never rises above r = 0.10 is reported UNRELIABLE and excluded
   rather than contributing a number read off noise.
5. Repeat under both possible readings of `data_timestamp` (packet start vs
   packet end) so the data decides rather than an assumption.

The measurement itself is validated in `test_drift_check.py`, which injects
offsets of known size and direction into synthetic data and confirms they are
recovered. That test caught a real bug during development: resampling 128 Hz to
10 Hz by averaging round(12.8) = 13-sample blocks yields 9.846 Hz, an error
that grows with elapsed time and reaches ~8 s by the middle of a 17-minute
session. Resampling is now done on the true time grid.

`check_annotation_drift.py` measured the lag on all 20 sessions:

| `data_timestamp` read as | Median lag | Range | Within ±1 s |
|---|---|---|---|
| packet **start** | **+0.10 s** | +0.00 … +0.20 | **20/20** |
| packet **end** | −7.90 s | −8.00 … −7.80 | 0/20 |

Peak correlation is identical either way (0.716 vs 0.715), which is what
distinguishes a displacement from a degradation.

THE MEASURED CLAIM, and the only one to make: on this database, read with
`data_timestamp` as the packet START, the annotations are aligned to within
0.2 s on every session. No correction is warranted.

A HYPOTHESIS, not established: the two readings differ by exactly one 8 s
packet, and the old training code computed `packet_start = data_timestamp -
data_duration`, i.e. read a start timestamp as an end. That would produce an
artefact of exactly the observed size. This is a plausible account of the old
pipeline only - the database was rebuilt in the meantime and the lab has its
own explanation, so do not present it as the cause of anything.

**No shift is applied anywhere. Do not reintroduce one.**

Note: the per-session plots come in pairs, `_start.png` and `_end.png`. The
`_end` ones peak at −8 *by design* — they reconstruct the bug. Don't mistake
them for a result.

## Evaluation protocol (why the numbers are lower-looking but better)

The old 0.788 came from **one 2-participant split of 249 windows** with the
phantom shift and the threshold fixed at 0.40. Now:

- Participant-level **4-fold cross-validation**, every participant held out once.
  Never split windows at random — consecutive 8 s slices of one meal are near
  duplicates and would leak.
- Threshold chosen on **inner validation participants**, never the test fold.
  Smoothing gets its own threshold (see below).
- Evaluation always on **non-overlapping** windows even when training used a
  half-window hop.
- **F1 and balanced accuracy lead**; accuracy is reported alongside.

Fold membership is recorded in each run's JSON — `participant_folds` re-deals
everyone when a participant is added, so results only compare like-for-like
within the same participant set.

## Negative result worth keeping

**Temporal median smoothing made things worse**, and this is now a FAIR test:
smoothing selects its own threshold independently, so the comparison is not
confounded. Model B F1 0.854 → 0.775 at 8 s.

A median over three 8 s windows spans 24 s and assumes a contiguity the labels
lack: eating is `bout OR bite`, and real meals are punctuated — bite, chew,
swallow, pause, talk — so at 8 s the label sequence genuinely flips on and off
and short but real bouts get erased. This is the same fact the lab's own pause
metrics (PADU) exist to measure. Publishable as a negative result with a
mechanism.

### Context windows, first attempt: also negative, and diagnosable

`--context-seconds 8` cost F1 0.854 -> 0.775. The pattern says why: precision
COLLAPSED 0.876 -> 0.683 while recall ROSE 0.834 -> 0.894. The model became
much more willing to call eating.

Diagnosis: it was given 24 s of signal and no indication of which 8 s it was
being asked about, with attention pooling free to attend anywhere in them. So
it learned the easier question - "is there eating SOMEWHERE in this 24 s?" -
which fires on every window adjacent to a meal, exactly where the extra false
positives are.

Fix implemented, NOT YET TESTED on real data: a fifth input channel that is 1
over the labelled window and 0 over the context (v2.mark_target_window). Run
`--context-seconds 8` again to test it. If it still loses, context is simply
not helping here and that is the finding - do not keep tuning it.

## Files

| File | What it is |
|---|---|
| `probe_database.py` | Read-only report of database contents |
| `check_annotation_drift.py` | Measures sensor/annotation lag per session |
| `plot_drift_evidence.py` | The 3-panel figure for the lab meeting |
| `ets_data.py` | Database → windows. The data model lives here |
| `ets_eval.py` | Metrics, thresholds, smoothing, reporting |
| `ets_train.py` | The CV harness both models share |
| `train_food_intake_v1_keras.py` | Model A — previous architecture, new data |
| `train_food_intake_v2.py` | Model B — time-domain CNN + attention |
| `compare_runs.py` | Paired comparison of two result JSONs, participant by participant (Wilcoxon + participant bootstrap of pooled F1) |
| `test_pipeline.py`, `test_drift_check.py` | Tests, no database needed |

Run the tests after any change: they guard silent failures (participant
leakage, windows built from un-annotated time, `-1` treated as a reading,
smoothing across recording gaps, context altering labels). They have already
caught three real bugs.

## Correction (9/24): Model B's input is NOT band-passed

`normalise(X, high_pass=True)` only subtracts the window mean, and the z-score
subtracts it again, so no frequency filter is applied anywhere. The input is
the raw signal z-scored per window and per channel. Earlier notes, slides and
the first literature review said "band-passed"; all have been corrected.
Whether a real filter (e.g. AIM-2's 0.1 Hz high-pass / 3 Hz low-pass) helps is
an open experiment. Also: the channel attention (squeeze-excite) reweights the
64 LEARNED feature channels after conv block 3, not the 4 sensors directly.

Full end-to-end description, precedent table and formal review:
`literature_review/Model_B_Technical_Description.docx`,
`literature_review/Model_B_Slides.pptx`,
`literature_review/Model_B_Literature_and_Lineage.docx`.

## Leave-one-subject-out (added 9/23)

Both training scripts take `--loso`: every participant is its own fold. The
protocol is otherwise unchanged. Three things came with it:

- **Inner validation rotates under LOSO.** The old rule took the first two
  participants in sorted order, which under LOSO hands the same two people
  early stopping and the threshold in 18 of 20 folds. `--loso` defaults to
  `--inner-selection rotate` (hash of seed, fold, participant). 4-fold keeps
  `first`, so earlier 4-fold runs still reproduce.
- **Per-participant metrics** are written to every result JSON under
  `participants`, with counts, so `compare_runs.py` can test any two runs.
- **Chew-count scale now comes from the fit participants only.** It used to be
  the 99th percentile over ALL windows, test participant included. Tiny, but a
  label statistic crossing the split. This moves Model B's numbers slightly
  even at 4 folds, so re-run before comparing against the table above.

Output files now carry the CV scheme and ablation flags in the name
(`model_v2_metrics_w8_loso_noaux.json`), so runs no longer overwrite each other.
The old unsuffixed files are not read by anything.

```bat
REM Model A and Model B under LOSO (about 5x the 4-fold runtime)
python train_food_intake_v1_keras.py --config <path>\jitai_config.ini --loso
python train_food_intake_v2.py      --config <path>\jitai_config.ini --loso

REM is B better than A, participant by participant?
python compare_runs.py model_v1_outputs\model_v1_metrics_w8_loso.json model_v2_outputs\model_v2_metrics_w8_loso.json

REM the ablations, now with a paired test each
python train_food_intake_v2.py --config ... --loso --fft-input
python train_food_intake_v2.py --config ... --loso --no-auxiliary
python train_food_intake_v2.py --config ... --loso --no-attention
python train_food_intake_v2.py --config ... --loso --no-overlap
python compare_runs.py model_v2_outputs\model_v2_metrics_w8_loso_noaux.json model_v2_outputs\model_v2_metrics_w8_loso.json
```

Check the first line the run prints: "N participants, M sessions". If M > N,
some participants have two sessions, and LOSO correctly holds both out.

## Next steps, in order

```bat
REM re-run both with the fixed harness (the numbers above predate it)
python train_food_intake_v1_keras.py --config <path>\jitai_config.ini
python train_food_intake_v2.py --config <path>\jitai_config.ini

REM the two untested improvements
python train_food_intake_v2.py --config ... --context-seconds 8
python train_food_intake_v2.py --config ... --context-seconds 8 --ensemble 3

REM window-length sweep: 2 / 4 / 8 / 16 s. 4 s is the prediction
python train_food_intake_v2.py --config ... --sweep-windows

REM ablations for the paper
python train_food_intake_v2.py --config ... --no-auxiliary
python train_food_intake_v2.py --config ... --no-overlap
```

Not yet built: a results comparison figure across models (offered, not started).

## Gotchas that have already cost time

- **TensorFlow 2.16 has no GPU on native Windows** (Google dropped Windows GPU
  builds after 2.10; `tensorflow-intel` is CPU-only). Use WSL2 for GPU. Every
  run prints the device it got. CPU is fine at this model size.
- Passing a **directory** to `--config` instead of the `.ini` file. Explorer
  hides the extension.
- `git` is not on PATH on that machine — GitHub Desktop bundles its own.
- **Versions are law**: only what is pinned in `requirements_ETS_full.txt`.
- No database password in this repository — `--config` or `ETS_DB_PASSWORD`.
  (Worth rotating that password eventually; it has been sitting in a config
  file.)

## The bigger constraint

20 annotated sessions is the binding limit, not the architecture. Ten more
annotated participants would likely beat every modelling trick left on the
list. That is an argument for annotation resources, not an excuse.

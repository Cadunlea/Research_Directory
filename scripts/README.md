# Food-intake detection from AIM-2 sensor data

Everything reads from the ETS database: no annotation CSVs, no hardcoded
participant list, and no time shift. TensorFlow / Keras. The model and the
literature behind every design choice are in
`../docs/literature_review/literature_review.pdf`.

## Files

| File | What it is |
|---|---|
| `train_food_intake.py` | The model, and every experiment for the paper as a flag (table below) |
| `compare_runs.py` | One table of every run, each paired against a reference: F1 change with 95% interval, participants better/worse, Wilcoxon p. Writes `results_summary.csv` |
| `rescore_runs.py` | Re-scores finished runs on the corrected per-session evaluation grid (fix of 20/10/26) into `results_fixed\`, from saved predictions; no retraining |
| `operating_curve.py` | False alarms per hour against recall for finished runs, from their saved predictions; no retraining |
| `plot_confusion.py` | Confusion matrices of one or more runs side by side (counts and row shares), with precision, recall and false alarms per hour under each |
| `episode_metrics.py` | Episode-level results from saved predictions: share of eating episodes detected (by length), delay to first detection, false and edge episodes per hour, episode precision |
| `spectrum_figure.py` | Mean power spectrum of eating vs non-eating windows per channel, chewing band shaded (explanatory figure, reads the window cache) |
| `exclude_participants.py` | Pooled metrics with and without the weakest participants (or named ones), against the run's own 95% interval; repeats the paired comparisons without them |
| `error_breakdown.py` | Sorts every wrong window of a run by its neighbours (bout edge, mid-meal, short bout; isolated or sustained false alarms), with the F1 if each kind were fixed and false episodes per hour |
| `plot_results.py` | Grayscale Arial-11 figures: per-participant F1, prediction timelines, run comparison |
| `ets_data.py` | Database → training windows. The only place the data model lives |
| `ets_eval.py` | Metrics, threshold selection, smoothing, per-participant and paired statistics |
| `ets_train.py` | The cross-validation harness (leave-one-subject-out by default) |
| `probe_database.py` | Read-only report of what the database holds |
| `check_annotation_drift.py` | Measures the sensor/annotation lag per session |
| `plot_drift_evidence.py` | The 3-panel drift figure |
| `test_pipeline.py`, `test_drift_check.py` | Tests, no database needed |

Outputs (all ignored by git): `ets_cache/` windows cache, `results/` one JSON
plus one `_predictions.npz` per run, `figures/`.

## Connecting

Docker Desktop runs the stack locally. Python talks to **MariaDB on 30071**
(`db_ets`). 30073 is phpMyAdmin (browser), 30072 the web API, 30074 the web
interface. Credentials: `--config path\to\jitai_config.ini` or the
`ETS_DB_PASSWORD` environment variable. None are stored here.

## Every flag

The default is the model in the paper. Each flag changes one dimension and
holds everything else fixed.

| Flag | Default | What it tests | Paper question |
|---|---|---|---|
| `--input fft` | `time` | FFT magnitude spectrum instead of the time series | Does keeping *when* rhythms happen matter? (largest effect so far) |
| `--sample-rate 64/32/16` | `128` | Anti-aliased downsampling; network scaled to the same 64 steps and receptive field in seconds | How much of the signal is needed? |
| `--window-seconds 2/4/16` | `8` | Window length | Resolution against context |
| `--context-seconds 8` | `0` | Extra signal each side, with a marker channel for the labelled window | Can neighbouring signal help without inflating false positives? |
| `--no-overlap` | overlap on | Train on non-overlapping windows only | Is the overlap augmentation worth it? |
| `--no-channel-attention` | on | Remove the squeeze-and-excitation gate | Does channel weighting help? |
| `--pooling average` | `attention` | Global average instead of attention pooling | Does temporal attention help? |
| `--transformer-layers 1/2` | `0` | Self-attention between the 64 time steps | Do Transformers add anything at this data size? |
| `--transformer-heads N` | `4` | Heads per Transformer layer | (with the above) |
| `--chew-weight 0/0.1/0.5/1` | `0.2` | Weight of the chew-count head; 0 removes it | How much should counting chews shape the features? |
| `--architecture fcn/resnet` | `attention-cnn` | Published baselines (Wang et al. 2017), as compared on AIM-2 by Ghosh & Sazonov (2022). Slow: 270k / 510k parameters | Where does the model stand against prior work, on identical folds? |
| `--smoothing 3` | `1` (off) | Median filter over windows | Negative result: kept to report it |
| `--ensemble N` | `1` | N models per fold, averaged | Stability, not capacity |
| `--repeats N` | `1` | Whole cross-validation at N seeds; per-participant counts pooled across seeds | Is an effect bigger than initialisation noise? |
| `--seed N` | `9` | Starting seed | |
| `--folds K` | LOSO | K-fold instead of leave-one-subject-out | Quick checks only; report LOSO |
| `--inner-participants N` | `2` | Participants held out of training for early stopping and the threshold | |
| `--inner-selection first/rotate` | rotate under LOSO | Which participants do that job | |
| `--sweep NAME` | | Every level of one dimension, one results file each: `windows`, `sample-rates`, `chew-weights`, `transformer-layers`, `context` | |
| `--monitor auc` | `loss` | Early stopping on validation ROC AUC instead of loss | Is early stopping keeping the right epoch? |
| `--learning-rate X` | `0.001` | Adam learning rate; lower spreads learning over more epochs | Does the model peak too early to be tuned? |
| `--augment` | off | Random per-channel amplitude scaling and Gaussian noise, training only | Does stopping person-specific shortcuts help new people? |
| `--soft-labels` | off | Train on the share of each window that is eating instead of 0/1; evaluation unchanged. Needs a cache built with this version: `--rebuild-cache --cache-dir ets_cache_soft` | Do bout-edge windows, 40% of the errors, improve when the model is told they are partly eating? |
| `--sequence N` | 0 (off) | Reads N consecutive windows at once: the window model on each, then a bidirectional GRU across them, trained end to end; each window keeps its own probability | Can context across minutes remove isolated false alarms (41-43% of them) without erasing short real bouts? |
| `--threshold-objective f0.5` | `f1` | Threshold chosen for F0.5 (precision weighted twice recall) | Fewer false positives at a small recall cost |
| `--threshold-objective false-alarms --max-false-alarms N` | 15 | Most recall within N false alarms per hour of non-eating, chosen on validation participants | The operating point a deployment needs |
| `--epochs N`, `--patience N` | `80`, `12` | Training length, early stopping | |
| `--threads N` | all cores | CPU threads for this run; use 4 to run four runs side by side | |
| `--tag NAME` | | Prefix for the results file name | |
| `--rebuild-cache` | | Rebuild windows from the database. Once, after new annotations | |
| `--output-dir`, `--cache-dir` | `results`, `ets_cache` | | |

Results file names spell out the configuration, e.g.
`time_se_attnpool_chew0.2_w8_128hz_loso_seed9.json`, so no run overwrites
another.

## Order to run things

```bat
REM 0. tests, no database needed (seconds)
python test_pipeline.py
python test_drift_check.py

REM 1. what the database holds now (read-only)
python probe_database.py --config path\to\jitai_config.ini

REM 2. the model, rebuilding the window cache from the new annotations.
REM    Check the line "N session(s) with ground truth" - that is how many
REM    sessions are really annotated.
python train_food_intake.py --config path\to\jitai_config.ini --rebuild-cache

REM 3. any experiment, e.g.
python train_food_intake.py --config path\to\jitai_config.ini --input fft
python train_food_intake.py --config path\to\jitai_config.ini --sweep sample-rates

REM 4. the table and the figures
python compare_runs.py results\*.json
python plot_results.py results\*.json
python plot_results.py results\*.json --compare
```

## What the database holds (measured 2026-09-15; re-run probe_database.py)

- **Sensor**: `aim_raw_data.numeric_data`, `XYZO`, ~8 s packets of `(4, 1022)`
  int16 at 128 Hz: Acc X/Y/Z and Optical.
- **Ground truth**: `study_data.numeric_data`, `CHGT` / `BOGT` / `BIGT`
  (chew / bout / bite), one whole day per row at 10 Hz (864000 samples)
  indexed from midnight.
- **`1` eating, `0` not eating, `-1` NOT RECORDED**, everywhere. `-1` windows
  are dropped, never treated as not eating.
- Sessions join on `(participantID, studyID)`. `record_hash` does **not** match
  between the two tables.
- `data_sampling_frequency` is a **period** (1/128) in the raw table but a
  **frequency** (10.0) in the study table; `data_duration` is **seconds** (8.0)
  in the raw table but a **sample count** (864000) in the study table. Both
  are handled in `ets_data.py`.
- A row in `study_data.participant_info` is an **enrolled** session. Only
  sessions with CHGT/BOGT/BIGT rows are annotated and can be trained on.

## No time shift

`check_annotation_drift.py` measured the sensor/annotation lag on every
annotated session: median +0.10 s with `data_timestamp` read as the packet
start, all sessions within 0.2 s. No correction is applied anywhere.

## GPU

TensorFlow ≥ 2.11 has no GPU support on native Windows; `tensorflow-intel` is
CPU-only. Use WSL2 for GPU. Every run prints the device it got. The default
model is small enough that CPU is fine; the FCN and ResNet baselines are
several times slower.

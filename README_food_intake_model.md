# Food-intake detection from AIM-2 sensor data

Rebuild of the food-intake classifier: reads everything from the ETS database
(no annotation CSVs), TensorFlow/Keras instead of PyTorch, and no hardcoded
time shift.

## Order to run things

```bat
REM 1. what is actually in the database (read-only, ~10 s)
python probe_database.py --config path\to\jitai_config.ini

REM 2. measure the sensor/annotation lag (read-only, ~2 min)
python check_annotation_drift.py --config path\to\jitai_config.ini

REM 3. baseline: the previous architecture on the new data
python train_food_intake_v1_keras.py --config path\to\jitai_config.ini

REM 4. the improved model
python train_food_intake_v2.py --config path\to\jitai_config.ini

REM 5. window-length sweep (2 / 4 / 8 / 16 s) - slower
python train_food_intake_v2.py --config path\to\jitai_config.ini --sweep-windows
```

The first training run builds a window cache in `ets_cache/`; later runs reuse
it. Pass `--rebuild-cache` after new annotations are uploaded.

No password is stored in this repository. Use `--config`, or set
`ETS_DB_PASSWORD`.

## Files

| File | What it is |
|---|---|
| `probe_database.py` | Read-only report of what the database contains |
| `check_annotation_drift.py` | Measures the sensor/annotation lag per session |
| `ets_data.py` | Database → training windows. The one place the data model lives |
| `ets_eval.py` | Metrics, threshold selection, temporal smoothing, reporting |
| `ets_train.py` | The cross-validation harness both models share |
| `train_food_intake_v1_keras.py` | **Model A** — previous architecture, new data source |
| `train_food_intake_v2.py` | **Model B** — time-domain CNN with attention |
| `test_pipeline.py`, `test_drift_check.py` | Tests, no database needed |

## What the database holds (measured 2026-09-15)

- **Sensor**: `aim_raw_data.numeric_data`, `XYZO`, ~8 s packets of `(4, 1022)`
  int16 at 128 Hz — Acc X/Y/Z and Optical. 43 participants, 61 sessions.
- **Ground truth**: `study_data.numeric_data`, `CHGT` / `BOGT` / `BIGT` —
  chew / bout / bite, one whole day per row at 10 Hz (864000 samples) indexed
  from midnight. 20 annotated sessions, ~16 min each, all with 100% sensor
  coverage.
- **`1` eating, `0` not eating, `-1` NOT RECORDED**, everywhere.
- Sessions join on `(participantID, studyID)`. `record_hash` does **not** match
  between the two tables.

Two traps, both handled explicitly in `ets_data.py`:

- `data_sampling_frequency` is a **period** (0.0078125 = 1/128) in the raw
  table but a real **frequency** (10.0) in the study table.
- `data_duration` is **seconds** (8.0) in the raw table but a **sample count**
  (864000) in the study table.

## The 8-second shift does not exist

The previous model hardcoded `ANNOTATION_TIME_SHIFT_SECONDS = -8.0`.
`check_annotation_drift.py` measured the real lag on all 20 sessions:

| `data_timestamp` read as | Median lag | Range | Within ±1 s |
|---|---|---|---|
| packet **start** | **+0.10 s** | +0.00 … +0.20 | **20/20** |
| packet **end** | −7.90 s | −8.00 … −7.80 | 0/20 |

Peak correlation is the same either way (0.716 vs 0.715). Genuine drift would
degrade the correlation; a units error displaces it while preserving its shape.
`data_timestamp` is the packet **start**; the old code computed
`packet_start = data_timestamp - data_duration`, reading a start as an end, and
that one subtraction produced an artefact of exactly one packet — 8 seconds.

**No shift is applied anywhere in this pipeline.**

## Evaluation protocol

The previous 80.3% / F1 0.788 came from a **single 2-participant test split of
249 windows**, with the phantom shift applied and the threshold fixed at 0.40.
That number has a very wide confidence interval. Here:

- **Participant-level 4-fold cross-validation** — every participant is held out
  exactly once. Windows are never split at random: consecutive 8 s slices of
  one meal are near-duplicates, and splitting them would report a score that
  says nothing about a new participant.
- **The decision threshold is chosen on inner validation participants**, never
  on the test fold.
- **Evaluation always uses non-overlapping windows**, even when training used a
  half-window hop, so the numbers stay comparable.
- **F1 and balanced accuracy lead**; accuracy is reported alongside. Per-session
  eating fraction runs 17%–74%, so accuracy alone can look good for a model
  that has collapsed to one class.

Targets: **F1 0.81** (AIM-2 random forest paper) and the previous model's
0.788 / 80.3%.

## Model B, and why each piece is there

1. **Time-domain input** rather than FFT magnitude — magnitude discards phase,
   and chewing is a rhythmic envelope while a bite is a transient.
2. **Cross-channel attention** — optical carries chewing, accelerometers carry
   motion that is sometimes signal and often artefact.
3. **Attention pooling** — global average pooling buries a 2 s bite in a 16 s
   window.
4. **Auxiliary chew-count head** — regresses CHGT marks per window. One bit per
   window is thin supervision for ~5 hours of data; predicting *how many* chews
   forces the trunk to represent rhythm instead of participant-specific
   amplitude. Discarded at inference.
5. **Overlapping training windows** — a half-window hop roughly doubles the
   training set; evaluation stays on the non-overlapping grid.
6. **Temporal median smoothing** — eating is contiguous; isolated positives are
   almost always errors. Reported with and without.

No transformer, no pretrained backbone, no large ensemble: with this much data
they would overfit and be harder to defend.

Ablations: `--no-auxiliary`, `--no-overlap`, `--smoothing 1`.

## GPU note

**TensorFlow 2.16 has no GPU support on native Windows** — Google stopped
shipping Windows GPU builds after 2.10, and `tensorflow-intel` (pinned in
`requirements_ETS_full.txt`) is CPU-only. Run under WSL2 for GPU. Every run
prints which device it got. The models are small enough that CPU is fine.

# AIM-2 Food Intake Detection

Deep-learning food intake detection from the Automatic Ingestion Monitor
Version 2 (AIM-2), an eyeglass-mounted wearable with a jaw-muscle optical
sensor and a 3-axis accelerometer. Part of the Eating Trajectory Study (ETS)
at The University of Alabama, carried out through the Randall Research
Scholars Program.

A compact convolutional neural network reads raw 8 s sensor windows and
decides whether the wearer is eating. Detected eating is then summarized into
eating episodes, chew counts and chewing rate per meal.

## Results

Leave-one-subject-out cross-validation on 61 annotated sessions from 43
participants (3 seeds):

| Measure | Result |
|---|---|
| F1 (pooled over all windows) | 0.86 |
| Precision / recall | 0.849 / 0.870 |
| Per-participant F1, mean ± SD (median) | 0.844 ± 0.101 (0.863) |
| Eating episodes detected | 91.9% (all episodes of 64 s or longer) |
| Median delay to first detection | 0 s |
| Agreement between two human annotators (one session) | F1 0.912 |
| Chewing rate from the optical signal, median error per meal | 7% (constant-rate baseline: 10%) |

Raw time-domain input outperforms a frequency-domain (FFT) input on the same
folds (F1 0.854 against 0.791, with half the false alarms); the attention
components and other architectural options change F1 by less than the
seed-to-seed variation. Full results, with the run behind each number, are in
`docs/compendium/results_compendium.pdf`.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/` | All code: data access, training and evaluation, inference, analysis and figures. `scripts/README.md` describes every script |
| `docs/compendium/` | Results compendium: every result, table and figure with its source |
| `docs/paper/` | Paper in progress (IEEE SoutheastCon template), its figures, and the scripts that build the document |
| `docs/literature_review/` | Literature review and model description (LaTeX, with `refs.bib`) |
| `docs/presentation/` | Lab meeting slide decks |
| `docs/progress_reports/` | Progress reports |
| `HANDOFF.md` | Current state of the project, decisions and open items |
| `NOTEBOOK.md` | Dated research notebook from the start of the project |

No data is stored in this repository. Window caches, results, figures and
trained models are written next to the scripts and ignored by git.

## Getting started

Requirements: Python 3.12 with the packages pinned in the lab's
`requirements_ETS_full.txt` (TensorFlow/Keras, NumPy, SciPy, matplotlib,
mysql-connector-python), and access to the ETS database.

Database credentials are read from the lab's `jitai_config.ini` (or the
`ETS_DB_PASSWORD` environment variable) and are never stored here. On
Windows, from `scripts/`:

```
set CFG=path\to\jitai_config.ini
```

## Usage

All commands run from `scripts/`.

**Evaluate the model** (leave-one-subject-out, final setup):

```
python train_food_intake.py --config "%CFG%" --threads 8 --monitor auc --learning-rate 0.0003 --repeats 3 --inner-participants 5
```

**Train the model that ships** (saved to `models\food_intake_final\`):

```
python train_final_model.py --config "%CFG%" --threads 8
```

**Run inference on every recording** and save eating episodes and per-meal summaries:

```
python process_database.py --config "%CFG%"
python process_database.py --config "%CFG%" --database dry-run
```

`--database dry-run` also checks the database and reports what would be
stored; `--database write` stores the detected episodes as new rows (insert
only). The storage format awaits approval before the first write.

**One session**, with a plot against its annotations:

```
python detect_eating.py --config "%CFG%" --participant <ID> --study ETS_lunch --compare
python plot_sensor_vs_annotations.py --config "%CFG%" --participant <ID> --study ETS_lunch --windows detections\<ID>_ETS_lunch_windows.csv
```

**Tests** (no database needed):

```
python test_pipeline.py
```

## Status

- Model development and evaluation: complete; the model is frozen.
- Inference: complete; writing results back to the database awaits approval
  of the table and format.
- Paper: in progress, targeting IEEE SoutheastCon 2027.

## Contact

Caelan Dunlea, The University of Alabama (cadunlea@crimson.ua.edu).
Supervised by Dr. Edward Sazonov.

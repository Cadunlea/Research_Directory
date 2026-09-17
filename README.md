# Research_Directory
Research materials for Eating Trajectory study stored here

---

## NOTEBOOK:

### Most recent:

**16 September 2026 | 6:40 pm - 12:20 am**
**Primary objective:** Verify Tuesday's results, decompose where the improvement actually came from, and record everything for reproducibility

Tuesday's numbers looked good enough that I wanted to be sure they weren't an artifact of how I was measuring before I showed anyone. That turned out to be the right instinct. Re-running with a corrected evaluation harness moved the numbers around twice, and both times the cause was something in the protocol rather than the model. Threshold selection was the culprit: it was being chosen on unsmoothed predictions and then applied to smoothed ones, and at one point I was spending a whole training participant to stabilize a number that belonged in the estimator instead. Once those were sorted out, Model B settled at F1 0.854 / accuracy 0.905, essentially where it started, but now on a protocol I can defend line by line.

A significant part of tonight's research was the ablation study. Model B changed four things at once relative to Model A, and "it got better" isn't an answer to why. I added `--fft-input`, `--no-attention`, `--no-overlap` and `--no-auxiliary` to `train_food_intake_v2.py` so each component could be removed one at a time against identical folds. The result was unambiguous and not what I expected: the time-domain representation is doing nearly all the work, and the attention machinery contributes almost nothing on its own.

The most important single result is the `--fft-input` run. Model B's architecture fed Model A's FFT features scores 0.767, *worse* than Model A's much simpler network on the same features (0.779). Attention pooling over frequency bins is meaningless; attention pooling over time is not. The architecture wasn't wrong, it was answering a question the representation couldn't pose.

**Ablation results** (Model B with one component removed, same folds):

| Variant | F1 | Δ | Precision |
|---|---|---|---|
| **Model B, full** | **0.854** | — | 0.876 |
| − time-domain input (FFT instead) | 0.767 | **−0.087** | 0.696 |
| − overlap augmentation | 0.829 | −0.025 | 0.834 |
| − attention | 0.847 | −0.007 | 0.857 |
| − auxiliary chew head | 0.849 | −0.006 | 0.857 |
| *Model A, for reference* | 0.779 | | 0.730 |

With a fold-to-fold spread of ±0.047, only the representation effect is large enough to assert. Overlap augmentation is suggestive; attention and the chew head are inside noise and have to be reported as measured but not separable from variance at n=20.

**Work Completed & Code Adjustments:**
* **Fixed Threshold Selection in `ets_eval.py` / `ets_train.py`:** Smoothed predictions now select their own threshold instead of inheriting the unsmoothed one, and the threshold is taken at the centre of the plateau rather than the exact argmax — the objective curve is nearly flat at its peak, so the argmax on a small validation set was landing wherever the noise was. Simulated the change over 400 trials before keeping it; it is worth +0.0009 F1, i.e. nothing, and is kept only so reported thresholds stop jumping between 0.16 and 0.50 for no visible reason.
* **Reverted an Expensive Protocol Change:** Raising the inner validation set from two participants to three cost both models ground (A 0.777→0.765, B 0.853→0.828) because a held-out participant is ~7% of the training data. Threshold stability now lives in the estimator, so it no longer has to be bought with data.
* **Added Ablation Flags to `train_food_intake_v2.py`:** `--fft-input`, `--no-attention`, `--no-overlap`, `--no-auxiliary`. Output heads factored into a shared `_heads()` so an ablation can't accidentally differ in how it's trained rather than in what's ablated.
* **Ran the Full Ablation Table:** Four runs, same harness, same folds; results above.
* **Diagnosed a Failed Experiment:** `--context-seconds 8` (showing the model 8 s either side of the labelled window) cost 0.080 F1. Precision collapsed 0.876→0.683 while recall *rose* to 0.894 — the signature of a model given 24 s of signal with no indication of which 8 s it was being asked about, so it learned "is there eating somewhere in here?" instead. Added a marker channel identifying the labelled window; untested so far.
* **Wrote `HANDOFF.md`:** State of everything: Measured database facts, the drift result, the ablation table, protocol history, and the gotchas — so a cold session or a future me doesn't have to rediscover any of it.

---

**15 September 2026 | 3:11 pm - 6:45 pm**
**Primary objective:** Build the new model architecture and get it reading from the database

The big day. I rebuilt the whole training pipeline around the database instead of the annotation CSVs on my Desktop, moved everything from PyTorch to TensorFlow/Keras 3.15, and dropped `ALLOWED_PARTICIPANTS` entirely. Whoever is in the database is in the study now is being used for the machine learning, so the participant list can't go "stale".

The pipeline is split into three shared modules so the two models can't disagree about anything. `ets_data.py` owns the data model and is the only place that knows how the database is laid out, which matters more than I expected: the raw and study tables use *opposite* conventions for both `data_duration` and `data_sampling_frequency` (a period in one table, a real frequency in the other), so anything reading both with one assumption is wrong by a factor of 128. `ets_eval.py` holds metrics and threshold selection, `ets_train.py` holds the cross-validation harness, and both training scripts run through the identical harness so any difference between them is the model rather than the measurement.

I built two models deliberately. `train_food_intake_v1_keras.py` is my old architecture, layer for layer, just fed from the database — that isolates "what did fixing the data change?" from "what did changing the model change?", which is the first thing anyone will ask. `train_food_intake_v2.py` is the new one from the 4/22 slides: time-domain input instead of FFT magnitude, cross-channel attention, attention pooling, plus an auxiliary chew-count head that regresses CHGT marks per window and gets discarded at inference.

Model B came in at **F1 0.854, accuracy 0.905, balanced accuracy 0.887** — past the 90% accuracy I was hoping for and past the 0.81 F1 from the AIM-2 random forest paper. The biggest single change is precision: 0.684 → 0.876, with false positives down to 139.

**Work Completed & Code Adjustments:**
* **Built `ets_data.py`:** Database → training windows. Handles the two unit traps between tables, drops the `-1` (not recorded) windows rather than treating them as negatives, and joins sessions on `(participantID, studyID)` since `record_hash` doesn't match across the two tables.
* **Built `ets_eval.py` and `ets_train.py`:** Metrics, threshold selection, temporal smoothing, and a participant-level 4-fold cross-validation harness shared by both models.
* **Model A — `train_food_intake_v1_keras.py`:** My previous FFT-CNN ported to Keras, new data source, as the honest baseline.
* **Model B — `train_food_intake_v2.py`:** Time-domain CNN with cross-channel attention, attention pooling, an auxiliary chew-count head, and overlapping training windows (evaluation stays on the non-overlapping grid so numbers remain comparable).
* **Switched to Participant-Level Cross-Validation:** Every participant held out exactly once, threshold chosen on inner validation participants and never on the test fold. The old 0.788 came from a single 2-participant split of 249 windows; this runs on 3,503 held-out windows across all 20.
* **Wrote `test_pipeline.py`:** Tests for the silent failure modes — participant leakage between folds, windows built from un-annotated time, `-1` treated as a real reading, smoothing across a recording gap. Caught a real design flaw in how folds were being assigned.
* **Found a Negative Result:** Temporal median smoothing *lowers* F1 (0.854 → 0.775). At 8 s resolution the eating label isn't temporally contiguous — meals are punctuated by pauses between bites — so a median filter erases short but genuine bouts. Same fact the lab's PADU pause metrics exist to measure.

---

**14 September 2026 | 3:13 pm - 5:29 pm & 8:12 pm - 9:47 pm**
**Primary objective:** Re-run the original model against the new database and settle the 8-second annotation drift

Before rebuilding anything I wanted to know what was actually in the new database rather than what I assumed was in it, so I wrote `probe_database.py` — read-only, just reports what the records contain. Good thing, too: it answered several things I'd have guessed wrong. Un-annotated time is stored as `-1` everywhere (so those windows get dropped, not counted as "not eating"), `CHGT`/`BOGT`/`BIGT` are chew/bout/bite ground truth at 10 Hz indexed from midnight, sessions join on `(participantID, studyID)` rather than `record_hash`, and there are 20 annotated sessions totalling about 7.8 hours, all with full sensor coverage over their annotated windows.

The main event was the drift. My previous model hardcoded `ANNOTATION_TIME_SHIFT_SECONDS = -8.0` because food-like sensor activity seemed to show up ~8 seconds before the annotation said eating had started. Nobody had ever measured it — it was a patch on a symptom. `check_annotation_drift.py` measures it properly, cross-correlating a chewing-band optical envelope against the annotated eating mask across a ±30 s grid, per session.

**It turned out the drift was never in the data.** The repository disagreed with itself about what `data_timestamp` means — the plotting scripts treat it as the packet's start, my training code computed `packet_start = data_timestamp - data_duration` and treated it as the end. A packet is 8 seconds long, so reading a start timestamp as an end timestamp displaces every sensor sample by exactly one packet.

| `data_timestamp` read as | Median lag | Range | Within ±1 s |
|---|---|---|---|
| packet **start** | **+0.10 s** | +0.00 … +0.20 | **20/20** |
| packet **end** | −7.90 s | −8.00 … −7.80 | 0/20 |

Peak correlation is identical either way (0.716 vs 0.715), which is the giveaway — real clock drift degrades the correlation, a units error just slides it while preserving its shape. My annotations were correctly aligned the whole time.

**Work Completed & Code Adjustments:**
* **Wrote `probe_database.py`:** Read-only inspection of both databases — array shapes, what fills un-annotated time, whether `record_hash` matches across tables, and how many sessions have both sensor data and ground truth.
* **Wrote `check_annotation_drift.py`:** Per-session lag measurement by cross-correlation, evaluating both readings of `data_timestamp` so the data decides rather than an assumption.
* **Wrote `test_drift_check.py`:** Validates the measurement against synthetic data with known injected offsets. This caught a real bug — downsampling 128 Hz to 10 Hz by averaging `round(12.8) = 13`-sample blocks actually produces 9.846 Hz, and the error grows to roughly 8 seconds by the middle of a 17-minute session. My own tool would have manufactured exactly the drift it was built to measure.
* **Removed the Hardcoded Shift:** No time shift is applied anywhere in the new pipeline, and the reasoning is documented so nobody re-adds it.
* **Wrote `plot_drift_evidence.py`:** Three-panel figure for the lab meeting — all 20 correlation curves under both conventions, the per-session lags, and a worked example where the alignment can be checked by eye against the annotated eating blocks.
* **Credential Hygiene:** Scripts read the database password from `jitai_config.ini` or `ETS_DB_PASSWORD` rather than storing it, since this repo is on GitHub and a committed credential stays in the history forever.

---

**13 September 2026 | 1:05 pm - 5:22 pm**
**Primary objective:** Assemble project context for the model rebuild and revisit the AIM-2 literature

Spent the day pulling together everything the rebuild would need in one place. The database structure, the existing worker code and its conventions, `posixtime.py` and the GMT/UTC timestamp handling, the 4/22 lab meeting slides describing the proposed architecture, and the environment and dependency pins. A lot of this was scattered across folders and old scripts, and writing it all out made it obvious how many assumptions in my current model came from the old CSV-based setup rather than from the database.

Also went back through the AIM-2 literature to re-ground the approach, since the sensor modality and the chewing-detection framing matter for how the windows and labels should be defined.

**Work Completed:**
* **Gathered Project Context:** Collected database schema, worker conventions, the timestamp handling in `posixtime.py`, the dependency pins in `requirements_ETS_full.txt`, and the 4/22 architecture slides into one working set.
* **Reviewed Current Model Assumptions:** Identified which parts of the existing training script were tied to the CSV workflow (the hardcoded participant list, the Desktop annotation paths, the fixed 0.40 threshold, the −8 s shift) and would need to change.
* **Literature Review:** Revisited the AIM/AIM-2 chewing-detection work to re-ground the window definition and labelling rule.


**9 September 2026 | 8:29 am - 9:03 pm**
Primary objective: Update write ups on github, create slideshow, look into and read more literature on the AIM-2 topic
Specifically https://www.nature.com/articles/s41598-024-51687-3

**8 September 2026 | 3:23 pm - 6:48 pm**
**Primary objective:** Try Claude Code, remote into the lab machine, get the Python environment set up

Since I hadn't been able to get into the lab over the long weekend, my first task today was getting remote access working reliably. Once I was in, I set up the Python environment for the machine learning work from scratch on this machine. I also used the afternoon to try out Claude Code for the first time, since I'd heard about it being used for coding tasks — I ran through it fairly quickly and burned through the available tokens faster than expected, so I'll need to budget my usage better next time or look into how the limits work before relying on it for anything time-sensitive.

**Work Completed & Code Adjustments:**
* **Remote Access Configured:** Set up and verified remote connection into the lab computer so work can continue outside of in-person lab hours.
* **Python Environment Rebuilt:** Installed and configured the machine learning environment (Python packages, dependencies) on the lab machine.
* **Evaluated Claude Code:** Tested Claude Code as a potential development tool; ran into token limits quickly and will need to plan usage more carefully going forward.

---

Timeskip from 9/5/26 -> 9/7/26 Labor Day Weekend (went home)

---

**4 September 2026 | 12:16 pm - 1:37 pm**
**Primary objective:** Complete RRSP Phase II contract

My main task today was finishing the Phase II project contract, which lays out the plan and risk assessment for the semester's work. I sent it to Dr. Sazonov for review shortly after finishing it, and heard back the same day with his approval, which was a relief given the earlier delays with Phase I. The contract was formally submitted on 9/8/26.

**Work Completed & Code Adjustments:**
* **Finalized Phase II Contract:** Completed the project plan and risk assessment sections and sent the contract to Dr. Sazonov for review.
* **Contract Approved:** Received approval from Dr. Sazonov the same day it was sent.
* **Submission:** Formally submitted the approved contract on 9/8/26.

---

**3 September 2026 | 3:10 pm - 5:36 pm**
**Primary objective:** Computer setup in the lab, debug any issues that arise with database access

Today was mostly about getting the lab computer properly set up and making sure I could actually pull data before relying on it for anything else. I ran into a few issues connecting to the database initially, so I spent some time tracing down where the connection was failing and fixing it. By the end of the session I was able to confirm the data-access scripts were pulling data correctly.

**Work Completed & Code Adjustments:**
* **Lab Computer Setup:** Configured the lab workstation for ongoing project work.
* **Debugged Database Access:** Identified and resolved connectivity issues preventing the scripts from reaching the raw database.
* **Verified Data Access:** Confirmed the existing data-access scripts run as intended and retrieve data correctly.

---

**2 September 2026 | 11:00 am - 11:53 am**
**Primary objective:** Attend meeting, write notes about RRSP II contract

Short session today — mainly attended the lab meeting and used the time afterward to write up notes on what still needed to go into the RRSP Phase II contract before I could finalize it.

**Work Completed & Code Adjustments:**
* **Attended Lab Meeting:** Participated in the scheduled lab meeting.
* **Contract Notes:** Wrote notes outlining remaining items for the RRSP Phase II contract, used to complete it on 9/4.

---

Timeskip: Week of 8/26/26-9/2/26
Primary objective: Work on video annotations for the study
Completed 2 participants, attempted 5 (too long, lag issues)

**25 August 2026 | 3:58 pm - 8:12 pm**
**Primary objective:** Create slides, update GitHub documentation, test scripts, and build hyperparameter tuning pipeline

My primary objective for the day was to get some testing done, now that I had access to the computer. In the middle of this, I found out that my github documentation was on the wrong account. I had to restart it from scratch, completely scrapping the other account's documents for consistency.
Below is a description of how that hyperparameter tuning testing worked. It took a long hours to run in the background. My goal was to obtain graphs I could use for the slides. As I waited for the script to finish, I created documentation for the github and created the slides that I will use for tomorrow's meeting.

**Work Completed & Code Adjustments:**
* **Created Hyperparameter Sweep Script:** Imports the baseline training script as a module, reusing dataset construction, FFT feature extraction, train/val/test splits, and threshold-sweep evaluation functions.
* **Parameterized Model Architecture:** Implemented `ParamFrequencyCNN` and `train_model_with_params` to accept variable arguments (learning rate, batch size, dropout, channel widths, and positive-class weight scale) rather than using hardcoded values.
* **Fixed Data Split:** Pre-computes the train/val/test participant split once at startup so all grid search configurations train and validate on identical data.
* **Grid Search & Multi-Seed Runs:** Evaluates every hyperparameter combination across multiple random seeds to measure performance stability and account for weight initialization/batch shuffling noise.
* **Threshold Evaluation & Logging:** Tests models against the baseline threshold grid (0.30–0.90) for direct comparison. Progressively logs individual runs to CSV and aggregates mean/standard deviation metrics per configuration.
* **Slides and Documentation:** Corrected the GitHub documentation issue, and worked on slides for tomorrow's meeting

---

**25 August 2026, 8:55 am - 9:45 pm**  
**Primary objective:** Experiment with variations of the current model

Since I was not able to access the computer in the lab due to Kenzy using it, I worked with variations of the current model. I looked for potential bugs and adjusted parts of the code that Siavash asked for.

**Work Completed & Code Adjustments:**
* **Added Configurable label rule `compute_food_indicator()`**
  * Was hardcoded to the three-way OR; now a switchable rule (default "bout_or_bite")
  * **bout_or_bite:** food = (Chew Bout > 0) OR (Bite > 0)
  * **bout_only:** food = (Chew Bout > 0)
  * **bite_only:** food = (Bite > 0)
  * **count_or_bout_or_bite:** original rule, food = (Chew Count > 0) OR (Chew Bout > 0) OR (Bite > 0)
* **Added classification error rate in `metrics_from_probabilities()`**
  * `error_rate = 1 - accuracy`
* **Mean absolute error in `metrics_from_probabilities()`**
  * `mean_absolute_error = mean( |y_true_i - probability_i| )` over all i windows
  * This is the literal "absolute error," computed on the raw sigmoid probability output, not the thresholded 0/1 prediction so it's sensitive to how confident the model was.
* **Per-participant missing-data table `participant_diagnostics`, written to `participant_missing_fraction_report.csv`**
  * **Per participant/session:** `mean_missing_fraction` = mean over that session's kept windows of (fraction of samples equal to -1) plus span overlap, window counts, and whether it was excluded.
  * This was already printed to console in the original version. Now it is also collected into a table and written to CSV.

---

**23 August 2026, 4:05 pm - 6:44 pm**  
**Primary objective:** Work on a parameter tuning script prototype

I created a working prototype of parameter tuning to go along with the machine learning model previously made. At first, I thought about putting the parameter tuning inside of the ML script, but decided against it for convenience. I am not yet sure if it is compatible with the script, since it appeared that Kenzy was using the computer at this time. I also worked on learning how to create parameter tuning scripts, and how to possibly get the data into a csv.

**Work Completed:**
* Created parameter tuning script to go along with the current working model
* Investigated options for the parameter tuning script, such as which parameters to tune, turning the data into a csv, and where to put the tuning (in or out of the ML script)

---

**19 August 2026, 6:19 pm - 7:28 pm**  
**Primary objective:** Convert to Github, relearn structure of code

I worked on porting my changes to github, including this journal. I looked into methods I could use to port google slides over, but I could not find any way except by using a URL. For now, those will have to stay in the google drive that I created.  
Lastly, I relearned the purpose and the structure of the code, going over old notes and slides to do this.

**Work Completed:**
* Port changes to github and create a ReadMe (will add scripts later)
* Studied the structure of the machine learning algorithm
* Relearned terms, label rules, and design choices

---

**19 August 2026, 11:15 pm - 1:05 pm**  
**Primary objective:** Lab meeting, emails, RRSP contracts

I discussed my project ideas in our research meeting today. Additionally, I presented my slides for the August 19th 2026 lab meeting. The professor suggested that I move my documentation to GitHub.  
Lastly, I worked on emails to write to Dr. Sazonov.

**Work Completed:**
* Finished the first drafts phase 1 and phase 2 contracts
* Sent them to Dr. Sazonov for review, alongside my CWID
* Discussed scheduling and weekly hourly commitment

---

**18 August 2026, 9:41 pm - 11:35 pm**  
**Primary objective:** Research setup and housekeeping

I worked on the creation of the original notebook in google docs. Additionally, I worked on slides to present at tomorrow’s meeting (August 19th 2026). To have one home for my notebook and my presentations for the future, I created a google drive folder which will be shared with my instructors.  
Lastly, I worked on the draft of an email to write to Dr. Sazonov and Siavash. The contents of this email were focused on RRSP Contracts I and II.  
I worked on and off, with periods where I paused to do other things such as put my classes in my calendar. Worked in total for about an hour and a half.

**Work Completed:**
* Set up the structure of this notebook
* Wrote the first notebook entry
* Created a google drive folder to organize my research contents
* Drafted and later sent an email discussing Contract I and II for RRSP

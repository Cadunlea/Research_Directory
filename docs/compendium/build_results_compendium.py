"""Results compendium, 4 November 2026: every result for the paper in one place.

Builds results_compendium.docx in this folder. A reference for writing, not
paper text: each section is a table or figure with its source run and its
caveats, and the last section lists what still has to be recomputed or
confirmed. Every number is copied from the runs named in each table
(HANDOFF.md records them); nothing is estimated.

Arial 11, grayscale, US Letter, matching docs/progress_reports.

    python build_results_compendium.py
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "paper" / "figures"
FONT, INK, NOTE = "Arial", RGBColor(0x22, 0x22, 0x22), RGBColor(0x70, 0x70, 0x70)

FINAL_RUN = "results_fixed\\time_se_attnpool_chew0.2_w8_128hz_loso_seed9_x3_stopauc_lr0.0003_inner5"
REF_RUN = "results_fixed\\time_se_attnpool_chew0.2_w8_128hz_loso_seed9_stopauc_lr0.0003 (2 validation participants)"

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Inches(0.9))


def style_font(style, size, bold=None):
    style.font.name, style.font.size, style.font.color.rgb = FONT, Pt(size), INK
    if bold is not None:
        style.font.bold = bold
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for key in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(key), FONT)
    for key in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if fonts.get(qn(key)) is not None:
            del fonts.attrib[qn(key)]
    ppr = style.element.find(qn("w:pPr"))
    if ppr is not None:
        for border in ppr.findall(qn("w:pBdr")):
            ppr.remove(border)


style_font(doc.styles["Normal"], 11)
doc.styles["Normal"].paragraph_format.space_after = Pt(4)
for name, size in (("Title", 18), ("Heading 1", 14), ("Heading 2", 12)):
    style_font(doc.styles[name], size, bold=True)
    doc.styles[name].paragraph_format.space_before = Pt(14 if name != "Title" else 0)
    doc.styles[name].paragraph_format.space_after = Pt(4)


def text(body: str, size=None, italic=False, gray=False, after=None):
    p = doc.add_paragraph()
    run = p.add_run(body)
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    if gray:
        run.font.color.rgb = NOTE
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def bullets(items):
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def source(body: str):
    text(f"Source: {body}", size=9, gray=True, after=10)


def shade(cell, fill="E7E6E6"):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


TABLES = [0]


def table(caption: str, header, rows, widths):
    TABLES[0] += 1
    p = text(f"Table {TABLES[0]}. {caption}", size=10, after=3)
    p.runs[0].bold = True
    p.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        run.bold, run.font.size = True, Pt(10)
        shade(cell)
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(value))
            run.font.size = Pt(10)
    # Fixed widths: Word and LibreOffice otherwise split the columns evenly.
    t.autofit = False
    grid = t._tbl.tblGrid
    for i, w in enumerate(widths):
        t.columns[i].width = Inches(w)
        grid.gridCol_lst[i].w = Inches(w)
    for r, row in enumerate(t.rows):
        for i, w in enumerate(widths):
            row.cells[i].width = Inches(w)
            # Keep each table on one page with its caption.
            if r < len(t.rows) - 1:
                for par in row.cells[i].paragraphs:
                    par.paragraph_format.keep_with_next = True


FIGURES = [0]


def figure(name: str, width: float, caption: str):
    FIGURES[0] += 1
    doc.add_picture(str(FIG / name), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.keep_with_next = True
    p = text(f"Figure {FIGURES[0]}. {caption}", size=10, after=10)
    p.runs[0].bold = False


# --------------------------------------------------------------------------- #
doc.add_paragraph("Results Compendium", style="Title")
text("AIM-2 food intake detection, ETS dataset. Caelan Dunlea, 4 November 2026.", after=2)
text("Every result for the paper in one place, with the run it comes from and its caveats. "
     "Reference material for writing, not paper text. Section 10 lists what must be "
     "recomputed or confirmed before submission.", size=10, gray=True, after=8)

text("Conventions used throughout", after=2).runs[0].bold = True
bullets([
    "Leave-one-subject-out (LOSO): 43 folds, each holding out one participant with all of "
    "their sessions; 5 of the remaining participants choose early stopping and the threshold.",
    "Pooled metrics are computed over all held-out windows together; participant metrics are "
    "computed per participant, then summarized.",
    "Windows are 8 s. Training uses a 4 s hop; every score uses non-overlapping windows set "
    "up per session.",
    "“Final configuration” is the frozen setup with 5 validation participants, run with "
    "3 seeds. “Reference configuration” is the earlier 2-validation-participant setup; "
    "numbers from it are marked.",
])

# --------------------------------------------------------------------------- #
doc.add_paragraph("1. Dataset", style="Heading 1")
table("Dataset summary.", ["Item", "Value"], [
    ["Participants", "43"],
    ["Sessions", "61: breakfast 8, lunch 18, dinner 12, snack 23"],
    ["Sessions with eating", "60 (AIM125068 snack has no window labeled eating)"],
    ["Annotated recording", "24.2 h, of which 9.6 h eating"],
    ["Annotated span per session", "2 to 55 min"],
    ["Eating share per session", "0% to 86%"],
    ["Full recordings (sensor data)", "0.09 to 12.5 h per session"],
    ["Training windows (8 s, 4 s hop)", "18,227 (7,014 eating, 38.5%)"],
    ["Scored windows (8 s, non-overlapping)", "9,133"],
    ["Annotation", "Video, 10 Hz: chewing bouts, bites, individual chews"],
    ["Label", "Eating = bout or bite; window eating if more than half of its annotated "
              "samples are eating"],
    ["Excluded windows", "Under half annotated, or over half of sensor samples missing"],
    ["Inter-annotator agreement", "F1 0.912 on 8 s windows, one snack session "
                                  "(C. Dunlea and K. Nguyen)"],
], [2.4, 4.3])
source("ets_cache\\ets_windows_w8_h4.npz; process_database.py --list; literature review "
       "model description.")

# --------------------------------------------------------------------------- #
doc.add_paragraph("2. Model and training", style="Heading 1")
figure("fig_architecture.png", 3.4, "Model architecture (from the literature review). The "
       "chew-count head is trained jointly; its rate output is not used (Section 8).")
table("Layers and parameters.", ["Layer", "Configuration", "Output", "Parameters"], [
    ["Input", "Acc X, Y, Z, optical", "1024 × 4", "–"],
    ["Conv block 1", "32 filters, kernel 9, stride 2, BN, max-pool 2", "256 × 32", "1,312"],
    ["Conv block 2", "64 filters, kernel 9, stride 2, BN, max-pool 2", "64 × 64", "18,752"],
    ["Conv block 3", "64 filters, kernel 5, BN", "64 × 64", "20,800"],
    ["Channel attention", "Squeeze-excite, 64 → 16 → 64", "64 × 64", "2,128"],
    ["Attention pooling", "1 × 1 conv, softmax over time", "64", "65"],
    ["Dense", "64, ReLU, dropout 0.3", "64", "4,160"],
    ["Eating head", "1, sigmoid", "1", "65"],
    ["Total at inference", "", "", "47,282"],
    ["Chew-count head", "32 → 1, softplus (trained jointly)", "1", "2,113"],
], [1.45, 3.0, 1.05, 1.2])
table("Training settings.", ["Setting", "Value"], [
    ["Preprocessing", "Per window and channel: mean removed, z-score over valid samples, "
                      "gaps set to 0. No band-pass filter."],
    ["Optimizer", "Adam, learning rate 0.0003, batch 64, class-balanced sample weights"],
    ["Stopping", "Up to 80 epochs; early stopping on validation ROC AUC, patience 12, "
                 "best weights restored; learning rate halved on plateau"],
    ["Chew target", "Annotated chews per window divided by its 99th percentile; loss weight 0.2"],
    ["Threshold", "Chosen per fold on the 5 validation participants to maximize F1"],
    ["Seeds", "3 (9, 10, 11)"],
], [1.6, 5.1])

# --------------------------------------------------------------------------- #
doc.add_paragraph("3. Window-level results (LOSO)", style="Heading 1")
table("Final configuration, 43 participants, 3 seeds.", ["Metric", "Value"], [
    ["F1", "0.86 (0.859 to 0.861 across seeds)"],
    ["Accuracy", "0.890"],
    ["Balanced accuracy", "0.886"],
    ["Precision", "0.849"],
    ["Recall (sensitivity)", "0.870"],
    ["Specificity", "0.902 (from balanced accuracy and recall)"],
    ["Median participant F1", "0.862"],
    ["False alarms per hour of non-eating", "43.8 windows"],
    ["Seed-to-seed variation", "about ±0.007 F1"],
    ["Human agreement (one session)", "0.912"],
], [3.0, 3.7])
source(FINAL_RUN + "; docs/presentation/final_metrics_table.pptx.")
figure("fig_participants.png", 6.6, "F1 per held-out participant, final configuration, with "
       "the pooled F1 dashed. Participant IDs must become P1 to P43 in the paper.")
table("Lowest participants and causes (F1 read from Figure 2).", ["Participant", "F1",
      "Cause"], [
    ["AIM132097", "~0.50", "Chews about 4 s, pauses about 3 s to talk, repeats: bouts shorter "
                           "than a window; errors at edges"],
    ["AIM126487", "~0.62", "Not investigated; the lunch is the weak session (re-annotation planned)"],
    ["AIM147560", "~0.63", "Soft food (oatmeal) barely moves the jaw sensor; 6 min session, "
                           "46 windows"],
    ["AIM122571", "~0.65", "Short eating bouts"],
    ["AIM119072", "~0.70", "Not investigated"],
    ["AIM107903", "~0.73", "Weak at both meals"],
    ["AIM127703", "~0.73", "Long high-probability stretches between annotated bouts; possible "
                           "unannotated eating (check video at 1.8–3, 11.3–12.3, "
                           "15.5–16.3 min)"],
], [1.2, 0.6, 4.9])
text("AIM112400 was 0.48 with 2 validation participants (threshold too strict) and is about "
     "0.80 in the final configuration.", size=10, gray=True)

# --------------------------------------------------------------------------- #
doc.add_paragraph("4. Episode-level results", style="Heading 1")
table("Final configuration. An episode is a run of consecutive eating windows.",
      ["Measure", "Value"], [
          ["Eating episodes detected", "91.9%"],
          ["Episodes of 64 s or longer detected", "100%"],
          ["Episodes of 8 s detected", "82%"],
          ["Median delay to first detection", "0 s"],
          ["False episodes per hour (away from meals)", "9.9"],
          ["Episode precision", "87.8%"],
          ["Sessions with eating that had a detection", "60 of 60"],
      ], [3.6, 3.1])
source("episode_metrics.py on the final run; docs/presentation/episode_table_final.pptx. "
       "Confirm the merge gap used (0 or 32 s) before reporting.")
figure("fig_example.png", 6.6, "Held-out example, AIM146271 lunch (final model): annotated "
       "eating above, model probability below with the 0.75 threshold dashed. F1 0.886, "
       "precision 0.925, recall 0.849.")

# --------------------------------------------------------------------------- #
doc.add_paragraph("5. Where errors occur", style="Heading 1")
table("Error location (reference configuration, one seed; recompute on the final configuration).",
      ["Measure", "Value"], [
          ["Errors at bout edges", "62% of all errors"],
          ["False-alarm windows at edges", "75%"],
          ["Missed windows at edges", "49%"],
          ["F1 with edge windows forgiven", "0.944"],
          ["Isolated false alarms", "19% of false alarms"],
          ["False episodes per hour", "8.9"],
          ["Fold thresholds", "0.06 to 0.93 (with 2 validation participants)"],
      ], [3.6, 3.1])
source(REF_RUN + "; error_breakdown.py.")

# --------------------------------------------------------------------------- #
doc.add_paragraph("6. Design comparisons", style="Heading 1")
table("Same LOSO folds; differences in F1 from the reference unless stated.",
      ["Change", "Result"], [
          ["Frequency (FFT) input instead of time domain",
           "F1 0.854 → 0.791; false alarms 41 → 84 per hour; 5 of 43 participants better"],
          ["5 validation participants instead of 2",
           "+0.008 [0.001, 0.016], 30 better / 12 worse, p = 0.004"],
          ["8 validation participants instead of 5", "+0.001 (no difference)"],
          ["Recurrent model over 2 min of windows", "−0.003 (no difference)"],
          ["Soft labels", "+0.001 (no difference)"],
          ["32 Hz / 16 Hz instead of 128 Hz", "−0.013 / −0.020"],
          ["16 s windows instead of 8 s", "−0.014 [−0.030, +0.002]"],
          ["Within seed variation", "Average pooling, no channel attention, chew weight 0 or 1, "
                                    "transformer layer, 3-model ensemble, augmentation, "
                                    "4 s windows, 64 Hz"],
          ["Excluding the 3 participants below 0.55", "0.854 → 0.864, inside the 95% CI"],
      ], [2.7, 4.0])
source("compare_runs.py on results_fixed\\ (paired bootstrap over participants, Wilcoxon).")
figure("fig_operating_curve.png", 4.6, "Recall against false alarms per hour, time-domain vs "
       "FFT input, same network. Dots: operating points chosen on validation participants.")

# --------------------------------------------------------------------------- #
doc.add_paragraph("7. Meal type", style="Heading 1")
table("Final configuration, LOSO.", ["Meal", "Sessions", "F1 (pooled)", "Recall",
      "Median bout"], [
    ["Breakfast", "8", "0.882", "0.909", "16 s"],
    ["Dinner", "12", "0.884", "0.931", "24 s"],
    ["Lunch", "18", "0.829 (precision 0.841)", "0.819", "16 s"],
    ["Snack", "23 (22 with eating)", "0.877", "0.891", "20 s"],
], [1.2, 1.5, 1.8, 0.9, 1.3])
bullets([
    "Per-session median F1: lunch 0.815 vs 0.890 for other meals, Mann-Whitney p = 0.009.",
    "Participants with a lunch and another session (9): 7 lower at lunch, median difference "
    "−0.081, Wilcoxon p = 0.098.",
    "Some lunches were eaten alone: this is a meal-type result, not a group-eating result.",
])
source("session_type_analysis.py and plot_meal_types.py on " + FINAL_RUN + ".")
figure("fig_meal_types.png", 6.6, "Left: every session's F1 by meal type, medians marked. "
       "Right: the 9 participants with a lunch and another session.")

# --------------------------------------------------------------------------- #
doc.add_paragraph("8. Final model, inference and chewing", style="Heading 1")
table("The shipped model.", ["Item", "Value"], [
    ["Training participants", "38"],
    ["Validation participants", "AIM146271, AIM128555, AIM109345, AIM147560, AIM116725 "
                                "(early stopping and threshold only)"],
    ["Threshold", "0.750"],
    ["Best epoch", "3 of 15"],
    ["Whole database", "61 recordings, 370 eating episodes; about 1 s per recording "
                       "(4 s for 12.5 h)"],
    ["Median F1, 8 held-out sessions", "0.854 (LOSO median participant F1: 0.862)"],
    ["Median F1, 53 trained-on sessions", "0.889 (optimistic: the model trained on them)"],
], [2.4, 4.3])
table("Held-out sessions (final model).", ["Participant", "Meal", "F1"], [
    ["AIM109345", "Lunch", "0.792"], ["AIM109345", "Snack", "0.864"],
    ["AIM116725", "Lunch", "0.823"], ["AIM116725", "Snack", "0.865"],
    ["AIM128555", "Breakfast", "0.844"], ["AIM128555", "Snack", "0.915"],
    ["AIM146271", "Lunch", "0.886"], ["AIM147560", "Breakfast", "0.667"],
    ["Median", "", "0.854"],
], [2.2, 2.2, 2.2])
source("train_final_model.py; process_database.py --overwrite, 4 November 2026.")
table("Chews and chewing rate per meal against the annotations.",
      ["Estimate", "Windows", "Meals", "r", "Median error"], [
          ["Chew count, model output", "Detected", "8 held out / 53 trained", "0.92 / 0.97",
           "14% / 10%"],
          ["Chewing rate, model output", "Detected", "8 / 52", "−0.23 / 0.42", "20% / 12%"],
          ["Chewing rate, optical peak count", "Annotated", "59", "0.72", "7%"],
          ["Chewing rate, optical spectral peak", "Annotated", "59", "0.67", "15% (16% high)"],
          ["Chewing rate, optical peak count", "Detected", "8 / 52", "0.39 / 0.75", "7% / 7%"],
          ["Same rate for every meal (baseline)", "–", "60", "–", "10%"],
      ], [2.3, 0.9, 1.5, 0.9, 1.1])
bullets([
    "The chew-count output is right mainly because eating time is right: it predicts about the "
    "same chews per window for everyone, so its rate is nearly constant.",
    "Optical peak count: optical signal band-passed to 0.94–2.17 Hz (Po et al. 2011), peaks "
    "at least 0.46 s apart counted. No training; settings fixed from the literature and not "
    "tuned on these meals. It needs no held-out split; the r of 0.39 is on 8 meals only.",
    "Annotated rate: chews in annotated eating windows per minute of those windows "
    "(61 to 112 chews/min, mean 85).",
])
source("plot_meals.py and chew_rate_check.py, 4 November 2026.")
figure("fig_chews.png", 5.6, "Chews per meal and chewing rate from the model's chew output "
       "(earlier two-panel version). Replace with figures\\chew_check.png from the lab machine, "
       "which adds the optical rate, and add figures\\chew_rate_check.png.")

# --------------------------------------------------------------------------- #
doc.add_paragraph("9. Prior work for comparison", style="Heading 1")
table("Not directly comparable: data, epochs and protocols differ.",
      ["Study", "Data and method", "Result"], [
          ["Doulah et al. 2021 (AIM-2)", "30 participants, 10 s epochs, hand-crafted features, "
           "SVM, LOSO", "F1 81.8 ± 10.1% (mean over participants); episode detection "
           "accuracy 82.7%"],
          ["Ghosh and Sazonov 2022", "AIM-2 time series, five deep architectures",
           "Best (ResNet) balanced accuracy 93.47%"],
          ["Ghosh et al. 2024", "AIM-2 images plus accelerometer, free living",
           "F1 80.77% combined (sensitivity 94.59%, precision 70.47%)"],
          ["Kyritsis et al. 2021", "Smartwatch inertial data, CNN plus LSTM, LOSO",
           "Bite F1 0.923; meal start/end Jaccard 0.820"],
          ["This work", "43 participants, 8 s windows, compact CNN, LOSO",
           "F1 0.86 pooled; median participant 0.862; 91.9% of episodes"],
      ], [1.6, 2.6, 2.5])
text("To compare with Doulah et al., also report this work's mean ± standard deviation of "
     "per-participant F1 (not yet computed for the final configuration).", size=10, gray=True)

# --------------------------------------------------------------------------- #
doc.add_paragraph("10. To recompute or confirm before submission", style="Heading 1")
bullets([
    "Error location (Section 5) and the fold threshold range on the final configuration.",
    "Mean ± SD of per-participant F1, final configuration.",
    "Episode metrics: which merge gap the reported table used.",
    "AIM125068 snack: why no window is labeled eating (near-empty snack or missing annotation).",
    "Sensor type: the ETS database calls the channel optical; Doulah 2021 describes a flex "
    "sensor. Confirm which AIM-2 version ETS used.",
    "Whether Doulah 2021 or Ghosh 2022 compared against inter-annotator agreement.",
    "Inter-annotator agreement on more than one session (re-annotation of the weakest "
    "participants planned).",
    "Relabel participants P1 to P43 in figures.",
    "With Dr. Sazonov: author list and order, funding and grant number, whether ETS data can "
    "be published, SoutheastCon deadline, AI-use disclosure wording.",
])

out = HERE / "results_compendium.docx"
doc.save(out)
print(f"wrote {out}: {TABLES[0]} tables, {FIGURES[0]} figures")

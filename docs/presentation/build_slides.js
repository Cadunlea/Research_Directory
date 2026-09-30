// New slides for the lab meeting. Grayscale only, Arial throughout, tables and
// chart text at 11 pt, matching the existing deck (white, gray table headers).
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "Food-intake detection: lab meeting update";

const FONT = "Arial";
const INK = "222222", MUTED = "666666", RULE = "BFBFBF", HEAD = "EFEFEF";
const DARK = "404040", LIGHT = "A6A6A6";
const FIG = "/home/user/Research_Directory/docs/literature_review/figure_architecture.png";

function title(slide, text) {
  slide.addText(text, { x: 0.5, y: 0.3, w: 9, h: 0.6, fontFace: FONT, fontSize: 28,
    color: INK, margin: 0, isTextBox: true });
}

function table(slide, rows, opts) {
  const body = rows.map((row, r) => row.map((cell) => ({
    text: String(cell),
    options: { bold: r === 0, fill: { color: r === 0 ? HEAD : "FFFFFF" },
               align: r === 0 && !opts.leftHeader ? "center" : "left" } })));
  slide.addTable(body, Object.assign({
    fontFace: FONT, fontSize: 11, color: INK, valign: "middle",
    border: { type: "solid", pt: 0.75, color: RULE }, margin: 0.05,
  }, opts));
}

function bullets(slide, items, box) {
  slide.addText(items.map((t, i) => ({ text: t, options: {
      bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6 } })),
    Object.assign({ fontFace: FONT, fontSize: 14, color: INK, valign: "top",
                    isTextBox: true }, box));
}

// ------------------------------------------------------------------ 1 ----
// Evidence drawn only from the ten papers on the literature slides. Where a
// component is this project's own addition, the slide says so.
{
  const s = pres.addSlide();
  title(s, "How the Literature Shaped the Model");
  table(s, [
    ["Model component", "What the reviewed papers found", "Papers"],
    ["Learned features from the normalised time series, not hand-crafted features",
     "Deep nets on AIM-2 optical + accelerometer signals reached 93.5% balanced accuracy; end-to-end learning on raw IMU beat hand-engineered methods (bite F1 0.923)",
     "Ghosh & Sazonov 2022; Kyritsis et al. 2021; Doulah et al. 2021"],
    ["Optical and accelerometer together, weighted per window (channel attention)",
     "Walking at 1\u20133 Hz looks like chewing and causes false positives; motion signals are needed to reject it",
     "Diou et al. 2022; Doulah et al. 2021"],
    ["Light attention on a CNN, Transformer tested as an option",
     "CNN + self-attention works on 61 people; a Transformer had specificity of only 0.50; Transformer ECG results used beat-level splits",
     "Wang et al. 2024; Vedovelli et al. 2026; Ikram et al. 2025"],
    ["Participant-level (leave-one-out) validation",
     "AIM-2, smartwatch bite and Transformer eating studies all test on held-out people; the 97% ECG result did not split by patient",
     "Doulah et al. 2021; Kyritsis et al. 2021; Vedovelli et al. 2026; Ikram et al. 2025"],
    ["Operating point and negatives aimed at false positives",
     "Hard negative examples and sensor-image fusion both reduced false positives",
     "Stankoski et al. 2021; Ghosh et al. 2024"],
    ["Chew-count head; time series vs FFT",
     "Our additions: few labels motivate extra supervision; the FFT comparison is our own result (F1 0.858 vs 0.796)",
     "This work (motivated by Diou et al. 2022)"],
  ], { x: 0.5, y: 1.0, w: 9, colW: [2.6, 4.2, 2.2], leftHeader: true });
  s.addNotes(
    "Everything here comes from the papers on the previous two slides. Ghosh & Sazonov ran deep " +
    "networks directly on AIM-2 signals, the same sensor we use, and Kyritsis showed end-to-end learning " +
    "beating hand-engineered features. Diou's review is the reason both sensors stay in: walking at 1-3 Hz " +
    "looks like chewing and causes false positives. On Transformers the evidence is mixed: Wang 2024 made " +
    "attention work on 61 people, Vedovelli's Transformer had poor specificity, and the 97% ECG result " +
    "was not split by patient. So I use light attention by default and test a Transformer as an option. " +
    "Two pieces are ours, and I say so: the chew-count head, and the time-series-versus-FFT comparison, " +
    "which is our own experimental result.");
}

// ------------------------------------------------------------------ 2 ----
{
  const s = pres.addSlide();
  title(s, "Model Architecture");
  const h = 4.55, w = h * (146.80 / 159.12);
  s.addImage({ path: FIG, x: 0.5, y: 0.95, w, h });
  bullets(s, [
    "Reads 8 s of raw signal: 3 accelerometer axes and the optical sensor at 128 Hz",
    "Three convolution blocks learn their own filters; each output step covers about one chew cycle",
    "Two attention steps: which sensor matters, and which moment matters",
    "Chew-count head used only in training",
    "47,282 parameters: small enough to train on CPU in about a minute per fold",
  ], { x: 5.0, y: 1.1, w: 4.5, h: 4.2 });
  s.addNotes(
    "Top to bottom: raw signal in, normalised per window, three convolution blocks, then the two " +
    "attention mechanisms (shaded). The dashed box is the chew-count head: it forces the network to " +
    "learn the chewing rhythm during training and is thrown away afterwards. The whole model is 47k " +
    "parameters, deliberately small for the amount of annotated data.");
}

// ------------------------------------------------------------------ 3 ----
{
  const s = pres.addSlide();
  title(s, "Dataset: All 61 Sessions Now Annotated");
  const stats = [["61", "annotated sessions"], ["43", "participants"],
                 ["24.2 h", "annotated recording"], ["9.6 h", "of eating"]];
  stats.forEach(([big, small], i) => {
    const x = 0.5 + i * 2.3;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.1, w: 2.05, h: 1.45,
      fill: { color: "F5F5F5" }, line: { color: "F5F5F5" } });
    s.addText(big, { x, y: 1.2, w: 2.05, h: 0.8, fontFace: FONT, fontSize: 36, bold: true,
      color: INK, align: "center", margin: 0, isTextBox: true });
    s.addText(small, { x, y: 1.95, w: 2.05, h: 0.45, fontFace: FONT, fontSize: 13,
      color: MUTED, align: "center", margin: 0, isTextBox: true });
  });
  table(s, [
    ["Meal", "Sessions"], ["Breakfast", "8"], ["Lunch", "18"], ["Dinner", "12"], ["Snack", "23"],
  ], { x: 0.5, y: 2.9, w: 3.2, colW: [1.9, 1.3] });
  bullets(s, [
    "Up from 20 sessions (7.8 h): three times the data",
    "18 participants have two sessions; each test fold holds out a whole person, both sessions together",
    "Sessions run 2 to 55 minutes; eating share runs 0% to 86%",
  ], { x: 4.1, y: 2.9, w: 5.4, h: 2.3 });
  s.addNotes(
    "The database now has all 61 sessions annotated, from 43 people. That's why the logs say 43 " +
    "participants: 18 people came in twice. Splits are always by person, never by window, so the " +
    "model is never tested on someone it trained on.");
}

// ------------------------------------------------------------------ 4 ----
{
  const s = pres.addSlide();
  title(s, "Key Finding: Time Series Beats the Spectrum");
  s.addChart(pres.charts.BAR, [
    { name: "Time series (model)", labels: ["F1", "Accuracy", "Precision", "Recall"],
      values: [0.858, 0.891, 0.884, 0.834] },
    { name: "FFT spectrum", labels: ["F1", "Accuracy", "Precision", "Recall"],
      values: [0.796, 0.829, 0.754, 0.843] },
  ], { x: 0.5, y: 1.0, w: 5.4, h: 4.2, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
       chartColors: [DARK, LIGHT], valAxisMinVal: 0.6, valAxisMaxVal: 1.0,
       valAxisLabelFormatCode: "0.00", showValue: true, dataLabelPosition: "outEnd",
       dataLabelFormatCode: "0.000", dataLabelFontSize: 11, dataLabelFontFace: FONT,
       catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, catAxisLabelFontFace: FONT,
       valAxisLabelFontFace: FONT, catAxisLabelColor: INK, valAxisLabelColor: MUTED,
       valGridLine: { color: "E0E0E0", size: 0.5 }, catGridLine: { style: "none" },
       showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: FONT });
  s.addText("81 → 32", { x: 6.2, y: 1.2, w: 3.3, h: 0.8, fontFace: FONT, fontSize: 40,
    bold: true, color: INK, margin: 0, isTextBox: true });
  s.addText("false alarms per hour of non-eating, spectrum → time series", { x: 6.2, y: 2.0,
    w: 3.3, h: 0.6, fontFace: FONT, fontSize: 13, color: MUTED, margin: 0, isTextBox: true });
  bullets(s, [
    "Same model, only the input changed",
    "Time series wins in all 5 folds",
    "Replicates the 20-session result on three times the data",
  ], { x: 6.2, y: 2.85, w: 3.3, h: 2.2 });
  s.addNotes(
    "This is the headline. Everything else held fixed, only the input representation changes. The " +
    "spectrum tells the model which rhythms are in the window but not when they happen, so it fires " +
    "on any rhythmic head movement: 81 false alarms per hour of non-eating versus 32. Recall is about " +
    "the same; the whole gain is precision. We saw the same thing on the first 20 sessions, and it " +
    "holds on all 61, winning in every fold. 5-fold participant cross-validation, 10,891 test windows.");
}

// ------------------------------------------------------------------ 5 ----
{
  const s = pres.addSlide();
  title(s, "Training Experiments");
  table(s, [
    ["Run", "F1", "Accuracy", "Recall", "False alarms / h", "Worst fold F1"],
    ["Default", "0.858", "0.891", "0.834", "32", "0.772"],
    ["Early stopping on AUC", "0.858", "0.891", "0.835", "33", "0.772"],
    ["AUC + learning rate 0.0003", "0.864", "0.893", "0.861", "39", "0.817"],
    ["AUC + augmentation", "0.862", "0.893", "0.846", "34", "0.814"],
    ["Augmentation + learning rate 0.0003", "0.854", "0.886", "0.842", "39", "0.789"],
  ], { x: 0.5, y: 1.05, w: 9, colW: [3.0, 0.9, 1.1, 1.0, 1.6, 1.4] });
  bullets(s, [
    "Early stopping kept the epoch-1 model in 3 of 5 folds: on new people the model peaks almost immediately",
    "A lower learning rate moved the best epoch to 2-6 and lifted the weakest fold from 0.772 to 0.817",
    "All differences are within about 0.006 F1; to be confirmed with paired per-participant tests",
  ], { x: 0.5, y: 3.0, w: 9, h: 1.9 });
  s.addNotes(
    "Diagnosing the training: the logs showed early stopping keeping the very first epoch. That " +
    "wasn't an artefact of what it monitored (switching to AUC changed nothing), so the model really " +
    "does generalise best after one pass, then starts fitting the training people. A lower learning " +
    "rate helped most, especially on the hardest fold, and F1 is now 0.864. But these gains are small " +
    "and I'll only claim them after a paired test across participants.");
}

// ------------------------------------------------------------------ 5b ---
{
  const s = pres.addSlide();
  title(s, "Reducing False Positives");
  table(s, [
    ["Operating point", "How the threshold is chosen"],
    ["Now: best F1", "Balances false alarms and misses equally"],
    ["F0.5", "Counts a false alarm twice as costly as a miss"],
    ["False-alarm budget", "Catches as much eating as possible within N false alarms per hour"],
  ], { x: 0.5, y: 1.05, w: 4.4, colW: [1.5, 2.9], leftHeader: true });
  bullets(s, [
    "Current model: 32 false alarms per hour of non-eating, precision 0.88",
    "Threshold always chosen on validation participants, never the test person",
    "No retraining needed to see the trade-off: the curve comes from saved predictions",
  ], { x: 0.5, y: 2.75, w: 4.4, h: 2.4 });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.05, w: 4.3, h: 4.1,
    fill: { color: "FFFFFF" }, line: { color: RULE, width: 0.75, dashType: "dash" } });
  s.addText("Paste figures\\operating_curve.png here", { x: 5.2, y: 2.9, w: 4.3, h: 0.4,
    fontFace: FONT, fontSize: 11, color: MUTED, align: "center", margin: 0, isTextBox: true });
  s.addNotes(
    "False positives are the failure that matters in free living, so the threshold should be set " +
    "for them. The model outputs a probability; where we draw the line decides the trade-off. Best " +
    "F1 treats a false alarm and a miss as equally bad. The other two options favour precision: " +
    "F0.5, or a hard budget of false alarms per hour. The curve shows, for any budget, how much " +
    "eating we still catch. Next step: train with the budget chosen on validation participants, " +
    "so the reported number is not tuned on the test set. Negative examples from other ETS " +
    "studies are the other lever, and they attack false positives directly.");
}

// ------------------------------------------------------------------ 6 ----
{
  const s = pres.addSlide();
  title(s, "Where the Errors Are");
  s.addChart(pres.charts.BAR, [
    { name: "Default", labels: ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
      values: [0.896, 0.904, 0.842, 0.851, 0.772] },
    { name: "Learning rate 0.0003", labels: ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
      values: [0.898, 0.898, 0.829, 0.855, 0.817] },
  ], { x: 0.5, y: 1.0, w: 5.4, h: 4.2, barDir: "col", barGrouping: "clustered", barGapWidthPct: 40,
       chartColors: [LIGHT, DARK], valAxisMinVal: 0.6, valAxisMaxVal: 1.0,
       valAxisLabelFormatCode: "0.00", showValue: true, dataLabelPosition: "outEnd",
       dataLabelFormatCode: "0.00", dataLabelFontSize: 11, dataLabelFontFace: FONT,
       catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, catAxisLabelFontFace: FONT,
       valAxisLabelFontFace: FONT, catAxisLabelColor: INK, valAxisLabelColor: MUTED,
       valGridLine: { color: "E0E0E0", size: 0.5 }, catGridLine: { style: "none" },
       showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: FONT,
       showTitle: true, title: "F1 by fold", titleFontSize: 11, titleFontFace: FONT, titleColor: INK });
  bullets(s, [
    "Same folds every run, so the same people are always the hardest",
    "Fold 5 is long lunches with little eating (20-31% of the session): recall drops",
    "Three participants are lowest in every run: AIM132097, AIM122571, AIM147560",
    "Spread between people is larger than any design change so far",
  ], { x: 6.2, y: 1.1, w: 3.3, h: 4.1 });
  s.addNotes(
    "The folds are fixed on purpose, so runs are directly comparable. The weakest fold is mostly " +
    "long lunches where eating is a small part of the session, and the model misses eating there. " +
    "Three people are at the bottom in every run; I'll check their sensor data next, which also " +
    "feeds the data-quality assessment. The main message: differences between people are bigger " +
    "than differences between model choices.");
}

// ------------------------------------------------------------------ 7 ----
{
  const s = pres.addSlide();
  title(s, "Action Items from Last Meeting");
  table(s, [
    ["Action item (9/23)", "Status"],
    ["Connect the model to the literature review", "Done: every component mapped to a paper; review written"],
    ["Explain the model, graphical representation", "Done: architecture figure and full design table"],
    ["Leave-one-subject-out validation", "Built in as the default; 43-fold run in progress"],
    ["Different sampling rates (e.g. 32 Hz)", "Implemented (128 / 64 / 32 / 16 Hz); next run"],
    ["Detection window 4 s / 16 s", "Implemented (2-16 s sweep); next run"],
    ["Weight of chew count vs detection", "Implemented (0 to 1.0 sweep); next run"],
    ["Look into transformers", "Implemented as an option; to run on the full dataset"],
    ["Add negative examples from other ETS studies", "Needs the data uploaded; to plan together"],
  ], { x: 0.5, y: 1.05, w: 9, colW: [4.0, 5.0], leftHeader: true });
  s.addNotes(
    "Every item from last week is either done, implemented and queued, or waiting on data. The " +
    "sampling-rate, window and chew-weight experiments are one flag each now, so they can all run " +
    "this week. The negative-examples idea is the one I need to plan with Siavash, since those " +
    "recordings need to go into the same database first.");
}

pres.writeFile({ fileName: "/home/user/Research_Directory/docs/presentation/lab_meeting_new_slides.pptx" })
  .then((f) => console.log("wrote", f));

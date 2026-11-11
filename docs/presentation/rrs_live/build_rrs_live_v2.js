// RRS Live deck, second pass: the layout of the microelectronics deck
// (centred bold title, short bold bullets, figures with small captions,
// outline slide, footer bar), in 16:9 with UA crimson instead of navy.
// Run from the repo root with NODE_PATH pointing at pptxgenjs:
//   node docs/presentation/rrs_live/build_rrs_live_v2.js
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/root/.claude/skills/synced/36909922-a146-4648-a906-6f07f81474b4_bd9816ec-0994-463a-bbd4-819843ae18dc/pptx/scripts/apply_theme.js");

const HERE = __dirname;
const FIG = path.join(HERE, "..", "..", "paper", "figures");
const OUT = path.join(HERE, "rrs_live_v2.pptx");
const DATE = "12/02/2026";          // 11/12/2026 for the practice run
const W = 13.333, H = 7.5;

const THEME = {
  name: "RRS Live v2", headFontFace: "Calibri", bodyFontFace: "Calibri",
  colors: { dk1: "000000", lt1: "FFFFFF", dk2: "595959", lt2: "F2F2F2",
            accent1: "9E1B32", accent2: "7F7F7F", accent3: "D9D9D9", accent4: "404040",
            accent5: "BFBFBF", accent6: "262626", hlink: "9E1B32", folHlink: "6E1323" },
};

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";
  pres.author = "Caelan Dunlea";
  pres.title = "RRS Live";
  pres.theme = { headFontFace: "Calibri", bodyFontFace: "Calibri" };
  const C = pres.SchemeColor;
  const LOGO = path.join(HERE, "rrsp_logo.png");            // 2000 x 229

  const bar = [
    { rect: { x: 0, y: H - 0.42, w: W, h: 0.42, fill: { color: C.accent1 } } },
    { text: { text: `Caelan Dunlea | Randall Research Scholars Program | ${DATE}`,
      options: { x: 0, y: H - 0.42, w: W, h: 0.42, fontSize: 12, bold: true,
                 color: C.background1, align: "center", valign: "middle" } } },
  ];
  pres.defineSlideMaster({
    title: "CONTENT", background: { color: C.background1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.25, w: W - 1, h: 0.8,
        fontSize: 30, bold: true, color: C.text1, align: "center", valign: "middle" }, text: "" } },
      ...bar,
    ],
    slideNumber: { x: W - 0.8, y: H - 0.39, w: 0.5, h: 0.36, fontSize: 11, bold: true,
                   color: C.background1, align: "right" },
  });
  pres.defineSlideMaster({ title: "PLAIN", background: { color: C.background1 }, objects: bar });

  const slide = (title) => {
    const s = pres.addSlide({ masterName: "CONTENT" });
    s.addText(title, { placeholder: "title" });
    return s;
  };
  // Short bold bullets, as in the microelectronics deck.
  const bullets = (s, items, opts = {}) => s.addText(
    items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
    { x: opts.x ?? 0.8, y: opts.y ?? 1.15, w: opts.w ?? W - 1.6, h: opts.h ?? 0.45 * items.length + 0.2,
      fontSize: opts.size ?? 22, bold: true, color: C.text1, paraSpaceAfter: 6, valign: "top",
      margin: 0, isTextBox: true });
  const caption = (s, text, x, y, w) => s.addText(text, { x, y, w, h: 0.35, fontSize: 13,
    bold: true, color: C.text2, align: "center", margin: 0, isTextBox: true });
  const picture = (s, file, x, y, w, ratio, alt) => s.addImage(
    { path: path.join(FIG, file), x, y, w, h: w / ratio, altText: alt });
  const todo = (s, x, y, w, h, label) => {
    s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: C.background2 },
      line: { color: C.accent5, width: 1, dashType: "dash" } });
    s.addText(label, { x, y, w, h, fontSize: 14, italic: true, color: C.text2, align: "center",
      valign: "middle", isTextBox: true });
  };

  // 1. Title ------------------------------------------------------------------
  let s = pres.addSlide({ masterName: "PLAIN" });
  s.addImage({ path: LOGO, x: (W - 6) / 2, y: 0.45, w: 6, h: 6 * 229 / 2000 });
  s.addText("Detecting Eating with a Wearable Sensor and Deep Learning",
    { x: (W - 9.6) / 2, y: 1.75, w: 9.6, h: 1.5, fontSize: 36, bold: true, color: C.accent1, align: "center",
      valign: "middle", isTextBox: true });
  s.addText([
    { text: "Caelan Dunlea | [Major]", options: { fontSize: 24, bold: true, breakLine: true } },
    { text: " ", options: { fontSize: 10, breakLine: true } },
    { text: "Project Director: Dr. Edward Sazonov", options: { fontSize: 18, bold: true, breakLine: true } },
    { text: "Department of Electrical and Computer Engineering", options: { fontSize: 18, bold: true, breakLine: true } },
    { text: " ", options: { fontSize: 10, breakLine: true } },
    { text: "The University of Alabama", options: { fontSize: 18, bold: true, italic: true, breakLine: true } },
    { text: "December 2nd, 2026", options: { fontSize: 16, bold: true } },
  ], { x: 1, y: 3.5, w: W - 2, h: 3.4, color: C.text1, align: "center", valign: "top", isTextBox: true });
  s.addNotes("~15 s. Name, major, project director, one sentence on the project.");

  // 2. Outline ------------------------------------------------------------------
  s = slide("Research Presentation Outline");
  bullets(s, ["Motivation", "AIM-2 Sensor & Data", "Model", "Testing on New People", "Results",
              "Where It Struggles", "Human Annotator Agreement", "Chewing Rate",
              "Conclusions & Future Work"], { x: 4.2, w: 5, size: 22 });
  s.addNotes("~10 s. Don't read the list; one sentence on where the talk goes.");

  // 3. Motivation ---------------------------------------------------------------
  s = slide("Motivation");
  bullets(s, ["Food diaries rely on memory, and intake is often under-reported [1]",
              "Wearable sensors can record eating automatically",
              "Goal: detect eating from the AIM-2's signals in people the model has never seen"]);
  todo(s, 3.4, 2.75, 6.5, 3.9, "[Photo: AIM-2 worn on glasses]\n[cite the source if not your own]");
  s.addNotes("~45 s. Why objective measurement of eating matters; what the goal is.");

  // 4. Sensor and data ----------------------------------------------------------
  s = slide("AIM-2 Sensor & Data");
  bullets(s, ["Optical jaw-muscle sensor and 3-axis accelerometer, 128 Hz",
              "43 participants, 61 supervised lab meals",
              "Video annotated at 10 Hz: chews, chewing bouts, bites"]);
  todo(s, 0.8, 2.75, 11.7, 3.5, "[Sensor vs annotation plot from the 10/21 deck, participant ID cropped out]");
  caption(s, "Sensor signals (top) and annotated eating (bottom)", 0.8, 6.3, 11.7);
  s.addNotes("~45 s. The device, the study, and that the video annotations are the answer key.");

  // 5. Model --------------------------------------------------------------------
  s = slide("Model");
  bullets(s, ["Input: raw 8 s windows of all four signals",
              "Compact CNN with attention, 47,282 parameters",
              "Output: probability of eating for each window",
              "Built in TensorFlow/Keras"], { w: 6.6, h: 3 });
  picture(s, "fig_architecture.png", 8.1, 1.1, 4.9, 1734 / 1880,
          "Model architecture: convolutional blocks, channel attention, attention pooling, eating and chew-count heads");
  caption(s, "Model architecture", 8.1, 6.45, 4.9);
  s.addNotes("~50 s. Cut into 8 s pieces, network scores each piece, threshold gives eating or not. " +
             "Point at the three conv blocks and the attention parts; don't read every box.");

  // 6. Testing on new people ----------------------------------------------------
  s = slide("Testing on New People");
  bullets(s, ["Leave-one-subject-out: train on the others, test on one person",
              "Repeated for all 43 participants, with 3 random seeds",
              "The model is never graded on someone it learned from"]);
  const n = 43, sq = 0.22, gap = 0.04, sx = (W - (n * sq + (n - 1) * gap)) / 2;
  for (let i = 0; i < n; i++) {
    const role = i === 21 ? C.accent1 : (i >= 2 && i < 7 ? C.accent2 : C.accent3);
    s.addShape(pres.shapes.RECTANGLE, { x: sx + i * (sq + gap), y: 4.0, w: sq, h: sq,
      fill: { color: role }, line: { type: "none" } });
  }
  caption(s, "Test: 1 participant", sx + 21 * (sq + gap) - 1.0, 3.5, 2.2);
  caption(s, "Validation: 5 (stopping and threshold)", sx + 2 * (sq + gap) - 0.6, 4.35, 3.4);
  caption(s, "Training: the other 37", 7.6, 4.35, 3.6);
  s.addNotes("~45 s. Each fold: 1 test person, 5 for deciding when to stop and the threshold, 37 to train on.");

  // 7. Results ------------------------------------------------------------------
  s = slide("Results");
  bullets(s, ["F1 0.86 (precision 0.849, recall 0.870)",
              "Every meal with eating was detected; 91.9% of eating episodes found"]);
  picture(s, "fig_example.png", 0.5, 2.4, 12.3, 2600 / 634,
          "Annotated eating and model probability over time for one lunch session");
  caption(s, "Example meal: gray = annotated eating, line = model probability, dashed = threshold",
          0.5, 5.5, 12.3);
  s.addNotes("~60 s. F1 balances misses and false alarms. Point at the example: the line rises when " +
             "the gray (real eating) starts. Per participant the mean is 0.844, above the 81.8% " +
             "reported for the earlier AIM-2 method [2].");

  // 8. Where it struggles ---------------------------------------------------------
  s = slide("Where It Struggles");
  bullets(s, ["Per-participant F1: mean 0.844 ± 0.101, median 0.863",
              "Lowest scores: soft food (oatmeal), and chewing broken up by talking"]);
  picture(s, "fig_participants_anon.png", 1.15, 2.25, 11, 2000 / 680,
          "Bar chart of F1 for each of the 43 held-out participants");
  caption(s, "F1 for each held-out participant (P1 to P43)", 1.15, 6.05, 11);
  s.addNotes("~55 s. Most people above 0.8. Oatmeal barely moves the jaw; one person chewed in " +
             "4 s bursts between talking, shorter than a window.");

  // 9. Human agreement ------------------------------------------------------------
  s = slide("Human Annotator Agreement");
  bullets(s, ["Two annotators labeled the same 3 meals independently",
              "They agreed at F1 0.81 to 0.93; the model scores 0.86",
              "Most disagreement is at the start and end of chewing bouts"]);
  const hdr = { bold: true, color: C.background1, fill: { color: C.accent1 }, align: "center" };
  const cell = { color: C.text1, align: "center" };
  s.addTable([
    [{ text: "Meal", options: hdr }, { text: "Agreement (F1, 8 s windows)", options: hdr }],
    [{ text: "Snack", options: cell }, { text: "0.91", options: cell }],
    [{ text: "Breakfast", options: cell }, { text: "0.81", options: cell }],
    [{ text: "Snack", options: cell }, { text: "0.93", options: cell }],
  ], { x: 3.4, y: 3.2, w: 6.5, colW: [2.5, 4.0], fontSize: 18, fontFace: "Calibri",
       border: { type: "solid", pt: 1, color: "BFBFBF" }, rowH: 0.5 });
  s.addNotes("~45 s. The answer key isn't perfect either. Ignoring a second either side of each bout " +
             "edge, agreement rose to 0.97 to 0.99. So the model is in the range of two people.");

  // 10. Chewing rate --------------------------------------------------------------
  s = slide("Chewing Rate");
  bullets(s, ["The model's chew output gave nearly the same rate for every meal",
              "Instead: count the jaw sensor's chewing peaks (0.94 to 2.17 Hz band)",
              "Median error 7% per meal, against 10% for a constant rate"]);
  todo(s, 3.0, 2.75, 7.3, 3.5, "[figures\\chew_rate_check.png from the lab machine]");
  caption(s, "Estimated against annotated chewing rate, per meal", 3.0, 6.3, 7.3);
  s.addNotes("~45 s. No training: band-pass the optical signal to the natural chewing band and count peaks.");

  // 11. Conclusions ---------------------------------------------------------------
  s = slide("Conclusions & Future Work");
  bullets(s, ["Detected eating in new people at F1 0.86, close to human agreement",
              "Model output saved to the study database for all 61 recordings",
              "Chewing rate measured directly from the jaw sensor",
              "Paper in progress for IEEE SoutheastCon 2027",
              "Future work: more double-annotated meals, shorter windows for brief chewing"],
          { h: 3.6 });
  s.addNotes("~45 s. Answer the research question, then what's next.");

  // 12. References ----------------------------------------------------------------
  s = slide("References");
  s.addText([
    { text: "[1] B. M. Bell et al., “Automatic, wearable-based, in-field eating detection approaches for public health research: A scoping review,” npj Digit. Med., vol. 3, Art. no. 38, 2020.", options: { breakLine: true } },
    { text: "[2] A. Doulah, T. Ghosh, D. Hossain, M. H. Imtiaz, and E. Sazonov, “‘Automatic Ingestion Monitor Version 2’ – A novel wearable device for automatic food intake detection and passive capture of food images,” IEEE J. Biomed. Health Inform., vol. 25, no. 2, pp. 568–576, 2021.", options: { breakLine: true } },
    { text: "[3] J. M. C. Po et al., “Time-frequency analysis of chewing activity in the natural environment,” J. Dent. Res., vol. 90, no. 10, pp. 1206–1210, 2011.", options: { breakLine: true } },
    { text: "[4] [Source of the AIM-2 photo, if not your own]" },
  ], { x: 0.8, y: 1.2, w: W - 1.6, h: 5, fontSize: 16, color: C.text1, paraSpaceAfter: 12,
       valign: "top", margin: 0, isTextBox: true });
  s.addNotes("Not presented.");

  // 13. Questions -----------------------------------------------------------------
  s = pres.addSlide({ masterName: "PLAIN" });
  s.addText([
    { text: "Thank you!", options: { fontSize: 36, bold: true, breakLine: true } },
    { text: "Questions?", options: { fontSize: 32, bold: true, italic: true } },
  ], { x: 1, y: 1.4, w: W - 2, h: 2, color: C.accent1, align: "center", valign: "middle",
       isTextBox: true });
  s.addText([
    { text: "Caelan Dunlea: cadunlea@crimson.ua.edu", options: { breakLine: true } },
    { text: "Dr. Edward Sazonov: [email]", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Thanks to Siavash Esfandiari Fard and Kenzy Nguyen", options: { italic: true } },
  ], { x: 1, y: 3.8, w: W - 2, h: 2, fontSize: 18, bold: true, color: C.text1, align: "center",
       valign: "top", isTextBox: true });
  s.addNotes("Pre-loaded questions (print and hand out): 1) Why 8-second windows? 2) Why a jaw " +
             "sensor instead of a camera or microphone? 3) What causes most of the remaining errors? " +
             "4) How could this be used in daily life outside the study?");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote " + OUT);
})();

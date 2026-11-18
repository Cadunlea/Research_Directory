// RRS Live deck, third pass: the practice-run feedback applied to the v2 layout.
//   AIM2_PHOTO=path/to/aim2_photos.jpg node docs/presentation/rrs_live/build_rrs_live_v3.js
// The AIM-2 photo strip (from the CU Anschutz article, reference 4) is not in
// the repository; without AIM2_PHOTO a placeholder box is drawn instead.
// Figures still to be drawn on the lab machine are placeholder boxes that name
// the file to insert.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/root/.claude/skills/synced/36909922-a146-4648-a906-6f07f81474b4_bd9816ec-0994-463a-bbd4-819843ae18dc/pptx/scripts/apply_theme.js");

const HERE = __dirname;
const FIG = path.join(HERE, "..", "..", "paper", "figures");
const OUT = process.env.OUT || path.join(HERE, "rrs_live_v3.pptx");
const PHOTO = process.env.AIM2_PHOTO || "";
const DATE = "12/02/2026";
const W = 13.333, H = 7.5;
const THEME = {
  name: "RRS Live v3", headFontFace: "Calibri", bodyFontFace: "Calibri",
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
  const LOGO = path.join(HERE, "rrsp_logo.png");               // 2000 x 229

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
  // A bullet is a string or an array of parts; {sup: "1"} becomes a superscript
  // citation, written as a Unicode superscript digit so each bullet stays one run
  // (pptxgenjs splits mixed-format runs into separate bulleted paragraphs).
  const SUP = { 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  const flat = (item) => (Array.isArray(item) ? item : [item]).map((r) =>
    typeof r === "string" ? r : [...r.sup].map((d) => SUP[d] || d).join("")).join("");
  const bullets = (s, items, o = {}) => s.addText(
    items.map((item, i) => ({ text: flat(item),
      options: { bullet: true, breakLine: i < items.length - 1 } })),
    { x: o.x ?? 0.8, y: o.y ?? 1.15, w: o.w ?? W - 1.6,
      h: o.h ?? 0.48 * items.length + 0.2, fontSize: o.size ?? 22, bold: true, color: C.text1,
      paraSpaceAfter: 6, valign: "top", margin: 0, isTextBox: true });
  const caption = (s, text, x, y, w) => s.addText(text, { x, y, w, h: 0.35, fontSize: 14,
    bold: true, color: C.text2, align: "center", margin: 0, isTextBox: true });
  const footnote = (s, text) => s.addText(text, { x: 0.5, y: H - 0.85, w: W - 1, h: 0.38,
    fontSize: 10, color: C.text2, valign: "bottom", margin: 0, isTextBox: true });
  const picture = (s, file, x, y, w, ratio, alt) => s.addImage(
    { path: file, x, y, w, h: w / ratio, altText: alt });
  const todo = (s, x, y, w, h, label) => {
    s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: C.background2 },
      line: { color: C.accent5, width: 1, dashType: "dash" } });
    s.addText(label, { x, y, w, h, fontSize: 14, italic: true, color: C.text2, align: "center",
      valign: "middle", isTextBox: true });
  };
  const photo = (s, x, y, w) => (PHOTO && fs.existsSync(PHOTO))
    ? s.addImage({ path: PHOTO, x, y, w, h: w * 585 / 1920, altText: "AIM-2 on eyeglasses" })
    : todo(s, x, y, w, w * 585 / 1920, "[AIM-2 photos from slide 3 of the old deck]");

  // 1. Title: larger text and the device photo fill the page --------------------
  let s = pres.addSlide({ masterName: "PLAIN" });
  s.addImage({ path: LOGO, x: (W - 6) / 2, y: 0.35, w: 6, h: 6 * 229 / 2000 });
  s.addText("Deep Learning Food Intake Detection Method from the AIM-2 Wearable Sensor",
    { x: 1.2, y: 1.2, w: W - 2.4, h: 1.4, fontSize: 38, bold: true, color: C.accent1,
      align: "center", valign: "middle", isTextBox: true });
  s.addText([
    { text: "Caelan Dunlea | Electrical and Computer Engineering", options: { fontSize: 24, bold: true, breakLine: true } },
    { text: "Project Director: Dr. Edward Sazonov, Department of Electrical and Computer Engineering", options: { fontSize: 18, bold: true, breakLine: true } },
    { text: "The University of Alabama | December 2nd, 2026", options: { fontSize: 18, bold: true, italic: true } },
  ], { x: 0.8, y: 2.65, w: W - 1.6, h: 1.4, color: C.text1, align: "center", valign: "top",
       paraSpaceAfter: 4, isTextBox: true });
  photo(s, (W - 8.6) / 2, 4.15, 8.6);

  // 2. Outline as a diagram ----------------------------------------------------
  s = slide("Research Presentation Outline");
  const stages = [
    ["Background", ["Why Monitor Eating?", "Research Goal", "AIM-2 Sensor & Data"]],
    ["Approach", ["Model Structure", "Methods for Testing"]],
    ["Findings", ["Results", "Model Weaknesses", "Annotator Agreement", "Chewing Rate"]],
    ["Wrap-up", ["Key Findings", "Next Steps"]],
  ];
  const bw = 2.75, gap = 0.45, x0 = (W - (4 * bw + 3 * gap)) / 2, y0 = 1.9, bh = 3.6;
  stages.forEach(([name, items], i) => {
    const x = x0 + i * (bw + gap);
    s.addShape(pres.shapes.RECTANGLE, { x, y: y0, w: bw, h: 0.75, fill: { color: C.accent1 },
      line: { type: "none" } });
    s.addText(`${i + 1}. ${name}`, { x, y: y0, w: bw, h: 0.75, fontSize: 22, bold: true,
      color: C.background1, align: "center", valign: "middle", isTextBox: true });
    s.addShape(pres.shapes.RECTANGLE, { x, y: y0 + 0.75, w: bw, h: bh - 0.75,
      fill: { color: C.background2 }, line: { color: C.accent3, width: 1 } });
    s.addText(items.map((t, k) => ({ text: t, options: { breakLine: k < items.length - 1 } })),
      { x: x + 0.15, y: y0 + 1.0, w: bw - 0.3, h: bh - 1.2, fontSize: 18, bold: true,
        color: C.text1, align: "center", valign: "top", paraSpaceAfter: 14, margin: 0,
        isTextBox: true });
    if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + bw + 0.08, y: y0 + 0.2,
      w: gap - 0.16, h: 0.35, fill: { color: C.accent2 }, line: { type: "none" } });
  });

  // 3. Background --------------------------------------------------------------
  s = slide("Why Monitor Eating?");
  bullets(s, [
    ["Eating habits matter for health, nutrition and obesity research", { sup: "1" }],
    ["Food diaries rely on memory, and intake is often under-reported", { sup: "1" }],
    ["Wearable sensors can record eating automatically, with no effort from the wearer", { sup: "1" }],
    ["The AIM-2, built in Dr. Sazonov's lab, clips onto ordinary glasses", { sup: "2" }],
  ]);
  photo(s, (W - 8.2) / 2, 3.35, 8.2);
  caption(s, "The AIM-2 worn on eyeglasses⁴", (W - 8.2) / 2, 3.35 + 8.2 * 585 / 1920 + 0.05, 8.2);
  footnote(s, "¹ Bell et al., NPJ Digit. Med., 2020   ² Doulah et al., IEEE J. Biomed. Health Inform., 2021   ⁴ Shaw, CU Anschutz Dept. of Medicine News, 2024");

  // 4. Research goal -----------------------------------------------------------
  s = slide("Research Goal");
  s.addText("Automatically detect eating from the AIM-2's signals, in people the model has never seen",
    { x: 1.2, y: 1.35, w: W - 2.4, h: 1.3, fontSize: 28, bold: true, color: C.accent1,
      align: "center", valign: "middle", isTextBox: true });
  bullets(s, [
    "Why: objective eating data for nutrition studies, without food diaries",
    "Built: a deep learning model that finds eating in every 8 s of sensor data",
    "Added: chewing rate measured from the jaw sensor",
    "Delivered: results for every recording saved to the lab's database",
  ], { x: 1.6, y: 3.15, w: W - 3.2, h: 2.8 });

  // 5. Sensor and data: bigger figures, fewer words ----------------------------
  s = slide("AIM-2 Sensor & Data");
  bullets(s, [
    "Optical jaw-muscle sensor and 3-axis accelerometer, 128 samples per second",
    "43 participants, 61 supervised lab meals, each video annotated at 10 Hz",
    "SQL database stores the raw sensor signals and the annotations",
  ], { size: 20 });
  todo(s, 0.6, 2.65, 5.9, 3.85, "[figures\\sensor_vs_annotations_..._sensors.png]");
  todo(s, 6.85, 2.65, 5.9, 3.85, "[figures\\sensor_vs_annotations_..._annotations.png]");
  caption(s, "Sensor signals (shaded: eating)", 0.6, 6.55, 5.9);
  caption(s, "Annotations from the video", 6.85, 6.55, 5.9);

  // 6. Model -------------------------------------------------------------------
  s = slide("Model Structure");
  bullets(s, [
    "Compact convolutional neural network (CNN) with attention, 47,282 parameters",
    "Input: raw 8 s windows of all four signals",
    "Output: probability of eating for each window",
  ]);
  picture(s, path.join(HERE, "fig_architecture_wide.png"), 0.5, 2.85, 12.3, 3690 / 1125,
          "Model architecture, left to right");

  // 7. Testing -----------------------------------------------------------------
  s = slide("Methods for Testing");
  bullets(s, [
    "Leave-one-subject-out: train on the others, test on one person",
    "Repeated for all 43 participants, with 3 random seeds",
    "The model is never tested on someone it learned from",
    "The 5 validation participants change every fold",
  ]);
  const n = 43, sq = 0.22, sg = 0.04, sx = (W - (n * sq + (n - 1) * sg)) / 2;
  const fold = (y, test, val, label) => {
    for (let i = 0; i < n; i++) {
      const color = i === test ? C.accent1 : (val.includes(i) ? C.accent2 : C.accent3);
      s.addShape(pres.shapes.RECTANGLE, { x: sx + i * (sq + sg), y, w: sq, h: sq,
        fill: { color }, line: { type: "none" } });
    }
    caption(s, label, sx, y - 0.4, 6);
  };
  fold(3.95, 21, [2, 3, 9, 30, 37], "Fold 1");
  fold(5.15, 34, [5, 11, 12, 19, 26], "Fold 2");
  caption(s, "Red: test person   Dark gray: 5 validation   Light gray: 37 training   …and so on",
          sx, 5.55, n * (sq + sg));

  // 8. Results: F1 defined, bigger figure --------------------------------------
  s = slide("Results");
  bullets(s, [
    "F1 0.86 (precision 0.849, recall 0.870)",
    "F1 balances missed eating and false alarms; 1.0 is perfect",
    "Every meal with eating was detected; 91.9% of eating episodes found",
  ]);
  todo(s, 0.5, 2.7, 12.3, 3.75, "[figures\\sensor_vs_annotations_..._example.png]");
  caption(s, "Example meal: shaded = annotated eating, line = model probability, dashed = threshold",
          0.5, 6.55, 12.3);

  // 9. Weaknesses --------------------------------------------------------------
  s = slide("Model Weaknesses");
  bullets(s, [
    "Per-participant F1: mean 0.844 ± 0.101, median 0.863",
    "Lowest scores: soft food (oatmeal), and chewing broken up by talking",
  ]);
  picture(s, path.join(FIG, "fig_participants_anon.png"), 1.15, 2.25, 11, 2000 / 680,
          "F1 for each of the 43 held-out participants");
  caption(s, "F1 for each held-out participant (P1 to P43)", 1.15, 6.05, 11);

  // 10. Annotator agreement ----------------------------------------------------
  s = slide("Human Annotator Agreement");
  bullets(s, [
    "Two annotators labeled the same four meals independently",
    "They agreed at F1 0.81 to 0.93; the model scores 0.86",
    "Most disagreement is at the start and end of chewing bouts",
  ]);
  const hdr = { bold: true, color: C.background1, fill: { color: C.accent1 }, align: "center" };
  const cell = { color: C.text1, align: "center" };
  s.addTable([
    [{ text: "Meal", options: hdr }, { text: "Agreement (F1 score)", options: hdr }],
    ...[["Snack", "0.91"], ["Breakfast", "0.81"], ["Snack", "0.93"], ["Lunch", "0.89"]]
      .map(([a, b]) => [{ text: a, options: cell }, { text: b, options: cell }]),
  ], { x: 3.4, y: 3.1, w: 6.5, colW: [2.5, 4.0], fontSize: 20, fontFace: "Calibri",
       border: { type: "solid", pt: 1, color: "BFBFBF" }, rowH: 0.55 });

  // 11. Chewing rate: superscript citation, footnote ---------------------------
  s = slide("Chewing Rate");
  bullets(s, [
    "Each chew makes the jaw signal rise and fall once",
    ["Count the peaks in the chewing band (0.94 to 2.17 Hz)", { sup: "3" }],
    "Over the whole meal: r = 0.97, median error 13% per meal",
  ]);
  picture(s, path.join(HERE, "chew_rate_slide_trend.png"), 1.9, 2.55, 9.5, 3150 / 1380,
          "Estimated against annotated chewing rate, per meal");
  footnote(s, "³ Po et al., J. Dent. Res., 2011");

  // 12. Key findings -----------------------------------------------------------
  s = slide("Key Findings");
  bullets(s, [
    "Detects eating in new people: F1 0.86",
    "In the range of two human annotators (0.81 to 0.93)",
    "Chewing rate over the whole meal: r = 0.97",
    "Results for all 61 recordings saved to the lab database",
  ], { x: 1.8, y: 2.0, w: W - 3.6, h: 3.6, size: 28 });

  // 13. Next steps -------------------------------------------------------------
  s = slide("Next Steps");
  bullets(s, [
    "Paper for IEEE SoutheastCon 2027",
    "More meals labeled by two annotators",
    "Shorter windows to catch brief chewing",
    "Test outside the lab, in everyday eating",
  ], { x: 1.8, y: 2.0, w: W - 3.6, h: 3.6, size: 28 });

  // 14. References (shortened) -------------------------------------------------
  s = slide("References");
  s.addText([
    { text: "[1] Bell BM, et al. Automatic, wearable-based, in-field eating detection approaches for public health research: a scoping review. NPJ Digit Med. 2020;3:38. doi: 10.1038/s41746-020-0246-2.", options: { breakLine: true } },
    { text: "[2] Doulah A, Ghosh T, Hossain D, Imtiaz MH, Sazonov E. \"Automatic Ingestion Monitor Version 2\" – A novel wearable device for automatic food intake detection and passive capture of food images. IEEE J Biomed Health Inform. 2021;25(2):568-576. doi: 10.1109/JBHI.2020.2995473.", options: { breakLine: true } },
    { text: "[3] Po JM, Kieser JA, Gallo LM, Tésenyi AJ, Herbison P, Farella M. Time-frequency analysis of chewing activity in the natural environment. J Dent Res. 2011;90(10):1206-1210. doi: 10.1177/0022034511416669.", options: { breakLine: true } },
    { text: "[4] Shaw T. Want to lose weight? An eyeglass device can help track your calories. CU Anschutz Dept. of Medicine News. Apr. 4, 2024." },
  ], { x: 0.8, y: 1.3, w: W - 1.6, h: 5, fontSize: 16, color: C.text1, paraSpaceAfter: 14,
       valign: "top", margin: 0, isTextBox: true });

  // 15. Questions --------------------------------------------------------------
  s = pres.addSlide({ masterName: "PLAIN" });
  s.addText([
    { text: "Thank you!", options: { fontSize: 36, bold: true, breakLine: true } },
    { text: "Questions?", options: { fontSize: 32, bold: true, italic: true } },
  ], { x: 1, y: 1.4, w: W - 2, h: 2, color: C.accent1, align: "center", valign: "middle",
       isTextBox: true });
  s.addText([
    { text: "Caelan Dunlea: cadunlea@crimson.ua.edu", options: { breakLine: true } },
    { text: "Dr. Edward Sazonov: esazonov@eng.ua.edu", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Thanks to Dr. Sazonov, Siavash Esfandiari Fard and Kenzy Nguyen", options: { italic: true } },
  ], { x: 1, y: 3.8, w: W - 2, h: 2, fontSize: 18, bold: true, color: C.text1, align: "center",
       valign: "top", isTextBox: true });

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote " + OUT);
})();

// RRS Live (single timeslot, 8-9 min) first-pass draft.
// Plain white, Arial, UA crimson used sparingly. Bracketed text = still to fill in.
// Run from the repo root:  node docs/presentation/rrs_live/build_rrs_live.js
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const { FaBook, FaGlasses } = require("react-icons/fa");
const { applyTheme } = require("/root/.claude/skills/synced/36909922-a146-4648-a906-6f07f81474b4_bd9816ec-0994-463a-bbd4-819843ae18dc/pptx/scripts/apply_theme.js");

const HERE = __dirname;
const FIG = path.join(HERE, "..", "..", "paper", "figures");
const OUT = path.join(HERE, "rrs_live_draft.pptx");
const DATE = "12/02/2026";   // change to 11/12/2026 for the practice run

const THEME = {
  name: "RRS Live",
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  colors: {
    dk1: "222222", lt1: "FFFFFF", dk2: "666666", lt2: "F2F2F2",
    accent1: "9E1B32", accent2: "777777", accent3: "BBBBBB", accent4: "444444",
    accent5: "D9D9D9", accent6: "5A5A5A", hlink: "9E1B32", folHlink: "6E1323",
  },
};
const HEX = THEME.colors;

async function icon(Component, color) {
  const svg = ReactDOMServer.renderToStaticMarkup(
    React.createElement(Component, { color: "#" + color, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";            // 13.333 x 7.5 in, 16:9
  pres.author = "Caelan Dunlea";
  pres.title = "RRS Live draft";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;
  const W = 13.333;
  const FOOTER = `Caelan Dunlea | Randall Research Scholars Program | ${DATE}`;
  const LOGO = path.join(HERE, "rrsp_logo.png");     // 2000 x 229
  const footer = [
    { text: { text: FOOTER, options: { x: 0.6, y: 6.95, w: 9, h: 0.35, fontSize: 11,
      color: C.text2, margin: 0 } } },
  ];
  const number = { x: 12.2, y: 6.95, w: 0.6, h: 0.35, fontSize: 11, color: C.text2, align: "right" };

  pres.defineSlideMaster({
    title: "RRS_TITLE",
    background: { color: C.background1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.2, w: W - 1.6, h: 1.8,
        fontSize: 40, bold: true, color: C.text1, align: "center", valign: "middle" }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 1.5, y: 3.2, w: W - 3, h: 2.3,
        fontSize: 18, color: C.text1, align: "center", valign: "top" }, text: "" } },
      { image: { path: LOGO, x: (W - 5.6) / 2, y: 5.85, w: 5.6, h: 5.6 * 229 / 2000 } },
    ],
  });
  pres.defineSlideMaster({
    title: "RRS_CONTENT",
    background: { color: C.background1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: W - 1.2, h: 0.9,
        fontSize: 34, bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 },
        text: "" } },
      ...footer,
    ],
    slideNumber: number,
  });
  pres.defineSlideMaster({
    title: "RRS_STATEMENT",
    background: { color: C.background1 },
    objects: [
      { placeholder: { options: { name: "kicker", type: "body", x: 0.6, y: 1.3, w: W - 1.2, h: 0.6,
        fontSize: 20, color: C.accent1, align: "center" }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 1.4, y: 2.2, w: W - 2.8, h: 3,
        fontSize: 36, color: C.text1, align: "center", valign: "middle" }, text: "" } },
      ...footer,
    ],
    slideNumber: number,
  });

  const content = (title, section) => {
    const s = pres.addSlide({ masterName: "RRS_CONTENT", sectionTitle: section });
    s.addText(title, { placeholder: "title" });
    return s;
  };
  // Gray box marking where a picture still has to go.
  const todoBox = (s, x, y, w, h, label) => {
    s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: C.background2 },
      line: { color: C.accent3, width: 1, dashType: "dash" }, objectName: "Picture to add" });
    s.addText(label, { x, y, w, h, fontSize: 14, italic: true, color: C.text2, align: "center",
      valign: "middle", isTextBox: true });
  };
  const card = (s, x, y, w, h, name) => s.addShape(pres.shapes.ROUNDED_RECTANGLE,
    { x, y, w, h, rectRadius: 0.08, fill: { color: C.background2 }, line: { type: "none" },
      objectName: name });
  const stat = (s, x, y, w, big, label) => {
    s.addText(big, { x, y, w, h: 1.0, fontSize: 54, bold: true, color: C.accent1, align: "left",
      valign: "bottom", margin: 0, isTextBox: true });
    s.addText(label, { x, y: y + 1.05, w, h: 0.6, fontSize: 16, color: C.text1, align: "left",
      valign: "top", margin: 0, isTextBox: true });
  };

  // 1. Title ---------------------------------------------------------------
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "RRS_TITLE", sectionTitle: "Introduction" });
  s.addText("Detecting Eating with a Wearable Sensor and Deep Learning", { placeholder: "title" });
  s.addText([
    { text: "Caelan Dunlea | [Major] | cadunlea@crimson.ua.edu", options: { breakLine: true } },
    { text: "Randall Research Scholars Program", options: { breakLine: true } },
    { text: "Project Director: Dr. Edward Sazonov", options: { breakLine: true } },
    { text: "Department of Electrical and Computer Engineering" },
  ], { placeholder: "body", paraSpaceAfter: 6 });
  s.addNotes("~15 s. Name, major, project director. One sentence: I built a model that tells " +
    "when someone is eating from a sensor on their glasses.");

  // 2. Why -----------------------------------------------------------------
  s = content("Why Measure Eating?", "Introduction");
  const book = await icon(FaBook, HEX.accent2), glasses = await icon(FaGlasses, HEX.accent1);
  const cols = [
    { x: 0.6, img: book, head: "Food diaries",
      lines: ["Depend on memory and honesty", "Intake is often under-reported [1]"] },
    { x: 6.85, img: glasses, head: "Wearable sensors",
      lines: ["Record eating automatically", "Measure when, how long and how fast someone eats"] },
  ];
  for (const c of cols) {
    card(s, c.x, 1.6, 5.9, 3.9, c.head);
    s.addImage({ data: c.img, x: c.x + 0.5, y: 2.0, w: 0.9, h: 0.9 });
    s.addText(c.head, { x: c.x + 0.5, y: 3.05, w: 5, h: 0.6, fontSize: 24, bold: true,
      color: C.text1, margin: 0, isTextBox: true });
    s.addText(c.lines.map((t, i) => ({ text: t, options: { bullet: true,
      breakLine: i < c.lines.length - 1 } })), { x: c.x + 0.5, y: 3.75, w: 5.1, h: 1.5,
      fontSize: 18, color: C.text1, paraSpaceAfter: 8, margin: 0, valign: "top", isTextBox: true });
  }
  s.addText("Goal: detect eating automatically, so diet can be studied without relying on memory",
    { x: 0.6, y: 5.85, w: W - 1.2, h: 0.6, fontSize: 18, italic: true, color: C.text2, margin: 0,
      isTextBox: true });
  s.addNotes("~50 s. Most diet research still relies on people writing down what they ate. " +
    "People forget and under-report. A wearable sensor can record eating on its own. " +
    "[Use the self-report citation from the paper intro for [1].]");

  // 3. Device --------------------------------------------------------------
  s = content("The AIM-2 Sensor", "Introduction");
  todoBox(s, 0.6, 1.6, 6.0, 4.9, "[Photo of the AIM-2 on glasses]\n[cite source if not your own photo]");
  const parts = [
    ["Optical sensor", "Sees the jaw muscle move during chewing"],
    ["Accelerometer", "Head motion in three directions"],
    ["On ordinary glasses", "128 readings per second, all day"],
  ];
  parts.forEach(([h, t], i) => {
    const y = 1.8 + i * 1.55;
    s.addText(h, { x: 7.2, y, w: 5.5, h: 0.5, fontSize: 22, bold: true, color: C.text1, margin: 0,
      isTextBox: true });
    s.addText(t, { x: 7.2, y: y + 0.5, w: 5.5, h: 0.6, fontSize: 18, color: C.text1, margin: 0,
      isTextBox: true });
  });
  s.addNotes("~40 s. Automatic Ingestion Monitor, version 2, developed in Dr. Sazonov's lab [2]. " +
    "Two signals the model uses: the jaw-muscle optical sensor and the accelerometer.");

  // 4. Data ----------------------------------------------------------------
  pres.addSection({ title: "Methods" });
  s = content("The Data", "Methods");
  [["43", "participants"], ["61", "meals recorded"], ["3", "meal types: breakfast,\nlunch, snack"]]
    .forEach(([b, l], i) => stat(s, 0.6 + i * 4.15, 1.4, 3.8, b, l));
  s.addText("Every meal was video recorded and annotated by hand: each chewing bout and bite, " +
    "to a tenth of a second", { x: 0.6, y: 3.25, w: W - 1.2, h: 0.6, fontSize: 18, color: C.text1,
    margin: 0, isTextBox: true });
  todoBox(s, 0.6, 4.0, W - 1.2, 2.65, "[Sensor vs annotation plot from the 10/21 deck, " +
    "participant ID cropped out of the title line]");
  s.addNotes("~40 s. Eating Trajectory Study. The hand annotations are the answer key the model " +
    "learns from and is graded against.");

  // 5. Research question ---------------------------------------------------
  s = pres.addSlide({ masterName: "RRS_STATEMENT", sectionTitle: "Methods" });
  s.addText("Research Question", { placeholder: "kicker" });
  s.addText("Can a small neural network detect eating from raw sensor signals in people it has " +
    "never seen?", { placeholder: "body" });
  s.addNotes("~15 s. Read it, then: 'never seen' is the important part, and the next two slides " +
    "show how I tested that. (Cut this slide if the talk runs long.)");

  // 6. Model ---------------------------------------------------------------
  s = content("How the Model Works", "Methods");
  const steps = [
    ["8 seconds of sensor data", "jaw muscle + head motion"],
    ["Neural network", "learns the patterns of chewing"],
    ["Eating or not", "a score from 0 to 1"],
    ["Summary per meal", "eating episodes, chews, chewing rate"],
  ];
  const bw = 2.6, gap = 0.55, x0 = (W - (4 * bw + 3 * gap)) / 2;
  steps.forEach(([h, t], i) => {
    const x = x0 + i * (bw + gap);
    card(s, x, 2.3, bw, 2.2, h);
    s.addText(h, { x: x + 0.2, y: 2.5, w: bw - 0.4, h: 0.9, fontSize: 20, bold: true,
      color: i === 2 ? C.accent1 : C.text1, align: "center", valign: "middle", margin: 0,
      isTextBox: true });
    s.addText(t, { x: x + 0.2, y: 3.4, w: bw - 0.4, h: 0.9, fontSize: 16, color: C.text1,
      align: "center", valign: "top", margin: 0, isTextBox: true });
    if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + bw + 0.1, y: 3.2, w: gap - 0.2, h: 0.4,
      fill: { color: C.accent2 }, line: { type: "none" } });
  });
  s.addText("A compact convolutional neural network (about 47,000 parameters) that reads the raw " +
    "signal, with no hand-made features", { x: 0.6, y: 5.1, w: W - 1.2, h: 0.8, fontSize: 18,
    color: C.text2, align: "left", margin: 0, isTextBox: true });
  s.addNotes("~50 s. The recording is cut into 8-second pieces. The network looks at each piece " +
    "and says how likely it is that the person is eating. Pieces above a threshold are joined " +
    "into eating episodes. Keep it simple; the full architecture is in the paper.");

  // 7. Testing -------------------------------------------------------------
  s = content("Testing on New People", "Methods");
  const n = 43, sq = 0.24, sg = 0.045, sx = (W - (n * sq + (n - 1) * sg)) / 2;
  for (let i = 0; i < n; i++) {
    s.addShape(pres.shapes.RECTANGLE, { x: sx + i * (sq + sg), y: 2.5, w: sq, h: sq,
      fill: { color: i === 20 ? C.accent1 : C.accent5 }, line: { type: "none" } });
  }
  s.addText("Train on 42 people", { x: sx, y: 2.9, w: 5, h: 0.4, fontSize: 16, color: C.text2,
    margin: 0, isTextBox: true });
  s.addText("Test on 1", { x: sx + 20 * (sq + sg) - 0.6, y: 2.0, w: 1.5, h: 0.4, fontSize: 16,
    bold: true, color: C.accent1, align: "center", margin: 0, isTextBox: true });
  s.addText([
    { text: "Repeated 43 times, so every person is the test person once", options: { bullet: true, breakLine: true } },
    { text: "The model is never graded on someone it learned from", options: { bullet: true, breakLine: true } },
    { text: "This estimates how it would do for a brand-new wearer", options: { bullet: true } },
  ], { x: 0.6, y: 3.9, w: W - 1.2, h: 2.2, fontSize: 22, color: C.text1, paraSpaceAfter: 12,
    margin: 0, valign: "top", isTextBox: true });
  s.addNotes("~45 s. Leave-one-subject-out cross-validation. If the same person were in training " +
    "and testing, the score would look better than it really is for new people.");

  // 8. Results -------------------------------------------------------------
  pres.addSection({ title: "Results" });
  s = content("Results", "Results");
  stat(s, 0.6, 1.25, 3.8, "0.86", "F1 score (1.0 = perfect)");
  stat(s, 4.75, 1.25, 3.8, "100%", "of meals detected");
  stat(s, 8.9, 1.25, 3.8, "91.9%", "of eating episodes found");
  s.addImage({ path: path.join(FIG, "fig_example.png"), x: 0.6, y: 3.15, w: W - 1.2,
    h: (W - 1.2) * 634 / 2600 });
  s.addText("Example meal. Gray: annotated eating. Line: model output. Dashed: decision threshold.",
    { x: 0.6, y: 6.2, w: W - 1.2, h: 0.4, fontSize: 12, color: C.text2, margin: 0, isTextBox: true });
  s.addNotes("~60 s. F1 balances missed eating against false alarms. Every meal was detected, and " +
    "every eating episode of a minute or longer. Per participant the mean F1 is 0.844, above the " +
    "81.8% reported for the earlier AIM-2 method [2]. Point at the example: the line jumps up " +
    "when the gray (real eating) starts.");

  // 9. Struggles -----------------------------------------------------------
  s = content("Where It Struggles", "Results");
  s.addImage({ path: path.join(FIG, "fig_participants_anon.png"), x: 0.6, y: 1.3, w: 7.6,
    h: 7.6 * 680 / 2000 });
  s.addText("F1 for each held-out participant", { x: 0.6, y: 3.95, w: 7.6, h: 0.4, fontSize: 12,
    color: C.text2, margin: 0, isTextBox: true });
  const why = [
    ["Soft food", "Oatmeal barely moves the jaw muscle, so the optical signal stays flat"],
    ["Talking between bites", "4 s of chewing, 3 s of talking, repeated: shorter than one 8 s window"],
  ];
  why.forEach(([h, t], i) => {
    const y = 1.3 + i * 1.6;
    card(s, 8.6, y, 4.15, 1.4, h);
    s.addText(h, { x: 8.85, y: y + 0.15, w: 3.7, h: 0.45, fontSize: 18, bold: true, color: C.text1,
      margin: 0, isTextBox: true });
    s.addText(t, { x: 8.85, y: y + 0.6, w: 3.7, h: 0.75, fontSize: 14, color: C.text1, margin: 0,
      valign: "top", isTextBox: true });
  });
  todoBox(s, 0.6, 4.55, W - 1.2, 2.1, "[Optional: cropped sensor plot of the soft-food meal, " +
    "labelled Participant A, no ID]");
  s.addNotes("~60 s. Most people score above 0.8. I looked at the lowest ones by hand: one ate " +
    "oatmeal, which hardly moves the jaw; another chewed in short bursts between talking. Short " +
    "meals also swing the score, since a few missed windows matter more.");

  // 10. Human agreement ----------------------------------------------------
  s = content("How Good Can It Get?", "Results");
  card(s, 0.6, 1.6, 5.9, 2.9, "Human agreement");
  card(s, 6.85, 1.6, 5.9, 2.9, "Model agreement");
  s.addText("0.81 to 0.93", { x: 1.0, y: 1.9, w: 5.1, h: 1.1, fontSize: 48, bold: true,
    color: C.text1, margin: 0, isTextBox: true });
  s.addText("Two human annotators labelling the same 3 meals", { x: 1.0, y: 3.1, w: 5.1, h: 1.2,
    fontSize: 18, color: C.text1, margin: 0, valign: "top", isTextBox: true });
  s.addText("0.86", { x: 7.25, y: 1.9, w: 5.1, h: 1.1, fontSize: 48, bold: true, color: C.accent1,
    margin: 0, isTextBox: true });
  s.addText("Model against the annotations, all 61 meals, people it never trained on",
    { x: 7.25, y: 3.1, w: 5.1, h: 1.2, fontSize: 18, color: C.text1, margin: 0, valign: "top",
      isTextBox: true });
  s.addText("Most human disagreement is about exactly when a chewing bout starts and stops",
    { x: 0.6, y: 4.9, w: W - 1.2, h: 0.6, fontSize: 18, italic: true, color: C.text2, margin: 0,
      isTextBox: true });
  s.addNotes("~45 s. The answer key itself isn't perfect. When Kenzy and I annotated the same meals " +
    "we agreed at F1 0.81 to 0.93, and almost all of the disagreement was at the edges of bouts. " +
    "So the model is in the same range as two people. [Update the range if more meals are compared.]");

  // 11. Conclusions --------------------------------------------------------
  pres.addSection({ title: "Conclusion" });
  s = content("Conclusions and Next Steps", "Conclusion");
  const lists = [
    { x: 0.6, head: "Done", items: ["Detects eating in new people (F1 0.86)",
      "Runs on every recording in the study database",
      "Chewing rate from the jaw sensor: 7% median error"] },
    { x: 6.85, head: "Next", items: ["Paper for IEEE SoutheastCon 2027",
      "More meals annotated by two people",
      "Store the results in the study database"] },
  ];
  for (const l of lists) {
    card(s, l.x, 1.6, 5.9, 3.7, l.head);
    s.addText(l.head, { x: l.x + 0.5, y: 1.9, w: 5, h: 0.6, fontSize: 26, bold: true,
      color: l.head === "Done" ? C.text1 : C.accent1, margin: 0, isTextBox: true });
    s.addText(l.items.map((t, i) => ({ text: t, options: { bullet: true,
      breakLine: i < l.items.length - 1 } })), { x: l.x + 0.5, y: 2.7, w: 5.1, h: 3.2,
      fontSize: 18, color: C.text1, paraSpaceAfter: 14, margin: 0, valign: "top", isTextBox: true });
  }
  s.addNotes("~45 s. Answer the research question: yes, a small network can detect eating in new " +
    "people, about as well as two people agree with each other. Then the next steps.");

  // 12. References ---------------------------------------------------------
  s = content("References", "Conclusion");
  s.addText([
    { text: "[1] [Self-report under-reporting reference from the paper introduction]",
      options: { breakLine: true } },
    { text: "[2] A. Doulah, T. Ghosh, D. Hossain, M. H. Imtiaz, and E. Sazonov, “‘Automatic " +
      "Ingestion Monitor Version 2’ – A novel wearable device for automatic food intake " +
      "detection and passive capture of food images,” IEEE J. Biomed. Health Inform., vol. 25, " +
      "no. 2, pp. 568–576, 2021.", options: { breakLine: true } },
    { text: "[3] [Image source for the AIM-2 photo, if not your own]" },
  ], { x: 0.6, y: 1.5, w: W - 1.2, h: 4.8, fontSize: 14, color: C.text1, paraSpaceAfter: 12,
    margin: 0, valign: "top", isTextBox: true });
  s.addNotes("Not presented; skip past it. Check [2] against the paper's reference list.");

  // 13. Questions ----------------------------------------------------------
  s = content("Questions?", "Conclusion");
  s.addText([
    { text: "Caelan Dunlea", options: { bold: true, breakLine: true } },
    { text: "cadunlea@crimson.ua.edu", options: { breakLine: true } },
    { text: " ", options: { breakLine: true } },
    { text: "Dr. Edward Sazonov", options: { bold: true, breakLine: true } },
    { text: "[email]" },
  ], { x: 0.6, y: 1.7, w: 6, h: 2.8, fontSize: 20, color: C.text1, margin: 0, valign: "top",
    isTextBox: true });
  s.addText([
    { text: "Thank you", options: { bold: true, breakLine: true } },
    { text: "Siavash Esfandiari Fard", options: { breakLine: true } },
    { text: "Kenzy Nguyen (co-annotator)" },
  ], { x: 7.0, y: 1.7, w: 5.7, h: 2.0, fontSize: 20, color: C.text1, margin: 0, valign: "top",
    isTextBox: true });
  s.addImage({ path: LOGO, x: 0.6, y: 5.6, w: 5.6, h: 5.6 * 229 / 2000 });
  s.addNotes("Pre-loaded questions (print and hand out): " +
    "1) Why use 8-second windows? " +
    "2) Why a jaw sensor instead of a camera or microphone? " +
    "3) What causes most of the remaining errors? " +
    "4) How could this be used in daily life outside the study?");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote " + OUT);
})();

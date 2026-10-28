const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/root/.claude/skills/synced/36909922-a146-4648-a906-6f07f81474b4_bd9816ec-0994-463a-bbd4-819843ae18dc/pptx/scripts/apply_theme.js");

// Grayscale, Arial, plain white, matching the 10/21 deck.
const THEME = {
  name: "Lab Grayscale",
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  colors: {
    dk1: "000000", lt1: "FFFFFF", dk2: "595959", lt2: "F2F2F2",
    accent1: "404040", accent2: "7F7F7F", accent3: "A6A6A6",
    accent4: "BFBFBF", accent5: "262626", accent6: "D9D9D9",
    hlink: "404040", folHlink: "7F7F7F",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.45, y: 0.25, w: 9.1, h: 0.7,
      fontSize: 28, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 9.3, y: 5.2, w: 0.5, h: 0.3, fontSize: 14, color: C.text2, align: "right" },
});

const BULLET = 16;
function bullets(slide, items, opts) {
  const size = opts.fontSize || BULLET;
  const runs = items.map((it, i) => {
    const sub = typeof it === "object" && it.sub;
    const text = typeof it === "object" ? it.text : it;
    return { text, options: { bullet: sub ? { indent: 18 } : true, indentLevel: sub ? 1 : 0,
      fontSize: sub ? size - 2 : size, color: C.text2,
      paraSpaceAfter: 8, breakLine: i < items.length - 1 } };
  });
  slide.addText(runs, Object.assign({ isTextBox: true, valign: "top", margin: 0.05 }, opts));
}

function add(section, title) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}

const HEAD = { bold: true, fill: { color: "F2F2F2" }, color: "000000" };
function table(slide, header, rows, opts, colW) {
  const data = [header.map((t) => ({ text: t, options: HEAD }))]
    .concat(rows.map((r) => r.map((t) => (typeof t === "object" ? t : { text: String(t) }))));
  slide.addTable(data, Object.assign({ fontSize: 13, color: "262626", border: { type: "solid", pt: 0.75, color: "A6A6A6" },
    margin: [3, 6, 3, 6], colW, valign: "middle" }, opts));
}

function stat(slide, x, y, value, label, w = 1.8) {
  slide.addText(value, { isTextBox: true, x, y, w, h: 0.75, fontSize: 40, bold: true, color: "000000", margin: 0 });
  slide.addText(label, { isTextBox: true, x, y: y + 0.75, w, h: 0.5, fontSize: 12, color: C.text2, margin: 0, valign: "top" });
}

// 1 ---------------------------------------------------------------------
pres.addSection({ title: "Update" });
let s = add("Update", "Project Updates");
s.addImage({ path: "contract.png", x: 0.45, y: 1.1, w: 5.2, h: 5.2 * 1180 / 1532, altText: "Randall Research Scholars Program project contract, activities 1 to 7" });
bullets(s, [
  "Model development is complete (Activity 6, due today)",
  "Trained the final model: 38 participants for training, 5 held out to pick the stopping point and threshold (0.75)",
  "Inference now runs on the whole database in one command",
  "Activity 7 (due 11/04): only writing results back to the database is left",
], { x: 5.95, y: 1.15, w: 3.65, h: 3.9 });
s.addNotes("Activity 6 closes today. The final model is the validated setup trained once on all annotated participants, with 5 held out for early stopping and the threshold.");

// 2 ---------------------------------------------------------------------
s = add("Update", "Inference Demo: AIM146271 Lunch");
s.addImage({ path: "a146271_crop.png", x: 0.45, y: 1.0, w: 9.1, h: 9.1 * 634 / 2600, altText: "AIM146271 lunch: annotated eating and the final model's probability over time" });
stat(s, 0.45, 3.45, "0.886", "F1, against my annotation");
stat(s, 2.35, 3.45, "0.925", "Precision");
stat(s, 4.25, 3.45, "0.849", "Recall");
bullets(s, [
  "Held out from training (only used to pick the threshold)",
  "Misses are short bouts and pauses (around 13:58); one false spike after the meal",
], { x: 6.2, y: 3.45, w: 3.4, h: 1.7 });
s.addNotes("1.7 h recording, 7 eating episodes, 9.3 min of eating detected; 153 windows skipped for missing sensor data. Annotations were read straight from the database. Shaded: annotated eating. Dashed: threshold 0.75.");

// 3 ---------------------------------------------------------------------
s = add("Update", "Whole-Database Inference");
table(s, ["Held-out participant", "Meal", "F1"], [
  ["AIM109345", "Lunch", "0.792"], ["AIM109345", "Snack", "0.864"],
  ["AIM116725", "Lunch", "0.823"], ["AIM116725", "Snack", "0.865"],
  ["AIM128555", "Breakfast", "0.844"], ["AIM128555", "Snack", "0.915"],
  ["AIM146271", "Lunch", "0.886"], ["AIM147560", "Breakfast", "0.667"],
  [{ text: "Median", options: { bold: true } }, "", { text: "0.854", options: { bold: true } }],
], { x: 0.45, y: 1.45, w: 4.6, rowH: 0.36 }, [2.1, 1.4, 1.1]);
s.addText("Scored: only the 8 sessions the model never trained on (the other 53 would be inflated)", {
  isTextBox: true, x: 0.45, y: 1.0, w: 4.6, h: 0.4, fontSize: 11, color: C.text2, margin: 0, valign: "top" });
bullets(s, [
  "One command ran the model on all 61 recordings in the database (about 1 s each)",
  "Saves every eating episode (start, end, duration, chews) and one row per meal",
  "Can be stopped and restarted; finished recordings are skipped",
  "Participants the final model never saw: median F1 0.854, close to the leave-one-out 0.862",
], { x: 5.4, y: 1.15, w: 4.2, h: 3.9 });
s.addNotes("The 0.854 is on the 5 participants held out of the final model's training. They did choose the threshold, so it is slightly optimistic, but it matches cross-validation.");

// 4 ---------------------------------------------------------------------
pres.addSection({ title: "Chewing" });
s = add("Chewing", "Chews per Meal");
s.addImage({ path: "chew_check.png", x: 0.45, y: 1.0, w: 6.2, h: 6.2 * 919 / 2000, altText: "Model versus annotated chews per meal and chewing rate" });
stat(s, 6.95, 1.05, "r = 0.92", "Chews per meal, held-out meals", 2.7);
stat(s, 6.95, 2.35, "r = −0.23", "Chewing rate, held-out meals", 2.7);
bullets(s, [
  "Chew counts match my annotations: median error 14%, slight undercount",
  "Chewing rate does not: the model gives about 70–100 chews/min to everyone, real rates are 75–130",
  "So the count is right mostly because eating time is right",
], { x: 0.45, y: 4.0, w: 9.1, h: 1.2 });
s.addNotes("The chew head was an auxiliary training target, never tuned on its own. Filled dots are the 8 held-out meals; open dots are the 53 meals of participants the model trained on (counts r = 0.97, median error 10%; rate r = 0.42), which are optimistic.");

// 5 ---------------------------------------------------------------------
s = add("Chewing", "Idea: Chewing Rate from the Optical Signal");
s.addImage({ path: "optical.png", x: 0.45, y: 1.05, w: 5.3, h: 5.3 * 700 / 1000, altText: "Optical sensor spectrum, eating versus not eating" });
s.addText("Solid: eating   Dashed: not eating   Shaded: chewing band (0.94–2.17 Hz)", { isTextBox: true,
  x: 0.45, y: 4.85, w: 5.3, h: 0.3, fontSize: 10, color: C.text2, margin: 0 });
bullets(s, [
  "During eating the optical (jaw) signal peaks at about 1.7 Hz, inside the chewing band",
  "1.7 Hz is about 100 chews/min, right in the annotated range",
  "Idea: read chewing rate from the peak frequency in each eating window",
  "Can be checked against the chew annotations I already have",
], { x: 6.05, y: 1.15, w: 3.55, h: 3.9 });
s.addNotes("Average spectra over all 8 s windows. The accelerometers barely differ in the chewing band, which is why the optical channel matters most for chewing.");

// 6 ---------------------------------------------------------------------
pres.addSection({ title: "Analysis" });
s = add("Analysis", "Performance by Meal Type");
s.addImage({ path: "meal_types.png", x: 0.8, y: 0.95, w: 8.4, h: 8.4 * 880 / 2200, altText: "F1 per session by meal type, and lunch versus the same participant's other meal" });
bullets(s, [
  "Lunches score lower: median F1 0.815 vs 0.890 for other meals (p = 0.009); the drop is mostly missed eating (recall 0.82)",
  "Some lunches were solo, so this is not yet a group-eating result; asked Siavash where session setting is recorded",
], { x: 0.8, y: 4.45, w: 8.4, h: 0.85, fontSize: 14 });
s.addNotes("Right panel: 9 participants with a lunch and another meal; 7 of 9 lower at lunch, median difference -0.081, Wilcoxon p = 0.10 (too few people to be significant). The two that improved had their weakest session at the other meal. Snack shows 22 sessions: AIM125068's snack has no annotated eating, so no F1. Pooled: breakfast 0.882, dinner 0.884, lunch 0.829, snack 0.877.");

// 7 ---------------------------------------------------------------------
s = add("Analysis", "Next Steps");
const cols = [
  ["Done", ["Final model trained and validated", "Whole-database inference", "Chews per meal + check"]],
  ["Waiting on", ["Which table and format to write results to", "Where group vs solo is recorded", "Is food type recorded?"]],
  ["Next", ["Chewing rate from the optical peak", "Annotate the 4 worst participants", "Paper for SoutheastCon, Methods first (12/02)"]],
];
cols.forEach(([head, items], i) => {
  const x = 0.45 + i * 3.1;
  s.addShape(pres.ShapeType.rect, { x, y: 1.15, w: 2.9, h: 3.0, fill: { color: "F2F2F2" }, line: { color: "F2F2F2" }, objectName: `${head} card` });
  s.addText(head, { isTextBox: true, x: x + 0.2, y: 1.3, w: 2.5, h: 0.45, fontSize: 18, bold: true, color: "000000", margin: 0 });
  bullets(s, items, { x: x + 0.15, y: 1.9, w: 2.6, h: 3.0 });
});

// Backup ------------------------------------------------------------------
pres.addSection({ title: "Backup" });
s = add("Backup", "Backup: All 61 Meals");
s.addImage({ path: "meals_full.png", x: 0.8, y: 0.95, w: 8.4, h: 8.4 * 576 / 1110.24, altText: "Eating time, chews and chewing rate per meal for all 61 meals" });
s.addNotes("Bars cover the whole recording; circles cover only the annotated stretch. Long recordings (AIM134628 snack, 12.5 h) show more detected than annotated eating; some of that is false alarms over many hours, some may be unannotated eating.");

s = add("Backup", "Backup: Time Domain vs FFT Input");
s.addImage({ path: "operating_curve.png", x: 0.45, y: 1.05, w: 5.7, h: 5.7 * 1050 / 1625, altText: "Recall versus false alarms per hour, time domain versus FFT" });
bullets(s, [
  "Same model, two inputs: raw signal (time domain) vs frequency spectrum (FFT)",
  "Time domain finds more eating at every false-alarm rate",
  "At the chosen threshold: about 41 vs 84 false alarms per hour at similar recall",
], { x: 6.45, y: 1.15, w: 3.15, h: 3.9 });

(async () => {
  await pres.writeFile({ fileName: "lab_meeting_10_28.pptx" });
  await applyTheme("lab_meeting_10_28.pptx", THEME);
  console.log("written");
})();

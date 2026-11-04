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
  "Activity 7: inference done except writing results back to the database",
  "Activity 8: all metrics, tables and figures collected in one results document",
  "Activity 9: paper set up in the IEEE template; abstract and introduction drafted",
], { x: 5.95, y: 1.15, w: 3.65, h: 3.9 });

// 2 ---------------------------------------------------------------------
pres.addSection({ title: "Paper" });
s = add("Paper", "Paper Progress");
table(s, ["Section", "Status"], [
  ["Abstract", "Drafted"],
  ["I. Introduction", "Background and prior work drafted; gap paragraph next"],
  ["II. Methods", "Next: data, windows, model, training, chewing rate"],
  ["III. Results", "Tables and figures chosen (3 tables, 3 to 4 figures)"],
  ["IV. Discussion, V. Conclusion", "Not started"],
], { x: 0.45, y: 1.1, w: 5.6, rowH: 0.5 }, [2.0, 3.6]);
bullets(s, [
  "Target: IEEE SoutheastCon 2027, 6-page full paper",
  "Every number traced to its run in the results document",
  "Before submission: recompute edge errors and per-participant mean and SD on the final model",
], { x: 6.35, y: 1.15, w: 3.25, h: 3.9 });

// 3 ---------------------------------------------------------------------
pres.addSection({ title: "Results" });
s = add("Results", "Chewing Rate from the Optical Signal");
table(s, ["Method", "Meals", "r", "Median error"], [
  ["Same rate for every meal (baseline)", "60", "–", "10%"],
  ["Model's chew output (held out)", "8", "−0.23", "20%"],
  ["Optical peak count, annotated windows", "59", "0.72", "7%"],
  ["Optical peak count, model's detected windows", "60", "0.39 / 0.75", "7%"],
], { x: 0.45, y: 1.1, w: 5.9, rowH: 0.55 }, [3.2, 0.8, 0.9, 1.0]);
bullets(s, [
  "Chewing is a rhythm the jaw sensor sees directly",
  "Filter to the chewing band, count peaks: no training",
  "Beats a constant rate; the model's chew output does not",
  "Now part of the inference script",
], { x: 6.65, y: 1.15, w: 2.95, h: 3.9 });
s.addNotes("Detected windows: r 0.39 on the 8 held-out meals, 0.75 on the 52 trained-on meals; median error 7% in both. The rate method has no training, so only which windows were detected differs.");

// 4 ---------------------------------------------------------------------
s = add("Results", "Literature Check and Questions");
bullets(s, [
  "Farooq and Sazonov 2016: chew count and chewing rate from a piezoelectric jaw sensor (15% chew-count error)",
  "Chew counting annotation agreement measured before (intra-class correlation 0.988, 3 raters)",
  "Likely new in this paper: agreement on eating windows (F1 0.91), where errors occur, rate from the AIM-2 optical rhythm (still checking the AIM-2 papers)",
], { x: 0.45, y: 1.15, w: 5.0, h: 3.9 });
s.addShape(pres.ShapeType.rect, { x: 5.8, y: 1.15, w: 3.75, h: 3.0, fill: { color: "F2F2F2" }, line: { color: "F2F2F2" }, objectName: "questions card" });
s.addText("Questions", { isTextBox: true, x: 6.0, y: 1.3, w: 3.4, h: 0.45, fontSize: 18, bold: true, color: "000000", margin: 0 });
bullets(s, [
  "Author list and order",
  "ETS funding and grant number",
  "Jaw sensor: optical or flex?",
  "SoutheastCon deadline",
  "Mention the Randall program?",
], { x: 5.95, y: 1.9, w: 3.5, h: 2.6 });

(async () => {
  await pres.writeFile({ fileName: "lab_meeting_update.pptx" });
  await applyTheme("lab_meeting_update.pptx", THEME);
  console.log("written");
})();

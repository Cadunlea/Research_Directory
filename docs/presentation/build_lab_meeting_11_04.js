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
  "Activity 7 (due today): inference runs on the whole database; only writing results back is left",
  "Activity 8 (metrics, 11/11): window, episode, meal-type and chew results are in",
  "Activity 9 (paper, 12/02): first full draft written",
], { x: 5.95, y: 1.15, w: 3.65, h: 3.9 });
s.addNotes("Activity 7 is complete apart from the database write, which is waiting on the table and format.");

// 2 ---------------------------------------------------------------------
pres.addSection({ title: "Paper" });
s = add("Paper", "Paper Draft v0.1");
s.addImage({ path: "paper_p1.png", x: 0.45, y: 1.0, w: 3.15, h: 3.15 * 11 / 8.5, altText: "First page of the paper draft" });
s.addShape(pres.ShapeType.rect, { x: 0.45, y: 1.0, w: 3.15, h: 3.15 * 11 / 8.5, fill: { type: "none" }, line: { color: "A6A6A6", width: 0.75 }, objectName: "page border" });
bullets(s, [
  "Target: IEEE SoutheastCon",
  "Full draft: abstract, introduction, methods, results, discussion, conclusion",
  "6 figures, 4 tables, 12 references from my literature review",
  "Every number comes from the leave-one-out runs; open items are marked in gray",
  "Next: move it into the IEEE template and cut to the page limit",
], { x: 3.95, y: 1.15, w: 5.65, h: 3.9 });

// 3 ---------------------------------------------------------------------
s = add("Paper", "What the Paper Shows");
const cards = [
  ["0.86", "F1, leave-one-subject-out on 43 participants; two annotators agree at 0.91"],
  ["91.9%", "Of eating episodes detected; every episode of a minute or more, 0 s median delay"],
  ["0.854 vs 0.791", "Raw signal beats frequency input, with half the false alarms"],
  ["62%", "Of errors sit at the edges of chewing bouts; lunches are detected worse"],
];
cards.forEach(([value, label], i) => {
  const x = 0.45 + (i % 2) * 4.65, y = 1.05 + Math.floor(i / 2) * 2.05;
  s.addShape(pres.ShapeType.rect, { x, y, w: 4.45, h: 1.85, fill: { color: "F2F2F2" }, line: { color: "F2F2F2" }, objectName: `card ${i + 1}` });
  s.addText(value, { isTextBox: true, x: x + 0.25, y: y + 0.2, w: 4.0, h: 0.75, fontSize: 36, bold: true, color: "000000", margin: 0 });
  s.addText(label, { isTextBox: true, x: x + 0.25, y: y + 1.0, w: 4.0, h: 0.7, fontSize: 14, color: C.text2, margin: 0, valign: "top" });
});
s.addNotes("These are the four contributions in the introduction. The 62% edge figure is from one seed of the reference configuration and is marked to be recomputed on the final configuration.");

// 4 ---------------------------------------------------------------------
pres.addSection({ title: "Plan" });
s = add("Plan", "Plan to 12/02 and Questions");
table(s, ["Date", "Step"], [
  ["11/04", "Activity 7: inference done except database write"],
  ["11/11", "Activity 8: recompute the gray items on the final model"],
  ["11/18", "Paper in the IEEE template, cut to the page limit"],
  ["11/25", "Draft to Dr. Sazonov for review"],
  ["12/02", "Activity 9: revised draft"],
], { x: 0.45, y: 1.1, w: 4.9, rowH: 0.42 }, [0.9, 4.0]);
bullets(s, [
  "Is SoutheastCon the right venue, and what is its deadline?",
  "Author list and order (Kenzy for annotation?)",
  "Paper scope: detection only, or also chews and meal type?",
  "Database write: which table and format?",
], { x: 5.7, y: 1.15, w: 3.9, h: 3.9 });

(async () => {
  await pres.writeFile({ fileName: "lab_meeting_11_04.pptx" });
  await applyTheme("lab_meeting_11_04.pptx", THEME);
  console.log("written");
})();

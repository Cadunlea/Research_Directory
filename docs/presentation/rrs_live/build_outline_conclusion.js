// Two replacement slides for the RRS Live deck, in its layout (16:9, centred
// bold title, crimson footer bar): the outline as a left-to-right diagram, and
// the conclusions in two columns. Copy them into rrs_live_v2.pptx.
//   node docs/presentation/rrs_live/build_outline_conclusion.js
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/root/.claude/skills/synced/36909922-a146-4648-a906-6f07f81474b4_bd9816ec-0994-463a-bbd4-819843ae18dc/pptx/scripts/apply_theme.js");

const OUT = path.join(__dirname, "outline_conclusion_slides.pptx");
const DATE = "12/02/2026";
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
  pres.theme = { headFontFace: "Calibri", bodyFontFace: "Calibri" };
  const C = pres.SchemeColor;
  pres.defineSlideMaster({
    title: "CONTENT", background: { color: C.background1 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.25, w: W - 1, h: 0.8,
        fontSize: 30, bold: true, color: C.text1, align: "center", valign: "middle" }, text: "" } },
      { rect: { x: 0, y: H - 0.42, w: W, h: 0.42, fill: { color: C.accent1 } } },
      { text: { text: `Caelan Dunlea | Randall Research Scholars Program | ${DATE}`,
        options: { x: 0, y: H - 0.42, w: W, h: 0.42, fontSize: 12, bold: true,
                   color: C.background1, align: "center", valign: "middle" } } },
    ],
    slideNumber: { x: W - 0.8, y: H - 0.39, w: 0.5, h: 0.36, fontSize: 11, bold: true,
                   color: C.background1, align: "right" },
  });

  // Outline as a diagram: four stages, left to right.
  let s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Research Presentation Outline", { placeholder: "title" });
  const stages = [
    ["Background", ["Motivation", "AIM-2 Sensor & Data"]],
    ["Approach", ["Model Structure", "Methods for Testing"]],
    ["Findings", ["Results", "Model Weaknesses", "Human Annotator Agreement", "Chewing Rate"]],
    ["Wrap-up", ["Conclusions & Future Work"]],
  ];
  const bw = 2.75, gap = 0.45, x0 = (W - (4 * bw + 3 * gap)) / 2, y0 = 2.0, bh = 3.4;
  stages.forEach(([name, items], i) => {
    const x = x0 + i * (bw + gap);
    s.addShape(pres.shapes.RECTANGLE, { x, y: y0, w: bw, h: 0.75, fill: { color: C.accent1 },
      line: { type: "none" }, objectName: `${name} header` });
    s.addText(`${i + 1}. ${name}`, { x, y: y0, w: bw, h: 0.75, fontSize: 22, bold: true,
      color: C.background1, align: "center", valign: "middle", isTextBox: true });
    s.addShape(pres.shapes.RECTANGLE, { x, y: y0 + 0.75, w: bw, h: bh - 0.75,
      fill: { color: C.background2 }, line: { color: C.accent3, width: 1 },
      objectName: `${name} topics` });
    s.addText(items.map((t, k) => ({ text: t, options: { breakLine: k < items.length - 1 } })),
      { x: x + 0.2, y: y0 + 1.0, w: bw - 0.4, h: bh - 1.2, fontSize: 18, bold: true,
        color: C.text1, align: "center", valign: "top", paraSpaceAfter: 14, isTextBox: true });
    if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + bw + 0.08, y: y0 + 0.2,
      w: gap - 0.16, h: 0.35, fill: { color: C.accent2 }, line: { type: "none" } });
  });

  // Conclusions in two columns.
  s = pres.addSlide({ masterName: "CONTENT" });
  s.addText("Conclusions & Future Work", { placeholder: "title" });
  const cols = [
    ["Key Findings", ["Detects eating in new people: F1 0.86",
                      "In the range of two human annotators (0.81 to 0.93)",
                      "Chewing rate from the jaw: r = 0.97",
                      "Results for all 61 recordings saved to the lab database"]],
    ["Next Steps", ["Paper for IEEE SoutheastCon 2027",
                    "More meals labeled by two annotators",
                    "Shorter windows for brief chewing",
                    "Test outside the lab, in everyday eating"]],
  ];
  const cw = 5.8, cg = 0.5, cx0 = (W - (2 * cw + cg)) / 2;
  cols.forEach(([head, items], i) => {
    const x = cx0 + i * (cw + cg);
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.7, w: cw, h: 0.7,
      fill: { color: i === 0 ? C.accent1 : C.accent4 }, line: { type: "none" },
      objectName: `${head} header` });
    s.addText(head, { x, y: 1.7, w: cw, h: 0.7, fontSize: 22, bold: true, color: C.background1,
      align: "center", valign: "middle", isTextBox: true });
    s.addShape(pres.shapes.RECTANGLE, { x, y: 2.4, w: cw, h: 3.5, fill: { color: C.background2 },
      line: { color: C.accent3, width: 1 }, objectName: `${head} items` });
    s.addText(items.map((t, k) => ({ text: t, options: { bullet: true,
      breakLine: k < items.length - 1 } })),
      { x: x + 0.35, y: 2.7, w: cw - 0.7, h: 3.1, fontSize: 20, bold: true, color: C.text1,
        valign: "top", paraSpaceAfter: 16, isTextBox: true });
  });

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote " + OUT);
})();

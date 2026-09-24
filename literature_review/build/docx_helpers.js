// Shared Word building blocks. Arial and the same steel-blue table header as
// the lab slides, so the documents and the deck read as one set.
const {
  Paragraph, TextRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, BorderStyle, ImageRun, Footer, PageNumber,
} = require('docx');

const FONT = 'Arial', MONO = 'Roboto Mono';
const HEAD = '9CBBCB', CODE = '38761D';
const W = 10080; // Letter, 1.1 in margins

function runs(spec, size, extra = {}) {
  const parts = Array.isArray(spec) ? spec : [spec];
  return parts.map(p => {
    if (typeof p === 'string') return new TextRun({ text: p, font: FONT, size, ...extra });
    if (p.b) return new TextRun({ text: p.b, bold: true, font: FONT, size, ...extra });
    if (p.i) return new TextRun({ text: p.i, italics: true, font: FONT, size, ...extra });
    if (p.code) return new TextRun({ text: p.code, font: MONO, size: size - 2, color: CODE });
    return new TextRun({ text: '', font: FONT, size });
  });
}
const P = (spec, o = {}) => new Paragraph({
  children: runs(spec, o.size || 21, o.run || {}),
  spacing: { before: o.before ?? 80, after: o.after ?? 120, line: o.line || 280 },
  alignment: o.align || AlignmentType.JUSTIFIED,
  indent: o.indent,
  keepNext: o.keepNext,
});
const H1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })] });
const H2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })] });
const Caption = t => new Paragraph({
  children: [new TextRun({ text: t, bold: true, font: FONT, size: 18 })],
  spacing: { before: 200, after: 80 }, keepNext: true,
});
const Note = t => new Paragraph({ children: [new TextRun({ text: t, italics: true, font: FONT, size: 16, color: '595959' })], spacing: { before: 60, after: 200 } });

const edge = { style: BorderStyle.SINGLE, size: 4, color: '000000' };
const borders = { top: edge, bottom: edge, left: edge, right: edge };

// codeCols: column indexes rendered as green monospace
function table(headers, rows, widths, { codeCols = [], size = 16, boldFirst = false } = {}) {
  const sum = widths.reduce((a, b) => a + b, 0);
  if (sum !== W) throw new Error(`table widths ${sum} != ${W} (${headers[0]})`);
  const cell = (text, w, { head = false, code = false, bold = false } = {}) => new TableCell({
    width: { size: w, type: WidthType.DXA }, borders,
    shading: head ? { fill: HEAD, type: ShadingType.CLEAR, color: 'auto' } : undefined,
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    children: [new Paragraph({
      spacing: { after: 0, line: 240 },
      children: [new TextRun({
        text: String(text), font: code ? MONO : FONT, size: code ? size - 2 : size,
        bold: head || bold, color: head ? 'FFFFFF' : code ? CODE : '000000',
      })],
    })],
  });
  return new Table({
    width: { size: W, type: WidthType.DXA }, columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, widths[i], { head: true })) }),
      ...rows.map(r => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, widths[i], { code: codeCols.includes(i), bold: boldFirst && i === 0 })) })),
    ],
  });
}

function image(data, widthIn, heightIn) {
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: 'png', data, transformation: { width: widthIn * 96, height: heightIn * 96 } })],
  });
}

const STYLES = {
  default: { document: { run: { font: FONT, size: 21 } } },
  paragraphStyles: [
    { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
      run: { size: 28, bold: true, font: FONT }, paragraph: { spacing: { before: 320, after: 120 }, outlineLevel: 0, keepNext: true } },
    { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
      run: { size: 23, bold: true, font: FONT }, paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1, keepNext: true } },
  ],
};
const PAGE = { size: { width: 12240, height: 15840 }, margin: { top: 1300, bottom: 1300, left: 1080, right: 1080 } };
const footer = () => ({ default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: '595959' })] })] }) });

function references(REFS, keys) {
  // keys: which references to list, in numeric order of the shared list
  const out = [];
  REFS.forEach(([k, r], i) => {
    if (keys && !keys.has(k)) return;
    out.push(new Paragraph({
      spacing: { after: 70, line: 250 }, indent: { left: 480, hanging: 480 },
      children: [new TextRun({ text: `[${i + 1}]  `, font: FONT, size: 17 }), new TextRun({ text: r, font: FONT, size: 17 })],
    }));
  });
  return out;
}

module.exports = { P, H1, H2, Caption, Note, table, image, runs, STYLES, PAGE, footer, references, W, FONT };

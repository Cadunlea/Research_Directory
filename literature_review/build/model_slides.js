// Model B slides in the style of the 4/22 and 5/20 lab decks: white slides,
// Arial, 25 pt black titles, steel-blue table headers, green monospace
// parameters, pastel flow boxes with thin coloured borders.
const pptxgen = require('pptxgenjs');
const { REFS, cite } = require('./refs');
const C = require('./model_content');

const OUT = process.argv[2] || 'Model_B_Slides.pptx';
const FONT = 'Arial', MONO = 'Roboto Mono';
const HEAD_FILL = '9CBBCB', CODE = '38761D', GREY = '595959';
const BOX = {
  data: { fill: 'D9EAD3', line: '38761D' },
  conv: { fill: 'CFE2F3', line: '3D85C6' },
  attn: { fill: 'D9D2E9', line: '351C75' },
  out: { fill: 'FCE5CD', line: 'B45F06' },
  aux: { fill: 'F3F3F3', line: '999999' },
};

const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9'; // 10 x 5.625 in, same as the Google Slides decks

function title(s, text) {
  s.addText(text, { x: 0.35, y: 0.2, w: 9.3, h: 0.6, fontFace: FONT, fontSize: 25, color: '000000', margin: 0, isTextBox: true, valign: 'middle' });
}
function note(s, text, y = 5.28) {
  s.addText(text, { x: 0.35, y, w: 9.3, h: 0.3, fontFace: FONT, fontSize: 9, italic: true, color: GREY, margin: 0, isTextBox: true });
}

// cells: plain strings, or {code: '...'} for green monospace parameters
function table(s, headers, rows, colW, { y = 0.95, fontSize = 9, rowH = 0.3, codeCols = [] } = {}) {
  const head = headers.map(h => ({ text: h, options: { bold: true, color: 'FFFFFF', fill: { color: HEAD_FILL }, fontSize: fontSize } }));
  const body = rows.map(r => r.map((c, j) => codeCols.includes(j)
    ? { text: c, options: { fontFace: MONO, fontSize: fontSize - 1, color: CODE } }
    : { text: c, options: {} }));
  s.addTable([head, ...body], {
    x: 0.35, y, w: colW.reduce((a, b) => a + b, 0), colW, rowH,
    fontFace: FONT, fontSize, color: '000000', valign: 'middle',
    border: { type: 'solid', pt: 0.5, color: '000000' }, margin: [0.03, 0.06, 0.03, 0.06],
  });
}

function box(s, x, y, w, h, kind, label, detail, dashed = false) {
  const st = BOX[kind];
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: st.fill }, line: { color: st.line, width: 1, dashType: dashed ? 'dash' : 'solid' } });
  s.addText([
    { text: label, options: { fontFace: FONT, fontSize: 10, bold: true, color: '000000', breakLine: true } },
    { text: detail, options: { fontFace: MONO, fontSize: 7.5, color: CODE } },
  ], { x: x + 0.04, y, w: w - 0.08, h, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
}
function arrow(s, x1, y1, x2, y2, head = true) {
  s.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001,
    flipH: x2 < x1, flipV: y2 < y1,
    line: { color: GREY, width: 1.25, endArrowType: head ? 'triangle' : 'none' },
  });
}

// ------------------------------------------------------------------ 1 diagram
{
  const s = pres.addSlide();
  title(s, 'Model B End-to-End Pipeline');
  const W = 1.62, H = 0.74, G = 0.3, X0 = 0.4;
  const col = i => X0 + i * (W + G);
  const R = [0.95, 2.1, 3.25];

  const rows = [
    [['data', 'ETS database', 'labels 10 Hz\nsensor 128 Hz'], ['data', '8 s windows', 'label: > 50% eating'],
     ['data', 'Quality filter', '>= 50% labelled\n<= 50% missing'], ['data', 'Z-score', 'per window,\nper channel'],
     ['data', 'Input', '(1024, 4)']],
    [['conv', 'Conv block 1', '32 x k9, stride 2\n> (256, 32)'], ['conv', 'Conv block 2', '64 x k9, stride 2\n> (64, 64)'],
     ['conv', 'Conv block 3', '64 x k5\n> (64, 64)'], ['attn', 'Channel attention', 'squeeze-excite\n> (64, 64)'],
     ['attn', 'Attention pooling', 'softmax over time\n> (64)']],
    [['attn', 'Dense + dropout', 'Dense 64, 0.3'], ['out', 'Food head', 'sigmoid > p(eating)'],
     ['out', 'Threshold', 'set on validation\nparticipants'], ['out', 'Output', 'eating / not eating']],
  ];
  rows.forEach((r, ri) => r.forEach(([k, a, b], ci) => {
    box(s, col(ci), R[ri], W, H, k, a, b);
    if (ci > 0) arrow(s, col(ci - 1) + W, R[ri] + H / 2, col(ci), R[ri] + H / 2);
  }));
  // row wraps: end of row -> start of next row
  for (const ri of [0, 1]) {
    const last = rows[ri].length - 1;
    const xs = col(last) + W / 2, ys = R[ri] + H, ymid = R[ri] + H + (R[ri + 1] - R[ri] - H) / 2;
    const xe = col(0) + W / 2;
    arrow(s, xs, ys, xs, ymid, false);
    arrow(s, xs, ymid, xe, ymid, false);
    arrow(s, xe, ymid, xe, R[ri + 1]);
  }
  // chew head branch, training only
  const cy = 4.38;
  box(s, col(0), cy, W, H, 'aux', 'Chew head', 'training only\ncount of chews', true);
  arrow(s, col(0) + W / 2, R[2] + H, col(0) + W / 2, cy);
  s.addText('Loss = BCE(food) + 0.2 x MSE(chews). The chew head is discarded after training.',
    { x: col(1), y: cy, w: 3 * W + 2 * G, h: H, fontFace: FONT, fontSize: 10, color: GREY, valign: 'middle', margin: 0, isTextBox: true });
  s.addNotes('Every box maps to a step in the summary tables that follow. Shapes are (time steps, channels). 49,395 trainable parameters with the chew head, 47,282 used at prediction time.');
}

// ------------------------------------------------------------------ 2-4 summary tables
const W4 = [0.45, 1.75, 2.75, 4.35];
function pipelineSlide(stageNames, t, noteText) {
  const s = pres.addSlide();
  title(s, t);
  const rows = [];
  C.PIPELINE.filter(g => stageNames.includes(g.stage)).forEach(g => g.rows.forEach(r => rows.push(r)));
  table(s, ['Step', 'Operation', 'Parameters', 'Description'], rows, W4, { codeCols: [2], fontSize: 8.5, rowH: 0.36 });
  if (noteText) note(s, noteText);
  return s;
}
pipelineSlide(['Data', 'Windows', 'Preprocessing'], 'Model B Summary Table (1/3): Data to Input');
pipelineSlide(['Network'], 'Model B Summary Table (2/3): Network',
  `Total parameters: ${C.TOTALS.withChew} during training, ${C.TOTALS.inference} at prediction time.`);
pipelineSlide(['Training', 'Evaluation'], 'Model B Summary Table (3/3): Training and Evaluation');

// ------------------------------------------------------------------ 5-6 precedents
const WP = [1.45, 2.45, 3.85, 1.55];
const half = Math.ceil(C.PRECEDENTS.length / 2);
[[0, half, '(1/2)'], [half, C.PRECEDENTS.length, '(2/2)']].forEach(([a, b, tag]) => {
  const s = pres.addSlide();
  title(s, `Every Design Choice Has a Precedent ${tag}`);
  table(s, ['Component', 'Model B', 'Precedent', 'Status'], C.PRECEDENTS.slice(a, b), WP, { fontSize: 8, rowH: 0.36 });
  note(s, 'Numbers in brackets refer to the reference slides at the end.');
});

// ------------------------------------------------------------------ 7 changes since 4/22
{
  const s = pres.addSlide();
  title(s, 'Changes Since the 4/22 Proposal');
  table(s, ['Component', '4/22 proposal', 'Model B now'], C.CHANGES, [2.2, 3.2, 3.9], { fontSize: 10, rowH: 0.36 });
  note(s, 'The core idea is unchanged: 1D convolutions, cross-channel attention, attention pooling and a sigmoid classifier.');
}

// ------------------------------------------------------------------ 8 false positives
{
  const s = pres.addSlide();
  title(s, 'Priority: Fewer False Positives');
  table(s, ['Evaluation', 'False positives', 'Non-eating windows', 'False positive rate', 'Precision'],
    C.FP_NOW, [2.2, 1.6, 1.9, 1.8, 1.8], { fontSize: 10, rowH: 0.3 });
  table(s, ['Approach', 'What it does', 'Precedent', 'Note'], C.FP_PLAN, [2.0, 3.6, 1.9, 1.8], { y: 2.05, fontSize: 8.5, rowH: 0.42 });
  note(s, 'Goal: a clearly lower false positive rate at roughly the same accuracy.');
}

// ------------------------------------------------------------------ 9 literature added
{
  const s = pres.addSlide();
  title(s, 'Literature Added');
  const added = [
    ['Stankoski et al. 2024', 'Optical sensors on smart glasses, ConvLSTM, F1 0.91, real-life precision 0.95', cite('stankoski2024')],
    ['Zhang and Amft 2018', 'EMG in eyeglasses over the temporalis, free-living chewing', cite('zhang2018')],
    ['Bedri et al. 2020', 'FitByte eyeglasses, multimodal fusion to cut false positives, F1 0.89', cite('bedri2020')],
    ['Kyritsis et al. 2021', 'End-to-end CNN and LSTM on raw smartwatch data', cite('kyritsis2021')],
    ['Farooq and Sazonov 2016', 'Automatic chew counting, basis for the chew head', cite('farooq2016')],
    ['Karim et al. 2019', 'Squeeze and excitation in time series classification', cite('karim2019')],
    ['Murahari and Plötz 2018', 'Attention over time for wearable activity recognition', cite('murahari2018')],
    ['Dehghani et al. 2019', 'Overlapping windows and subject-independent evaluation', cite('dehghani2019')],
    ['Saeb et al. 2017', 'Why splits must be by subject', cite('saeb2017')],
    ['Shavit and Klein 2021, Dirgová Luptáková et al. 2022', 'Transformers on wearable inertial data', cite('shavit2021', 'luptakova2022')],
    ['Zerveas et al. 2021, Haresamudram et al. 2020', 'Self-supervised pretraining for sensor time series', cite('zerveas2021', 'haresamudram2020')],
    ['Yamane et al. 2025', 'Lower sampling rates keep accuracy, supports the 32 Hz test', cite('yamane2025')],
  ];
  table(s, ['Paper', 'Why it matters here', 'Ref'], added, [3.0, 5.6, 0.7], { fontSize: 9, rowH: 0.33 });
}

// ------------------------------------------------------------------ 10-11 references
const perSlide = Math.ceil(REFS.length / 2);
[[0, perSlide, '(1/2)'], [perSlide, REFS.length, '(2/2)']].forEach(([a, b, tag]) => {
  const s = pres.addSlide();
  title(s, `References ${tag}`);
  s.addText(REFS.slice(a, b).map(([, r], i) => ({
    text: `[${a + i + 1}] ${r}`, options: { breakLine: true, paraSpaceAfter: 2 },
  })), { x: 0.35, y: 0.85, w: 9.3, h: 4.6, fontFace: FONT, fontSize: 7, color: '000000', valign: 'top', margin: 0, isTextBox: true });
});

pres.writeFile({ fileName: OUT }).then(f => console.log('wrote', f));

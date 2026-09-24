const pptxgen = require('pptxgenjs');
const OUT = process.argv[2] || 'Literature_Review_Slide.pptx';

const NAVY = '1F3864', MID = '2E5597', ICE = 'E8EEF7', GREY = '5A6270', FONT = 'Calibri';

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE'; // 13.33 x 7.5
const s = pres.addSlide();
s.background = { color: 'FFFFFF' };

s.addText('Literature Review', { x: 0.5, y: 0.35, w: 8, h: 0.7, fontFace: FONT, fontSize: 36, bold: true, color: NAVY, margin: 0, isTextBox: true });
s.addText('Where the AIM work has been, and where Model B sits', { x: 0.5, y: 1.02, w: 8.5, h: 0.4, fontFace: FONT, fontSize: 16, color: MID, margin: 0, isTextBox: true });

// ---- comparison table (left) ----
const hdr = t => ({ text: t, options: { bold: true, color: 'FFFFFF', fill: { color: NAVY } } });
const rows = [
  ['Study', 'Input', 'Model', 'Validation', 'F1'].map(hdr),
  ['Doulah et al. 2021', 'Handcrafted features', 'Linear SVM', 'LOSO, 30', '0.818'],
  ['Ghosh and Sazonov 2022', 'Raw time series', 'ResNet', 'LOSO, 17', '0.906'],
  ['Ghosh et al. 2024', 'Wavelet scalogram', '2D CNN', 'LOSO, 30', '0.776'],
  ['Diou et al. 2022', 'Raw wrist IMU, bites', 'CNN and LSTM', 'LOSO', '0.923'],
  ['Model A, previous design', 'FFT spectrum', '1D CNN', '4 fold', '0.779'],
  ['Model B, this work', 'Z-scored time series', '1D CNN, attention', '4 fold', '0.854'],
];
const body = rows.map((r, i) => i === 0 ? r : r.map((c, j) => ({
  text: c,
  options: {
    fill: { color: i === 6 ? ICE : (i % 2 ? 'FFFFFF' : 'F7F9FC') },
    bold: i === 6 || j === 0,
    color: i === 6 ? NAVY : '222222',
    align: j === 4 ? 'center' : 'left',
  },
})));
body[0][4].options.align = 'center';
s.addTable(body, {
  x: 0.5, y: 1.75, w: 8.1, colW: [2.2, 2.0, 1.65, 1.35, 0.9],
  fontFace: FONT, fontSize: 13, rowH: 0.52, valign: 'middle',
  border: { type: 'solid', pt: 0.75, color: 'B7C2D0' }, margin: [0, 0.08, 0, 0.08],
});
s.addText('Different datasets, labels and validation schemes. Figures are not directly comparable. Diou et al. is bite detection on a smartwatch.', {
  x: 0.5, y: 5.55, w: 8.1, h: 0.5, fontFace: FONT, fontSize: 11, italic: true, color: GREY, margin: 0, isTextBox: true,
});

// ---- takeaway cards (right) ----
const cards = [
  ['Established', 'Raw time series input on AIM was introduced by Ghosh and Sazonov in 2022.'],
  ['What this adds', 'A controlled ablation. Same architecture on FFT input loses 0.087 F1 and 0.180 precision.'],
  ['Next', 'Reproduce the 2022 ResNet on the new database, then pretrain on the 41 unannotated sessions.'],
];
cards.forEach(([head, text], i) => {
  const y = 1.75 + i * 1.45;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.95, y, w: 3.9, h: 1.35, fill: { color: ICE }, line: { color: ICE }, rectRadius: 0.08 });
  s.addShape(pres.shapes.OVAL, { x: 9.12, y: y + 0.17, w: 0.42, h: 0.42, fill: { color: NAVY }, line: { color: NAVY } });
  s.addText(String(i + 1), { x: 9.12, y: y + 0.17, w: 0.42, h: 0.42, fontFace: FONT, fontSize: 14, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  s.addText(head, { x: 9.68, y: y + 0.14, w: 3.05, h: 0.46, fontFace: FONT, fontSize: 16, bold: true, color: NAVY, valign: 'middle', margin: 0, isTextBox: true });
  s.addText(text, { x: 9.12, y: y + 0.62, w: 3.6, h: 0.68, fontFace: FONT, fontSize: 12, color: '333333', valign: 'top', margin: 0, isTextBox: true });
});

// ---- recommendation band ----
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 6.2, w: 12.35, h: 0.8, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.08 });
s.addText([
  { text: 'Recommendation   ', options: { bold: true, color: 'FFFFFF' } },
  { text: 'A compact 1D CNN on z-scored four channel time series, with subject independent validation. No transformer at this data scale.', options: { color: 'FFFFFF' } },
], { x: 0.75, y: 6.2, w: 11.9, h: 0.8, fontFace: FONT, fontSize: 15, valign: 'middle', margin: 0, isTextBox: true });

s.addNotes([
  'Seven papers reviewed. Three from the AIM line, one from the Thessaloniki group, one survey, and two from outside the field for architectural ideas.',
  'The lab moved to raw time series input in 2022. The previous in house model used an FFT magnitude spectrum, which was a step away from that. My ablation measures the cost under matched conditions.',
  'Model B is below the 2022 ResNet figure, but the data are very different, 7.8 hours from 20 annotated sessions against 372 hours from 17 participants. The fair comparison is to train that ResNet on the new database under the same harness.',
  'The attention and chew count components did not separate from noise. The representation did nearly all of the work.',
].join('\n\n'));

pres.writeFile({ fileName: OUT }).then(f => console.log('wrote', f));

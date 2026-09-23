const fs = require('fs');
const pptxgen = require('pptxgenjs');
const OUT = process.argv[2] || 'Results_Slides.pptx';
const CSV = process.argv[3];

const NAVY = '1F3864', MID = '2E5597', ICE = 'E8EEF7', GREY = '5A6270', FONT = 'Calibri';
const GOOD = '1E7B4A', WARN = 'B26B00', BAD = 'B03A2E';

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE'; // 13.33 x 7.5

function title(s, t, sub) {
  s.addText(t, { x: 0.5, y: 0.35, w: 12.3, h: 0.7, fontFace: FONT, fontSize: 34, bold: true, color: NAVY, margin: 0, isTextBox: true });
  if (sub) s.addText(sub, { x: 0.5, y: 1.02, w: 12.3, h: 0.4, fontFace: FONT, fontSize: 16, color: MID, margin: 0, isTextBox: true });
}
function footnote(s, t, y = 6.85) {
  s.addText(t, { x: 0.5, y, w: 12.3, h: 0.4, fontFace: FONT, fontSize: 11, italic: true, color: GREY, margin: 0, isTextBox: true });
}
const hdr = t => ({ text: t, options: { bold: true, color: 'FFFFFF', fill: { color: NAVY }, align: 'center' } });
function body(rows, { highlight = [], firstBold = true, colors = {} } = {}) {
  return rows.map((r, i) => r.map((c, j) => ({
    text: c,
    options: {
      fill: { color: highlight.includes(i) ? ICE : (i % 2 ? 'F7F9FC' : 'FFFFFF') },
      bold: highlight.includes(i) || (firstBold && j === 0),
      color: (colors[i] && colors[i][j]) || (highlight.includes(i) ? NAVY : '222222'),
      align: j === 0 ? 'left' : 'center',
    },
  })));
}
const tableOpts = (x, y, w, colW, rowH, fontSize = 14) => ({
  x, y, w, colW, rowH, fontFace: FONT, fontSize, valign: 'middle',
  border: { type: 'solid', pt: 0.75, color: 'B7C2D0' }, margin: [0, 0.1, 0, 0.1],
});
function card(s, x, y, w, h, head, text, headColor = NAVY) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: ICE }, line: { color: ICE }, rectRadius: 0.08 });
  s.addText(head, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: 0.4, fontFace: FONT, fontSize: 16, bold: true, color: headColor, margin: 0, isTextBox: true });
  s.addText(text, { x: x + 0.2, y: y + 0.55, w: w - 0.4, h: h - 0.65, fontFace: FONT, fontSize: 13, color: '333333', valign: 'top', margin: 0, isTextBox: true });
}

// ------------------------------------------------------------------ slide 1
{
  const s = pres.addSlide();
  title(s, 'Model A vs Model B', 'Same data, same folds, same evaluation. Only the model changes.');
  const rows = [
    ['Metric', 'Model A (previous design)', 'Model B (new)', 'Change'].map(hdr),
    ...body([
      ['F1 score', '0.779', '0.854', '+0.075'],
      ['Accuracy', '0.841', '0.905', '+0.064'],
      ['Balanced accuracy', '0.840', '0.887', '+0.047'],
      ['Precision', '0.730', '0.876', '+0.146'],
      ['Recall', '0.836', '0.834', '-0.002'],
    ], { highlight: [0], colors: { 0: { 3: GOOD }, 1: { 3: GOOD }, 2: { 3: GOOD }, 3: { 3: GOOD }, 4: { 3: GREY } } }),
  ];
  s.addTable(rows, tableOpts(0.5, 1.75, 7.9, [2.3, 2.2, 1.9, 1.5], 0.62, 16));

  card(s, 8.8, 1.75, 4.05, 1.75, 'Where the gain is',
    'Almost all in precision. Model B raises far fewer false alarms while catching the same share of eating.');
  card(s, 8.8, 3.7, 4.05, 1.85, 'New. Leave one subject out',
    'Model A under LOSO, 20 folds. F1 0.758, accuracy 0.818, balanced accuracy 0.826. Model B under LOSO is running next.');

  footnote(s, '20 participants, 3,503 windows of 8 s. Participant level 4 fold cross validation, threshold chosen on held out validation participants, never the test fold.', 6.35);
  s.addNotes('Model A is the original architecture retrained on the new database, so this compares models, not data sources. Pooled over all 3,503 held out windows. Model B confusion matrix tp 980, fp 139, fn 195, tn 2189. Per fold F1 for Model B was 0.920, 0.801, 0.807, 0.852, a spread of about 0.05. LOSO numbers for Model A came in today. The two participants with the lowest F1 under LOSO were AIM122571 and AIM126487, both near 0.43.');
}

// ------------------------------------------------------------------ slide 2
{
  const s = pres.addSlide();
  title(s, 'Ablation Study', 'Model B with one component removed or added at a time. Same folds, same evaluation.');
  const rows = [
    ['Variant', 'F1', 'Change', 'Precision', 'Verdict'].map(hdr),
    ...body([
      ['Model B, full', '0.854', '', '0.876', ''],
      ['Remove time domain input (use FFT)', '0.767', '-0.087', '0.696', 'Real effect'],
      ['Remove overlapping training windows', '0.829', '-0.025', '0.834', 'Suggestive'],
      ['Remove attention', '0.847', '-0.007', '0.857', 'Within noise'],
      ['Remove chew count head', '0.849', '-0.006', '0.857', 'Within noise'],
      ['Add median smoothing', '0.775', '-0.079', '0.765', 'Hurts'],
      ['Add 8 s context each side', '0.775', '-0.080', '0.683', 'Hurts'],
    ], {
      highlight: [0],
      colors: { 1: { 4: GOOD }, 2: { 4: WARN }, 3: { 4: GREY }, 4: { 4: GREY }, 5: { 4: BAD }, 6: { 4: BAD } },
    }),
  ];
  s.addTable(rows, tableOpts(0.5, 1.75, 8.4, [3.4, 0.95, 1.1, 1.4, 1.55], 0.56, 15));

  card(s, 9.25, 1.75, 3.6, 1.55, 'Headline',
    'The input representation does nearly all the work. Same architecture on FFT input loses 0.087 F1.');
  card(s, 9.25, 3.45, 3.6, 1.45, 'Why smoothing hurts',
    'Meals pause between bites. A 24 s median erases short real bouts.');
  card(s, 9.25, 5.05, 3.6, 1.3, 'Next',
    'Rerun under LOSO for a paired test per participant.');

  footnote(s, 'Fold to fold spread in F1 is about 0.048. Effects smaller than that cannot be separated from noise with 4 folds.', 6.55);
  s.addNotes('Real means larger than the fold spread, with a matching drop in precision. Suggestive means about half the spread. Attention and the chew head are inside noise and are not claimed as contributions. The FFT row is the key one. Model B architecture on FFT input scores 0.767, below Model A on the same input at 0.779, so the architecture only helps once the input keeps its timing. Context lost precision because the model answered whether there was eating anywhere in 24 s, instead of in the labelled 8 s.');
}

// ------------------------------------------------------------------ slide 3
{
  const s = pres.addSlide();
  title(s, 'Literature Review Progress', 'Seven papers read so far. A starting set, not yet a full survey.');
  const rows = [
    ['Paper', 'What they did', 'Key finding'].map(hdr),
    ...body([
      ['Doulah et al. 2021, JBHI', 'AIM-2 introduced. Handcrafted features and SVM', 'F1 0.818. Camera captures only during eating'],
      ['Ghosh and Sazonov 2022, EMBC', 'Five deep nets on raw accelerometer and optical signals', 'ResNet best, F1 0.906. Closest to my work'],
      ['Ghosh et al. 2024, Sci. Reports', 'Wavelet image of accelerometer, fused with camera', 'Sensor F1 0.776. Fusion cuts false positives'],
      ['Diou et al. 2022, Appetite', 'Smartwatch and in ear audio, CNN with LSTM', 'Self supervised learning works with few labels'],
      ['Usman and Chen 2021, survey', 'Review of wearable intake sensors', 'Most work still uses handcrafted features'],
      ['Yang et al. 2026, Phys. Comm.', 'Transformer for radio signals', 'Augmentation helps. Needs about 100,000 signals'],
      ['Ikram et al. 2025, Front. Med.', 'Transformer for ECG', '97% accuracy, but no patient level split'],
    ]).map(r => r.map((c, j) => (j > 0 ? { ...c, options: { ...c.options, align: 'left' } } : c))),
  ];
  rows[0].forEach(c => { c.options.align = 'left'; });
  s.addTable(rows, tableOpts(0.5, 1.6, 12.35, [3.3, 4.6, 4.45], 0.52, 13));

  const y = 5.95;
  const items = [
    ['Proven', 'CNNs on raw time series'],
    ['Gap', 'Little work with limited annotation'],
    ['Direction', 'Label efficiency on AIM data'],
  ];
  items.forEach(([h, t], i) => {
    const x = 0.5 + i * 4.18;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 3.99, h: 0.9, fill: { color: i === 2 ? NAVY : ICE }, line: { color: i === 2 ? NAVY : ICE }, rectRadius: 0.08 });
    s.addText([
      { text: h + '   ', options: { bold: true, color: i === 2 ? 'FFFFFF' : NAVY } },
      { text: t, options: { color: i === 2 ? 'FFFFFF' : '333333' } },
    ], { x: x + 0.2, y, w: 3.6, h: 0.9, fontFace: FONT, fontSize: 14, valign: 'middle', margin: 0, isTextBox: true });
  });
  s.addNotes('Three papers from the AIM line, one from the Thessaloniki group, one survey, and two from outside the field read for architecture ideas. The lab moved to raw time series input in 2022, so my FFT ablation is a controlled measurement of that choice rather than a new idea. The fair next baseline is the 2022 ResNet trained on the new database under my evaluation. The two transformer papers do not separate subjects and use far more data than we have, so I do not recommend a transformer at this scale. Figures across papers come from different datasets and are not directly comparable.');
}

// ------------------------------------------------------------------ slide 4
{
  const s = pres.addSlide();
  title(s, 'Annotation Alignment Check', 'Verified on the new database before training. No time correction is needed.');

  const steps = [
    'Build the eating mask from the database labels, bout or bite, at 10 Hz',
    'Turn the optical signal into chewing activity. Filter to 0.8 to 3 Hz, take the envelope, resample to 10 Hz',
    'Slide one against the other from -30 s to +30 s in 0.1 s steps and measure the correlation',
    'The best lag is the offset. Sessions with a weak peak are flagged unreliable',
    'Repeat reading each timestamp as packet start and as packet end',
  ];
  steps.forEach((t, i) => {
    const y = 1.65 + i * 0.72;
    s.addShape(pres.shapes.OVAL, { x: 0.5, y: y + 0.04, w: 0.42, h: 0.42, fill: { color: NAVY }, line: { color: NAVY } });
    s.addText(String(i + 1), { x: 0.5, y: y + 0.04, w: 0.42, h: 0.42, fontFace: FONT, fontSize: 14, bold: true, color: 'FFFFFF', align: 'center', valign: 'middle', margin: 0, isTextBox: true });
    s.addText(t, { x: 1.1, y, w: 5.2, h: 0.62, fontFace: FONT, fontSize: 13, color: '333333', valign: 'middle', margin: 0, isTextBox: true });
  });

  // per-session lags from the CSV, as a native chart
  const rows = fs.readFileSync(CSV, 'utf8').trim().split(/\r?\n/).slice(1).map(l => l.split(','));
  const start = rows.filter(r => r[2] === 'start');
  const end = rows.filter(r => r[2] === 'end');
  const labels = start.map((r, i) => String(i + 1));
  s.addChart(pres.charts.BAR, [
    { name: 'Timestamp read as packet start', labels, values: start.map(r => +r[7]) },
    { name: 'Timestamp read as packet end', labels, values: end.map(r => +r[7]) },
  ], {
    x: 6.6, y: 1.55, w: 6.25, h: 2.75, barDir: 'col', barGrouping: 'clustered',
    chartColors: [NAVY, 'A9B8CF'], showLegend: true, legendPos: 'b', legendFontSize: 10, legendFontFace: FONT,
    showTitle: true, title: 'Measured lag per session (s). Start reading sits at 0', titleFontSize: 12, titleColor: NAVY, titleFontFace: FONT,
    valAxisMinVal: -9, valAxisMaxVal: 1, valAxisMajorUnit: 2, valAxisLabelFontSize: 10, valAxisLabelColor: GREY,
    catAxisLabelFontSize: 9, catAxisLabelColor: GREY, catAxisTitle: 'Session', showCatAxisTitle: true, catAxisTitleFontSize: 10,
    valGridLine: { color: 'E3E7EE', size: 0.5 }, catGridLine: { style: 'none' },
  });

  const t = [
    ['Timestamp read as', 'Median lag', 'Within 1 s', 'Peak correlation'].map(hdr),
    ...body([
      ['Packet start', '+0.10 s', '20 of 20', '0.716'],
      ['Packet end', '-7.90 s', '0 of 20', '0.715'],
    ], { highlight: [0] }),
  ];
  s.addTable(t, tableOpts(6.6, 4.5, 6.25, [1.9, 1.35, 1.3, 1.7], 0.45, 13));

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 5.45, w: 5.8, h: 1.25, fill: { color: NAVY }, line: { color: NAVY }, rectRadius: 0.08 });
  s.addText([
    { text: 'Result   ', options: { bold: true } },
    { text: 'Aligned within 0.2 s on all 20 sessions. The same peak under both readings means the data are displaced, not degraded, so no shift is applied.' },
  ], { x: 0.7, y: 5.45, w: 5.4, h: 1.25, fontFace: FONT, fontSize: 13, color: 'FFFFFF', valign: 'middle', margin: 0, isTextBox: true });

  footnote(s, 'The method is tested on synthetic data with offsets of known size and direction, and it recovers them within 1 s.', 6.85);
  s.addNotes('This was a precondition before training on the new database, not a headline finding. The two readings of the timestamp differ by exactly one 8 s packet, which is why both were measured. Correlation is computed only over samples valid in both series, so missing sensor data and unannotated time cannot create agreement. The test suite injects offsets of 0, plus or minus 8, 5 and minus 12 seconds and checks each is recovered, and checks that pure noise is reported as unreliable. Those tests also caught a resampling error during development that would have grown to about 8 s over a session.');
}

pres.writeFile({ fileName: OUT }).then(f => console.log('wrote', f));

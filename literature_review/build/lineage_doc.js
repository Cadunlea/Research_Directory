// Literature and lineage document, laid out after Section 2 of Ikram et al.
// (Frontiers in Medicine 2025): narrative literature review, a grey-header
// References / Techniques / Goals / Findings table, a contributions list and a
// closing challenges paragraph. Two further tables trace every stage of the
// Model B pipeline to its origin and mark what this work adds.
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType,
  ShadingType, AlignmentType, BorderStyle, LevelFormat, HeadingLevel,
} = require('docx');
const { image, STYLES, PAGE, footer, FONT } = require('./docx_helpers');
const { REFS, INDEX } = require('./refs');

const OUT = process.argv[2] || 'Model_B_Literature_and_Lineage.docx';
const FIG = fs.readFileSync(path.join(__dirname, '..', 'figures', 'model_b_pipeline.png'));
const W = 10080;

// Frontiers style in-text citations, "(2, 14)"
// Numbered by first appearance, as in the Frontiers paper.
const order = [];
const c = (...keys) => '(' + keys.map(k => {
  if (!(k in INDEX)) throw new Error(k);
  if (!order.includes(k)) order.push(k);
  return order.indexOf(k) + 1;
}).join(', ') + ')';
const cc = c;

const P = (text, o = {}) => new Paragraph({
  children: (Array.isArray(text) ? text : [text]).map(t => typeof t === 'string'
    ? new TextRun({ text: t, font: FONT, size: 21 })
    : new TextRun({ text: t.b, bold: true, font: FONT, size: 21 })),
  spacing: { before: 60, after: 140, line: 288 }, alignment: AlignmentType.LEFT, ...o,
});
const H1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })] });
const Bullet = t => new Paragraph({ numbering: { reference: 'dots', level: 0 }, spacing: { after: 80, line: 276 }, children: [new TextRun({ text: t, font: FONT, size: 21 })] });

// ECG-paper table: dark grey header, white bold header text, light rules,
// caption under the table.
const rule = { style: BorderStyle.SINGLE, size: 4, color: 'BFBFBF' };
function etable(label, headers, rows, widths, caption) {
  const sum = widths.reduce((a, b) => a + b, 0);
  if (sum !== W) throw new Error(`${label} widths ${sum}`);
  const cell = (t, w, head) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    borders: { top: rule, bottom: rule, left: rule, right: rule },
    shading: head ? { fill: '6B6B6B', type: ShadingType.CLEAR, color: 'auto' } : undefined,
    margins: { top: 70, bottom: 70, left: 100, right: 100 },
    children: [new Paragraph({ spacing: { after: 0, line: 250 }, children: [new TextRun({ text: String(t), font: FONT, size: head ? 17 : 16, bold: head, color: head ? 'FFFFFF' : '262626' })] })],
  });
  return [
    new Paragraph({ keepNext: true, spacing: { before: 200, after: 80 }, children: [new TextRun({ text: label, font: FONT, size: 19, bold: true })] }),
    new Table({
      width: { size: W, type: WidthType.DXA }, columnWidths: widths,
      rows: [
        new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, widths[i], true)) }),
        ...rows.map(r => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, widths[i], false)) })),
      ],
    }),
    new Paragraph({ spacing: { before: 80, after: 240 }, children: [new TextRun({ text: caption, font: FONT, size: 17, color: '595959' })] }),
  ];
}

const k = [];
k.push(new Paragraph({ children: [new TextRun({ text: 'Model B Literature and Lineage', bold: true, font: FONT, size: 36 })], spacing: { after: 60 } }));
k.push(new Paragraph({ children: [new TextRun({ text: 'Where each part of the AIM-2 food intake model comes from, and what this work adds', font: FONT, size: 22, color: '595959' })], spacing: { after: 60 } }));
k.push(new Paragraph({
  children: [new TextRun({ text: 'Caelan Dunlea  |  Working document compiled for the Sazonov laboratory  |  September 2026', font: FONT, size: 17, color: '595959' })],
  spacing: { after: 240 }, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: '6B6B6B', space: 4 } },
}));

// ================================================================ 1 literature review
k.push(H1('1 Literature review'));
k.push(P(`The automatic detection of food intake remains an important area of research because self-report cannot capture when, how long or how often a person eats ${cc('doulah2021', 'diou2022')}. Over the past decade, methods for sensor-based intake detection have ranged from handcrafted features with conventional classifiers to end-to-end deep networks, with the shared goal of improving accuracy, generalisation to new people and robustness in free living. Doulah et al. ${cc('doulah2021')} introduced the AIM-2 eyeglass sensor and detected food intake with a linear support vector machine applied to 38 handcrafted time and frequency features per channel, as summarised in Table 1. Farooq and Sazonov ${cc('farooq2016')} extracted peak-based features from a piezoelectric sensor to count chews, and a survey of wearable intake monitoring found that most systems still relied on handcrafted features and classical classifiers ${cc('usman2021')}. These methods generalise poorly to new conditions because their performance depends on how the features were designed and on the data used to select them.`));

k.push(...etable('Table 1', ['References', 'Techniques', 'Goals', 'Findings'], [
  [`Doulah et al. ${c('doulah2021')}`, 'Handcrafted features, linear SVM', 'Detect intake to trigger a camera', 'F1 0.818 in the laboratory'],
  [`Farooq and Sazonov ${c('farooq2016')}`, 'Piezoelectric sensor, peak-based chew counting', 'Chew count and chewing rate', 'Mean absolute error 10.4% to 15.0%'],
  [`Zhang and Amft ${c('zhang2018')}`, 'EMG electrodes in eyeglass frames', 'Chewing in free living', 'About 94% in the laboratory, about 80% in free living'],
  [`Ghosh and Sazonov ${c('ghosh2022')}`, 'Raw accelerometer and optical signals, ResNet and four other deep networks', 'End-to-end intake detection', 'ResNet F1 0.906'],
  [`Kyritsis et al. ${c('kyritsis2021')}`, 'CNN with LSTM on raw smartwatch data', 'Bite and meal detection in the wild', 'Bite F1 0.923'],
  [`Ghosh et al. ${c('ghosh2024')}`, 'Wavelet scalogram with 2D CNN, fused with image detection', 'Reduce false positives', 'Sensor F1 0.776, fused F1 0.808'],
  [`Stankoski et al. ${c('stankoski2024')}`, 'Optical sensors in smart glasses, ConvLSTM', 'Chewing detection in the laboratory and real life', 'F1 0.91, real-life precision 0.95'],
  [`Bedri et al. ${c('bedri2020')}`, 'Multimodal sensing on eyeglasses', 'Eating in unconstrained settings', 'F1 0.89'],
  [`Shavit and Klein ${c('shavit2021')}`, 'Transformer encoder on inertial data', 'Activity recognition', 'Outperformed CNN and LSTM'],
  [`Yang et al. ${c('yang2026')}`, 'Windowed transformer with FFT period block', 'Radio signal classification', 'Accuracy 95.0%'],
  [`Ikram et al. ${c('ikram2025')}`, 'Transformer with PCA', 'ECG arrhythmia classification', 'Accuracy 97.1%'],
], [2250, 3050, 2450, 2330], 'State-of-the-art methods for wearable food intake detection and related sensor classification.'));

k.push(P(`To address the limitations of handcrafted features, Ghosh and Sazonov ${cc('ghosh2022')} applied five end-to-end deep networks directly to the raw accelerometer and optical signals of the AIM-2.1 and found that a residual network performed best, with more convolutional layers giving better detection. Kyritsis et al. ${cc('kyritsis2021')} trained convolutional and recurrent layers together on raw smartwatch data to detect individual bites in the wild. Ghosh et al. ${cc('ghosh2024')} converted accelerometer windows into wavelet scalograms for a two-dimensional network and fused the result with food detection in egocentric images to reduce false positives. Stankoski et al. ${cc('stankoski2024')} showed that optical sensors in smart glasses support chewing detection with high precision in real life. Outside food intake, one-dimensional convolutional networks on normalised raw series form strong baselines for time series classification ${cc('wang2017', 'fawaz2019')}.`));
k.push(P(`Attention mechanisms have recently gained traction in sensor time series because they let a network weight the most informative channels and moments. Squeeze-and-excitation blocks ${cc('hu2018')} were added to fully convolutional networks for multivariate time series ${cc('karim2019')}, and attention over time improved wearable activity recognition ${cc('murahari2018')}. Transformers extend this idea to the whole sequence. Shavit and Klein ${cc('shavit2021')} and Dirgová Luptáková et al. ${cc('luptakova2022')} reported transformers that matched or exceeded convolutional and recurrent networks on inertial data, and Zerveas et al. ${cc('zerveas2021')} showed that masked pretraining lets a transformer learn from unlabelled series. Yang et al. ${cc('yang2026')} and Ikram et al. ${cc('ikram2025')} applied transformers to radio and ECG signals, whose rhythm near 1 to 2 Hz resembles chewing.`));
k.push(P(`Despite their promise, transformers also come with challenges. They need large labelled datasets and careful tuning, and both cross-domain studies split data at the sample level, so their accuracies do not show performance on new people ${cc('saeb2017')}. Annotated eating data is scarce, which favours methods that learn from unlabelled recordings ${cc('haresamudram2020', 'diou2022')}. False positives from chewing gum and other chewing-like activity remain the main practical weakness of sensor-only detection ${cc('ghosh2024', 'bedri2020')}. The contributions of this work are the following.`));
k.push(Bullet('A time-domain convolutional network with channel and temporal attention for AIM-2 data, evaluated with leave-one-subject-out validation on 20 participants, reaching an F1 score of 0.838 and an accuracy of 0.890.'));
k.push(Bullet('A controlled comparison of input representations on AIM data. With the architecture and folds held fixed, replacing the time series with its FFT magnitude lowered F1 by 0.087 and precision by 0.180, identifying the representation as the dominant design choice.'));
k.push(Bullet('A single network that learns intake detection and chew counting together, using the laboratory chew annotations as an auxiliary training target that is discarded at prediction time.'));
k.push(Bullet('A verified sensor to annotation alignment on the laboratory database, with all 20 sessions aligned within 0.2 s, so that no time correction is applied to the labels.'));
k.push(Bullet('A design aimed at a low false positive rate, with planned experiments on external negative recordings, a decision threshold set for a target false positive rate, and self-supervised pretraining on unannotated sessions.'));
k.push(P('Despite notable advances in convolutional, recurrent and attention-based methods, several challenges persist. These include limited labelled data, the drop in performance between the laboratory and free living, false positives from chewing-like activity, and evaluation protocols that let the same person appear in training and testing. Overcoming these is essential for intake detection that is accurate for people the model has never seen.'));

// ================================================================ 2 lineage
k.push(H1('2 Lineage of the proposed model'));
k.push(P('Figure 1 shows the proposed model from the database to the decision. Table 2 traces each stage to the work that introduced it, the work that showed it succeeds, and what this work changes. The step numbers match the full model description.'));
k.push(image(FIG, 6.5, 3.13));
k.push(new Paragraph({ spacing: { before: 40, after: 240 }, children: [new TextRun({ text: 'Figure 1. The proposed model end to end. Shapes are time steps by channels. The dashed chew head is used only in training.', font: FONT, size: 17, color: '595959' })] }));

k.push(...etable('Table 2', ['Steps', 'Component', 'Introduced by', 'Shown successful in', 'This work'], [
  ['1 to 3', 'Video-based ground truth, eating as bout or bite', `AIM ground truth ${c('doulah2021')}`, `${c('doulah2021', 'ghosh2022')}`, 'Only annotated time is used, never labelled as not eating'],
  ['4, 5', 'Accelerometer and optical signals at 128 Hz', `AIM-2.1 ${c('ghosh2022')}`, `F1 0.906 ${c('ghosh2022')}. Optical glasses F1 0.91 ${c('stankoski2024')}`, 'Same channels. Alignment verified within 0.2 s'],
  ['6, 8', '8 s windows with a majority label', `8 s ${c('ghosh2022')}. Majority rule ${c('doulah2021')}`, `${c('ghosh2022')}. Window size matters ${c('banos2014')}`, 'Unchanged. 4 and 16 s to be tested'],
  ['6', 'Overlapping windows for training only', `Sliding windows ${c('dehghani2019')}`, `Valid under subject cross-validation ${c('dehghani2019')}`, 'Tested on non-overlapping windows. +0.025 F1'],
  ['7, 10', 'Per-window z-score, missing samples masked', `Z-normalised series ${c('wang2017')}`, `${c('wang2017', 'fawaz2019')}`, 'Missing samples excluded from the statistics'],
  ['11 to 14', 'Strided 1D convolutions with BatchNorm', `${c('wang2017', 'ioffe2015', 'springenberg2015')}`, `${c('fawaz2019', 'ghosh2022')}`, 'Receptive field about 0.9 s, close to one chew cycle'],
  ['15', 'Squeeze-and-excitation channel attention', `${c('hu2018')}`, `Time series ${c('karim2019')}. Radio signals ${c('yang2026')}`, 'First use on AIM data among the works reviewed'],
  ['16', 'Attention pooling over time', `${c('ilse2018')}. In activity recognition ${c('murahari2018')}`, `${c('murahari2018')}`, 'Replaces global average pooling used on AIM'],
  ['17', 'Dropout', `${c('srivastava2014')}`, 'Standard practice', 'Rate 0.3'],
  ['19, 20', 'Auxiliary chew count head', `Multitask learning ${c('caruana1997')}`, `Chew counting ${c('farooq2016')}`, 'Detection and chew counting learned in one network'],
  ['21', 'Class weighting', `${c('he2009')}`, `Cost-sensitive training on AIM ${c('ghosh2024')}`, 'Weights set per fold'],
  ['22, 23', 'Adam and early stopping', `${c('kingma2015', 'prechelt1998')}`, 'Standard practice', 'Best weights restored from validation'],
  ['24, 25', 'Leave-one-subject-out validation', `Subject-wise evaluation ${c('saeb2017')}`, `${c('doulah2021', 'ghosh2022', 'ghosh2024')}`, 'Rotating validation participants and a paired test per participant'],
  ['26', 'Threshold chosen on validation data', `${c('lipton2014')}`, `${c('lipton2014')}`, 'Planned threshold for a target false positive rate'],
  ['27', 'F1 and balanced accuracy', `${c('brodersen2010')}`, `${c('diou2022', 'ghosh2022')}`, 'Reported pooled and per participant'],
], [900, 2300, 2100, 2330, 2450], 'Origin, evidence and adaptation for each stage of the proposed model.'));

// ================================================================ 3 novelty
k.push(H1('3 What the combination adds'));
k.push(P('No single component of the model is new. What is new is the combination on AIM data and the evidence behind it. Table 3 separates what the reviewed literature has already reported from what this work adds, and marks which additions are complete and which are planned.'));
k.push(...etable('Table 3', ['Element', 'Reported in the works reviewed', 'Status in this work'], [
  ['End-to-end CNN on raw AIM signals', `Yes ${c('ghosh2022')}`, 'Complete. Used as the starting point'],
  ['Matched comparison of time and FFT input on AIM', 'Not found', 'Complete. FFT input costs 0.087 F1'],
  ['Channel and temporal attention on AIM', `Not found. Used in other sensor tasks ${c('karim2019', 'murahari2018')}`, 'Complete. Effect within noise at 20 participants'],
  ['Chew count as an auxiliary target in the classifier', `Not found. Chew counting as a separate stage ${c('farooq2016')}`, 'Complete. Effect within noise at 20 participants'],
  ['Verified sensor to annotation alignment', 'Not found', 'Complete. 20 of 20 sessions within 0.2 s'],
  ['Self-supervised pretraining on unannotated AIM sessions', `Not found. Used in other sensor tasks ${c('haresamudram2020', 'zerveas2021')}`, 'Planned'],
  ['Operating point and external negatives for a low false positive rate', `Not found. Image fusion used instead ${c('ghosh2024')}`, 'Planned'],
], [3500, 3700, 2880], 'Elements of the proposed approach and whether the reviewed literature reports them.'));
k.push(P('Together, the planned elements point to a clear and citable aim, which is a label-efficient AIM-2 detector with a low false positive rate. It would learn from recordings that were never annotated, use chew counting as extra supervision, and be tuned to the false positive rate rather than to accuracy alone, all under subject-independent evaluation. None of the reviewed works combines these, and each piece has precedent that makes the combination defensible.'));

// ================================================================ references
k.push(H1('References'));
const TEXT = Object.fromEntries(REFS);
order.forEach((key, i) => {
  k.push(new Paragraph({
    spacing: { after: 70, line: 250 }, indent: { left: 480, hanging: 480 },
    children: [new TextRun({ text: `${i + 1}. `, font: FONT, size: 17 }), new TextRun({ text: TEXT[key], font: FONT, size: 17 })],
  }));
});

const doc = new Document({
  creator: 'Caelan Dunlea', title: 'Model B Literature and Lineage', styles: STYLES,
  numbering: { config: [{ reference: 'dots', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{ properties: { page: PAGE }, footers: footer(), children: k }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync(OUT, b); console.log('wrote', OUT); });

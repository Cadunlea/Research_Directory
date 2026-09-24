const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  WidthType, ShadingType, AlignmentType, HeadingLevel, BorderStyle,
  LevelFormat, Footer, PageNumber,
} = require('docx');

const OUT = process.argv[2] || 'Literature_Review.docx';
const W = 10224; // content width in DXA, Letter with 0.9 in margins
const FONT = 'Calibri';

// ---- helpers -------------------------------------------------------------

function runs(spec, size) {
  // spec: string, or array of strings / {b: 'bold text'} / {i: 'italic'}
  const parts = Array.isArray(spec) ? spec : [spec];
  return parts.map(p => {
    if (typeof p === 'string') return new TextRun({ text: p, font: FONT, size });
    if (p.b) return new TextRun({ text: p.b, bold: true, font: FONT, size });
    if (p.i) return new TextRun({ text: p.i, italics: true, font: FONT, size });
    return new TextRun({ text: '', font: FONT, size });
  });
}

const P = (spec, opts = {}) => new Paragraph({
  children: runs(spec, opts.size || 22),
  spacing: { before: opts.before ?? 120, after: opts.after ?? 140, line: 276 },
  alignment: opts.align || AlignmentType.LEFT,
  ...(opts.bullet ? { numbering: { reference: 'bullets', level: 0 } } : {}),
});

const H1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })], spacing: { before: 300, after: 120 } });
const H2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })], spacing: { before: 220, after: 100 } });
const Caption = t => new Paragraph({ children: [new TextRun({ text: t, bold: true, font: FONT, size: 20, color: '1F3864' })], spacing: { before: 160, after: 80 }, keepNext: true });
const Note = t => new Paragraph({ children: [new TextRun({ text: t, italics: true, font: FONT, size: 17, color: '555555' })], spacing: { before: 60, after: 200 } });

const border = { style: BorderStyle.SINGLE, size: 4, color: 'B7C2D0' };
const borders = { top: border, bottom: border, left: border, right: border };

function cell(spec, width, { header = false, fill, bold = false } = {}) {
  const content = Array.isArray(spec) && spec.every(s => Array.isArray(s)) ? spec : [spec];
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    borders,
    shading: header ? { fill: '1F3864', type: ShadingType.CLEAR, color: 'auto' }
                    : fill ? { fill, type: ShadingType.CLEAR, color: 'auto' } : undefined,
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    children: content.map(c => new Paragraph({
      children: header
        ? [new TextRun({ text: String(c), bold: true, color: 'FFFFFF', font: FONT, size: 17 })]
        : (bold ? [new TextRun({ text: String(c), bold: true, font: FONT, size: 17 })] : runs(c, 17)),
      spacing: { after: 0, line: 252 },
    })),
  });
}

// rows: array of arrays. highlight: set of row indexes to shade.
function table(headers, rows, widths, { highlight = [], boldFirst = false } = {}) {
  const sum = widths.reduce((a, b) => a + b, 0);
  if (sum !== W) throw new Error(`widths sum ${sum} != ${W} for ${headers[0]}`);
  return new Table({
    width: { size: W, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, widths[i], { header: true })) }),
      ...rows.map((r, ri) => new TableRow({
        cantSplit: true,
        children: r.map((c, i) => cell(c, widths[i], {
          fill: highlight.includes(ri) ? 'E8EEF7' : (ri % 2 ? 'F7F9FC' : undefined),
          bold: (boldFirst && i === 0) || highlight.includes(ri) && i === 0,
        })),
      })),
    ],
  });
}

// ---- content -------------------------------------------------------------

const children = [];

children.push(new Paragraph({
  children: [new TextRun({ text: 'Literature Review', bold: true, font: FONT, size: 40, color: '1F3864' })],
  spacing: { after: 60 },
}));
children.push(new Paragraph({
  children: [new TextRun({ text: 'Wearable Food Intake Detection and Model Selection for AIM-2', font: FONT, size: 26, color: '2E5597' })],
  spacing: { after: 60 },
}));
children.push(new Paragraph({
  children: [new TextRun({ text: 'Caelan Dunlea  |  Randall Research Scholars Program, Phase II, Activity 4  |  September 2026', font: FONT, size: 19, color: '666666' })],
  spacing: { after: 240 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: '1F3864', space: 4 } },
}));

children.push(H1('Purpose'));
children.push(P('This review supports Activity 4 of my Phase II contract. It covers seven papers on wearable food intake detection and on deep learning for one dimensional signals, and it places the model I have developed on the new AIM-2 database alongside them. The aim is not to adopt an existing approach. It is to establish what has already been done with this device family, to identify where the current work adds something new, and to recommend a direction that builds on those findings.'));
children.push(P('Three of the papers come from the AIM research line, two review or represent other intake monitoring groups, and two come from outside the food intake field and were read for their architectural ideas.'));

// Table 1
children.push(H1('Papers Reviewed'));
children.push(Caption('Table 1. Scope of each paper'));
children.push(table(
  ['#', 'Study', 'Domain and sensors', 'Data', 'Task'],
  [
    ['1', 'Doulah et al., 2021, IEEE JBHI', 'AIM-2 on eyeglasses. 3 axis accelerometer and flex sensor over the temporalis muscle, 128 Hz', '30 participants, one pseudo free living day and one free living day each', 'Epoch level intake detection used to trigger the camera'],
    ['2', 'Ghosh and Sazonov, 2022, IEEE EMBC', 'AIM-2.1. 3 axis accelerometer and optical sensor over the temporalis muscle, 128 Hz', '17 participants, 23 free living days, 372 h, 167,419 epochs', 'Epoch level intake detection'],
    ['3', 'Ghosh et al., 2024, Scientific Reports', 'AIM-2 accelerometer and egocentric camera', '30 participants, free living day only, 380 h, 111 meals', 'Intake and episode detection by fusing sensor and image classifiers'],
    ['4', 'Diou et al., 2022, Appetite', 'Commercial smartwatch IMU. In ear microphone with PPG and accelerometer', 'FIC with 12 subjects and 21 meals. FreeFIC with 77 h. ACE with 250 h', 'Bite, meal and chewing detection'],
    ['5', 'Usman and Chen, 2021, survey', 'Wearable intake sensors in general', 'Review of published studies', 'Taxonomy of tasks and sensors, future directions'],
    ['6', 'Yang et al., 2026, Physical Communication', 'Radio frequency waveforms, outside the food intake field', '100,000 emitter signals and the RML2018.01a benchmark', 'Signal classification with a time frequency transformer'],
    ['7', 'Ikram et al., 2025, Frontiers in Medicine', 'Electrocardiogram, outside the food intake field', 'MIT-BIH, five beat classes', 'Arrhythmia classification with a transformer'],
  ],
  [360, 2000, 2864, 2600, 2400],
));

// Table 2
children.push(H1('Methods and Reported Results'));
children.push(Caption('Table 2. Representation, model and evaluation in each study, with Model B for reference'));
children.push(table(
  ['#', 'Input representation', 'Window', 'Model', 'Validation', 'Best reported result'],
  [
    ['1', '38 handcrafted time and frequency features per channel, 195 in total, reduced by mRMR and forward selection', '10 s, non overlapping', 'Two stage linear SVM, gated by flex sensor amplitude', 'Leave one subject out, 30 folds', 'F1 0.818 (SD 0.101) on laboratory meals'],
    ['2', 'Raw 4 channel time series, three accelerometer axes and optical', '8 s', 'MLP, Time-CNN, FCN, ResNet and Inception compared', 'Leave one subject out, 17 folds', 'ResNet F1 0.906, balanced accuracy 0.935'],
    ['3', 'Continuous wavelet scalogram of the three accelerometer axes, resized to a 192 by 192 image', '15 s', '2D CNN, then a random forest that fuses sensor and image scores', 'Leave one subject out for the sensor model. Random holdout to select the fusion model', 'Sensor only F1 0.776. Episode level fusion F1 0.808'],
    ['4', 'Raw inertial signals for bites. Audio signal with learned features for chewing', '5 s', 'CNN followed by LSTM, trained end to end. Audio CNN, also with self supervised pretraining', 'Leave one subject out', 'Bite F1 0.923. Chewing F1 0.908 supervised and 0.86 self supervised'],
    ['5', 'Mostly handcrafted features', 'Varies', 'Mostly SVM and random forest, some CNN and LSTM', 'Varies', 'Not applicable'],
    ['6', 'Raw waveform, with an internal FFT step that selects dominant periods', '3,000 samples', 'Windowed multi scale transformer with squeeze and excitation gating', 'Random 60, 20, 20 split', 'Accuracy 0.950 over all noise levels'],
    ['7', 'Single beats, min max scaled, with PCA and correlation based selection', 'One beat', 'Transformer encoder with four attention heads', 'Sample level train and test split. Patient separation not reported', 'Accuracy 0.971, F1 0.95'],
    ['B', 'Z-scored 4 channel time series with no frequency filter, missing samples excluded then zeroed', '8 s, with a 4 s hop at training time only', '1D CNN with squeeze and excitation, attention pooling over time and an auxiliary chew count head', 'Participant level 4 fold CV, threshold set on inner validation participants', 'F1 0.854, balanced accuracy 0.887, precision 0.876'],
  ],
  [360, 2250, 1250, 2250, 2014, 2100],
  { highlight: [7] },
));
children.push(Note('Figures come from different datasets, labels and validation schemes and are not directly comparable. The section on evaluation and comparability discusses this.'));

// Table 3
children.push(H1('Input Representation Across the AIM Line of Work'));
children.push(P('The clearest thread through the AIM papers is the input representation. Table 3 lays out that progression together with the two models I trained on the new database.'));
children.push(Caption('Table 3. How the AIM input has been represented over time'));
children.push(table(
  ['Year', 'Work', 'Representation', 'Sensors', 'F1'],
  [
    ['2021', 'Doulah et al.', 'Handcrafted features and SVM', 'Accelerometer and flex', '0.818, laboratory meals'],
    ['2022', 'Ghosh and Sazonov', 'Raw time series and ResNet', 'Accelerometer and optical', '0.906, free living'],
    ['2024', 'Ghosh et al.', 'Wavelet scalogram and 2D CNN', 'Accelerometer only', '0.776, free living'],
    ['Prior', 'Previous in house model, Model A', 'FFT magnitude spectrum, 513 bins, and 1D CNN over frequency', 'Accelerometer and optical', '0.779, new database'],
    ['2026', 'Model B, this project', 'Z-scored time series and 1D CNN with attention', 'Accelerometer and optical', '0.854, new database'],
    ['2026', 'Model B architecture with FFT input', 'FFT magnitude spectrum', 'Accelerometer and optical', '0.767, new database'],
  ],
  [800, 2500, 3024, 2000, 1900],
  { highlight: [4] },
));
children.push(P(['The lab moved from handcrafted features to raw time series input in 2022, so the magnitude spectrum used by the previous in house model was a step away from established practice rather than the norm. My ablation therefore does not introduce time domain input as a new idea. What it contributes is a ', { b: 'controlled measurement' }, ' of what the spectrum costs. With the architecture, folds and threshold procedure held fixed, replacing the time series with its magnitude spectrum lowers F1 by 0.087 and precision by 0.180, and it removes any benefit from attention. None of the reviewed papers reports that comparison under matched conditions.']));
children.push(P('The 2024 scalogram result points the same way. A wavelet scalogram keeps time localisation, which a magnitude spectrum discards, but converting it to a resized image and using the accelerometer alone produced a lower F1 than the raw four channel input of 2022.'));

// Table 4
children.push(H1('Model B Components Against Prior Work'));
children.push(P('Table 4 takes each component of Model B and records the closest precedent I found, whether the idea is established, and what it measurably contributed in my ablations. The fold to fold spread in F1 is about 0.048, which sets the bar for what counts as a real effect.'));
children.push(Caption('Table 4. Precedent and measured effect for each component'));
children.push(table(
  ['Component', 'Closest precedent', 'Status', 'Measured effect in Model B'],
  [
    ['Time domain input', 'Raw four channel input in Ghosh and Sazonov 2022', 'Established in the lab', 'F1 +0.087, precision +0.180. Larger than the fold spread'],
    ['Squeeze and excitation channel weighting', 'Hu et al. 2018. Also used as the gating step in Yang et al. 2026', 'Standard building block', 'F1 +0.007 together with attention pooling. Inside noise'],
    ['Attention pooling over time', 'Attention based pooling in Ilse et al. 2018. The reviewed intake papers use global average pooling or an LSTM', 'Standard block, not used in the reviewed AIM papers', 'Counted with the row above'],
    ['Auxiliary chew count head', 'Chew counting from sensor signals by Farooq and Sazonov, summarised in Usman and Chen 2021, but as a separate stage', 'Not found in the reviewed papers as a training target inside the classifier', 'F1 +0.006. Inside noise'],
    ['Overlapping windows at training time only', 'Random shift and crop augmentation in Yang et al. 2026', 'Common augmentation', 'F1 +0.025. Suggestive, about half the spread'],
    ['Participant level validation', 'Leave one subject out in Doulah 2021, Ghosh 2022 and Ghosh 2024', 'Established in the lab. My 4 fold scheme is coarser', 'Not an ablation'],
    ['Sensor to annotation alignment check by cross correlation', 'None in the reviewed papers', 'Verification step not reported elsewhere', 'Median lag +0.10 s across 20 sessions. No correction applied'],
    ['Median smoothing across windows', 'Episode formation with a 150 s Gaussian kernel in Doulah 2021. Previous window scores in Ghosh 2024', 'Contrasting result', 'F1 minus 0.079 at the epoch level'],
    ['Neighbouring context without a target marker', 'Previous window scores in Ghosh 2024, which raised F1 under a random holdout', 'Contrasting result', 'F1 minus 0.080, with precision falling 0.193'],
  ],
  [2300, 3224, 2300, 2400],
  { boldFirst: true },
));
children.push(P('Two of the negative results sit in tension with the literature and deserve care in how they are presented. Doulah et al. smoothed detections to form eating episodes, which is a different question from classifying each 8 s window, so the two findings are compatible. Ghosh et al. 2024 reported gains from adding previous window scores, but that experiment used a random holdout that the authors note may place the same participant in both training and testing. Neighbouring windows of the same meal are close to duplicates, so a subject independent test of the same idea may not show the gain.'));

// Evaluation
children.push(H1('Evaluation and Comparability'));
children.push(P('The reported numbers cannot be ranked against each other directly, and this matters most for the comparison with the lab\'s 2022 ResNet. Model B reaches F1 0.854, below the 0.906 reported by Ghosh and Sazonov. That study used 372 hours from 17 participants with image based annotation, while Model B trains on 7.8 hours from 20 video annotated sessions. The labels, the amount of data and the validation scheme all differ, so neither number shows which architecture is better. The fair test is to train the 2022 ResNet on the new database under my harness, with the same folds and threshold procedure.'));
children.push(P('The two transformer papers need a different caution. Neither separates subjects between training and testing. Both train on tens of thousands of labelled examples, while this project evaluates on about 3,500 windows. Their architectures are useful as sources of ideas, but their headline accuracies are not evidence that a transformer would help at this scale.'));
children.push(P('Diou et al. reinforce two points that apply here. Leave one subject out evaluation is the norm across groups, and F1 is the preferred headline metric because eating is a minority class in free living data.'));

// Opportunities
children.push(H1('Opportunities to Build On'));
children.push(Caption('Table 5. Directions suggested by the reviewed work'));
children.push(table(
  ['Direction', 'Evidence', 'Application here', 'Novelty on AIM data'],
  [
    ['Reproduce the 2022 ResNet on the new database', 'Ghosh and Sazonov 2022', 'Same harness, folds and threshold procedure as Model B', 'Replication. Needed before any claim against the lab\'s best result'],
    ['Self supervised pretraining on unannotated sessions', 'Diou et al. 2022 reached F1 0.86 with precision 0.94 using self supervised audio features', '61 sensor sessions exist and only 20 are annotated. Pretrain the trunk on all of them, then fine tune on the annotated set', 'Not found in the reviewed AIM papers. Targets the main constraint, which is labelled data'],
    ['Signal augmentation such as time masking, noise and amplitude scaling', 'Augmentation raised accuracy from 0.922 to 0.950 in Yang et al. 2026. Orientation augmentation in Diou et al. 2022', 'Training time only, with an ablation for each type', 'Low novelty, likely gain'],
    ['Leave one subject out evaluation', 'Standard in all three AIM papers', 'Replace the 4 fold scheme so results sit on the lab\'s usual footing', 'Aligns reporting with the lab'],
    ['Window length sweep that includes 5 s', 'Diou et al. use 5 s so each window holds at least five chews at 1 Hz. The AIM papers used 8, 10 and 15 s', 'Already planned as a 2, 4, 8 and 16 s sweep. Adding 5 s links it to prior choices', 'Systematic comparison not reported for AIM'],
    ['Multi task learning with bite and chew targets', 'Chew counting work by Farooq and Sazonov. Bite labels are in the database', 'Extend the auxiliary head to bite count alongside chew count', 'Not found in the reviewed papers'],
    ['Period aware spectral module', 'The time frequency block in Yang et al. 2026 finds dominant periods by FFT', 'Chewing is quasi periodic near 1 to 2.5 Hz. A period aware block could add spectral cues without discarding phase', 'Exploratory'],
    ['Episode level metrics', 'Reported in Doulah 2021, Ghosh 2024 and Diou 2022', 'Report episodes alongside window metrics', 'Aligns reporting with the lab'],
  ],
  [2300, 2824, 2800, 2300],
  { boldFirst: true },
));

// Recommendation
children.push(H1('Recommendation'));
children.push(P(['I recommend continuing with a ', { b: 'compact one dimensional convolutional network on z-scored four channel time series' }, ', evaluated with subject independent validation. This agrees with the strongest result in the lab\'s own prior work and with the controlled ablation on the new database, where the representation accounted for nearly all of the improvement.']));
children.push(P('I do not recommend moving to a transformer at this stage. The reviewed transformer results come from far larger datasets and from splits that do not separate subjects, and my own attention components did not separate from noise with 20 sessions.'));
children.push(P('The places where this project can add something new are mostly outside the architecture. The first is a clean comparison against the 2022 ResNet on the new database. The second is using the unannotated sessions through self supervised pretraining, which addresses the limit on labelled data directly. The third is the set of controlled negative results on smoothing and context, which add a subject independent view to a question the lab has so far studied under different conditions.'));

// References
children.push(H1('References'));
const refs = [
  'A. Doulah, T. Ghosh, D. Hossain, M. H. Imtiaz, and E. Sazonov, "Automatic Ingestion Monitor Version 2 - A Novel Wearable Device for Automatic Food Intake Detection and Passive Capture of Food Images," IEEE Journal of Biomedical and Health Informatics, vol. 25, no. 2, pp. 568-576, 2021. doi 10.1109/JBHI.2020.2995473',
  'T. Ghosh and E. Sazonov, "A Comparative Study of Deep Learning Algorithms for Detecting Food Intake," in Proc. 44th Annual International Conference of the IEEE EMBC, Glasgow, 2022, pp. 2993-2996. doi 10.1109/EMBC48229.2022.9871278',
  'T. Ghosh, Y. Han, V. Raju, D. Hossain, M. A. McCrory, J. Higgins, C. Boushey, E. J. Delp, and E. Sazonov, "Integrated image and sensor-based food intake detection in free-living," Scientific Reports, vol. 14, 1665, 2024. doi 10.1038/s41598-024-51687-3',
  'C. Diou, K. Kyritsis, V. Papapanagiotou, and I. Sarafis, "Intake Monitoring in Free-Living Conditions. Overview and Lessons We Have Learned," Appetite, 2022. doi 10.1016/j.appet.2022.106096',
  'M. Usman and H. Chen, "Recent Trends in Food Intake Monitoring using Wearable Sensors," arXiv 2101.01378, 2021.',
  'J. Yang, Y. Wang, M. Yang, and H. Meng, "Signal classification based on multi-scale time-frequency transformer," Physical Communication, vol. 75, 103022, 2026. doi 10.1016/j.phycom.2026.103022',
  'S. Ikram et al., "Transformer-based ECG classification for early detection of cardiac arrhythmias," Frontiers in Medicine, vol. 12, 1600855, 2025. doi 10.3389/fmed.2025.1600855',
  'J. Hu, L. Shen, and G. Sun, "Squeeze-and-Excitation Networks," in Proc. IEEE CVPR, 2018.',
  'M. Ilse, J. Tomczak, and M. Welling, "Attention-based Deep Multiple Instance Learning," in Proc. ICML, 2018.',
];
refs.forEach((r, i) => children.push(new Paragraph({
  children: [new TextRun({ text: `[${i + 1}]  `, font: FONT, size: 18, bold: true }), new TextRun({ text: r, font: FONT, size: 18 })],
  spacing: { after: 80 },
  indent: { left: 440, hanging: 440 },
})));

const doc = new Document({
  creator: 'Caelan Dunlea',
  title: 'Literature Review',
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: '1F3864' }, paragraph: { spacing: { before: 300, after: 120 }, outlineLevel: 0, keepNext: true } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, font: FONT, color: '2E5597' }, paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1, keepNext: true } },
    ],
  },
  numbering: { config: [{ reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1080, left: 1008, right: 1008 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ children: ['Page ', PageNumber.CURRENT], font: FONT, size: 16, color: '888888' })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then(b => { fs.writeFileSync(OUT, b); console.log('wrote', OUT); });

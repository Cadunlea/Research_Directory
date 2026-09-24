// Model B technical description. Objectives 1 and 2 in writing. The same
// content as the slides, read from model_content.js.
const fs = require('fs');
const path = require('path');
const { Document, Packer, Paragraph, TextRun, AlignmentType, BorderStyle } = require('docx');
const { P, H1, Caption, Note, table, image, STYLES, PAGE, footer, references, FONT } = require('./docx_helpers');
const { REFS, cite } = require('./refs');
const C = require('./model_content');

const OUT = process.argv[2] || 'Model_B_Technical_Description.docx';
const FIG = fs.readFileSync(path.join(__dirname, '..', 'figures', 'model_b_pipeline.png'));

const k = [];
k.push(new Paragraph({ children: [new TextRun({ text: 'Model B Technical Description', bold: true, font: FONT, size: 40 })], spacing: { after: 60 } }));
k.push(new Paragraph({ children: [new TextRun({ text: 'A time-domain convolutional network with channel and temporal attention for AIM-2 food intake detection', font: FONT, size: 24, color: '595959' })], spacing: { after: 60 } }));
k.push(new Paragraph({
  children: [new TextRun({ text: 'Caelan Dunlea  |  Randall Research Scholars Program, Phase II  |  September 2026', font: FONT, size: 18, color: '595959' })],
  spacing: { after: 240 }, border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: '000000', space: 4 } },
}));

k.push(H1('1 Purpose'));
k.push(P('This document describes Model B from the database query to the final decision, with no step left out, and gives the published precedent for each design choice. It is written to support a paper on the model, so every number below comes from the training code or from a build of the network, and every citation was checked against the original source.'));

k.push(H1('2 Two Corrections'));
k.push(P(['Two statements in earlier notes and slides were inaccurate, and both are corrected here. First, ', { b: 'the input is not band-passed' }, '. The preprocessing subtracts each window mean and scales each channel to unit variance, and no frequency filter is applied. Earlier descriptions called the input band-passed, which it is not. Whether a real filter helps is now an open experiment rather than a property of the model.']));
k.push(P(['Second, ', { b: 'the channel attention weighs learned features, not sensors' }, '. It rescales the 64 feature channels produced by the third convolution block. Those features mix all four sensors, so the attention does not directly choose between the accelerometer and the optical channel as the April slides suggested.']));

k.push(H1('3 End-to-End Pipeline'));
k.push(P('Figure 1 shows the full path from the database to a decision. Table 1 lists every step in order, grouped by stage. Shapes are written as time steps by channels.'));
k.push(image(FIG, 6.6, 3.18));
k.push(Note('Figure 1. Model B end to end. Green boxes prepare the data, blue boxes extract features, purple boxes apply attention and combine features, orange boxes produce the decision, and the grey dashed box is used only during training.'));

k.push(Caption('Table 1. Every step of Model B'));
const rows = [];
C.PIPELINE.forEach(g => g.rows.forEach((r, i) => rows.push([i === 0 ? g.stage : '', ...r])));
k.push(table(['Stage', 'Step', 'Operation', 'Parameters', 'Description'], rows, [1350, 560, 1850, 2620, 3700], { codeCols: [3], size: 15, boldFirst: true }));
k.push(Note(`Total parameters ${C.TOTALS.withChew} during training and ${C.TOTALS.inference} at prediction time, counted from a Keras build of the model.`));

k.push(P('Two properties of the network are worth stating explicitly because they follow from the numbers above rather than from any design document. The first two blocks reduce the 1,024 input samples to 64 time steps, so each step of the attention layers represents 0.125 s of signal. The receptive field of a single feature after the third block is 115 samples, about 0.9 s, which is close to one chewing cycle at the 0.94 to 2 Hz chewing rate reported for this sensor placement ' + cite('doulah2021') + '. The attention pooling then chooses among 64 such overlapping views of the window.'));

k.push(H1('4 Precedent for Each Design Choice'));
k.push(P('Table 2 pairs each component with the published work it follows. The status column separates choices that are established practice from those that are new on AIM data and from those that remain open. Two entries have no direct precedent in the literature reviewed, and they are marked as such rather than given a weak citation.'));
k.push(Caption('Table 2. Design choices and their precedent'));
k.push(table(['Component', 'Model B', 'Precedent', 'Status'], C.PRECEDENTS, [1650, 2550, 3980, 1900], { size: 15, boldFirst: true }));

k.push(H1('5 Changes Since the April Proposal'));
k.push(P('The model presented on 22 April set out the core idea, which is unchanged. Table 3 lists every parameter that has moved since then, so the approved design and the running design can be compared directly.'));
k.push(Caption('Table 3. The 4/22 proposal against Model B as it now runs'));
k.push(table(['Component', '4/22 proposal', 'Model B now'], C.CHANGES, [2400, 3500, 4180], { size: 16, boldFirst: true }));

k.push(H1('6 Current Performance'));
k.push(P('Table 4 reports Model B under both evaluation schemes on the 20 annotated sessions. Leave one subject out is the lab standard and is the scheme used from here on. The 4-fold numbers are kept for continuity with earlier meetings.'));
k.push(Caption('Table 4. Model B results pooled over all held-out windows'));
k.push(table(['Evaluation', 'F1', 'Accuracy', 'Balanced accuracy', 'Precision', 'Recall', 'False positive rate'], [
  ['Leave one subject out', '0.838', '0.890', '0.879', '0.828', '0.848', '8.9%'],
  ['4-fold by participant', '0.854', '0.905', '0.887', '0.876', '0.834', '6.0%'],
], [2280, 1060, 1260, 1600, 1260, 1060, 1560], { size: 16, boldFirst: true }));
k.push(Note('3,503 non-overlapping 8 s windows, 1,175 eating and 2,328 not eating.'));

k.push(H1('7 Reducing False Positives'));
k.push(P('False positives are the priority. Under leave one subject out, 207 of 2,328 non-eating windows were called eating, a false positive rate of 8.9%. Table 5 lists the planned experiments in order of cost, each with its precedent. The first requires no retraining and sets the baseline for the rest.'));
k.push(Caption('Table 5. Planned experiments to lower the false positive rate'));
k.push(table(['Approach', 'What it does', 'Precedent', 'Note'], C.FP_PLAN, [2100, 3700, 2200, 2080], { size: 15, boldFirst: true }));

k.push(H1('References'));
references(REFS).forEach(p => k.push(p));

const doc = new Document({ creator: 'Caelan Dunlea', title: 'Model B Technical Description', styles: STYLES,
  sections: [{ properties: { page: PAGE }, footers: footer(), children: k }] });
Packer.toBuffer(doc).then(b => { fs.writeFileSync(OUT, b); console.log('wrote', OUT); });

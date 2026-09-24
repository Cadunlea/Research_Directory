// Formal literature review in the layout of Ikram et al., Frontiers in
// Medicine 2025: title block, abstract, keywords, numbered sections, a
// state-of-the-art table, a performance table, a limitations table.
const fs = require('fs');
const path = require('path');
const { Document, Packer, Paragraph, TextRun, AlignmentType, BorderStyle, ShadingType } = require('docx');
const { P, H1, H2, Note, table, image, STYLES, PAGE, footer, references, FONT } = require('./docx_helpers');
const { REFS, cite } = require('./refs');

const OUT = process.argv[2] || 'Literature_Review_Formal.docx';
const FIG = fs.readFileSync(path.join(__dirname, '..', 'figures', 'model_b_pipeline.png'));

// Frontiers-style table caption, "TABLE 1  Caption text."
const TCap = (n, t) => new Paragraph({
  keepNext: true, spacing: { before: 220, after: 80 },
  children: [new TextRun({ text: `TABLE ${n}  `, bold: true, font: FONT, size: 18 }), new TextRun({ text: t, font: FONT, size: 18 })],
});
const FCap = (n, t) => new Paragraph({
  spacing: { before: 60, after: 220 },
  children: [new TextRun({ text: `FIGURE ${n}  `, bold: true, font: FONT, size: 17 }), new TextRun({ text: t, font: FONT, size: 17 })],
});

const k = [];
// ---------------------------------------------------------------- title block
k.push(new Paragraph({ children: [new TextRun({ text: 'Review', font: FONT, size: 18, color: '595959' })], spacing: { after: 120 } }));
k.push(new Paragraph({ children: [new TextRun({ text: 'Deep learning for wearable food intake detection with the Automatic Ingestion Monitor. A review of signal representations, architectures and evaluation', bold: true, font: FONT, size: 36 })], spacing: { after: 160, line: 400 } }));
k.push(new Paragraph({ children: [new TextRun({ text: 'Caelan Dunlea', font: FONT, size: 22 }), new TextRun({ text: ' 1', font: FONT, size: 22, superScript: true })], spacing: { after: 60 } }));
k.push(new Paragraph({ children: [new TextRun({ text: '1 Department of Electrical and Computer Engineering, The University of Alabama, Tuscaloosa, AL, United States', font: FONT, size: 17, color: '595959' })], spacing: { after: 240 } }));

// ---------------------------------------------------------------- abstract
const abstract = 'Wearable sensors on eyeglasses can detect food intake from the activity of the temporalis muscle, but the choice of signal representation, network architecture and evaluation protocol varies widely between studies and is rarely justified against the literature. This review examines work on the Automatic Ingestion Monitor (AIM) together with related eyeglass, smartwatch and in-ear systems, deep learning methods for sensor time series, and two transformer studies from outside the food intake field. Across the AIM line, the input has moved from handcrafted features to raw time series and to wavelet images, and the strongest reported result comes from convolutional networks applied directly to the raw accelerometer and optical signals. Attention mechanisms, multitask learning and self-supervised pretraining are well established for sensor data but have not been reported on AIM data. Transformers match or exceed convolutional networks on large inertial datasets, yet the published results rely on far more labelled data than a typical annotated eating study, and several use random splits that place the same subject in training and testing. The review concludes that a compact one-dimensional convolutional network on the time-domain signal, evaluated with leave-one-subject-out cross-validation, is the best supported starting point for AIM-2 data. It identifies the use of unannotated recordings, a controlled comparison of input representations, and the reduction of false positives as the directions with the most room for new contributions.';
k.push(new Paragraph({
  spacing: { after: 140, line: 276 }, alignment: AlignmentType.JUSTIFIED,
  shading: { fill: 'F2F5F7', type: ShadingType.CLEAR, color: 'auto' },
  border: { left: { style: BorderStyle.SINGLE, size: 12, color: '9CBBCB', space: 8 } },
  indent: { left: 200, right: 200 },
  children: [new TextRun({ text: abstract, font: FONT, size: 19 })],
}));
k.push(new Paragraph({ children: [new TextRun({ text: 'KEYWORDS', bold: true, font: FONT, size: 17 })], spacing: { before: 120, after: 40 } }));
k.push(new Paragraph({ children: [new TextRun({ text: 'food intake detection, wearable sensors, Automatic Ingestion Monitor, chewing, deep learning, time series classification, attention, leave-one-subject-out validation', font: FONT, size: 17 })], spacing: { after: 280 } }));

// ---------------------------------------------------------------- 1 introduction
k.push(H1('1 Introduction'));
k.push(P(`Self-reported dietary intake is known to be inaccurate and cannot capture the timing or structure of individual meals ${cite('doulah2021', 'diou2022')}. Wearable sensors offer an objective alternative. Eyeglass-mounted devices are particularly attractive because the temple of the frame rests over the temporalis muscle, which contracts with every chew ${cite('zhang2018', 'doulah2021')}. The Automatic Ingestion Monitor version 2 (AIM-2) combines a three-axis accelerometer with a sensor over the temporalis and a gaze-aligned camera, and it has been evaluated in both laboratory and free-living conditions ${cite('doulah2021', 'ghosh2022', 'ghosh2024')}.`));
k.push(P(`The algorithms applied to AIM data have changed considerably. The original work used handcrafted time and frequency features with a support vector machine ${cite('doulah2021')}. Later work applied end-to-end deep networks to the raw signals ${cite('ghosh2022')} and to wavelet images of the accelerometer ${cite('ghosh2024')}. Outside the AIM line, deep learning for sensor time series has developed quickly, with convolutional baselines ${cite('wang2017', 'fawaz2019')}, attention mechanisms ${cite('murahari2018', 'karim2019')}, transformers ${cite('shavit2021', 'luptakova2022')} and self-supervised pretraining ${cite('haresamudram2020', 'zerveas2021')}. These developments have not been reviewed together with the AIM literature.`));
k.push(P('The objective of this review is to establish what has already been done with this device family, which design choices are supported by prior work, and where a new model can contribute. The review focuses on three questions. The first is how the sensor signal should be represented. The second is which architectural components have evidence behind them. The third is how performance should be evaluated so that results reflect a person the model has never seen.'));

// ---------------------------------------------------------------- 2 literature review
k.push(H1('2 Literature review'));
k.push(H2('2.1 Sensors and placement'));
k.push(P(`Chewing can be sensed at the jaw, the ear, the throat or the temple. The AIM line senses the temporalis from the temple of the eyeglass frame, first with a flex sensor ${cite('doulah2021')} and then with an optical sensor alongside the accelerometer ${cite('ghosh2022')}. Zhang and Amft placed electromyography electrodes in personalised eyeglass frames over the same muscle and reported chewing detection near 94% in the laboratory, falling to about 80% in free living ${cite('zhang2018')}. Stankoski et al. used optical tracking sensors in smart glasses and reached an F1 score of 0.91 for chewing in controlled conditions, with a precision of 0.95 and a recall of 0.82 for eating segments in real life ${cite('stankoski2024')}. FitByte combined inertial and optical sensing on eyeglasses to reduce false positives and reported an F1 score of 0.89 across five unconstrained situations ${cite('bedri2020')}. Wrist-worn inertial sensors ${cite('kyritsis2021', 'diou2022')} and in-ear microphones ${cite('diou2022')} are the main alternatives and detect intake gestures and chewing sounds rather than muscle activity.`));

k.push(H2('2.2 From handcrafted features to learned representations'));
k.push(P(`Early food intake systems relied on handcrafted features and conventional classifiers ${cite('usman2021')}. AIM-2 computed 38 features per channel, 195 in total, and selected a subset with mutual information ranking followed by forward selection ${cite('doulah2021')}. Ghosh and Sazonov replaced the features with end-to-end networks on the raw four-channel signal in 8 s windows and found that a residual network reached an F1 score of 0.906 under leave-one-subject-out validation on 17 participants ${cite('ghosh2022')}. Ghosh et al. later converted 15 s accelerometer windows into continuous wavelet scalograms for a two-dimensional network and reported an F1 score of 0.776 for the sensor classifier alone in free living ${cite('ghosh2024')}. In time series classification more broadly, one-dimensional convolutional networks on z-normalised raw series form strong baselines ${cite('wang2017')}, and a large empirical comparison found residual and fully convolutional networks among the most reliable architectures ${cite('fawaz2019')}.`));

k.push(H2('2.3 Convolutional, recurrent and attention architectures'));
k.push(P(`Convolutional layers extract local temporal patterns, and deeper stacks improved food intake detection on AIM data ${cite('ghosh2022')}. Recurrent layers model longer sequences and are combined with convolution in DeepConvLSTM for activity recognition ${cite('ordonez2016')} and in end-to-end bite detection from smartwatches ${cite('kyritsis2021')}. Squeeze-and-excitation blocks learn a weight for each feature channel ${cite('hu2018')} and were added to fully convolutional networks for multivariate time series classification ${cite('karim2019')}. Attention over time was introduced to wearable activity recognition by Murahari and Plötz ${cite('murahari2018')}, and attention-based pooling replaces a fixed average with learned weights over the elements of a sequence ${cite('ilse2018')}. Multitask learning with a related auxiliary target can improve the shared representation ${cite('caruana1997')}. Chew counting from wearable sensors is established in the Sazonov laboratory ${cite('farooq2016')}, but no reviewed work uses the chew count as an auxiliary target inside a food intake classifier.`));

k.push(H2('2.4 Transformers for wearable sensor data'));
k.push(P(`Transformers replace convolution and recurrence with self-attention across the whole sequence. On inertial data, Shavit and Klein reported better accuracy and generalisation than convolutional and recurrent baselines across several datasets ${cite('shavit2021')}, and Dirgová Luptáková et al. found that a transformer matched a convolutional recurrent network on smartphone motion data ${cite('luptakova2022')}. Two transformer studies from outside the field were also examined. Yang et al. combined windowed attention with a frequency-derived period block and squeeze-and-excitation gating for radio signals and reached 95.0% accuracy on 100,000 signals, with augmentation giving a larger gain than the architecture itself ${cite('yang2026')}. Ikram et al. applied a transformer with principal component analysis to single ECG beats and reported 97.1% accuracy ${cite('ikram2025')}. ECG is relevant because its rhythm near 1 to 2 Hz resembles the chewing rhythm. Both studies, however, split data at the sample level rather than by subject, so their accuracies do not show how a transformer generalises to new people.`));

k.push(H2('2.5 Learning with limited annotation'));
k.push(P(`Annotating eating from video is slow, so labelled datasets are small. Self-supervised pretraining learns a representation from unlabelled signals before fine-tuning on the labelled set. Masked reconstruction improved activity recognition with small labelled sets on two of four benchmarks ${cite('haresamudram2020')}. A transformer pretrained on masked time series outperformed supervised methods even with very few labels ${cite('zerveas2021')}. In food intake monitoring, self-supervised training of an audio chewing detector reached an F1 score of 0.86 with a precision of 0.94 ${cite('diou2022')}. None of the reviewed AIM studies uses unannotated recordings in this way.`));

k.push(H2('2.6 Evaluation protocols'));
k.push(P(`Windows cut from the same meal are highly correlated. When they are split at random, the same person appears in training and testing and the reported accuracy overstates performance on a new person ${cite('saeb2017')}. Dehghani et al. showed that overlapping windows inflate results under subject-dependent evaluation but not under subject-independent evaluation, and recommended subject cross-validation ${cite('dehghani2019')}. All three AIM studies use leave-one-subject-out validation ${cite('doulah2021', 'ghosh2022', 'ghosh2024')}. Because eating is the minority class in most recordings, F1 is the preferred headline metric ${cite('diou2022')}, balanced accuracy corrects accuracy for class imbalance ${cite('brodersen2010')}, and the F1-optimal decision threshold can be chosen from validation data ${cite('lipton2014')}.`));

k.push(TCap(1, 'State-of-the-art methods for wearable food intake detection and related sensor classification.'));
k.push(table(['References', 'Techniques', 'Goals', 'Findings'], [
  [`Doulah et al. ${cite('doulah2021')}`, 'Accelerometer and flex sensor on eyeglasses, 38 handcrafted features, linear SVM', 'Detect intake to trigger a camera', 'F1 0.818 in the laboratory, 82.7% of eating episodes detected'],
  [`Ghosh and Sazonov ${cite('ghosh2022')}`, 'Raw accelerometer and optical signals, MLP, CNN, FCN, ResNet and Inception', 'Compare end-to-end deep networks', 'ResNet F1 0.906 under leave-one-subject-out validation'],
  [`Ghosh et al. ${cite('ghosh2024')}`, 'Accelerometer wavelet scalogram with a 2D CNN, image detection, random forest fusion', 'Reduce false positives in free living', 'Sensor F1 0.776, fused episode F1 0.808'],
  [`Stankoski et al. ${cite('stankoski2024')}`, 'Optical tracking sensors in smart glasses, CNN and ConvLSTM', 'Chewing detection in the laboratory and real life', 'F1 0.91 in the laboratory, precision 0.95 in real life'],
  [`Zhang and Amft ${cite('zhang2018')}`, 'EMG electrodes in eyeglass frames over the temporalis', 'Chewing and eating in free living', 'About 94% in the laboratory, about 80% in free living'],
  [`Bedri et al. ${cite('bedri2020')}`, 'Inertial and optical sensing on eyeglasses, multimodal fusion', 'Eating and drinking in unconstrained settings', 'F1 0.89 across five situations'],
  [`Kyritsis et al. ${cite('kyritsis2021')}`, 'Smartwatch inertial data, end-to-end CNN with LSTM', 'Bite and meal detection in the wild', 'Bite F1 0.923 in meals'],
  [`Diou et al. ${cite('diou2022')}`, 'Review of smartwatch and in-ear audio methods', 'Lessons for free-living monitoring', 'Self-supervised chewing F1 0.86'],
  [`Farooq and Sazonov ${cite('farooq2016')}`, 'Piezoelectric sensor, automatic chew counting', 'Chew count and chewing rate', 'Mean absolute error 10.4% to 15.0%'],
  [`Karim et al. ${cite('karim2019')}`, 'FCN with squeeze-and-excitation and LSTM', 'Multivariate time series classification', 'Improved accuracy across benchmarks'],
  [`Murahari and Plötz ${cite('murahari2018')}`, 'Attention layers added to DeepConvLSTM', 'Wearable activity recognition', 'Higher performance on benchmark datasets'],
  [`Shavit and Klein ${cite('shavit2021')}`, 'Transformer encoder on inertial data', 'Activity and device location recognition', 'Better accuracy and generalisation than CNN and LSTM'],
  [`Zerveas et al. ${cite('zerveas2021')}`, 'Transformer with masked pretraining', 'Multivariate time series classification', 'Pretraining helps even with few labels'],
  [`Haresamudram et al. ${cite('haresamudram2020')}`, 'Masked reconstruction pretraining', 'Activity recognition with few labels', 'Gains on two of four datasets'],
  [`Yang et al. ${cite('yang2026')}`, 'Windowed transformer, FFT period block, squeeze-and-excitation', 'Radio signal classification', '95.0% accuracy, augmentation +2.8 points'],
  [`Ikram et al. ${cite('ikram2025')}`, 'Transformer with PCA and feature selection', 'ECG arrhythmia classification', '97.1% accuracy with a sample-level split'],
], [2150, 3150, 2250, 2530], { size: 15, boldFirst: true }));

// ---------------------------------------------------------------- 3 methods
k.push(H1('3 Materials and methods'));
k.push(P(`This is a narrative review. It began from seven papers selected with the supervising laboratory, covering the AIM device ${cite('doulah2021', 'ghosh2022', 'ghosh2024')}, food intake monitoring more broadly ${cite('diou2022', 'usman2021')} and two transformer studies from other signal domains ${cite('yang2026', 'ikram2025')}. It was extended by targeted searches of the ACM Digital Library, IEEE Xplore, PubMed, MDPI and arXiv for eyeglass-based chewing detection, attention and multitask learning for sensor time series, transformers for wearable data, self-supervised learning for activity recognition, and evaluation protocols for wearable studies. Works were included when they reported a method and a quantitative result on sensor signals, or when they are the original source of a component used in the model described in Section 5. Every citation was checked against the publisher, PubMed, arXiv or DBLP. In total, 38 works are cited.`));

// ---------------------------------------------------------------- 4 results
k.push(H1('4 Results of the review'));
k.push(P('Table 2 collects the results reported for food intake detection from temporalis and related sensors, together with the model developed in this project. The figures come from different datasets, labels and validation schemes and are therefore indicative rather than directly comparable.'));
k.push(TCap(2, 'Reported food intake detection performance.'));
k.push(table(['Study', 'Input representation', 'Model', 'Validation', 'Reported result'], [
  [`Doulah et al. ${cite('doulah2021')}`, 'Handcrafted features, 10 s', 'Linear SVM', 'LOSO, 30 participants', 'F1 0.818'],
  [`Ghosh and Sazonov ${cite('ghosh2022')}`, 'Raw time series, 8 s', 'ResNet', 'LOSO, 17 participants', 'F1 0.906'],
  [`Ghosh et al. ${cite('ghosh2024')}`, 'Wavelet scalogram, 15 s', '2D CNN', 'LOSO, 30 participants', 'F1 0.776'],
  [`Stankoski et al. ${cite('stankoski2024')}`, 'Optical sensor signals', 'ConvLSTM', 'Laboratory and real life', 'F1 0.91'],
  [`Bedri et al. ${cite('bedri2020')}`, 'Multimodal eyeglass signals', 'Fusion model', 'Five situations', 'F1 0.89'],
  ['This project, Model B', 'Z-scored time series, 8 s', '1D CNN with attention', 'LOSO, 20 participants', 'F1 0.838'],
], [2300, 2250, 1900, 1930, 1700], { size: 15, boldFirst: true }));
k.push(P('Three findings emerge. First, the input representation matters more than the classifier. Within the AIM line the raw time series outperformed both handcrafted features and wavelet images, and in this project replacing the time series with its FFT magnitude lowered F1 by 0.087 with the architecture held fixed. Second, attention and multitask components are well established for sensor data but untested on AIM data. Third, every AIM study uses subject-independent validation, while both transformer studies from other domains do not, which limits what their accuracies say about new people.'));

// ---------------------------------------------------------------- 5 implications
k.push(H1('5 Implications for the AIM-2 model'));
k.push(P(`The model developed in this project, Model B, was designed from these findings. Figure 1 shows it end to end. It takes 8 s windows of the four channels at 128 Hz, following ${cite('ghosh2022')}, and labels a window as eating when more than half of it is annotated as eating, following ${cite('doulah2021')}. Each channel is z-scored within the window ${cite('wang2017', 'fawaz2019')}. Three one-dimensional convolution blocks ${cite('wang2017', 'ioffe2015', 'springenberg2015')} reduce the window to 64 time steps. A squeeze-and-excitation block reweights the learned channels ${cite('hu2018', 'karim2019')}, and attention pooling weights the time steps ${cite('murahari2018', 'ilse2018')}. An auxiliary head predicts the chew count during training only ${cite('caruana1997', 'farooq2016')}. Evaluation is leave-one-subject-out ${cite('saeb2017', 'dehghani2019')} with the threshold chosen on held-out validation participants ${cite('lipton2014')}.`));
k.push(image(FIG, 6.4, 3.08));
k.push(FCap(1, 'Model B from the database to the decision. Shapes are time steps by channels. The dashed chew head is used only in training.'));
k.push(P('Under leave-one-subject-out validation on 20 participants, Model B reached an F1 score of 0.838, an accuracy of 0.890 and a precision of 0.828. The component analysis supports the review. The time-domain input accounts for nearly all of the gain, while the attention and the auxiliary head each changed F1 by less than 0.01, which cannot be separated from participant-to-participant variation at this sample size.'));

// ---------------------------------------------------------------- 6 discussion
k.push(H1('6 Discussion'));
k.push(P(`The literature supports a compact convolutional network on the time-domain signal as the starting point for AIM-2 data. This is where the strongest AIM result was obtained ${cite('ghosh2022')}, it is a strong general baseline for time series ${cite('wang2017', 'fawaz2019')}, and it is consistent with the controlled comparison in this project. The case for a transformer is weaker at the current scale. The inertial transformer studies report gains ${cite('shavit2021', 'luptakova2022')}, but on far larger labelled datasets, and the two studies from other domains do not separate subjects ${cite('yang2026', 'ikram2025')}. A transformer is still worth testing, because lowering the sampling rate from 128 Hz to 32 Hz shortens an 8 s window from 1,024 to 256 samples, and activity recognition accuracy holds at such rates ${cite('yamane2025')}. The test should use the same leave-one-subject-out protocol so that it is a fair comparison with the convolutional model.`));
k.push(P(`The clearest gaps concern data and false positives rather than architecture. The database holds 61 sensor sessions, of which 20 were annotated at the time of this analysis, and self-supervised pretraining on the remaining recordings has helped comparable sensor tasks ${cite('haresamudram2020', 'zerveas2021', 'diou2022')}. False positives are the main practical weakness of sensor-based intake detection. The AIM literature reports them from chewing gum ${cite('ghosh2024')}, and multimodal eyeglass systems were designed specifically to reduce them ${cite('bedri2020')}. Adding non-eating recordings from other studies as negative examples, choosing the decision threshold for a target false positive rate ${cite('lipton2014')}, and weighting false positives more heavily in the loss ${cite('lin2017', 'he2009')} are the most direct routes.`));

// ---------------------------------------------------------------- 7 limitations
k.push(H1('7 Limitations'));
k.push(P('This review is narrative rather than systematic, and it began from a small set of papers chosen by the laboratory. Reported results come from different devices, labels and protocols and cannot be ranked directly. Several of the most relevant recent works were identified through targeted searches and may not represent everything published on eyeglass-based detection. Table 3 summarises the limitations of each family of approaches and what they imply for this project.'));
k.push(TCap(3, 'Limitations of the approaches reviewed.'));
k.push(table(['Approach', 'Limitation reported or observed', 'Implication for this project'], [
  ['Handcrafted features', `Depend on feature design and selection data ${cite('doulah2021', 'usman2021')}`, 'Learn the representation from the signal'],
  ['Frequency and wavelet inputs', `Magnitude discards timing, and scalograms were resized to images ${cite('ghosh2024')}`, 'Use the time series, confirmed by ablation'],
  ['Image fusion', `Requires a camera and raises privacy concerns ${cite('doulah2021', 'ghosh2024')}`, 'Keep a sensor-only model as the baseline'],
  ['Transformers', `Need large labelled sets, and the cross-domain studies use sample-level splits ${cite('yang2026', 'ikram2025')}`, 'Test only under leave-one-subject-out, at 32 Hz'],
  ['Record-wise evaluation', `Overstates accuracy on new people ${cite('saeb2017', 'dehghani2019')}`, 'Leave-one-subject-out throughout'],
  ['Small annotated datasets', `Limit supervised learning ${cite('diou2022', 'haresamudram2020')}`, 'Pretrain on unannotated sessions'],
  ['Laboratory to free-living gap', `Performance falls outside the laboratory ${cite('zhang2018', 'stankoski2024')}`, 'Add negative examples from other recordings'],
], [2200, 4280, 3600], { size: 15, boldFirst: true }));

// ---------------------------------------------------------------- 8 future work
k.push(H1('8 Future work'));
k.push(P('Four directions follow from the review. The first is a controlled comparison of the residual network from the AIM literature with Model B on the same data and folds. The second is a study of sampling rate, window length and filtering. The third is self-supervised pretraining on the unannotated sessions. The fourth is a set of experiments aimed directly at the false positive rate, including negative examples from other studies and a threshold chosen for a target false positive rate.'));
k.push(H1('References'));
references(REFS).forEach(p => k.push(p));

const doc = new Document({ creator: 'Caelan Dunlea', title: 'Deep learning for wearable food intake detection', styles: STYLES,
  sections: [{ properties: { page: PAGE }, footers: footer(), children: k }] });
Packer.toBuffer(doc).then(b => { fs.writeFileSync(OUT, b); console.log('wrote', OUT); });

// Single source for everything the Model B slides and document say. Numbers
// come from the code (train_food_intake_v2.py, ets_data.py, ets_train.py) and
// from a Keras build of the model (layer shapes and parameter counts).
const { cite } = require('./refs');

// ---------------------------------------------------------------------------
// End-to-end steps, grouped by stage. [step, operation, parameters, what it does]
const PIPELINE = [
  { stage: 'Data', rows: [
    ['1', 'Load annotations', 'study_data.numeric_data, CHGT, BOGT, BIGT at 10 Hz', 'Reads chew, bout and bite labels for each session, one full day per record. A value of -1 means not annotated.'],
    ['2', 'Define eating per sample', 'eating = BOGT > 0 or BIGT > 0', 'A 10 Hz sample is eating when it falls in a chewing bout or a bite. Samples missing in both are left as -1.'],
    ['3', 'Find the annotated span', 'first to last non-missing sample', 'Only the video-annotated part of the day is used. Unannotated time is never treated as not eating.'],
    ['4', 'Load sensor packets', 'aim_raw_data XYZO, 4 x 1,022 samples per packet', 'Accelerometer X, Y, Z and optical at 128 Hz. data_timestamp is read as the packet start, as verified by the alignment check.'],
    ['5', 'Place samples on a 128 Hz grid', 'missing = -1', 'Each packet is written at its timestamp. The 2 samples between 1,022-sample packets stay -1, which gives about 99.8% coverage.'],
  ]},
  { stage: 'Windows', rows: [
    ['6', 'Cut windows', '8 s = 1,024 samples', 'Training uses a 4 s hop (50% overlap). Evaluation uses only the non-overlapping 8 s grid.'],
    ['7', 'Quality filter', 'keep if >= 50% labelled and <= 50% sensor missing', 'Drops windows that are mostly unannotated or mostly without sensor data.'],
    ['8', 'Window label', 'eating if > 50% of labelled samples are eating', 'One binary label per window, the same majority rule as the AIM-2 paper.'],
    ['9', 'Chew target', 'count of CHGT marks in the window', 'Used only by the auxiliary head during training.'],
  ]},
  { stage: 'Preprocessing', rows: [
    ['10', 'Per-window z-score', 'each channel, each window', 'Missing samples are excluded, the rest are shifted to mean 0 and scaled to standard deviation 1, and missing samples are then set to 0. No frequency filter is applied.'],
  ]},
  { stage: 'Network', rows: [
    ['11', 'Input', '(1024, 4)', '8 s of the four channels.'],
    ['12', 'Conv block 1', 'Conv1D 32 filters, kernel 9, stride 2 > BatchNorm > ReLU > MaxPool 2', 'Output (256, 32). Learns short local patterns and cuts the sequence by 4. 1,312 parameters.'],
    ['13', 'Conv block 2', 'Conv1D 64 filters, kernel 9, stride 2 > BatchNorm > ReLU > MaxPool 2', 'Output (64, 64). Longer patterns. 18,752 parameters.'],
    ['14', 'Conv block 3', 'Conv1D 64 filters, kernel 5, stride 1 > BatchNorm > ReLU', 'Output (64, 64). Each of the 64 time steps covers 0.125 s, and each feature sees about 0.9 s of signal, roughly one chew cycle. 20,800 parameters.'],
    ['15', 'Channel attention (squeeze and excitation)', 'average over time > Dense 16 ReLU > Dense 64 sigmoid > multiply', 'Scores each of the 64 learned feature channels between 0 and 1 and rescales them. It acts on learned features, not directly on the 4 sensors. 2,128 parameters.'],
    ['16', 'Attention pooling over time', 'Conv1D 1 x 1 > softmax over 64 steps > weighted sum', 'Gives each time step a weight and sums the sequence into one 64-value vector, so a short bite is not averaged away. 65 parameters.'],
    ['17', 'Shared dense layer', 'Dense 64 ReLU > Dropout 0.3', 'Combines the pooled features. Dropout reduces overfitting. 4,160 parameters.'],
    ['18', 'Food head', 'Dense 1 sigmoid', 'Probability that the window is eating. The only output used for prediction. 65 parameters.'],
    ['19', 'Chew head (training only)', 'Dense 32 ReLU > Dense 1 softplus', 'Predicts the scaled chew count. Discarded after training. 2,113 parameters.'],
  ]},
  { stage: 'Training', rows: [
    ['20', 'Loss', 'BCE(food) + 0.2 x MSE(chews)', 'Chew counts are divided by the 99th percentile of the training participants so both losses sit on a similar scale.'],
    ['21', 'Class balance', 'weight = total / (2 x class count)', 'Computed per fold, so neither class can be ignored.'],
    ['22', 'Optimiser', 'Adam, learning rate 0.001, batch 64', 'Standard defaults.'],
    ['23', 'Stopping and schedule', 'max 80 epochs, early stop patience 12, halve LR after 4 flat epochs, min 1e-5', 'Monitors validation loss and restores the best weights.'],
  ]},
  { stage: 'Evaluation', rows: [
    ['24', 'Cross-validation', 'leave one subject out, 20 folds', 'Each participant is tested once by a model that never saw them.'],
    ['25', 'Inner validation', '2 training participants per fold, rotated', 'Used for early stopping and the threshold. Never the test participant.'],
    ['26', 'Decision threshold', 'grid 0.05 to 0.95, maximise F1, centre of the plateau', 'Chosen on the inner validation participants, then applied unchanged to the test participant.'],
    ['27', 'Metrics', 'F1, balanced accuracy, precision, recall, specificity', 'Pooled over all held-out windows and reported per participant.'],
  ]},
];

const TOTALS = { withChew: '49,395', inference: '47,282' };

// ---------------------------------------------------------------------------
// Precedent for every design choice. [component, what Model B does, precedent, evidence or status]
const PRECEDENTS = [
  ['Sensors', 'Accelerometer X, Y, Z and optical over the temporalis', `Same four AIM-2.1 channels ${cite('ghosh2022')}. Optical sensing on smart glasses reached F1 0.91 ${cite('stankoski2024')}. Temporalis activity reflects chewing ${cite('zhang2018', 'doulah2021')}`, 'Established'],
  ['Sampling rate', '128 Hz, the device rate', `AIM-2 records at 128 Hz ${cite('doulah2021', 'ghosh2022')}. Accuracy holds down to about 10 to 25 Hz in activity recognition ${cite('yamane2025')}`, 'Established. 32 Hz test planned'],
  ['Window length', '8 s', `8 s epochs ${cite('ghosh2022')}. Others used 10 s ${cite('doulah2021')}, 15 s ${cite('ghosh2024')} and 5 s ${cite('diou2022')}. Window size changes results ${cite('banos2014')}`, 'Established. 4 and 16 s sweep planned'],
  ['Window label', 'Eating if more than 50% of the window is eating', `Same majority rule ${cite('doulah2021')}`, 'Established'],
  ['Annotated time only', 'Unannotated time is dropped, never labelled not eating', `Labels come only from ground truth ${cite('doulah2021', 'ghosh2022')}`, 'Established'],
  ['Missing samples', 'Excluded from the z-score, then set to 0', 'No direct precedent found', 'Design choice, covered by pipeline tests'],
  ['Normalisation', 'Z-score per window and per channel', `Normalisation for inter-subject variation ${cite('doulah2021')}. Z-normalised series are standard in time series classification ${cite('fawaz2019', 'wang2017')}`, 'Established'],
  ['Filtering', 'None. The input is not band-passed', `Raw input without filtering ${cite('ghosh2022')}. AIM-2 used a 0.1 Hz high-pass and 3 Hz low-pass ${cite('doulah2021')}`, 'Open. Filter test planned'],
  ['Input representation', 'Time series, not FFT magnitude', `Raw time series with CNNs ${cite('ghosh2022', 'wang2017', 'fawaz2019', 'kyritsis2021')}`, 'Own ablation. FFT input costs 0.087 F1'],
  ['Convolution trunk', 'Three 1D conv blocks with BatchNorm and ReLU', `1D CNNs for time series ${cite('wang2017', 'fawaz2019')}. Deeper conv stacks helped on AIM ${cite('ghosh2022')}. BatchNorm ${cite('ioffe2015')}`, 'Established'],
  ['Strided convolution', 'Stride 2 plus max pooling to shorten the sequence', `Strided convolution as downsampling ${cite('springenberg2015')}`, 'Established'],
  ['Channel attention', 'Squeeze and excitation on the 64 learned channels', `Introduced in ${cite('hu2018')}. Used in time series classification ${cite('karim2019')} and radio signals ${cite('yang2026')}`, 'Standard. Own ablation within noise'],
  ['Attention pooling', 'Softmax weights over 64 time steps', `Attention over time in activity recognition ${cite('murahari2018')}. Attention-based pooling ${cite('ilse2018')}. AIM work used global average pooling ${cite('ghosh2022')}`, 'New on AIM. Within noise'],
  ['Dropout', '0.3 after the shared dense layer', `${cite('srivastava2014')}`, 'Established'],
  ['Chew count head', 'Auxiliary regression, weight 0.2, training only', `Multitask learning ${cite('caruana1997')}. Chew counting from sensors ${cite('farooq2016')}`, 'New on AIM. Within noise'],
  ['Class weighting', 'Per-fold inverse class frequency', `Imbalanced learning ${cite('he2009')}. Cost-sensitive training on AIM ${cite('ghosh2024')}`, 'Established'],
  ['Optimiser and stopping', 'Adam 0.001, early stopping on validation loss', `${cite('kingma2015', 'prechelt1998')}`, 'Established'],
  ['Overlapping windows', '50% overlap for training, non-overlapping for testing', `Overlap inflates results only under subject-dependent evaluation, and subject cross-validation is recommended ${cite('dehghani2019')}`, 'Own ablation. +0.025 F1'],
  ['Cross-validation', 'Leave one subject out', `Used in ${cite('doulah2021', 'ghosh2022', 'ghosh2024', 'diou2022')}. Subject-wise splits are required for a valid estimate ${cite('saeb2017')}`, 'Established. Lab standard'],
  ['Threshold', 'Chosen on inner validation participants to maximise F1', `${cite('lipton2014')}`, 'Established'],
  ['Metrics', 'F1 and balanced accuracy lead', `F1 as the headline metric ${cite('diou2022', 'ghosh2022')}. Balanced accuracy ${cite('brodersen2010')}`, 'Established'],
  ['Alignment check', 'Cross-correlation of chewing energy with labels', 'No direct precedent found', 'Own method, validated on synthetic offsets'],
];

// ---------------------------------------------------------------------------
// What changed between the 4/22 proposal and Model B as it now runs.
const CHANGES = [
  ['Participants', '14', '20 (61 sessions available)'],
  ['Window', '10 s, 1,280 samples', '8 s, 1,024 samples'],
  ['Filtering', '0.1 Hz high-pass', 'None, z-score only'],
  ['Conv layers', '2 (16 and 32 filters, kernels 5 and 9)', '3 (32, 64, 64 filters, kernels 9, 9, 5), strided'],
  ['Channel attention', 'Reduction ratio 8', 'Squeeze and excitation, reduction 4'],
  ['Dense and dropout', 'Dense 16, dropout 0.2', 'Dense 64, dropout 0.3'],
  ['Outputs', 'Food only', 'Food plus chew count (training only)'],
  ['Batch, epochs, patience', '32, 30, 5', '64, 80, 12'],
  ['Training windows', 'Non-overlapping', '50% overlap, tested non-overlapping'],
  ['Validation', '4 folds plus 2 held-out participants', 'Leave one subject out'],
];

// ---------------------------------------------------------------------------
// False positives. Current numbers and planned experiments with precedent.
const FP_NOW = [
  ['4-fold', '139', '2,328', '6.0%', '0.876'],
  ['Leave one subject out', '207', '2,328', '8.9%', '0.828'],
];
const FP_PLAN = [
  ['Precision-targeted threshold', 'Pick the threshold on validation that keeps the false positive rate under a set target, for example 5%', cite('lipton2014'), 'No retraining. Costs some recall'],
  ['Negative examples from other datasets', 'Add non-eating recordings from other studies into ETS as extra negatives, split by participant', `${cite('ghosh2024')} reports false positives from chewing gum`, 'Agreed at the 9/23 meeting'],
  ['Cost-sensitive loss', 'Weight a false positive more than a false negative, or use focal loss', cite('ghosh2024', 'lin2017'), 'One parameter to sweep'],
  ['Signal-quality gating', 'Use accelerometer-only decisions when the optical signal is weak', cite('doulah2021'), 'Two-stage rule from AIM-2'],
  ['Augmentation', 'Noise, time masking and shifts during training', cite('yang2026'), 'Largest single gain in that paper'],
];

module.exports = { PIPELINE, TOTALS, PRECEDENTS, CHANGES, FP_NOW, FP_PLAN };

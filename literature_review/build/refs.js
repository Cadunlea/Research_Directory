// One numbered reference list shared by the slides and both documents, so a
// number means the same paper everywhere. Every entry was checked against the
// publisher, PubMed, arXiv or DBLP before it was added.
const REFS = [
  ['doulah2021', 'A. Doulah, T. Ghosh, D. Hossain, M. H. Imtiaz, E. Sazonov, "Automatic Ingestion Monitor Version 2 - A novel wearable device for automatic food intake detection and passive capture of food images," IEEE Journal of Biomedical and Health Informatics, vol. 25, no. 2, pp. 568-576, 2021. doi 10.1109/JBHI.2020.2995473'],
  ['ghosh2022', 'T. Ghosh, E. Sazonov, "A comparative study of deep learning algorithms for detecting food intake," Proc. 44th Annual Int. Conf. IEEE EMBC, Glasgow, 2022, pp. 2993-2996. doi 10.1109/EMBC48229.2022.9871278'],
  ['ghosh2024', 'T. Ghosh, Y. Han, V. Raju, D. Hossain, M. A. McCrory, J. Higgins, C. Boushey, E. J. Delp, E. Sazonov, "Integrated image and sensor-based food intake detection in free-living," Scientific Reports, vol. 14, 1665, 2024. doi 10.1038/s41598-024-51687-3'],
  ['diou2022', 'C. Diou, K. Kyritsis, V. Papapanagiotou, I. Sarafis, "Intake monitoring in free-living conditions. Overview and lessons we have learned," Appetite, 2022. doi 10.1016/j.appet.2022.106096'],
  ['usman2021', 'M. Usman, H. Chen, "Recent trends in food intake monitoring using wearable sensors," arXiv 2101.01378, 2021.'],
  ['yang2026', 'J. Yang, Y. Wang, M. Yang, H. Meng, "Signal classification based on multi-scale time-frequency transformer," Physical Communication, vol. 75, 103022, 2026. doi 10.1016/j.phycom.2026.103022'],
  ['ikram2025', 'S. Ikram et al., "Transformer-based ECG classification for early detection of cardiac arrhythmias," Frontiers in Medicine, vol. 12, 1600855, 2025. doi 10.3389/fmed.2025.1600855'],
  ['stankoski2024', 'S. Stankoski, I. Kiprijanovska, M. Gjoreski, F. Panchevski, B. Sazdov, B. Sofronievski, A. Cleal, M. Fatoorechi, C. Nduka, H. Gjoreski, "Controlled and real-life investigation of optical tracking sensors in smart glasses for monitoring eating behavior using deep learning. Cross-sectional study," JMIR mHealth and uHealth, vol. 12, e59469, 2024.'],
  ['zhang2018', 'R. Zhang, O. Amft, "Monitoring chewing and eating in free-living using smart eyeglasses," IEEE Journal of Biomedical and Health Informatics, vol. 22, no. 1, pp. 23-32, 2018. doi 10.1109/JBHI.2017.2698523'],
  ['bedri2020', 'A. Bedri, D. Li, R. Khurana, K. Bhuwalka, M. Goel, "FitByte. Automatic diet monitoring in unconstrained situations using multimodal sensing on eyeglasses," Proc. CHI Conference on Human Factors in Computing Systems, 2020. doi 10.1145/3313831.3376869'],
  ['farooq2016', 'M. Farooq, E. Sazonov, "Automatic measurement of chew count and chewing rate during food intake," Electronics, vol. 5, no. 4, 62, 2016. doi 10.3390/electronics5040062'],
  ['kyritsis2021', 'K. Kyritsis, C. Diou, A. Delopoulos, "A data driven end-to-end approach for in-the-wild monitoring of eating behavior using smartwatches," IEEE Journal of Biomedical and Health Informatics, vol. 25, no. 1, pp. 22-34, 2021.'],
  ['bahador2021', 'N. Bahador, D. Ferreira, S. Tamminen, J. Kortelainen, "Deep learning-based multimodal data fusion. Case study in food intake episodes detection using wearable sensors," JMIR mHealth and uHealth, vol. 9, no. 1, e21926, 2021.'],
  ['fawaz2019', 'H. Ismail Fawaz, G. Forestier, J. Weber, L. Idoumghar, P.-A. Muller, "Deep learning for time series classification. A review," Data Mining and Knowledge Discovery, vol. 33, pp. 917-963, 2019. doi 10.1007/s10618-019-00619-1'],
  ['wang2017', 'Z. Wang, W. Yan, T. Oates, "Time series classification from scratch with deep neural networks. A strong baseline," Proc. Int. Joint Conf. Neural Networks (IJCNN), 2017, pp. 1578-1585.'],
  ['karim2019', 'F. Karim, S. Majumdar, H. Darabi, S. Harford, "Multivariate LSTM-FCNs for time series classification," Neural Networks, vol. 116, pp. 237-245, 2019. doi 10.1016/j.neunet.2019.04.014'],
  ['hu2018', 'J. Hu, L. Shen, G. Sun, "Squeeze-and-excitation networks," Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2018, pp. 7132-7141.'],
  ['murahari2018', 'V. S. Murahari, T. Plötz, "On attention models for human activity recognition," Proc. ACM Int. Symposium on Wearable Computers (ISWC), 2018. doi 10.1145/3267242.3267287'],
  ['ilse2018', 'M. Ilse, J. Tomczak, M. Welling, "Attention-based deep multiple instance learning," Proc. Int. Conf. Machine Learning (ICML), 2018.'],
  ['caruana1997', 'R. Caruana, "Multitask learning," Machine Learning, vol. 28, pp. 41-75, 1997.'],
  ['dehghani2019', 'A. Dehghani, O. Sarbishei, T. Glatard, E. Shihab, "A quantitative comparison of overlapping and non-overlapping sliding windows for human activity recognition using inertial sensors," Sensors, vol. 19, no. 22, 5026, 2019. doi 10.3390/s19225026'],
  ['saeb2017', 'S. Saeb, L. Lonini, A. Jayaraman, D. C. Mohr, K. P. Kording, "The need to approximate the use-case in clinical machine learning," GigaScience, vol. 6, no. 5, gix019, 2017.'],
  ['lipton2014', 'Z. C. Lipton, C. Elkan, B. Narayanaswamy, "Optimal thresholding of classifiers to maximize F1 measure," Proc. ECML PKDD, LNCS vol. 8725, pp. 225-239, 2014.'],
  ['haresamudram2020', 'H. Haresamudram, A. Beedu, V. Agrawal, P. L. Grady, I. Essa, J. Hoffman, T. Plötz, "Masked reconstruction based self-supervision for human activity recognition," Proc. ACM ISWC, 2020. doi 10.1145/3410531.3414306'],
  ['zerveas2021', 'G. Zerveas, S. Jayaraman, D. Patel, A. Bhamidipaty, C. Eickhoff, "A transformer-based framework for multivariate time series representation learning," Proc. ACM SIGKDD Conf. Knowledge Discovery and Data Mining, 2021. doi 10.1145/3447548.3467401'],
  ['shavit2021', 'Y. Shavit, I. Klein, "Boosting inertial-based human activity recognition with transformers," IEEE Access, vol. 9, pp. 53540-53547, 2021. doi 10.1109/ACCESS.2021.3070646'],
  ['luptakova2022', 'I. Dirgová Luptáková, M. Kubovčík, J. Pospíchal, "Wearable sensor-based human activity recognition with transformer model," Sensors, vol. 22, no. 5, 1911, 2022. doi 10.3390/s22051911'],
  ['yamane2025', 'T. Yamane, M. Kimura, M. Morita, "Effects of sampling frequency on human activity recognition with machine learning aiming at clinical applications," Sensors, vol. 25, no. 12, 3780, 2025.'],
  ['ioffe2015', 'S. Ioffe, C. Szegedy, "Batch normalization. Accelerating deep network training by reducing internal covariate shift," Proc. ICML, 2015.'],
  ['kingma2015', 'D. P. Kingma, J. Ba, "Adam. A method for stochastic optimization," Proc. Int. Conf. Learning Representations (ICLR), 2015.'],
  ['srivastava2014', 'N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, R. Salakhutdinov, "Dropout. A simple way to prevent neural networks from overfitting," Journal of Machine Learning Research, vol. 15, pp. 1929-1958, 2014.'],
  ['lin2017', 'T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár, "Focal loss for dense object detection," Proc. IEEE Int. Conf. Computer Vision (ICCV), 2017, pp. 2980-2988.'],
  ['he2009', 'H. He, E. A. Garcia, "Learning from imbalanced data," IEEE Transactions on Knowledge and Data Engineering, vol. 21, no. 9, pp. 1263-1284, 2009.'],
  ['prechelt1998', 'L. Prechelt, "Early stopping, but when?," in Neural Networks. Tricks of the Trade, LNCS vol. 1524, Springer, 1998, pp. 55-69.'],
  ['ordonez2016', 'F. J. Ordóñez, D. Roggen, "Deep convolutional and LSTM recurrent neural networks for multimodal wearable activity recognition," Sensors, vol. 16, no. 1, 115, 2016.'],
  ['springenberg2015', 'J. T. Springenberg, A. Dosovitskiy, T. Brox, M. Riedmiller, "Striving for simplicity. The all convolutional net," ICLR Workshop, 2015.'],
  ['brodersen2010', 'K. H. Brodersen, C. S. Ong, K. E. Stephan, J. M. Buhmann, "The balanced accuracy and its posterior distribution," Proc. Int. Conf. Pattern Recognition (ICPR), 2010, pp. 3121-3124.'],
  ['banos2014', 'O. Banos, J.-M. Galvez, M. Damas, H. Pomares, I. Rojas, "Window size impact in human activity recognition," Sensors, vol. 14, no. 4, pp. 6474-6499, 2014.'],
];

const INDEX = Object.fromEntries(REFS.map(([k], i) => [k, i + 1]));
// cite('ghosh2022','doulah2021') -> "[2, 1]"
function cite(...keys) {
  return '[' + keys.map(k => {
    if (!(k in INDEX)) throw new Error('unknown reference ' + k);
    return INDEX[k];
  }).join(', ') + ']';
}
module.exports = { REFS, cite, INDEX };

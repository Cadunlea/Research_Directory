"""Methods section (II.A-E) as drafted on 11/10/26, with Table I and the
renumbered reference list, as a standalone .docx. Highlighted sentences are
drafts not yet revised by the author."""
from pathlib import Path
from docx import Document
from docx.enum.text import WD_COLOR_INDEX, WD_ALIGN_PARAGRAPH
from docx.shared import Pt

OUT = Path(__file__).with_name("methods_draft_11_10.docx")
doc = Document()
style = doc.styles["Normal"]
style.font.name, style.font.size = "Times New Roman", Pt(10)
style.paragraph_format.space_after = Pt(4)

def heading(text, size=10, italic=False, center=False, smallcaps=False):
    p = doc.add_paragraph()
    r = p.add_run(text); r.italic = italic; r.font.size = Pt(size); r.font.small_caps = smallcaps
    if center: p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)

def para(*parts):
    """parts: plain strings, or (text, True) for highlighted draft text."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Pt(10)
    for part in parts:
        text, draft = (part, False) if isinstance(part, str) else part
        r = p.add_run(text)
        if draft:
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW

t = doc.add_paragraph()
r = t.add_run("Methods draft, 11/10/26 (Caelan Dunlea). Highlighted: drafted, not yet revised. "
              "[Brackets]: information still needed.")
r.italic = True; r.font.size = Pt(9)

heading("II. METHODS", center=True, smallcaps=True)
heading("A. Data and Annotation", italic=True)
para("Data were collected in the Eating Trajectory Study (ETS), approved by the Institutional "
     "Review Board of The University of Alabama [protocol number]. Forty-three participants each "
     "ate one meal per session under supervision in a laboratory at The University of Alabama, "
     "across one or more sessions, for 61 sessions in total. Meals were chosen from a predefined "
     "menu spanning breakfasts, lunches, dinners and snacks, and varying in texture, size and style "
     "of cuisine. Participants wore the AIM-2, which recorded a 3-axis accelerometer and an optical "
     "sensor at 128 Hz, and each meal was video recorded. The videos were manually annotated at "
     "10 Hz in custom annotation software, played at one-third speed, following a written protocol "
     "that defined each label and how to handle edge cases. Three labels were annotated: individual "
     "chews, chewing bouts and bites; a sample was labeled as eating if it fell within a chewing "
     "bout or a bite. One session contained no annotated eating, and a second annotator "
     "independently labeled three sessions to estimate annotation reliability. The dataset is "
     "summarized in Table I.")

heading("TABLE I.   DATASET SUMMARY", size=8, center=True, smallcaps=True)
rows = [("Item", "Value"), ("Participants", "43"),
        ("Sessions", "61 (breakfast 8, lunch 18, dinner 12, snack 23)"),
        ("Annotated recording", "24.2 h, of which 9.6 h eating"),
        ("Signals", "Accelerometer (X, Y, Z) and optical, 128 Hz"),
        ("Annotation", "Video, 10 Hz; eating = chewing bout or bite"),
        ("Windows (8 s)", "18,227 for training (38.5% eating); 9,133 scored")]
table = doc.add_table(rows=len(rows), cols=2)
table.style = "Table Grid"
for i, (a, b) in enumerate(rows):
    for j, text in enumerate((a, b)):
        cell = table.cell(i, j); cell.text = text
        for r in cell.paragraphs[0].runs:
            r.font.size = Pt(8); r.bold = i == 0

heading("B. Windowing and Preprocessing", italic=True)
para("The signals were segmented into 8 s windows, each containing 1024 samples (8 s × 128 Hz) "
     "for each of the four channels. Windows were excluded when more than half of their sensor "
     "samples were missing or when they lay more than halfway outside the annotated portion of "
     "the video. A window was labeled as eating if more than half of its annotated samples were "
     "eating. The number of annotated chews in each window was also recorded as the target for an "
     "auxiliary output (Section II-C). For training, a new window started every 4 s, so "
     "consecutive windows overlapped by half; for scoring, windows did not overlap, so each moment "
     "was scored only once. Each channel of each window was z-scored over its valid samples, with "
     "missing samples set to zero, which removed differences in signal level between participants "
     "and sensor placements; no other filtering was applied to the model input. The resulting "
     "window counts are given in Table I.")

heading("C. Model Architecture", italic=True)
para("The model is a compact one-dimensional convolutional neural network (CNN) that takes a "
     "normalized 8 s window and outputs the probability of eating (Fig. 1). The model has 47,282 "
     "parameters in total and was implemented in TensorFlow/Keras. The feature extractor consists "
     "of three convolutional blocks, each with batch normalization and ReLU activation, a structure "
     "similar to the fully convolutional network of Wang et al. [8]. Strides and max-pooling in the "
     "first two convolutional blocks shorten the 1024 samples to 64 time steps of 125 ms each. "
     "Squeeze-and-excitation channel attention [9] reweights the 64 feature channels in each "
     "window, intended to let features carrying chewing count more than their motion-artifact "
     "counterparts. Attention pooling [10] learns one weight for every time step, so that a short "
     "bite is not averaged away over the entire 8 s window. A dense layer of 64 units with dropout "
     "of 0.3 feeds into a sigmoid classifier, which outputs the probability of eating between zero "
     "and one. Lastly, an auxiliary chew head [11] predicts the number of chews in each window "
     "during training, weighted 0.2 relative to the eating loss, to encourage the network to learn "
     "the chewing rhythm. This auxiliary chew head is not used at inference.")
heading("[Fig. 1. Model architecture.]", size=8, center=True)

heading("D. Training and Evaluation", italic=True)
para("The network was trained to minimize binary cross-entropy on the eating output plus 0.2 times "
     "the mean squared error of the auxiliary chew head's output. Training used the Adam optimizer "
     "[12] with a learning rate of 0.0003, a batch size of 64 and at most 80 epochs. To address "
     "class imbalance, eating and non-eating windows were weighted inversely to their share of the "
     "training data, with the weights recomputed in each fold and applied to the eating loss only. "
     "Training was stopped early when the area under the receiver operating characteristic curve "
     "(AUROC) on the validation participants had not improved for 12 consecutive epochs, and the "
     "weights from the epoch with the highest validation AUROC were restored. The learning rate was "
     "halved after 4 epochs without improvement, to a minimum of 10⁻⁵.")
para("The model was evaluated with leave-one-subject-out cross-validation [13]. There were 43 "
     "folds, one per participant: in each, all of that participant's sessions were held out for "
     "testing and excluded from both training and validation. Within each fold, 5 of the remaining "
     "42 participants, a different 5 in each fold, were held out as a validation set for early "
     "stopping and threshold selection, and the model was trained on the other 37. ",
     ("On the validation participants, F1 was computed at thresholds from 0.05 to 0.95 in steps of "
      "0.01, and the threshold at the centre of the range within 0.01 of the maximum F1 was "
      "selected. The entire procedure was repeated with three random seeds. Precision, recall and "
      "F1 were computed on the non-overlapping windows of each held-out participant and pooled "
      "across all participants; per-participant F1 is also reported as mean ± standard deviation, "
      "the form reported by Doulah et al. [3]. ", True),
     "An eating episode was defined as one or more consecutive eating windows, with no merging "
     "across pauses. A true (annotated) episode was counted as detected if at least one of its "
     "windows was predicted as eating. ",
     ("Episode precision is the fraction of predicted episodes that overlap true eating, and false "
      "episodes are predicted episodes that do not touch any true eating, expressed per hour of "
      "non-eating time. Results are also reported with pauses of up to 32 s merged into a single "
      "episode. Each design variant in Table III was compared with the final model on the same "
      "participants, using a 95% bootstrap confidence interval for the change in pooled F1 (2,000 "
      "resamples of participants) and a Wilcoxon signed-rank test on the per-participant "
      "differences in F1.", True))

heading("E. Chewing Rate", italic=True)
para("Chewing rate was estimated separately from the classifier, directly from the optical signal, "
     "with no training. The optical signal of each eating window was band-pass filtered to the "
     "natural chewing band of 0.94–2.17 Hz reported by Po et al. [14], using a second-order "
     "Butterworth filter applied forward and backward. Peaks were counted if they were at least "
     "0.46 s apart, corresponding to the band's upper limit, and had a prominence of at least 0.5 "
     "standard deviations of the filtered window. The chewing rate was the number of peaks "
     "multiplied by 60 and divided by the 8 s window length, in chews per minute, averaged over "
     "each meal's eating windows. Each meal's estimated rate was compared with its annotated rate, "
     "defined as the number of annotated chews in its eating windows per minute of those windows, "
     "and with a baseline that assigned every meal the same rate, the mean annotated rate.")

heading("REFERENCES", center=True, smallcaps=True)
refs = [
 ("B. M. Bell et al., “Automatic, wearable-based, in-field eating detection approaches for public health research: A scoping review,” ", "npj Digit. Med.", ", vol. 3, Art. no. 38, 2020, doi: 10.1038/s41746-020-0246-2."),
 ("J. M. Fontana, M. Farooq, and E. Sazonov, “Automatic Ingestion Monitor: A novel wearable device for monitoring of ingestive behavior,” ", "IEEE Trans. Biomed. Eng.", ", vol. 61, no. 6, pp. 1772–1779, 2014, doi: 10.1109/TBME.2014.2306773."),
 ("A. Doulah, T. Ghosh, D. Hossain, M. H. Imtiaz, and E. Sazonov, “‘Automatic Ingestion Monitor Version 2’ – A novel wearable device for automatic food intake detection and passive capture of food images,” ", "IEEE J. Biomed. Health Inform.", ", vol. 25, no. 2, pp. 568–576, 2021, doi: 10.1109/JBHI.2020.2995473."),
 ("T. Ghosh and E. Sazonov, “A comparative study of deep learning algorithms for detecting food intake,” in ", "Proc. 44th Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBC)", ", 2022, pp. 2993–2996, doi: 10.1109/EMBC48229.2022.9871278."),
 ("V. Papapanagiotou, C. Diou, and A. Delopoulos, “Chewing detection from an in-ear microphone using convolutional neural networks,” in ", "Proc. 39th Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBC)", ", 2017, pp. 1258–1261, doi: 10.1109/EMBC.2017.8037060."),
 ("K. Kyritsis, C. Diou, and A. Delopoulos, “A data driven end-to-end approach for in-the-wild monitoring of eating behavior using smartwatches,” ", "IEEE J. Biomed. Health Inform.", ", vol. 25, no. 1, pp. 22–34, 2021, doi: 10.1109/JBHI.2020.2984907."),
 ("M. Farooq and E. Sazonov, “Automatic measurement of chew count and chewing rate during food intake,” ", "Electronics", ", vol. 5, no. 4, Art. no. 62, 2016, doi: 10.3390/electronics5040062."),
 ("Z. Wang, W. Yan, and T. Oates, “Time series classification from scratch with deep neural networks: A strong baseline,” in ", "Proc. Int. Joint Conf. Neural Netw. (IJCNN)", ", 2017, pp. 1578–1585, doi: 10.1109/IJCNN.2017.7966039."),
 ("J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in ", "Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)", ", 2018, pp. 7132–7141, doi: 10.1109/CVPR.2018.00745."),
 ("M. Ilse, J. M. Tomczak, and M. Welling, “Attention-based deep multiple instance learning,” in ", "Proc. 35th Int. Conf. Mach. Learn. (ICML)", ", vol. 80, 2018, pp. 2127–2136."),
 ("R. Caruana, “Multitask learning,” ", "Mach. Learn.", ", vol. 28, no. 1, pp. 41–75, 1997, doi: 10.1023/A:1007379606734."),
 ("D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in ", "Proc. 3rd Int. Conf. Learn. Represent. (ICLR)", ", San Diego, CA, USA, 2015."),
 ("S. Saeb, L. Lonini, A. Jayaraman, D. C. Mohr, and K. P. Kording, “The need to approximate the use-case in clinical machine learning,” ", "GigaScience", ", vol. 6, no. 5, Art. no. gix019, 2017, doi: 10.1093/gigascience/gix019."),
 ("J. M. C. Po, J. A. Kieser, L. M. Gallo, A. J. Tésenyi, P. Herbison, and M. Farella, “Time-frequency analysis of chewing activity in the natural environment,” ", "J. Dent. Res.", ", vol. 90, no. 10, pp. 1206–1210, 2011, doi: 10.1177/0022034511416669."),
]
for i, (a, venue, c) in enumerate(refs, 1):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Pt(18); p.paragraph_format.first_line_indent = Pt(-18)
    for text, italic in ((f"[{i}]\t{a}", False), (venue, True), (c, False)):
        r = p.add_run(text); r.italic = italic; r.font.size = Pt(8)

doc.save(OUT)
print("wrote", OUT)

"""Paper draft v0.1 (4 November 2026): food-intake detection from AIM-2 signals.

Builds paper_draft.docx in this folder. Every number is taken from the
leave-one-subject-out results and inference runs recorded in HANDOFF.md and
the lab-meeting decks; nothing is estimated. Text in [[double brackets]] is a
note to resolve before submission and is printed in gray italics. {key} cites
an entry of ../literature_review/refs.bib; references are numbered in order of
first citation, IEEE style.

Arial 11, grayscale, US Letter, single column for drafting. The submission
version moves into the IEEE conference template.

    python build_paper_draft.py
"""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
BIB = HERE.parent / "literature_review" / "refs.bib"
FIG = HERE / "figures"
FONT, INK, NOTE = "Arial", RGBColor(0x22, 0x22, 0x22), RGBColor(0x80, 0x80, 0x80)

# --------------------------------------------------------------------------- #
# references
# --------------------------------------------------------------------------- #
def read_bib(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    entries = {}
    for match in re.finditer(r"@(\w+)\{(\w+),(.*?)\n\}", text, re.S):
        kind, key, body = match.groups()
        fields = {k.lower(): re.sub(r"\s+", " ", v).strip()
                  for k, v in re.findall(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", body)}
        fields["kind"] = kind.lower()
        entries[key] = fields
    return entries


def clean(text: str) -> str:
    text = text.replace("``", "\u201c").replace("''", "\u201d").replace("--", "\u2013")
    text = re.sub(r"\\'\{?(\w)\}?", lambda m: {"a": "\u00e1", "e": "\u00e9", "o": "\u00f3"}.get(m.group(1), m.group(1)), text)
    return text.replace("{", "").replace("}", "").replace("\\&", "&")


def ieee_authors(field: str) -> str:
    names = []
    for person in field.split(" and "):
        if "," in person:
            last, first = [p.strip() for p in person.split(",", 1)]
        else:
            parts = person.split()
            last, first = parts[-1], " ".join(parts[:-1])
        initials = " ".join(f"{p[0]}." for p in re.split(r"[\s-]+", first) if p)
        names.append(f"{initials} {last}".strip())
    if len(names) > 6:
        return f"{names[0]} et al."
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + ", and " + names[-1]


def ieee_reference(e: dict) -> str:
    title = clean(e.get("title", "")).replace("\u201c", "").replace("\u201d", "")
    parts = [ieee_authors(clean(e.get("author", ""))) + ",", f"\u201c{title},\u201d"]
    venue = e.get("journal") or e.get("booktitle")
    if venue:
        parts.append(("in " if e["kind"] == "inproceedings" else "") + clean(venue) + ",")
    if e.get("volume"):
        parts.append(f"vol. {e['volume']},")
    if e.get("number"):
        parts.append(f"no. {e['number']},")
    if e.get("pages"):
        parts.append(f"pp. {clean(e['pages'])},")
    parts.append(f"{e.get('year', '')}.")
    if e.get("doi"):
        parts.append(f"doi: {e['doi']}.")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# document helpers
# --------------------------------------------------------------------------- #
doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Inches(1.0))


def style_font(style, size, bold=None):
    style.font.name, style.font.size, style.font.color.rgb = FONT, Pt(size), INK
    if bold is not None:
        style.font.bold = bold
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for key in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(key), FONT)
    for key in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if fonts.get(qn(key)) is not None:
            del fonts.attrib[qn(key)]
    ppr = style.element.find(qn("w:pPr"))
    if ppr is not None:
        for border in ppr.findall(qn("w:pBdr")):
            ppr.remove(border)


style_font(doc.styles["Normal"], 11)
doc.styles["Normal"].paragraph_format.space_after = Pt(6)
doc.styles["Normal"].paragraph_format.line_spacing = 1.15
for name, size in (("Title", 16), ("Heading 1", 13), ("Heading 2", 11)):
    style_font(doc.styles[name], size, bold=True)
    doc.styles[name].paragraph_format.space_before = Pt(12 if name != "Title" else 0)
    doc.styles[name].paragraph_format.space_after = Pt(4)

BIBDATA = read_bib(BIB)
ORDER: list[str] = []


def number(key: str) -> int:
    if key not in BIBDATA:
        raise KeyError(f"{key} is not in {BIB}")
    if key not in ORDER:
        ORDER.append(key)
    return ORDER.index(key) + 1


def para(text: str, style=None, align=None, size=None, italic=False, after=None):
    """Paragraph with [[notes]] in gray italics and {key} citations as [n]."""
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    for piece in re.split(r"(\[\[.*?\]\])", text):
        if not piece:
            continue
        note = piece.startswith("[[")
        body = piece[2:-2] if note else piece
        body = re.sub(r"\{(\w+(?:,\s*\w+)*)\}",
                      lambda m: "[" + ", ".join(str(number(k.strip())) for k in m.group(1).split(",")) + "]",
                      body)
        run = p.add_run(f"[{body}]" if note else body)
        run.italic = note or italic
        if note:
            run.font.color.rgb = NOTE
        if size:
            run.font.size = Pt(size)
    return p


def heading(text: str, level: int = 1):
    doc.add_paragraph(text, style=f"Heading {level}")


FIGURES: list[str] = []
TABLES: list[str] = []


def figure(path: str, width: float, caption: str):
    doc.add_picture(str(FIG / path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.keep_with_next = True   # caption stays with it
    FIGURES.append(path)
    para(f"Fig. {len(FIGURES)}. {caption}", size=10, after=10)


def shade(cell, fill="F2F2F2"):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


def table(caption: str, header: list[str], rows: list[list[str]], widths: list[float]):
    TABLES.append(caption)
    roman = ["I", "II", "III", "IV", "V", "VI"][len(TABLES) - 1]
    para(f"TABLE {roman}. {caption}", size=10, after=4)
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, text in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        run.bold, run.font.size = True, Pt(10)
        shade(cell)
    for row in rows:
        cells = t.add_row().cells
        for i, text in enumerate(row):
            cells[i].text = ""
            note = text.startswith("[[")
            run = cells[i].paragraphs[0].add_run(text[2:-2] if note else text)
            run.font.size = Pt(10)
            if note:
                run.italic, run.font.color.rgb = True, NOTE
    for row in t.rows:
        for i, w in enumerate(widths):
            row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


# --------------------------------------------------------------------------- #
# front matter
# --------------------------------------------------------------------------- #
para("Working draft v0.1, 4 November 2026. Gray bracketed notes mark items to verify or "
     "decide before submission. Target venue: IEEE SoutheastCon [[confirm deadline and page "
     "limit]]; the submission version will use the IEEE conference template.",
     size=9, italic=True, after=12)
doc.add_paragraph("Food Intake Detection from Eyeglass-Mounted Jaw and Head-Motion Sensors "
                  "with a Compact Convolutional Neural Network", style="Title")
para("Caelan Dunlea, [[co-authors and order to be confirmed]], Edward Sazonov", after=0)
para("Department of Electrical and Computer Engineering, The University of Alabama, "
     "Tuscaloosa, AL, USA [[confirm affiliations]]", size=10, after=14)

heading("Abstract")
para("Automatic detection of food intake from wearable sensors enables objective monitoring "
     "of eating behavior without self-report. This paper presents a compact one-dimensional "
     "convolutional neural network that detects eating from the temporalis-muscle optical sensor "
     "and three-axis accelerometer of the Automatic Ingestion Monitor Version 2 (AIM-2), worn on "
     "eyeglasses. The network reads raw 8 s windows at 128 Hz and combines channel attention, "
     "attention pooling over time and an auxiliary chew-count output, with 47,282 parameters. "
     "Under leave-one-subject-out cross-validation on 61 annotated eating sessions from 43 "
     "participants, the model reached a window-level F1 score of 0.86 (precision 0.849, recall "
     "0.870) and detected 91.9% of eating episodes, including every episode of 64 s or longer, "
     "with a median detection delay of 0 s. A matched comparison showed that the raw time-domain "
     "input outperformed a frequency-domain input (F1 0.854 against 0.791, with 41 against 84 "
     "false alarms per hour). Errors concentrated at the boundaries of chewing bouts, and "
     "detection was lower in lunch sessions than in other meals (median session F1 0.815 against "
     "0.890, p = 0.009). The trained model processes a recording in about one second and reports "
     "eating episodes and per-meal chew counts.")
para("Keywords: food intake detection, wearable sensors, AIM-2, convolutional neural network, "
     "chewing, leave-one-subject-out", italic=True, size=10)

# --------------------------------------------------------------------------- #
heading("I. Introduction")
para("Measuring when and how people eat is central to research on obesity and eating behavior, "
     "yet self-report is burdensome and unreliable. Wearable sensors that detect chewing, bites "
     "and hand-to-mouth gestures offer an objective alternative, and approaches range from "
     "classical machine learning on hand-crafted features to end-to-end deep learning {bell2020scoping}. "
     "The Automatic Ingestion Monitor (AIM) combined jaw-motion, hand-gesture and accelerometer "
     "sensors {fontana2014aim}; its successor, AIM-2, moved the sensors onto eyeglasses, with a "
     "temporalis-muscle optical sensor and a three-axis accelerometer sampled at 128 Hz {doulah2021aim2}. "
     "With hand-crafted features and a support vector machine on 10 s epochs, AIM-2 reached F1 0.818 "
     "under leave-one-subject-out validation on 30 participants {doulah2021aim2}, and a comparison "
     "of five deep architectures on AIM-2 time series found a residual network best {ghosh2022comparative}.")
para("Deep networks that learn their own filters from raw signals have outperformed "
     "hand-crafted features in chewing and bite detection {papapanagiotou2017chewing, kyritsis2021endtoend} "
     "and form strong baselines for time-series classification {wang2017strong}. Their "
     "evaluation, however, is often limited to a single window-level score. For a monitoring "
     "system the relevant questions are also whether each eating episode is found, how quickly, "
     "how many false episodes occur, and where the remaining errors fall.")
para("This paper makes four contributions. First, it presents a compact convolutional network "
     "with two attention mechanisms and an auxiliary chew-count output for AIM-2 eating detection. "
     "Second, it evaluates the network with leave-one-subject-out cross-validation on 43 "
     "participants at the window, episode and meal level, against an inter-annotator benchmark. "
     "Third, it isolates the effect of the input representation and of the validation design in "
     "matched comparisons. Fourth, it analyzes where errors occur, by position within a chewing "
     "bout and by meal type, and reports the trained model as a tool that processes complete "
     "recordings.")

# --------------------------------------------------------------------------- #
heading("II. Methods")
heading("A. Data and annotation", 2)
para("The data come from the Eating Trajectory Study (ETS), in which participants wore AIM-2 "
     "during breakfast, lunch, dinner and snack sessions [[add setting, recruitment, IRB approval "
     "and demographics]]. AIM-2 records the optical temporalis-muscle signal and three-axis "
     "acceleration at 128 Hz in 8 s packets. Sessions were annotated from video at 10 Hz with "
     "three tracks: chewing bouts, bites and individual chews. A sample was labeled eating when "
     "it fell within a chewing bout or a bite. Table I summarizes the dataset.")
table("Dataset summary.", ["Item", "Value"], [
    ["Participants", "43"],
    ["Sessions", "61 (breakfast 8, lunch 18, dinner 12, snack 23)"],
    ["Annotated recording", "24.2 h, of which 9.6 h eating"],
    ["Annotated span per session", "2 to 55 min"],
    ["Eating share per session", "0% to 86%"],
    ["Training windows (8 s, 4 s hop)", "18,227 (38.5% eating)"],
    ["Evaluation windows (8 s, non-overlapping)", "9,133"],
], [2.8, 3.6])

heading("B. Windowing and preprocessing", 2)
para("Signals were cut into 8 s windows of 1024 samples by four channels. A window was labeled "
     "eating when more than half of its annotated samples were eating. Windows with less than half "
     "of their samples annotated, or with more than half of their sensor samples missing, were "
     "excluded. Training windows overlapped by half; evaluation used a non-overlapping grid set up "
     "separately for each session, so that no second of recording was scored twice. Each window "
     "and channel was normalized to zero mean and unit variance over its valid samples, and "
     "missing samples were set to zero. No band-pass filtering was applied.")

heading("C. Model", 2)
para("The network (Fig. 1) has three convolutional blocks (32, 64 and 64 filters; kernels 9, 9 "
     "and 5; batch normalization and ReLU; the first two with stride 2 and max pooling), which "
     "reduce each window to 64 time steps of 64 features. A squeeze-and-excitation gate "
     "{hu2018squeeze} reweights the feature channels, and attention pooling {ilse2018attention} "
     "forms a weighted sum over time steps. A dense layer of 64 units with dropout 0.3 feeds two "
     "heads: a sigmoid eating output and a softplus chew-count output trained jointly as an "
     "auxiliary task {caruana1997multitask} on the number of annotated chews per window. The "
     "network has 47,282 parameters excluding the chew head.")
figure("fig_architecture.png", 3.6, "Model architecture. Shaded blocks are the two attention "
       "mechanisms; the dashed chew-count head is trained jointly with the eating output.")

heading("D. Training and evaluation", 2)
para("Models were trained with Adam at a learning rate of 3 \u00d7 10\u207b\u2074, batch size 64 "
     "and class-balanced sample weights, for up to 80 epochs. Early stopping monitored validation "
     "ROC AUC with a patience of 12 epochs and restored the best weights; the learning rate was "
     "halved on plateaus. The chew target was divided by its 99th percentile and weighted 0.2 in "
     "the loss.")
para("Performance was estimated with leave-one-subject-out cross-validation {saeb2017need}: 43 "
     "folds, each holding out one participant with all of their sessions. Within each fold, five "
     "of the remaining participants formed a validation set, used for early stopping and for "
     "choosing the decision threshold that maximized F1; the model was trained on the other 37. "
     "Each configuration was run with three random seeds. Metrics were pooled over all held-out "
     "windows. An eating episode was defined as a run of consecutive eating windows, and an "
     "annotated episode counted as detected when at least one of its windows was predicted as "
     "eating [[confirm the definitions in episode_metrics.py, and the merge gap used for "
     "Table III]]. Configurations were compared on the same folds with a paired bootstrap over "
     "participants and the Wilcoxon signed-rank test; meal types were compared with the "
     "Mann-Whitney U test on per-session F1.")
para("Inter-annotator agreement was measured by having a second annotator label one snack "
     "session independently and scoring the two annotations against each other on the same 8 s "
     "windows [[extend to more sessions; see Section IV]].")

# --------------------------------------------------------------------------- #
heading("III. Results")
heading("A. Window-level detection", 2)
para("Table II gives the leave-one-subject-out results. The model reached F1 0.86 with "
     "precision 0.849 and recall 0.870; the three seeds ranged from 0.859 to 0.861. The two "
     "annotators agreed at F1 0.912 on the same window definition, which bounds the score that "
     "can be expected against a single annotator.")
table("Window-level leave-one-subject-out results (43 participants, 3 seeds).",
      ["Metric", "Value"], [
          ["F1", "0.86 (0.859 to 0.861 across seeds)"],
          ["Accuracy / balanced accuracy", "0.890 / 0.886"],
          ["Precision / recall", "0.849 / 0.870"],
          ["Median participant F1", "0.862"],
          ["False alarms per hour of non-eating", "43.8 windows"],
          ["Inter-annotator agreement (one session)", "F1 0.912"],
      ], [3.2, 3.2])
para("Performance varied more between participants than between any two configurations "
     "(Fig. 2). Four participants scored below 0.70. Review of their video identified chewing in "
     "short bursts between speech, soft food (oatmeal) that produced little temporalis-muscle "
     "activity, and very short sessions in which a few missed windows dominated the score. Their "
     "errors were predominantly missed eating rather than false alarms [[recompute the false-alarm "
     "count for these participants on the final configuration]].")
figure("fig_participants.png", 6.4, "Window-level F1 per held-out participant, with the pooled "
       "F1 as a dashed line. [[Replace participant IDs with P1 to P43 before submission.]]")
figure("fig_example.png", 6.4, "Example held-out lunch session. Top: annotated eating (chewing "
       "bout or bite). Bottom: model probability per 8 s window, with annotated eating shaded and "
       "the decision threshold dashed. On this session F1 was 0.886, precision 0.925 and recall "
       "0.849.")

heading("B. Episode-level detection", 2)
para("At the episode level (Table III) the model detected 91.9% of annotated eating episodes and "
     "every episode of 64 s or longer; the misses fell among short episodes, and 82% of "
     "single-window (8 s) episodes were detected. The median delay from the start of an episode to its first "
     "detected window was 0 s, and every one of the 60 meals with annotated eating had at least "
     "one detection. False episodes away from meals occurred 9.9 times per hour.")
table("Episode-level results.", ["Measure", "Value"], [
    ["Eating episodes detected", "91.9%"],
    ["Episodes of 64 s or longer detected", "100%"],
    ["Episodes of 8 s detected", "82%"],
    ["Median delay to first detection", "0 s"],
    ["False episodes per hour (away from meals)", "9.9"],
    ["Episode precision", "87.8%"],
    ["Meals with any detection", "60 of 60"],
], [3.2, 3.2])

heading("C. Where errors occur", 2)
para("Most errors lay at the boundaries of chewing bouts: 62% of all errors, including 75% of "
     "false-alarm windows and 49% of missed windows, were adjacent to a change in the annotation. "
     "Scoring with boundary windows excluded raised F1 to 0.944, and only 19% of false alarms "
     "were isolated from any eating [[these figures are from one seed of the reference "
     "configuration; recompute with the final configuration]]. Because an 8 s window that "
     "straddles the start or end of a bout is partly eating and partly not, a share of these "
     "errors reflects the label resolution rather than the detector.")

heading("D. Design comparisons", 2)
para("Table IV compares design choices on the same folds. Replacing the time series with its "
     "magnitude spectrum lowered F1 from 0.854 to 0.791 and doubled false alarms, and the "
     "time-domain input detected more eating at every false-alarm rate (Fig. 3). Choosing the "
     "threshold on five validation participants rather than two raised F1 by 0.008 (95% CI 0.001 "
     "to 0.016; 30 participants better, 12 worse; p = 0.004), mainly by stabilizing the threshold "
     "for the weakest participants. Other changes, including a recurrent model over two minutes of "
     "context, soft labels, a transformer layer, a three-model ensemble and data augmentation, "
     "stayed within the seed-to-seed variation of about \u00b10.007.")
table("Design comparisons on the same leave-one-subject-out folds.",
      ["Change from the reference", "Effect on F1"], [
          ["Frequency-domain (FFT) input instead of time domain", "0.854 \u2192 0.791; false alarms 41 \u2192 84 per hour"],
          ["Five validation participants instead of two", "+0.008 [0.001, 0.016], p = 0.004"],
          ["Eight validation participants instead of five", "+0.001 (no difference)"],
          ["Recurrent model over 2 min of windows", "\u22120.003 (no difference)"],
          ["Soft labels (eating fraction)", "+0.001 (no difference)"],
          ["Sampling rate 32 Hz / 16 Hz instead of 128 Hz", "\u22120.013 / \u22120.020"],
          ["16 s windows instead of 8 s", "\u22120.014 [\u22120.030, +0.002]"],
      ], [3.6, 2.8])
figure("fig_operating_curve.png", 5.0, "Share of eating windows detected against false alarms "
       "per hour of non-eating, for the same network with time-domain and frequency-domain input. "
       "Dots mark the operating points chosen on the validation participants.")

heading("E. Meal type", 2)
para("Detection was lower in lunch sessions than in other meals (Fig. 4): median session F1 "
     "0.815 against 0.890 (Mann-Whitney p = 0.009), driven by recall (0.819 at lunch against "
     "0.891 to 0.931 for the other meals). Among the nine participants with both a lunch and "
     "another session, seven scored lower at lunch (median difference \u22120.081, Wilcoxon "
     "p = 0.10). Median bout length did not differ between lunch and breakfast (16 s), so bout "
     "length alone does not explain the difference. Some lunches were eaten alone, so this result "
     "concerns meal type, not group eating [[test group against solo sessions once the session "
     "setting is available]].")
figure("fig_meal_types.png", 6.4, "Left: F1 of every session by meal type, with medians. Right: "
       "participants with a lunch and another session; dark lines are lower at lunch.")

heading("F. Chew counts and the trained model", 2)
para("The final model was trained once on all annotated participants, with five held out for "
     "early stopping and threshold selection (threshold 0.75). On the eight sessions of those five "
     "participants, which the model never trained on, median F1 was 0.854, close to the "
     "leave-one-subject-out median of 0.862. Applied to all 61 recordings, it processed each in "
     "about one second and produced a list of eating episodes with start, end, duration and chew "
     "count.")
para("Chews per meal from the auxiliary output agreed with the annotated counts on the eight "
     "held-out meals (r = 0.92, median error 14%, 8% undercount), but chewing rate did not "
     "(r = \u22120.23): the model assigned 70 to 100 chews per minute to every meal, while "
     "annotated rates ranged from 75 to 130 (Fig. 5). Because the number of chews is the product "
     "of eating time and rate, and eating time varies far more between meals than rate, the count "
     "is accurate mainly because eating time is.")
figure("fig_chews.png", 6.0, "Model against annotated chews per meal (left) and chewing rate "
       "(right). Filled: participants the model never trained on; open: participants it trained on.")

# --------------------------------------------------------------------------- #
heading("IV. Discussion")
para("The model approaches the agreement between two human annotators on the same windows, and "
     "its residual errors are concentrated where the labels themselves are least certain: at the "
     "edges of chewing bouts and in short, fragmented eating. At the episode level, which is what "
     "a monitoring application acts on, every eating episode of a minute or more was found "
     "without delay. The 0.86 F1 is not directly comparable with the 0.818 reported for AIM-2 "
     "with hand-crafted features {doulah2021aim2} or the balanced accuracy of the deep models in "
     "{ghosh2022comparative}, since datasets, epoch lengths and protocols differ "
     "[[re-run a baseline on this dataset if a direct comparison is wanted]].")
para("The comparison of input representations indicates that the learned time-domain filters, "
     "rather than any training refinement, carry the performance: no other change exceeded the "
     "seed-to-seed variation, while the spectral input cost 0.06 F1. The variation between "
     "participants was larger than any configuration difference, which supports participant-level "
     "validation as the basis for these estimates {saeb2017need}.")
para("The lower detection at lunch is consistent with eating broken up by conversation, which was "
     "observed on video for the weakest participants, but the present data cannot separate meal "
     "type from group setting, food or session length. The chew results suggest a direct route to "
     "chewing rate: the optical signal shows a clear spectral peak near 1.7 Hz during eating, "
     "inside the chewing band reported for natural chewing {po2011chewing}, and the peak frequency "
     "of each eating window could estimate rate without relying on the auxiliary output.")
para("Limitations. Most sessions were annotated by one person, and agreement was measured on a "
     "single session. Windows of 8 s limit the timing resolution at bout boundaries. Decision "
     "thresholds chosen per fold varied widely, reflecting differences in the probability scale "
     "between fold models [[report the range for the final configuration; it was 0.06 to 0.93 "
     "with two validation participants]]. [[Add: study setting relative to free living; "
     "number of participants; sessions without eating.]]")

heading("V. Conclusion")
para("A compact convolutional network operating on raw AIM-2 signals detected eating with F1 0.86 "
     "under leave-one-subject-out validation on 43 participants, found every eating episode of a "
     "minute or longer with no median delay, and outperformed the same network on spectral input. "
     "Errors concentrated at bout boundaries and in lunch sessions. Future work will extend the "
     "inter-annotator comparison, estimate chewing rate from the optical spectrum, and test "
     "whether eating in groups explains the lower detection at lunch.")

heading("Acknowledgment")
para("This work was supported by the Randall Research Scholars Program at The University of "
     "Alabama [[confirm wording and any grant numbers]]. The authors thank Kenzy Nguyen for "
     "annotation [[confirm]].")

heading("References")
for i, key in enumerate(ORDER, 1):
    para(f"[{i}] {ieee_reference(BIBDATA[key])}", size=9, after=3)

out = HERE / "paper_draft.docx"
doc.save(out)
print(f"wrote {out}: {len(FIGURES)} figures, {len(TABLES)} tables, {len(ORDER)} references")

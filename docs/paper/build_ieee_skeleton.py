"""Fill IEEE's US Letter conference template with the paper's skeleton.

    python build_ieee_skeleton.py conference-template-letter.docx paper_skeleton_ieee.docx

Takes the template exactly as IEEE distributes it (Word "Strict" format, which
python-docx cannot open, so the XML is edited directly with lxml) and keeps
its styles, columns and section breaks. Every new paragraph is a copy of one
of the template's own paragraphs, so it carries IEEE's style:

  * all of IEEE's guidance text is removed;
  * title, one-column author block, abstract and keywords are filled in;
  * every section and subsection of the planned paper is added, each with a
    gray italic note of what goes there and which results-compendium table
    or figure to use (delete each note as the section is written);
  * captions for the planned tables and figures are in place, already in
    the "table head" and "figure caption" styles, so they number themselves;
  * the funding box, acknowledgment and the 12 references are filled in.

Spots in the abstract and acknowledgment that still need a decision are
highlighted in yellow.
"""
from __future__ import annotations

import copy
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from lxml import etree

TITLE = "Deep Learning Food Intake Detection Method from the AIM-2 Wearable Sensor"
AUTHOR = ["Caelan Dunlea", "Department of Electrical and Computer Engineering",
          "The University of Alabama", "Tuscaloosa, AL, United States",
          "cadunlea@crimson.ua.edu"]
KEYWORDS = ("food intake detection, wearable sensors, convolutional neural network, "
            "chewing, leave-one-subject-out")

# The abstract as written, with the spots still to fix marked: (text, highlight).
ABSTRACT = [
    ("Self-report of food intake events is unreliable and susceptible to underreporting and "
     "recall errors. Additionally, human annotation of video eating events is time-intensive "
     "and does not scale to free-living conditions. The Automatic Ingestion Monitor Version 2 "
     "(AIM-2) is an eyeglass-mounted device that combines a jaw-muscle optical sensor with an "
     "accelerometer for passive eating monitoring. Food intake detection is the gateway "
     "algorithm on which other models and metrics depend. However, previous food intake "
     "detection methods for AIM-2 relied on hand-crafted features", False),
    (", and its performance has not been compared with the agreement between independent "
     "human annotators", True),
    (". This study presents a supervised deep learning approach, combining a convolutional "
     "neural network with attention and an auxiliary chew-count output. The model ", False),
    ("is", True),
    (" evaluated with leave-one-subject-out cross-validation on data spanning ", False),
    ("61 meals", True),
    (" and 43 participants. ", False),
    ("The time-domain input", True),
    (" outperformed a frequency-domain input, with an F1 of 0.854 against 0.791 and half the "
     "false alarms. Overall, the model achieved an F1 of ", False),
    ("0.86 approaching", True),
    (" the 0.91 agreement between two human annotators, with an accuracy of 0.89, and detected "
     "91.9% of all eating episodes. All 60 sessions containing eating were detected, as was "
     "every eating episode lasting a minute or longer. The model detects eating reliably "
     "enough to be ", False),
    ("utilized in parallel future models", True),
    (", such as those that identify meal timing, meal count, chew counts, and chewing rates.",
     False),
]

# (style, text). Style keys: h1, h2, note, table, figure.
OUTLINE = [
    ("h1", "Introduction"),
    ("note", "P1 (3 to 4 sentences): why measuring eating matters; self-report is unreliable. Cite [1]."),
    ("note", "P2: AIM [2] and AIM-2 [3]; AIM-2's hand-crafted-feature detector [3]; deep learning "
             "on AIM-2 [4]; raw-signal networks in other eating studies [5], [6]."),
    ("note", "P3: the gap. State only what you verified in [3] and [4]."),
    ("note", "P4: what this paper does: the model, how it was evaluated, the contributions."),
    ("h1", "Methods"),
    ("h2", "Data and Annotation"),
    ("note", "ETS study and IRB approval; 43 participants, 61 sessions by meal type; video annotation "
             "at 10 Hz (bouts, bites, chews); eating = bout or bite; one session without eating; "
             "second annotator on one session. Compendium Table 1."),
    ("table", "Dataset Summary"),
    ("note", "Table I here: trim compendium Table 1 to about six rows."),
    ("h2", "Windowing and Preprocessing"),
    ("note", "8 s windows at 128 Hz; label rule (over half eating); exclusions; 4 s hop for "
             "training, non-overlapping for scoring; per-window z-score, no filtering. "
             "Compendium Sections 1 and 2."),
    ("h2", "Model Architecture"),
    ("note", "Three convolutional blocks [7], channel attention [8], attention pooling [9], "
             "auxiliary chew-count output [10]; 47,282 parameters. Compendium Table 2."),
    ("figure", "Model architecture."),
    ("note", "Fig. 1 here: compendium Figure 1."),
    ("h2", "Training and Evaluation"),
    ("note", "Adam, early stopping, class weights; leave-one-subject-out [11]; 5 validation "
             "participants choose the threshold; 3 seeds; pooled metrics; episode definition; "
             "statistical tests. Compendium Table 3."),
    ("h2", "Chewing Rate"),
    ("note", "Optical signal band-passed to the chewing band [12], peaks counted, no training."),
    ("h1", "Results"),
    ("h2", "Detection Performance"),
    ("note", "Window-level results and human agreement; why the weakest participants fail. "
             "Compendium Tables 4 and 5."),
    ("table", "Window-Level Detection Results"),
    ("note", "Table II here: compendium Table 4."),
    ("figure", "F1 per held-out participant."),
    ("note", "Fig. 2 here: compendium Figure 2, with participants relabeled P1 to P43."),
    ("h2", "Episode Detection"),
    ("note", "91.9% of episodes, 100% of those 64 s or longer, 0 s median delay, 9.9 false "
             "episodes per hour. Compendium Table 6."),
    ("figure", "Example held-out session."),
    ("note", "Fig. 3 here: compendium Figure 3."),
    ("h2", "Input Representation"),
    ("note", "Time domain against FFT input: F1 0.854 against 0.791, half the false alarms. "
             "Compendium Table 8; operating curve (compendium Figure 4) only if there is room."),
    ("h2", "Error Analysis"),
    ("note", "Errors at bout edges (recompute on the final configuration first); lunch against "
             "other meals. Compendium Sections 5 and 7."),
    ("h2", "Chewing Rate"),
    ("note", "Peak count r = 0.72, median error 7% against 10% for a constant rate; 7% on the "
             "model's detected windows. Compendium Table 12."),
    ("h1", "Discussion"),
    ("note", "P1: close to human agreement; remaining errors at bout edges, where labels are "
             "uncertain."),
    ("note", "P2: comparison with prior AIM-2 work, and why it is not like for like [3], [4]. "
             "Compendium Table 13."),
    ("note", "P3: the lunch result (meal type, not group eating); chew output against optical rate."),
    ("note", "P4: limitations: one annotator per session, agreement on one session, 8 s "
             "resolution, recorded meals rather than free living."),
    ("h1", "Conclusion"),
    ("note", "Three to four sentences on what was shown; one line of future work."),
]

ACK = [("The author thanks Kenzy Nguyen for data annotation and Siavash Esfandiari Fard for "
        "data collection. Claude (Anthropic) was used to assist in software development and "
        "to provide editorial feedback on the manuscript. The ", False),
       ("author(s)", True), (" wrote the manuscript and ", False), ("take(s)", True),
       (" full responsibility for its content.", False)]

FUNDING = ("Funding source and grant number for the ETS study: ask Dr. Sazonov. "
           "If there is none, delete this box.")

# Each reference as (text, italic) runs; venue names are italic.
REFERENCES = [
    [("B. M. Bell ", False), ("et al.", True), (", “Automatic, wearable-based, in-field eating "
     "detection approaches for public health research: a scoping review,” ", False),
     ("npj Digital Medicine", True), (", vol. 3, Art. no. 38, 2020, doi: "
     "10.1038/s41746-020-0246-2.", False)],
    [("J. M. Fontana, M. Farooq, and E. Sazonov, “Automatic Ingestion Monitor: A novel "
      "wearable device for monitoring of ingestive behavior,” ", False),
     ("IEEE Trans. Biomed. Eng.", True), (", vol. 61, no. 6, pp. 1772–1779, 2014, doi: "
     "10.1109/TBME.2014.2306773.", False)],
    [("A. Doulah, T. Ghosh, D. Hossain, M. H. Imtiaz, and E. Sazonov, “‘Automatic "
      "Ingestion Monitor Version 2’ – A novel wearable device for automatic food intake "
      "detection and passive capture of food images,” ", False),
     ("IEEE J. Biomed. Health Inform.", True), (", vol. 25, no. 2, pp. 568–576, 2021, doi: "
     "10.1109/JBHI.2020.2995473.", False)],
    [("T. Ghosh and E. Sazonov, “A comparative study of deep learning algorithms for detecting "
      "food intake,” in ", False),
     ("Proc. 44th Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBC)", True),
     (", 2022, pp. 2993–2996, doi: 10.1109/EMBC48229.2022.9871278.", False)],
    [("V. Papapanagiotou, C. Diou, and A. Delopoulos, “Chewing detection from an in-ear "
      "microphone using convolutional neural networks,” in ", False),
     ("Proc. 39th Annu. Int. Conf. IEEE Eng. Med. Biol. Soc. (EMBC)", True),
     (", 2017, pp. 1258–1261, doi: 10.1109/EMBC.2017.8037060.", False)],
    [("K. Kyritsis, C. Diou, and A. Delopoulos, “A data driven end-to-end approach for "
      "in-the-wild monitoring of eating behavior using smartwatches,” ", False),
     ("IEEE J. Biomed. Health Inform.", True), (", vol. 25, no. 1, pp. 22–34, 2021, doi: "
     "10.1109/JBHI.2020.2984907.", False)],
    [("Z. Wang, W. Yan, and T. Oates, “Time series classification from scratch with deep neural "
      "networks: A strong baseline,” in ", False),
     ("Proc. Int. Joint Conf. Neural Netw. (IJCNN)", True),
     (", 2017, pp. 1578–1585, doi: 10.1109/IJCNN.2017.7966039.", False)],
    [("J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in ", False),
     ("Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)", True),
     (", 2018, pp. 7132–7141, doi: 10.1109/CVPR.2018.00745.", False)],
    [("M. Ilse, J. M. Tomczak, and M. Welling, “Attention-based deep multiple instance "
      "learning,” in ", False), ("Proc. 35th Int. Conf. Mach. Learn. (ICML)", True),
     (", vol. 80, 2018, pp. 2127–2136.", False)],
    [("R. Caruana, “Multitask learning,” ", False), ("Machine Learning", True),
     (", vol. 28, pp. 41–75, 1997, doi: 10.1023/A:1007379606734.", False)],
    [("S. Saeb, L. Lonini, A. Jayaraman, D. C. Mohr, and K. P. Kording, “The need to "
      "approximate the use-case in clinical machine learning,” ", False),
     ("GigaScience", True), (", vol. 6, no. 5, 2017, doi: 10.1093/gigascience/gix019.", False)],
    [("J. M. C. Po, J. A. Kieser, L. M. Gallo, A. J. Tésenyi, P. Herbison, and M. Farella, "
      "“Time-frequency analysis of chewing activity in the natural environment,” ", False),
     ("J. Dental Res.", True), (", vol. 90, no. 10, pp. 1206–1210, 2011, doi: "
     "10.1177/0022034511416669.", False)],
]

NOTE_COLOR = "808080"


def main(template: Path, out: Path) -> None:
    work = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(template) as z:
        names = z.namelist()
        z.extractall(work)
    doc_path = work / "word" / "document.xml"
    tree = etree.parse(str(doc_path))
    root = tree.getroot()
    W = root.nsmap["w"]
    q = lambda n: f"{{{W}}}{n}"                                   # noqa: E731
    body = root.find(q("body"))
    kids = list(body)

    def style(p):
        ppr = p.find(q("pPr"))
        s = ppr.find(q("pStyle")) if ppr is not None else None
        return s.get(q("val")) if s is not None else ""

    def text(p):
        return "".join(t.text or "" for t in p.iter(q("t")))

    def first(pred):
        return next(p for p in kids if p.tag == q("p") and pred(p))

    # Prototypes: the template's own paragraphs, copied before anything is removed.
    proto = {
        "h1": first(lambda p: style(p) == "Heading1"),
        "h2": first(lambda p: style(p) == "Heading2"),
        "body": first(lambda p: style(p) == "BodyText" and text(p).strip()),
        "table": first(lambda p: style(p) == "tablehead"),
        "figure": first(lambda p: style(p) == "figurecaption"),
        "h5": first(lambda p: style(p) == "Heading5"),
        "ref": first(lambda p: style(p) == "references"),
    }
    proto = {k: copy.deepcopy(v) for k, v in proto.items()}
    sponsor = first(lambda p: style(p) == "sponsors")
    title_p = first(lambda p: style(p) == "papertitle")
    abstract_p = first(lambda p: style(p) == "Abstract")
    keywords_p = first(lambda p: style(p) == "Keywords")
    last_ref = [p for p in kids if p.tag == q("p") and style(p) == "references"][-1]

    def run_props(p):
        r = p.find(q("r"))
        rpr = r.find(q("rPr")) if r is not None else None
        return copy.deepcopy(rpr) if rpr is not None else etree.Element(q("rPr"))

    def clear_runs(p):
        for child in list(p):
            if child.tag != q("pPr"):
                p.remove(child)

    # Order of run properties in the schema; Word rejects them out of order.
    RPR_ORDER = ["rStyle", "rFonts", "b", "bCs", "i", "iCs", "caps", "smallCaps", "strike",
                 "dstrike", "outline", "shadow", "emboss", "imprint", "noProof", "snapToGrid",
                 "vanish", "webHidden", "color", "spacing", "w", "kern", "position", "sz",
                 "szCs", "highlight", "u", "effect", "bdr", "shd", "fitText", "vertAlign",
                 "rtl", "cs", "em", "lang", "eastAsianLayout", "specVanish", "oMath"]

    def add_run(p, value, base=None, italic=None, bold=None, color=None, highlight=False):
        """A run copying base's formatting; italic/bold given as True or False
        override it, None keeps it."""
        r = etree.SubElement(p, q("r"))
        rpr = copy.deepcopy(base) if base is not None else etree.Element(q("rPr"))
        for tag, want in (("i", italic), ("b", bold)):
            if want is not None:
                for old in rpr.findall(q(tag)):
                    rpr.remove(old)
                if want:
                    etree.SubElement(rpr, q(tag))
        if color:
            for old in rpr.findall(q("color")):
                rpr.remove(old)
            etree.SubElement(rpr, q("color")).set(q("val"), color)
        if highlight:
            etree.SubElement(rpr, q("highlight")).set(q("val"), "yellow")
        rank = {name: i for i, name in enumerate(RPR_ORDER)}
        rpr[:] = sorted(rpr, key=lambda e: rank.get(etree.QName(e).localname, len(rank)))
        r.append(rpr)
        t = etree.SubElement(r, q("t"))
        t.text = value
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        return r

    def new_par(kind, value=None, runs=None, **kw):
        p = copy.deepcopy(proto[kind])
        base = run_props(p)
        clear_runs(p)
        if runs:
            for value_, flag in runs:
                add_run(p, value_, base, **({"italic": flag} if kind == "ref" else {"highlight": flag}))
        else:
            add_run(p, value, base, **kw)
        return p

    # ---- title -------------------------------------------------------------
    base = run_props(title_p)
    clear_runs(title_p)
    add_run(title_p, TITLE, base)

    # ---- author block: keep the first author paragraph, one column -------
    title_i = kids.index(title_p)
    abstract_i = kids.index(abstract_p)
    block = kids[title_i + 1:abstract_i]
    author_p = next(p for p in block if "1st Given Name" in text(p))
    base = run_props(author_p)
    for child in list(author_p):
        if child.tag != q("pPr"):
            author_p.remove(child)
    for i, line in enumerate(AUTHOR):
        r = add_run(author_p, line, base)
        if i < len(AUTHOR) - 1:
            etree.SubElement(r, q("br"))
    cols = author_p.find(q("pPr")).find(q("sectPr")).find(q("cols"))
    cols.set(q("num"), "1")
    for p in block:
        if p is author_p:
            continue
        ppr = p.find(q("pPr"))
        has_sect = ppr is not None and ppr.find(q("sectPr")) is not None
        if "Sub-titles" in text(p) or not has_sect:
            body.remove(p)                         # the note and the other authors
        else:
            clear_runs(p)                          # keep the section breaks, empty
            c = ppr.find(q("sectPr")).find(q("cols"))
            if c is not None and c.get(q("num")):
                c.set(q("num"), "1")

    # ---- abstract and keywords: keep the run-in heads ---------------------
    runs = abstract_p.findall(q("r"))
    head_base = run_props(abstract_p)
    body_base = copy.deepcopy(runs[-1].find(q("rPr"))) if runs and runs[-1].find(q("rPr")) is not None \
        else head_base
    clear_runs(abstract_p)
    add_run(abstract_p, "Abstract—", head_base)
    for value, flag in ABSTRACT:
        add_run(abstract_p, value, body_base, highlight=flag)
    runs = keywords_p.findall(q("r"))
    head_base = run_props(keywords_p)
    body_base = copy.deepcopy(runs[-1].find(q("rPr"))) if runs and runs[-1].find(q("rPr")) is not None \
        else head_base
    clear_runs(keywords_p)
    add_run(keywords_p, "Keywords—", head_base)
    add_run(keywords_p, KEYWORDS, body_base)

    # ---- body: drop everything between the keywords and the last reference
    kids = list(body)
    start, end = kids.index(keywords_p) + 1, kids.index(last_ref)
    for el in kids[start:end]:
        if el is not sponsor:
            body.remove(el)
    # The last reference paragraph holds the two-column section break: keep
    # it as an empty plain paragraph.
    ppr = last_ref.find(q("pPr"))
    ppr.find(q("pStyle")).set(q("val"), "BodyText")
    for num in ppr.findall(q("numPr")):
        ppr.remove(num)
    clear_runs(last_ref)

    # Funding box (a frame): new text, same position.
    base = run_props(sponsor)
    clear_runs(sponsor)
    add_run(sponsor, FUNDING, base, highlight=True)

    anchor = keywords_p
    def place(p):
        nonlocal anchor
        anchor.addnext(p)
        anchor = p

    body.remove(sponsor)
    place(sponsor)
    for kind, value in OUTLINE:
        if kind == "note":
            place(new_par("body", value, italic=True, color=NOTE_COLOR))
        else:
            place(new_par(kind, value))
    place(new_par("h5", "Acknowledgment"))
    place(new_par("body", runs=ACK))
    place(new_par("h5", "References"))
    for ref in REFERENCES:
        place(new_par("ref", runs=ref))

    tree.write(str(doc_path), xml_declaration=True, encoding="UTF-8", standalone=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(work / name, name)
    shutil.rmtree(work)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

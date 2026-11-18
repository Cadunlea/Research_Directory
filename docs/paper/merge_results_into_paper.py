"""Fill the Results section of the IEEE paper and bring Methods in line with it.

    python merge_results_into_paper.py paper_full_11_10.docx paper_full_results.docx

Changes (Strict OOXML, edited as XML; everything else is left as it is):
  * Results A-E: the note lines are replaced by text. Values still to be
    confirmed are in [brackets] and highlighted yellow.
  * Results E gets Fig. 5 (chewing rate per meal, both definitions).
  * Methods II-A: three double-annotated sessions -> four.
  * Methods II-E: describes both chewing-rate definitions and the 5-window rule.
  * Table II: human agreement over 4 sessions; two rows for episodes with
    pauses of up to 32 s merged.
"""
import copy
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from lxml import etree

HERE = Path(__file__).resolve().parent
FIG5 = HERE.parent / "presentation" / "rrs_live" / "chew_rate_slide_trend.png"   # 3150 x 1380
H = True        # highlighted: still to confirm

RESULTS = {
    "Detection Performance": [
        [("Across the 43 held-out participants, the model reached a pooled window-level F1 of "
          "0.86 (0.859 to 0.861 across the three seeds), with a precision of 0.849 and a recall "
          "of 0.870 (Table II). Specificity was 0.902 and balanced accuracy 0.886, corresponding "
          "to 43.8 false-alarm windows per hour of non-eating. Per-participant F1 had a mean of "
          "0.844 ± 0.101 and a median of 0.863 (Fig. 2); ", False),
         ("[N]", H), (" of the 43 participants scored below 0.80, and the lowest score was ", False),
         ("[min]", H), (".", False)],
        [("Two annotators independently labeled four sessions. Their window-level agreement "
          "ranged from F1 0.81 to 0.93 (Table II), a range that includes the model's pooled F1. "
          "On the two sessions compared at the 0.1 s sample level, excluding 1 s on either side "
          "of each chewing-bout boundary raised agreement from F1 0.83 and 0.87 to 0.97 and 0.99, "
          "respectively; in one of these sessions, 65 of the 70 s of disagreement fell within "
          "those boundary regions.", False)],
        [("The lowest-scoring participants were examined individually. In one, most missed "
          "windows occurred while the participant ate oatmeal, during which the optical signal "
          "remained nearly flat; this session was also short (46 scored windows). Another chewed "
          "in bursts of about 4 s separated by pauses of about 3 s filled by talking; talking "
          "alone produced only two false-alarm windows for this participant.", False)],
    ],
    "Episode Detection": [
        [("Of the 1,091 annotated eating episodes, 91.9% contained at least one window predicted "
          "as eating, including all 69 episodes of 64 s or longer and 82% of the 367 single-window "
          "(8 s) episodes (Table II). The median delay from the start of a detected episode to its "
          "first detected window was 0 s (90th percentile 0 s). Of the 1,159 predicted episodes, "
          "87.8% overlapped annotated eating; false episodes, which touched no annotated eating, "
          "occurred at 9.9 per hour of non-eating time, and a further 1.4 per hour touched an "
          "annotated episode without overlapping it. Every session that contained eating (60 of "
          "60) had at least one detection. With pauses of up to 32 s merged, the 222 resulting "
          "episodes were detected at 94.6%, with 2.4 false episodes per hour and an episode "
          "precision of 86.7%.", False)],
    ],
    "Input Representation": [
        [("The input representation had the largest effect of any design choice. In the "
          "reference configuration, the network given the raw time-domain signal reached an F1 "
          "of 0.854, against 0.791 for the same network given the magnitude spectrum, and "
          "produced half as many false alarms (41 against 84 per hour of non-eating; Fig. 4, "
          "Table III). Using two validation participants instead of five lowered pooled F1 by "
          "0.008 (95% CI −0.016 to −0.001; Wilcoxon p = 0.004), while eight gave no further gain "
          "(+0.001). Reducing the sampling rate to 32 or 16 Hz lowered F1 by 0.013 and 0.020, and "
          "16 s windows lowered it by 0.014 (95% CI −0.030 to +0.002). Removing channel attention, "
          "replacing attention pooling with average pooling, setting the chew-head weight to 0 or "
          "1, and adding a transformer layer, an ensemble, augmentation, a recurrent sequence "
          "model or soft labels each changed F1 by less than the seed-to-seed variation "
          "(±0.007).", False)],
    ],
    "Error Analysis": [
        [("In the reference configuration, 62% of window errors occurred at the edges of "
          "annotated eating bouts, and 75% of false-alarm windows were adjacent to annotated "
          "eating. ", False),
         ("[Recompute on the final configuration with error_breakdown.py before submission; "
          "add the lunch against other meals comparison or delete this note.]", H)],
    ],
    "Chewing Rate": [
        [("The auxiliary chew output did not track chewing rate: on the eight sessions from "
          "participants held out of the final model's training, its rate estimate correlated at "
          "r = −0.23 with the annotated rate (median error 20%). Peak counting was therefore "
          "evaluated on the 59 meals with at least 40 s of annotated eating (Fig. 5). Counted "
          "peaks matched the annotated number of chews per meal closely (r = 0.98, median error "
          "7%). Expressed per minute of eating windows, the estimated rate correlated at r = 0.72 "
          "with the annotated rate (median error 7%, against 10% for a constant rate equal to the "
          "mean), but the estimates were compressed toward the mean (trend slope 0.33). Expressed "
          "per minute of the whole meal, the estimate tracked the annotated rate closely (r = "
          "0.97, trend slope 0.86), with a median error of 13% against 23% for the constant-rate "
          "baseline.", False)],
    ],
}
FIG5_CAPTION = ("Estimated against annotated chewing rate for 59 meals: (left) chews per minute "
                "of eating windows; (right) chews per minute of the whole meal. Dashed: perfect "
                "agreement; solid: least-squares trend.")
METHODS_E = [
    ("Chewing rate was estimated separately from the classifier, directly from the optical "
     "signal, with no training. The optical signal of each eating window was band-pass filtered "
     "to the natural chewing band of 0.94–2.17 Hz reported by Po et al. [14], using a "
     "second-order Butterworth filter applied forward and backward. Peaks were counted if they "
     "were at least 0.46 s apart, corresponding to the band's upper limit, and had a prominence "
     "of at least 0.5 standard deviations of the filtered window; each meal's chew count was the "
     "sum over its eating windows. Two rates were derived from this count: chews per minute of "
     "the meal's eating windows, and chews per minute of the whole meal, from the start of its "
     "first eating window to the end of its last, so that pauses within the meal are included. "
     "Each was compared with the same quantity computed from the annotated chews, and with a "
     "baseline that assigned every meal the mean annotated rate. Meals with fewer than five "
     "eating windows (40 s) were excluded.", False)]


def main(src, dst):
    work = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        z.extractall(work)
    doc = work / "word" / "document.xml"
    tree = etree.parse(str(doc))
    root = tree.getroot()
    W = root.nsmap["w"]
    q = lambda n: f"{{{W}}}{n}"                                              # noqa: E731
    XML = "{http://www.w3.org/XML/1998/namespace}space"
    body = root.find(q("body"))

    def style(p):
        s = p.find(f"{q('pPr')}/{q('pStyle')}")
        return s.get(q("val")) if s is not None else ""

    def text(p):
        return "".join(p.itertext()).strip()

    def fill(p, runs):
        for r in p.findall(q("r")):
            p.remove(r)
        for t, hl in runs:
            r = etree.SubElement(p, q("r"))
            if hl:
                etree.SubElement(etree.SubElement(r, q("rPr")), q("highlight")).set(q("val"), "yellow")
            node = etree.SubElement(r, q("t"))
            node.text = t
            node.set(XML, "preserve")

    def section(name):
        kids = list(body)
        start = next(i for i, p in enumerate(kids) if style(p) == "Heading1" and text(p) == name)
        end = next(i for i, p in enumerate(kids) if i > start and style(p) == "Heading1")
        return kids[start:end]

    # Results A-E.
    results = section("Results")
    last_note = {}
    for i, p in enumerate(results):
        if style(p) == "Heading2" and text(p) in RESULTS:
            note = next(c for c in results[i + 1:] if style(c) == "BodyText")
            paras = RESULTS[text(p)]
            fill(note, paras[0])
            after = note
            for extra in paras[1:]:
                new = copy.deepcopy(note)
                fill(new, extra)
                after.addnext(new)
                after = new
            last_note[text(p)] = after

    # Methods: four double-annotated sessions; chewing-rate paragraph.
    methods = section("Methods")
    for p in methods:
        for t in p.iter(q("t")):
            if t.text and "independently labeled three sessions" in t.text:
                t.text = t.text.replace("labeled three sessions", "labeled four sessions")
    chew = [p for p in methods if style(p) == "Heading2" and text(p) == "Chewing Rate"][0]
    para = chew.getnext()
    while style(para) != "BodyText":
        para = para.getnext()
    fill(para, METHODS_E)

    # Table II.
    for tbl in body.iter(q("tbl")):
        rows = tbl.findall(q("tr"))
        labels = ["".join(tr.itertext()) for tr in rows]
        if not any(l.startswith("Episode precision") for l in labels):
            continue
        for t in tbl.iter(q("t")):
            if t.text == "Human agreement (3 sessions)":
                t.text = "Human agreement (4 sessions)"
        anchor = rows[[i for i, l in enumerate(labels) if l.startswith("Episode precision")][0]]
        for label, value in (("False episodes per hour, pauses ≤ 32 s merged", "2.4"),
                             ("Episodes detected, pauses ≤ 32 s merged", "94.6%")):
            new = copy.deepcopy(anchor)
            ts = list(new.iter(q("t")))
            ts[0].text, ts[-1].text = label, value
            for extra in ts[1:-1]:
                extra.text = ""
            anchor.addnext(new)

    # Fig. 5 after the chewing-rate results, copied from Fig. 4's paragraphs.
    fig4 = [p for p in body if p.find(".//" + q("drawing")) is not None][-1]
    cap4 = fig4.getnext()
    fig5, cap5 = copy.deepcopy(fig4), copy.deepcopy(cap4)
    width = 3154680
    height = int(width * 1380 / 3150)
    for el in fig5.iter():
        tag = etree.QName(el).localname
        if tag in ("extent", "ext") and el.get("cx"):
            el.set("cx", str(width))
            el.set("cy", str(height))
        if tag in ("docPr", "cNvPr"):
            el.set("id", "1005")
            el.set("name", "fig_chew_rate.png" if tag == "cNvPr" else "Figure 1005")
            if tag == "docPr":
                el.set("descr", "Estimated against annotated chewing rate per meal, two definitions")
        if tag == "blip":
            el.set(f"{{{root.nsmap['r']}}}embed", "rIdFig5")
    for t in cap5.iter(q("t")):
        t.text = ""
    list(cap5.iter(q("t")))[0].text = FIG5_CAPTION
    last_note["Chewing Rate"].addnext(fig5)
    fig5.addnext(cap5)

    shutil.copy(FIG5, work / "word" / "media" / "fig_chew_rate.png")
    names.append("word/media/fig_chew_rate.png")
    rels = work / "word" / "_rels" / "document.xml.rels"
    rtext = rels.read_text(encoding="utf-8")
    rtext = rtext.replace("</Relationships>",
                          '<Relationship Id="rIdFig5" Type="http://purl.oclc.org/ooxml/'
                          'officeDocument/relationships/image" Target="media/fig_chew_rate.png"/>'
                          "</Relationships>")
    rels.write_text(rtext, encoding="utf-8")

    tree.write(str(doc), xml_declaration=True, encoding="UTF-8", standalone=True)
    tmp = Path(dst).with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(work / n, n)
    shutil.move(tmp, dst)
    print("wrote", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

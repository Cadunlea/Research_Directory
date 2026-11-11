"""Put the Methods draft (methods_draft_11_10.docx, built by build_methods_draft.py)
into the full IEEE paper, in place of the Methods notes, and replace the
reference list with the renumbered one (Adam added as [12]).

    python merge_methods_into_paper.py paper_with_figures.docx paper_full.docx

The paper is Strict OOXML (python-docx cannot open it), so it is edited as XML.
Everything else in the paper is left as it is, apart from the Table II
human-agreement row, which now covers the three double-annotated sessions.
"""
import shutil, sys, tempfile, zipfile, copy
from pathlib import Path
from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from lxml import etree

HERE = Path(__file__).resolve().parent
SUBSECTIONS = ["Data and Annotation", "Windowing and Preprocessing", "Model Architecture",
               "Training and Evaluation", "Chewing Rate"]


def draft_paragraphs():
    """{subsection: [paragraph runs]} and [reference runs] from the methods draft;
    a run is (text, italic, highlighted)."""
    doc = Document(HERE / "methods_draft_11_10.docx")
    sections, refs, current = {}, [], None
    for p in doc.paragraphs:
        text = p.text.strip()
        heads = [s for s in SUBSECTIONS if text.endswith(s) and len(text) < len(s) + 4]
        if heads and current != "refs":
            current = heads[0]; sections[current] = []; continue
        if text == "REFERENCES":
            current = "refs"; continue
        if not text or text.startswith("[Fig.") or text.startswith("TABLE") or current is None:
            continue
        runs = [(r.text, bool(r.italic), r.font.highlight_color == WD_COLOR_INDEX.YELLOW)
                for r in p.runs]
        if current == "refs":
            runs[0] = (runs[0][0].split("\t", 1)[1], runs[0][1], False)   # drop "[n]\t"
            refs.append(runs)
        else:
            sections[current].append(runs)
    return sections, refs


def main(src, dst):
    sections, refs = draft_paragraphs()
    work = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(src) as z:
        names = z.namelist(); z.extractall(work)
    path = work / "word" / "document.xml"
    tree = etree.parse(str(path)); root = tree.getroot()
    W = root.nsmap["w"]; q = lambda n: f"{{{W}}}{n}"
    body = root.find(q("body"))

    def style(p):
        s = p.find(f"{q('pPr')}/{q('pStyle')}")
        return s.get(q("val")) if s is not None else ""

    def fill(p, runs):
        for r in p.findall(q("r")):
            p.remove(r)
        for text, italic, highlight in runs:
            r = etree.SubElement(p, q("r"))
            if italic or highlight:
                rpr = etree.SubElement(r, q("rPr"))
                if italic:
                    etree.SubElement(rpr, q("i"))
                if highlight:
                    etree.SubElement(rpr, q("highlight")).set(q("val"), "yellow")
            t = etree.SubElement(r, q("t")); t.text = text
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")

    # Methods: the BodyText note after each Heading2 of the Methods section.
    children = list(body)
    start = next(i for i, p in enumerate(children)
                 if style(p) == "Heading1" and "".join(p.itertext()) == "Methods")
    end = next(i for i, p in enumerate(children) if i > start and style(p) == "Heading1")
    for i in range(start, end):
        p = children[i]
        name = "".join(p.itertext()).strip()
        if style(p) == "Heading2" and name in sections:
            note = next(c for c in children[i + 1:end] if style(c) == "BodyText")
            paras = sections[name]
            fill(note, paras[0])
            after = note
            for extra in paras[1:]:
                new = copy.deepcopy(note); fill(new, extra)
                after.addnext(new); after = new

    # References: replace every "references" paragraph with the renumbered list.
    old = [p for p in body if p.tag == q("p") and style(p) == "references"]
    template = old[0]
    for runs in refs:
        new = copy.deepcopy(template); fill(new, runs)
        old[-1].addnext(new) if False else template.addprevious(new)
    for p in old:
        body.remove(p)

    # Table II: human agreement now covers three sessions.
    for t in root.iter(q("t")):
        if t.text == "Human agreement (one session)":
            t.text = "Human agreement (3 sessions)"
        elif t.text == "F1 0.912":
            t.text = "F1 0.81–0.93"

    tree.write(str(path), xml_declaration=True, encoding="UTF-8", standalone=True)
    tmp = Path(dst).with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(work / n, n)
    shutil.move(tmp, dst)
    print("wrote", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

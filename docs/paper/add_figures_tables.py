"""Insert the paper's figures and tables into the IEEE Word file, in place.

    python add_figures_tables.py paper_in.docx paper_out.docx

Works on the paper as written (IEEE Letter template, Word "Strict" format,
edited as XML because python-docx cannot open Strict files). Only these
changes are made; every other paragraph is left exactly as it is:

  * each figure goes directly above its "figure caption" paragraph, sized to
    one column; Fig. 4 (time against FFT input) and its caption are added at
    the end of the Input Representation subsection;
  * each table goes directly below its "table head" paragraph, built with
    IEEE's own table styles (tablecolhead, tablecopy) and the borders of the
    template's sample table; Table III (design comparisons) and its head are
    added in Input Representation;
  * the gray "Fig. N here" / "Table N here" notes are removed. Fig. 2's note
    is replaced by one about the anonymized version.

Numbers in the tables are from the results compendium; cells still to be
computed say so in gray.
"""
from __future__ import annotations

import copy
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

from lxml import etree
from PIL import Image

FIG = Path(__file__).resolve().parent / "figures"
EMU_PER_INCH = 914400
COLUMN_IN = 3.45                      # IEEE column is about 3.5 in
NOTE = "808080"

PIC = "http://purl.oclc.org/ooxml/drawingml/picture"
A = "http://purl.oclc.org/ooxml/drawingml/main"
IMAGE_REL = "http://purl.oclc.org/ooxml/officeDocument/relationships/image"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"

# (caption text that identifies the paragraph, image, width in inches, alt text)
FIGURES = [
    ("Model architecture", "fig_architecture.png", 2.9,
     "Model architecture: three convolutional blocks, channel attention, attention pooling, "
     "eating and chew-count heads"),
    ("F1 per held-out participant", "fig_participants_anon.png", COLUMN_IN,
     "Bar chart of F1 for each of the 43 held-out participants"),
    ("Example held-out session", "fig_example.png", COLUMN_IN,
     "Annotated eating and model probability over time for one held-out lunch session"),
]
NEW_FIGURE = ("Recall against false alarms per hour of non-eating for time-domain and "
              "frequency-domain input to the same network.", "fig_operating_curve.png",
              COLUMN_IN, "Operating curves for time-domain and frequency-domain input")

TABLE_I = (["Item", "Value"], [
    ["Participants", "43"],
    ["Sessions", "61 (breakfast 8, lunch 18, dinner 12, snack 23)"],
    ["Annotated recording", "24.2 h, of which 9.6 h eating"],
    ["Signals", "Accelerometer (X, Y, Z) and optical, 128 Hz"],
    ["Annotation", "Video, 10 Hz; eating = chewing bout or bite"],
    ["Windows (8 s)", "18,227 for training (38.5% eating); 9,133 scored"],
], [80, 170])
TABLE_II = (["Measure", "Value"], [
    ["Window level", ""],
    ["F1 (3 seeds)", "0.86 (0.859 to 0.861)"],
    ["Precision / recall", "0.849 / 0.870"],
    ["Specificity / balanced accuracy", "0.902 / 0.886"],
    ["Participant F1, median", "0.863"],
    ["Participant F1, mean ± SD", "0.844 ± 0.101"],
    ["False alarms per hour of non-eating", "43.8"],
    ["Episode level", ""],
    ["Episodes detected", "91.9%"],
    ["Episodes of 64 s or longer detected", "100%"],
    ["Median delay to first detection", "0 s"],
    ["False episodes per hour", "9.9"],
    ["Episode precision", "87.8%"],
    ["Human agreement (one session)", "F1 0.912"],
], [150, 100])
TABLE_III = (["Change from the final model", "Effect on F1"], [
    ["Frequency-domain (FFT) input", "0.854 → 0.791; false alarms 41 → 84 per hour"],
    ["Two validation participants instead of five", "−0.008 [−0.016, −0.001]"],
    ["Eight validation participants instead of five", "+0.001"],
    ["32 Hz / 16 Hz instead of 128 Hz", "−0.013 / −0.020"],
    ["16 s windows instead of 8 s", "−0.014 [−0.030, +0.002]"],
    ["No channel attention; average pooling; chew weight 0 or 1; transformer layer; "
     "ensemble; augmentation; recurrent model; soft labels",
     "Within seed variation (±0.007)"],
], [150, 100])

TBL_PR = ('<w:tblPr xmlns:w="{w}"><w:tblW w:w="0pt" w:type="dxa"/><w:jc w:val="center"/>'
          '<w:tblBorders><w:top w:val="single" w:sz="2" w:space="0" w:color="auto"/>'
          '<w:start w:val="single" w:sz="2" w:space="0" w:color="auto"/>'
          '<w:bottom w:val="single" w:sz="2" w:space="0" w:color="auto"/>'
          '<w:end w:val="single" w:sz="2" w:space="0" w:color="auto"/>'
          '<w:insideH w:val="single" w:sz="2" w:space="0" w:color="auto"/>'
          '<w:insideV w:val="single" w:sz="2" w:space="0" w:color="auto"/></w:tblBorders>'
          '<w:tblLayout w:type="fixed"/></w:tblPr>')


def main(src: Path, dst: Path) -> None:
    work = Path(tempfile.mkdtemp())
    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        z.extractall(work)
    doc_file = work / "word" / "document.xml"
    tree = etree.parse(str(doc_file))
    root = tree.getroot()
    W, R, WP = root.nsmap["w"], root.nsmap["r"], root.nsmap["wp"]
    q = lambda n: f"{{{W}}}{n}"                                   # noqa: E731
    body = root.find(q("body"))

    def style(p):
        ppr = p.find(q("pPr"))
        s = ppr.find(q("pStyle")) if ppr is not None else None
        return s.get(q("val")) if s is not None else ""

    def text(p):
        return "".join(t.text or "" for t in p.iter(q("t")))

    def paragraphs():
        return [p for p in body if p.tag == q("p")]

    def find(kind, start):
        return next(p for p in paragraphs() if style(p) == kind and text(p).startswith(start))

    def note(start):
        return next((p for p in paragraphs() if style(p) == "BodyText"
                     and text(p).startswith(start)), None)

    # ---- images: parts, relationships, content type --------------------------
    rels_file = work / "word" / "_rels" / "document.xml.rels"
    rels = etree.parse(str(rels_file))
    rel_root = rels.getroot()
    used = {r.get("Id") for r in rel_root}
    media = work / "word" / "media"
    media.mkdir(exist_ok=True)
    new_names = []
    doc_pr = [1000]

    def add_image(path: Path):
        target = f"media/{path.name}"
        shutil.copy(path, media / path.name)
        new_names.append(f"word/{target}")
        rid = next(f"rIdFig{i}" for i in range(1, 99) if f"rIdFig{i}" not in used)
        used.add(rid)
        el = etree.SubElement(rel_root, f"{{{PKG_REL}}}Relationship")
        el.set("Id", rid)
        el.set("Type", IMAGE_REL)
        el.set("Target", target)
        return rid

    ct_file = work / "[Content_Types].xml"
    ct = etree.parse(str(ct_file))
    if not any(d.get("Extension") == "png" for d in ct.getroot().iter(f"{{{CT}}}Default")):
        d = etree.Element(f"{{{CT}}}Default")
        d.set("Extension", "png")
        d.set("ContentType", "image/png")
        ct.getroot().insert(0, d)

    def picture_paragraph(image: str, width_in: float, alt: str):
        path = FIG / image
        w_px, h_px = Image.open(path).size
        cx = int(width_in * EMU_PER_INCH)
        cy = int(cx * h_px / w_px)
        rid = add_image(path)
        doc_pr[0] += 1
        n = doc_pr[0]
        xml = (
            f'<w:p xmlns:w="{W}" xmlns:wp="{WP}" xmlns:a="{A}" xmlns:pic="{PIC}" xmlns:r="{R}">'
            '<w:pPr><w:keepNext/><w:spacing w:before="6pt" w:after="3pt"/>'
            '<w:ind w:start="0pt" w:end="0pt" w:firstLine="0pt"/><w:jc w:val="center"/></w:pPr>'
            '<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{cx}" cy="{cy}"/>'
            f'<wp:docPr id="{n}" name="Figure {n}" descr="{alt}"/>'
            '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            f'<a:graphic><a:graphicData uri="{PIC}"><pic:pic>'
            f'<pic:nvPicPr><pic:cNvPr id="{n}" name="{image}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')
        return etree.fromstring(xml)

    # ---- tables --------------------------------------------------------------
    def cell(value, width_pt, head=False, bold=False, keep=True):
        tc = etree.Element(q("tc"))
        pr = etree.SubElement(tc, q("tcPr"))
        tcw = etree.SubElement(pr, q("tcW"))
        tcw.set(q("w"), f"{width_pt}pt")
        tcw.set(q("type"), "dxa")
        etree.SubElement(pr, q("vAlign")).set(q("val"), "center")
        p = etree.SubElement(tc, q("p"))
        ppr = etree.SubElement(p, q("pPr"))
        etree.SubElement(ppr, q("pStyle")).set(q("val"), "tablecolhead" if head else "tablecopy")
        if keep:
            etree.SubElement(ppr, q("keepNext"))          # keeps the table in one piece
        if value:
            pending = value.startswith("[[")
            r = etree.SubElement(p, q("r"))
            rpr = etree.SubElement(r, q("rPr"))
            if bold:
                etree.SubElement(rpr, q("b"))
            if pending:
                etree.SubElement(rpr, q("i"))
                etree.SubElement(rpr, q("color")).set(q("val"), NOTE)
            t = etree.SubElement(r, q("t"))
            t.text = value[2:-2] if pending else value
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        return tc

    def table(spec):
        header, rows, widths = spec
        tbl = etree.Element(q("tbl"))
        tbl.append(etree.fromstring(TBL_PR.format(w=W)))
        grid = etree.SubElement(tbl, q("tblGrid"))
        for w in widths:
            etree.SubElement(grid, q("gridCol")).set(q("w"), str(w * 20))
        for r, row in enumerate([header] + rows):
            tr = etree.SubElement(tbl, q("tr"))
            trpr = etree.SubElement(tr, q("trPr"))
            etree.SubElement(trpr, q("cantSplit"))
            if r == 0:
                etree.SubElement(trpr, q("tblHeader"))
            group = r > 0 and row[1] == ""             # "Window level" style rows
            last = r == len(rows)
            for value, w in zip(row, widths):
                tr.append(cell(value, w, head=(r == 0), bold=group, keep=not last))
        return tbl

    def keep_with_next(p):
        ppr = p.find(q("pPr"))
        if ppr is not None and ppr.find(q("keepNext")) is None:
            ppr.insert(1, etree.Element(q("keepNext")))

    def spacer():
        return etree.fromstring(f'<w:p xmlns:w="{W}"><w:pPr><w:spacing w:before="0pt" '
                                f'w:after="6pt"/></w:pPr></w:p>')

    def drop(p):
        if p is not None:
            body.remove(p)

    # Figures: above their captions; their notes removed.
    for caption, image, width, alt in FIGURES:
        cap = find("figurecaption", caption)
        cap.addprevious(picture_paragraph(image, width, alt))
    drop(note("Fig. 1 here"))
    drop(note("Fig. 3 here"))
    drop(note("Fig. 2 here"))              # Fig. 2 is the anonymized P1 to P43 chart

    # Tables: below their heads; their notes removed.
    for head, spec, note_start in (("Dataset Summary", TABLE_I, "Table I here"),
                                   ("Window-Level Detection Results", TABLE_II, "Table II here")):
        h = find("tablehead", head)
        keep_with_next(h)
        tbl = table(spec)
        h.addnext(tbl)
        tbl.addnext(spacer())
        drop(note(note_start))

    # Input Representation: Table III and Fig. 4, after the subsection's note.
    anchor = note("Time domain against FFT input")
    if anchor is not None:
        head = copy.deepcopy(find("tablehead", "Dataset Summary"))
        for t in head.iter(q("t")):
            t.text = ""
        next(head.iter(q("t"))).text = "Design Comparisons"
        keep_with_next(head)
        tbl = table(TABLE_III)
        cap = copy.deepcopy(find("figurecaption", "Model architecture"))
        for t in cap.iter(q("t")):
            t.text = ""
        next(cap.iter(q("t"))).text = NEW_FIGURE[0]
        pic = picture_paragraph(NEW_FIGURE[1], NEW_FIGURE[2], NEW_FIGURE[3])
        anchor.addnext(head)
        head.addnext(tbl)
        tbl.addnext(spacer())
        tbl.getnext().addnext(pic)
        pic.addnext(cap)

    tree.write(str(doc_file), xml_declaration=True, encoding="UTF-8", standalone=True)
    rels.write(str(rels_file), xml_declaration=True, encoding="UTF-8", standalone=True)
    ct.write(str(ct_file), xml_declaration=True, encoding="UTF-8", standalone=True)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for name in names + [n for n in new_names if n not in names]:
            z.write(work / name, name)
    shutil.rmtree(work)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

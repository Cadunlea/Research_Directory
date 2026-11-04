"""Arial 11 throughout, grayscale tables, US Letter, 0.87 in margins."""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document("raw.docx")
FONT = "Arial"

def set_font(style_or_run, size=None, bold=None, color=None):
    f = style_or_run.font
    f.name = FONT
    if size: f.size = Pt(size)
    if bold is not None: f.bold = bold
    if color: f.color.rgb = RGBColor.from_string(color)
    rpr = style_or_run.element.get_or_add_rPr() if hasattr(style_or_run.element, "get_or_add_rPr") else None
    if rpr is not None:
        fonts = rpr.find(qn("w:rFonts"))
        if fonts is None:
            fonts = OxmlElement("w:rFonts"); rpr.insert(0, fonts)
        for key in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            fonts.set(qn(key), FONT)
        for key in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
            if fonts.get(qn(key)) is not None:
                del fonts.attrib[qn(key)]

sizes = {"Title": (16, True), "Subtitle": (11, False), "Author": (11, False),
         "Date": (11, False), "Heading 1": (14, True), "Heading 2": (12, True),
         "Heading 3": (11, True)}
for style in doc.styles:
    if style.type != 1:          # paragraph styles
        continue
    size, bold = sizes.get(style.name, (11, None))
    set_font(style, size, bold, "222222")
    if style.name.startswith("Heading") or style.name == "Title":
        style.paragraph_format.space_before = Pt(12)
        style.paragraph_format.space_after = Pt(6)

for section in doc.sections:
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(section, side, Inches(0.87))

def all_paragraphs(container):
    for p in container.paragraphs:
        yield p
    for t in container.tables:
        for row in t.rows:
            for cell in row.cells:
                yield from all_paragraphs(cell)

for p in all_paragraphs(doc):
    for r in p.runs:
        size = r.font.size
        set_font(r)
        if p.style.name not in sizes:
            r.font.size = Pt(11)

def border(el, tag, color="A6A6A6", sz="4"):
    b = OxmlElement(f"w:{tag}")
    b.set(qn("w:val"), "single"); b.set(qn("w:sz"), sz); b.set(qn("w:color"), color)
    el.append(b)

widths = [[1.25, 1.75, 1.6, 2.16], [1.5, 3.16, 1.0, 1.1], [0.42, 1.2, 1.75, 1.95, 1.44]]
for index, table in enumerate(doc.tables):
    tbl = table._tbl
    tblPr = tbl.tblPr
    for old in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(old)
    borders = OxmlElement("w:tblBorders")
    for tag in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border(borders, tag)
    tblPr.append(borders)
    ncols = len(table.columns)
    table.autofit = False
    cols = widths[index] if index < len(widths) else None
    for i, row in enumerate(table.rows):
        for j, cell in enumerate(row.cells):
            if cols and j < len(cols):
                cell.width = Inches(cols[j])
            if i == 0:
                shd = OxmlElement("w:shd")
                shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto")
                shd.set(qn("w:fill"), "EFEFEF")
                cell._tc.get_or_add_tcPr().append(shd)
                for p in cell.paragraphs:
                    for r in p.runs:
                        r.font.bold = True
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.space_before = Pt(2)
    # group rows (Data, Input, Network, ...): one italic label spanning the row
    for row in table.rows[1:]:
        cells = row.cells
        if cells[0].text.strip() and not any(c.text.strip() for c in cells[1:]) and len(cells) == 5:
            merged = cells[0].merge(cells[-1])
            for p in merged.paragraphs[1:]:
                p._element.getparent().remove(p._element)
    if cols:
        grid = tbl.tblGrid
        for gc, w in zip(grid.findall(qn("w:gridCol")), cols):
            gc.set(qn("w:w"), str(int(w * 1440)))

doc.core_properties.author = "Caelan Dunlea"
doc.core_properties.title = "Food-intake detection from AIM-2 sensor signals: literature review and model description"
doc.save("/home/user/Research_Directory/docs/literature_review/literature_review.docx")
print("saved")

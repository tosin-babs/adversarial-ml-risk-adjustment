"""
Typeset the pandoc DOCX to match the PDF.

pandoc writes a plain Word file: default page, left-aligned body text, and
tables with no rules. This restyles it in place:

  page        A4, margins 22 mm top and bottom, 20 mm left and right
  body        Palatino 10.5 pt, justified, 1.15 line spacing, 6 pt after
  headings    title 17 pt; section headings 13 pt with a rule beneath;
              subsections 11 pt italic
  tables      Arial 8 pt, rules above and below the header row and below
              the last row (booktabs style), header bold, first column left
              and the rest right, full text width, rows kept together
  captions    "Table N." and "Figure N." bold, kept with the next paragraph
  figures     centered, at most the text width
  references  hanging indent, 9.5 pt

Requires python-docx.
"""

from __future__ import annotations

import re

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Mm, Pt, RGBColor

BODY_FONT = "Palatino Linotype"
TABLE_FONT = "Arial"
INK = RGBColor(0x11, 0x11, 0x11)


def _font(style_or_run, name, size=None, bold=None, italic=None):
    f = style_or_run.font
    f.name = name
    rpr = style_or_run.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        fonts.set(qn(a), name)
    if size is not None:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    f.color.rgb = INK


# Child order the WordprocessingML schema requires; Word rejects files that
# break it even where other readers do not.
ORDER = {
    "pPr": ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl", "numPr",
            "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens", "kinsoku", "wordWrap",
            "overflowPunct", "topLinePunct", "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd",
            "snapToGrid", "spacing", "ind", "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc",
            "textDirection", "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr",
            "sectPr", "pPrChange"],
    "tblPr": ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize", "tblStyleColBandSize",
              "tblW", "jc", "tblCellSpacing", "tblInd", "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook",
              "tblCaption", "tblDescription"],
    "tcPr": ["cnfStyle", "tcW", "gridSpan", "hMerge", "vMerge", "tcBorders", "shd", "noWrap", "tcMar",
             "textDirection", "tcFitText", "vAlign", "hideMark"],
}


def _child(parent, name):
    """Get or create parent's child w:<name>, inserted in schema order."""
    el = parent.find(qn(f"w:{name}"))
    if el is not None:
        return el
    el = OxmlElement(f"w:{name}")
    kind = parent.tag.split("}")[1]
    order = ORDER.get(kind)
    if order and name in order:
        after = order[order.index(name) + 1:]
        for i, c in enumerate(list(parent)):
            if c.tag.split("}")[1] in after:
                parent.insert(i, el)
                return el
    parent.append(el)
    return el


def _border(el, edge, sz=8, val="single"):
    """Add a border to a paragraph (w:pBdr) or table cell (w:tcBorders)."""
    tag = "w:pBdr" if el.tag.endswith("}pPr") else "w:tcBorders"
    box = _child(el, tag[2:])
    edges = ["top", "left", "start", "bottom", "right", "end", "insideH", "insideV", "tl2br", "tr2bl", "between", "bar"]
    b = box.find(qn(f"w:{edge}"))
    if b is None:
        b = OxmlElement(f"w:{edge}")
        later = edges[edges.index(edge) + 1:]
        for i, c in enumerate(list(box)):
            if c.tag.split("}")[1] in later:
                box.insert(i, b)
                break
        else:
            box.append(b)
    b.set(qn("w:val"), val)
    b.set(qn("w:sz"), str(sz))
    b.set(qn("w:space"), "1" if tag == "w:pBdr" else "0")
    b.set(qn("w:color"), "333333")


def _styles(doc):
    st = doc.styles
    for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
        if name in st:
            _font(st[name], BODY_FONT, 10.5)
    for name in ("Body Text", "First Paragraph", "Normal"):
        if name in st:
            pf = st[name].paragraph_format
            pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            pf.space_before = Pt(0)
            pf.space_after = Pt(6)
            pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
            pf.line_spacing = 1.15
            pf.first_line_indent = Cm(0)
    heads = {"Title": (17, True, False), "Heading 1": (17, True, False),
             "Heading 2": (13, True, False), "Heading 3": (11, True, True)}
    for name, (size, bold, italic) in heads.items():
        if name in st:
            _font(st[name], BODY_FONT, size, bold, italic)
            pf = st[name].paragraph_format
            pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
            pf.space_before = Pt(16 if name != "Heading 3" else 10)
            pf.space_after = Pt(5)
            pf.keep_with_next = True
    if "Heading 2" in st:
        _border(st["Heading 2"].element.get_or_add_pPr(), "bottom", sz=6)
    for name in ("Image Caption", "Table Caption"):
        if name in st:
            _font(st[name], BODY_FONT, 9, italic=False)
            st[name].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            st[name].paragraph_format.space_after = Pt(10)


def _page(doc):
    for s in doc.sections:
        s.page_width, s.page_height = Mm(210), Mm(297)
        s.top_margin = s.bottom_margin = Mm(22)
        s.left_margin = s.right_margin = Mm(20)


def _numeric(text):
    t = text.strip().replace(",", "").replace("$", "").replace("₦", "").replace("%", "")
    t = t.replace("−", "-").replace("(", "").replace(")", "").replace("+", "")
    return bool(re.fullmatch(r"-?\d+(\.\d+)?(e-?\d+)?", t)) or t in ("", "–", "-")


def _widths(t, total_dxa):
    """Column widths in twentieths of a point, from content: each column gets
    a share proportional to its longest cell text (capped), with a floor, so
    labels do not wrap mid-word and numbers stay on one line."""
    rows = list(t.rows)
    ncol = max(len(r.cells) for r in rows)
    want = []
    for j in range(ncol):
        texts = [r.cells[j].text.strip() for r in rows if j < len(r.cells)]
        head = texts[0] if texts else ""
        body = texts[1:] if len(texts) > 1 else texts
        longest_word = max((len(w) for x in texts for w in x.split()), default=4)
        longest = max((len(x) for x in body), default=4)
        cap = 42 if j == 0 else 16
        want.append(max(min(longest, cap), longest_word, 4, min(len(head), 10)))
    tot = sum(want)
    return [int(total_dxa * w / tot) for w in want]


def _tables(doc, width):
    total_dxa = int(170 * 56.7)
    for t in doc.tables:
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        widths = _widths(t, total_dxa)
        grid = t._tbl.tblGrid
        for g, wv in zip(grid.findall(qn("w:gridCol")), widths):
            g.set(qn("w:w"), str(wv))
        for row in t.rows:
            for j, cell in enumerate(row.cells):
                if j < len(widths):
                    tcPr = cell._tc.get_or_add_tcPr()
                    tcW = _child(tcPr, "tcW")
                    tcW.set(qn("w:w"), str(widths[j]))
                    tcW.set(qn("w:type"), "dxa")
                    if j > 0 and _numeric(cell.text):
                        _child(tcPr, "noWrap")
        tblPr = t._tbl.tblPr
        w = _child(tblPr, "tblW")
        w.set(qn("w:type"), "dxa")
        w.set(qn("w:w"), str(total_dxa))
        for extra in tblPr.findall(qn("w:tblLayout"))[1:]:
            tblPr.remove(extra)
        _child(tblPr, "tblLayout").set(qn("w:type"), "fixed")
        mar = _child(tblPr, "tblCellMar")
        for c in list(mar):
            mar.remove(c)
        for edge, v in (("top", 20), ("left", 60), ("bottom", 20), ("right", 60)):
            e = OxmlElement(f"w:{edge}")
            e.set(qn("w:w"), str(v))
            e.set(qn("w:type"), "dxa")
            mar.append(e)
        nrows = len(t.rows)
        for i, row in enumerate(t.rows):
            trPr = row._tr.get_or_add_trPr()
            if trPr.find(qn("w:cantSplit")) is None:
                trPr.append(OxmlElement("w:cantSplit"))
            if i == 0 and trPr.find(qn("w:tblHeader")) is None:
                trPr.append(OxmlElement("w:tblHeader"))
            for j, cell in enumerate(row.cells):
                tcPr = cell._tc.get_or_add_tcPr()
                if i == 0:
                    _border(tcPr, "top", sz=10)
                    _border(tcPr, "bottom", sz=6)
                if i == nrows - 1:
                    _border(tcPr, "bottom", sz=10)
                for p in cell.paragraphs:
                    pf = p.paragraph_format
                    pf.space_before = Pt(0)
                    pf.space_after = Pt(0)
                    pf.line_spacing = 1.0
                    if j == 0 or not _numeric(cell.text):
                        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    else:
                        pf.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                    if i == 0 and j > 0:
                        pf.alignment = WD_ALIGN_PARAGRAPH.RIGHT if _numeric_col(t, j) else WD_ALIGN_PARAGRAPH.LEFT
                    for r in p.runs:
                        _font(r, TABLE_FONT, 8, bold=True if i == 0 else None)


def _numeric_col(t, j):
    vals = [row.cells[j].text for row in list(t.rows)[1:] if j < len(row.cells)]
    return bool(vals) and sum(_numeric(v) for v in vals) >= 0.7 * len(vals)


def _paragraphs(doc, width):
    in_refs = False
    for p in doc.paragraphs:
        text = p.text.strip()
        if p.style.name.startswith("Heading") and text.lower().startswith("references"):
            in_refs = True
            continue
        if p.style.name.startswith("Heading"):
            in_refs = False
            continue
        if re.match(r"^(Table|Figure) [0-9A-Z]+[a-z]?\.", text):
            p.paragraph_format.keep_with_next = text.startswith("Table")
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.space_before = Pt(10)
            for r in p.runs:
                r.font.size = Pt(9.5)
        # italic table notes directly under a caption
        if p.runs and all(r.italic for r in p.runs if r.text.strip()) and len(text) > 40:
            for r in p.runs:
                r.font.size = Pt(8.5)
            p.paragraph_format.keep_with_next = True
        if in_refs and re.match(r"^\d+\.", text) or (in_refs and p.style.name in ("Compact", "Body Text", "First Paragraph")):
            pf = p.paragraph_format
            pf.left_indent = Cm(0.7)
            pf.first_line_indent = Cm(-0.7)
            pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
            pf.space_after = Pt(3)
            for r in p.runs:
                r.font.size = Pt(9.5)
        # author and affiliation block, keywords: not justified
        if text.startswith(("¹", "²", "³", "**", "Keywords", "Word count", "Email")) or "corresponding author" in text:
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        # figures
        for blip in p._p.iter(qn("a:blip")):
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
        for ext in p._p.iter(qn("wp:extent")):
            cx, cy = int(ext.get("cx")), int(ext.get("cy"))
            if cx > width:
                ext.set("cx", str(width))
                ext.set("cy", str(int(cy * width / cx)))
                for a in p._p.iter(qn("a:ext")):
                    a.set("cx", str(width))
                    a.set("cy", str(int(cy * width / cx)))


def _rules(doc):
    """pandoc writes a horizontal rule as a VML rectangle; replace it with a
    thin paragraph border."""
    for p in doc.paragraphs:
        picts = list(p._p.iter(qn("w:pict")))
        if picts and not p.text.strip() and "rect" in p._p.xml:
            for r in list(p._p.findall(qn("w:r"))):
                p._p.remove(r)
            _border(p._p.get_or_add_pPr(), "bottom", sz=4)
            p.paragraph_format.space_after = Pt(10)


def style(path):
    doc = Document(str(path))
    _page(doc)
    _styles(doc)
    width = int(Mm(170))
    _tables(doc, width)
    _paragraphs(doc, width)
    _rules(doc)
    # numbered list items in references are pandoc "Compact" list paragraphs
    doc.save(str(path))


if __name__ == "__main__":
    import sys
    for f in sys.argv[1:]:
        style(f)
        print(f"styled {f}")

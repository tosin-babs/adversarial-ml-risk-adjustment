"""
Build the submission documents from the markdown source.

Produces, in manuscript/:
  Paper7_Ologunbaba.docx  - what most journals want uploaded
  Paper7_Ologunbaba.pdf   - for reading and for the preprint server

The tables are regenerated from the analysis CSVs first (make_tables.py), so a
rebuild after a rerun of the pipeline cannot leave stale numbers in the tables.
The prose is written by hand and checked by check_manuscript.py.

Requires pandoc. The PDF goes through HTML and headless Chrome.
"""

from __future__ import annotations

import shutil
import subprocess

import config
import make_tables

MS = config.ROOT / "manuscript"
SRC = MS / "Paper7_manuscript.md"
TABLES = MS / "tables.md"
COMBINED = MS / ".combined.md"
DOCX = MS / "Paper7_Ologunbaba.docx"
PDF = MS / "Paper7_Ologunbaba.pdf"
HTML = MS / ".paper.html"

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 22mm 20mm; }
body { font-family: "Palatino", "Palatino Linotype", Georgia, serif; font-size: 10.5pt;
       line-height: 1.5; color: #111; }
h1 { font-size: 17pt; line-height: 1.25; margin: 0 0 4pt; }
h2 { font-size: 13pt; margin: 20pt 0 6pt; border-bottom: 1px solid #bbb;
     padding-bottom: 3pt; }
h3 { font-size: 11.2pt; margin: 14pt 0 4pt; font-style: italic; font-weight: 600; }
p  { margin: 0 0 7pt; text-align: justify; hyphens: auto; }
img { max-width: 100%; display: block; margin: 10pt auto 4pt; }
table { border-collapse: collapse; width: 100%; font-size: 8.2pt; margin: 8pt 0 14pt;
        font-family: "Helvetica Neue", Arial, sans-serif; }
th, td { border-bottom: 0.5pt solid #ccc; padding: 3pt 5pt; text-align: right; }
th:first-child, td:first-child { text-align: left; }
thead th { border-bottom: 1pt solid #333; border-top: 1pt solid #333; font-weight: 600; }
code { font-family: "SF Mono", Menlo, monospace; font-size: 8.6pt; background: #f2f2f2;
       padding: 0 2pt; }
hr { border: 0; border-top: 0.5pt solid #bbb; margin: 16pt 0; }
ol { padding-left: 18pt; } ol li { margin-bottom: 3pt; font-size: 9.4pt; }
"""


def main():
    if shutil.which("pandoc") is None:
        raise SystemExit("pandoc not found: brew install pandoc")
    make_tables.main()
    COMBINED.write_text(SRC.read_text() + "\n\n" + TABLES.read_text())
    css = MS / ".paper.css"
    css.write_text(CSS)
    common = ["--resource-path", f"{MS}:{config.ROOT}"]
    subprocess.run(["pandoc", str(COMBINED), "-o", str(DOCX), *common], check=True)
    import docx_style
    docx_style.style(DOCX)
    print(f"wrote {DOCX.relative_to(config.ROOT)} ({DOCX.stat().st_size / 1024:,.0f} KB)")
    subprocess.run(["pandoc", str(COMBINED), "-s", "--metadata", "title=",
                    "-c", str(css), "--embed-resources", *common,
                    "-o", str(HTML)], check=True)
    if shutil.os.path.exists(CHROME):
        subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-sandbox",
                        "--no-pdf-header-footer", f"--print-to-pdf={PDF}",
                        HTML.as_uri()], check=True, capture_output=True)
        print(f"wrote {PDF.relative_to(config.ROOT)} ({PDF.stat().st_size / 1024:,.0f} KB)")
    else:
        print("Chrome not found; skipped the PDF. The DOCX is the submission file.")
    for f in (COMBINED, HTML, css):
        f.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

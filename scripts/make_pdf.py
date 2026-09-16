#!/usr/bin/env python3
"""Render report.md to a PDF: Markdown -> styled HTML -> headless Chrome.

Chrome is the only PDF engine on this machine (no pandoc, no wkhtmltopdf), so
the print stylesheet below is what makes the output read as a document rather
than a web page. Figures come from figures/*.png, which the notebook writes.

    python3 scripts/make_pdf.py                      # report.md -> report.pdf
    python3 scripts/make_pdf.py --source notes.md --out notes.pdf

Needs: pip install -r requirements-notebook.txt
"""

import argparse
import pathlib
import shutil
import subprocess
import sys

import markdown

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Print stylesheet. Same ink and rule colours as the figures, so the tables and
# the charts look like one document.
CSS = """
@page { size: A4; margin: 18mm 15mm 16mm; }

body {
  font-family: "DejaVu Sans", system-ui, -apple-system, sans-serif;
  font-size: 10pt; line-height: 1.5; color: #0b0b0b; background: #ffffff;
  margin: 0;
}

h1 { font-size: 19pt; line-height: 1.25; margin: 0 0 2pt; letter-spacing: -0.2pt; }
h2 { font-size: 13pt; margin: 20pt 0 6pt; padding-top: 9pt; border-top: 0.5pt solid #e1e0d9; }
h3 { font-size: 11pt; margin: 14pt 0 4pt; }
h1, h2, h3 { break-after: avoid; }
h2 + p, h3 + p, h2 + table, h3 + table { break-before: avoid; }

/* The subtitle and the dateline that open the report. */
h1 + p { font-size: 11.5pt; color: #52514e; margin: 0 0 2pt; }
h1 + p + p { color: #898781; margin: 0 0 6pt; }

p, li { orphans: 2; widows: 2; }
ul, ol { padding-left: 16pt; }
li { margin: 2pt 0; }

blockquote {
  margin: 10pt 0; padding: 2pt 0 2pt 10pt;
  border-left: 2pt solid #c3c2b7; color: #52514e;
}

table {
  border-collapse: collapse; width: 100%; font-size: 8.5pt;
  margin: 8pt 0 12pt;
  /* Long tables flow across pages instead of being pushed whole onto the next
     one, which left a half-empty page. Rows stay intact; headers repeat. */
  break-inside: auto;
}
thead { display: table-header-group; }
tr { break-inside: avoid; }
th {
  text-align: left; padding: 4pt 6pt; font-weight: 600;
  border-bottom: 0.8pt solid #c3c2b7;
}
td { padding: 3.5pt 6pt; border-bottom: 0.4pt solid #e1e0d9; vertical-align: top; }
tbody tr:nth-child(even) { background: #faf9f6; }

img { max-width: 100%; display: block; margin: 10pt auto 12pt; break-inside: avoid; }

code {
  font-family: "DejaVu Sans Mono", monospace; font-size: 8.5pt;
  background: #f2f1ed; padding: 0.5pt 2.5pt; border-radius: 2pt;
}
pre {
  background: #f7f6f3; border: 0.4pt solid #e1e0d9; border-radius: 3pt;
  padding: 8pt 10pt; font-size: 8.5pt; line-height: 1.45;
  break-inside: avoid; white-space: pre-wrap;
}
pre code { background: none; padding: 0; }

hr { border: 0; border-top: 0.5pt solid #e1e0d9; margin: 16pt 0; }
a { color: #2a78d6; text-decoration: none; }
strong { font-weight: 600; }
"""

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<base href="{base}">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
{body}
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="report.md")
    ap.add_argument("--out", default="report.pdf")
    args = ap.parse_args()

    source = (ROOT / args.source).resolve()
    out_pdf = (ROOT / args.out).resolve()
    if not source.exists():
        sys.exit(f"{source} not found")

    body = markdown.markdown(
        source.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    html = build / f"{source.stem}.html"
    # base href = repo root, so figures/*.png resolve from the build directory.
    html.write_text(
        TEMPLATE.format(base=ROOT.as_uri() + "/", title=source.stem, css=CSS, body=body),
        encoding="utf-8",
    )

    chrome = next((shutil.which(c) for c in
                   ("google-chrome", "chromium", "chromium-browser", "google-chrome-stable")
                   if shutil.which(c)), None)
    if chrome is None:
        sys.exit("no Chrome/Chromium found - install one, or open build/report.html and print to PDF")

    result = subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
         "--no-pdf-header-footer",
         "--virtual-time-budget=20000",      # let every figure finish loading
         f"--print-to-pdf={out_pdf}", html.as_uri()],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not out_pdf.exists():
        print(result.stdout + result.stderr)
        sys.exit("Chrome failed to print the PDF")

    print(f"{source.name} -> {out_pdf.relative_to(ROOT)} "
          f"({out_pdf.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

"""README.md → README.pdf. LTR, no page frames."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "README.md"
HTML = ROOT / "README.html"
PDF = ROOT / "README.pdf"

CSS = """
* { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
html, body { direction: ltr; }
body {
  font-family: "Segoe UI", Calibri, Arial, sans-serif;
  font-size: 11pt;
  line-height: 1.55;
  color: #1a1a1a;
  max-width: 100%;
}
h1 { font-size: 18pt; margin: 0 0 0.55em; color: #143a6b; page-break-after: avoid; }
h2 { font-size: 13.5pt; margin: 1.15em 0 0.35em; color: #143a6b; page-break-after: avoid; }
h3 { font-size: 12pt; margin: 0.9em 0 0.25em; color: #1e3a5f; page-break-after: avoid; }
p { margin: 0.4em 0; }
ul, ol { margin: 0.35em 0 0.35em 1.25em; }
li { margin: 0.1em 0; }
code, pre { font-family: Consolas, "Courier New", monospace; font-size: 8.8pt; }
pre { background: #f4f4f4; padding: 8px 10px; white-space: pre-wrap; word-break: break-word; }
table { border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: 0.6em 0; }
th, td { border: none; padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #143a6b; color: #fff; }
td { background: #f7f7f7; }
blockquote { margin: 0.5em 0; padding: 0.25em 0.8em; background: #f7f5ef; }
mjx-container { direction: ltr !important; }
a { color: #143a6b; }
"""


def latex_to_plain(text: str) -> str:
    text = text.replace(r"\varepsilon", "ε").replace(r"\epsilon", "ε")
    text = text.replace(r"\sigma", "σ").replace(r"\tau", "τ")
    text = text.replace(r"\in", "∈").replace(r"\mathrm{mean}", "mean")
    text = text.replace(r"\bar d", "d̄").replace(r"\cdot", "·")
    text = re.sub(r"\\\((.+?)\\\)", r"\1", text, flags=re.S)
    text = re.sub(r"\\\[(.+?)\\\]", r"\1", text, flags=re.S)
    return text


def main() -> None:
    body = markdown.markdown(
        latex_to_plain(MD.read_text(encoding="utf-8")),
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>README</title>
<style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""
    HTML.write_text(html, encoding="utf-8")
    tmp = PDF.with_name("README.tmp.pdf")
    if tmp.exists():
        tmp.unlink()
    printer = ROOT / "scripts" / "_print_pdf.mjs"
    subprocess.run(
        ["node", str(printer), str(HTML.resolve()), str(tmp.resolve())],
        check=True,
        cwd=str(ROOT / "scripts"),
    )
    if PDF.exists():
        try:
            PDF.unlink()
        except OSError:
            pass
    tmp.replace(PDF)
    print(f"wrote {PDF} ({PDF.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

"""Persian report PDF: cover (IUT logo), linked TOC, RTL body."""
from __future__ import annotations

import base64
import re
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "REPORT_FA.md"
HTML = ROOT / "REPORT_FA.html"
PDF = ROOT / "REPORT_FA.pdf"
LOGO = ROOT / "assets" / "iut-logo.png"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

NAVY = "#143a6b"
GOLD = "#c4a35a"
FONT_DIR = ROOT / "assets" / "fonts"


def vazir_faces() -> str:
    chunks = []
    for fname, weight in (
        ("Vazirmatn-Regular.ttf", 400),
        ("Vazirmatn-Medium.ttf", 500),
        ("Vazirmatn-Bold.ttf", 700),
        ("Vazirmatn-Bold.ttf", 800),
    ):
        b64 = base64.b64encode((FONT_DIR / fname).read_bytes()).decode("ascii")
        chunks.append(
            f"@font-face{{font-family:Vazirmatn;src:url(data:font/ttf;base64,{b64})"
            f" format('truetype');font-weight:{weight};font-style:normal;font-display:block;}}"
        )
    return "\n".join(chunks)


CSS = rf"""
* {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
@page {{ size: A4; margin: 20mm 18mm 20mm 18mm; }}
html {{ direction: rtl; }}
body {{
  font-family: "B Nazanin", Vazirmatn, Tahoma, sans-serif;
  font-size: 11pt;
  line-height: 1.85;
  color: #1a1a1a;
  text-align: justify;
}}

h1 {{
  font-size: 16pt; line-height: 1.45; margin: 1.15em 0 0.4em;
  page-break-after: avoid; color: {NAVY}; font-weight: 700;
}}
h1.chapter {{
  page-break-before: always;
  margin-top: 0;
  padding-top: 6mm;
  border-top: 1.35pt solid {NAVY};
}}
h2 {{
  font-size: 13pt; margin: 1em 0 0.3em; page-break-after: avoid; color: {NAVY};
}}
h3 {{ font-size: 11.4pt; margin: 0.8em 0 0.25em; page-break-after: avoid; color: #1e3a5f; }}
p {{ margin: 0.42em 0; }}
ul, ol {{ margin: 0.35em 0; padding-right: 1.35em; padding-left: 0; }}
li {{ margin: 0.12em 0; }}
hr {{
  border: 0;
  border-top: 0.9pt solid {GOLD};
  margin: 1.5em 0 0.3em;
}}
code, pre {{
  font-family: Consolas, "Courier New", monospace;
  direction: ltr; text-align: left; font-size: 8.5pt; unicode-bidi: isolate;
}}
pre {{ background: #f4f4f4; padding: 8px; white-space: pre-wrap; }}
table {{
  border-collapse: collapse; width: 100%; font-size: 7.1pt;
  line-height: 1.3; margin: 0.7em 0; word-break: break-word;
}}
thead {{ display: table-header-group; }}
tr {{ page-break-inside: avoid; }}
th, td {{ border: none; padding: 3px 4px; text-align: center; vertical-align: top; }}
th {{ background: {NAVY}; color: #fff; font-weight: 600; }}
td {{ background: #f7f7f7; }}
blockquote {{
  background: #f7f5ef;
  margin: 0.55em 0; padding: 0.35em 0.85em;
}}
.math, .math-display {{
  unicode-bidi: isolate;
  direction: ltr;
  font-style: italic;
  font-family: "Cambria Math", "Segoe UI", Times, serif;
}}
.math {{ white-space: nowrap; }}
.math-display {{
  display: block;
  text-align: center;
  margin: 0.7em 0;
}}
a {{ color: {NAVY}; }}

.cover {{
  page-break-after: always;
  min-height: 228mm;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  justify-content: flex-start;
  padding: 6mm 8mm 4mm;
}}
.cover .bism {{ font-size: 13.5pt; color: {NAVY}; margin: 0 0 7mm; font-weight: 600; }}
.cover img.logo {{ width: 44mm; height: auto; margin: 0 0 5mm 0; }}
.cover .rule {{
  width: 78%; height: 8mm; margin: 2mm 0 5mm;
}}
.cover .uni {{ font-size: 17pt; font-weight: 800; color: {NAVY}; margin: 0.15em 0; }}
.cover .fac {{ font-size: 12pt; margin: 0.2em 0 8mm; color: #243047; line-height: 1.7; }}
.cover .kind {{
  font-size: 12.5pt; font-weight: 700; color: #fff;
  background: {NAVY};
  padding: 0.35em 1.4em;
  letter-spacing: 0.08em;
  margin: 0 0 8mm;
}}
.cover .title {{
  font-size: 16pt; font-weight: 800; line-height: 1.6;
  color: {NAVY}; margin: 0 10mm 0.5em;
}}
.cover .sub {{ font-size: 10.5pt; color: #3a3a3a; margin: 0 14mm 10mm; line-height: 1.7; }}
.cover .card {{
  width: 78%;
  background: #fbfaf6;
  padding: 4.5mm 6mm;
  text-align: right;
  font-size: 11.5pt;
  line-height: 2.05;
  color: #1a1a1a;
}}
.cover .card b {{ color: {NAVY}; }}
.cover .year {{ font-size: 12pt; color: {NAVY}; margin-top: 10mm; font-weight: 700; }}

.toc {{ page-break-after: always; }}
.toc > h1 {{
  text-align: center; border-right: 0;
  padding-right: 0; margin-top: 0;
}}
.toc ol {{ padding-right: 0.4em; list-style: none; }}
.toc a {{ color: {NAVY}; text-decoration: none; }}
.toc .h1 {{ font-weight: 700; margin: 0.45em 0 0.12em; font-size: 10.6pt; }}
.toc .h2 {{ font-size: 9.6pt; margin: 0.12em 1.1em 0.12em 0; color: #333; font-weight: 400; }}
"""


CMDS = {
    "varepsilon": "ε",
    "epsilon": "ε",
    "sigma": "σ",
    "tau": "τ",
    "phi": "φ",
    "cdot": "·",
    "times": "×",
    "in": "∈",
    "notin": "∉",
    "subset": "⊂",
    "subseteq": "⊆",
    "leq": "≤",
    "le": "≤",
    "geq": "≥",
    "ge": "≥",
    "neq": "≠",
    "approx": "≈",
    "sim": "∼",
    "simeq": "≃",
    "gg": "≫",
    "ll": "≪",
    "pm": "±",
    "to": "→",
    "mapsto": "↦",
    "infty": "∞",
    "sum": "∑",
    "prod": "∏",
    "cap": "∩",
    "cup": "∪",
    "emptyset": "∅",
    "ldots": "…",
    "cdots": "⋯",
    "quad": " ",
    "qquad": "  ",
    " ": " ",
    ",": " ",
}


def _brace(s: str, i: int) -> tuple[str, int]:
    if i >= len(s) or s[i] != "{":
        return "", i
    depth = 0
    j = i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1 : j], j + 1
        j += 1
    return s[i + 1 :], len(s)


def convert_tex(s: str) -> str:
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        if s.startswith("\\left", i):
            i += 5
            if i < n and s[i] in ".()[]|":
                if s[i] != ".":
                    out.append(s[i])
                i += 1
            continue
        if s.startswith("\\right", i):
            i += 6
            if i < n and s[i] in ".()[]|":
                if s[i] != ".":
                    out.append(s[i])
                i += 1
            continue
        if s.startswith("\\frac", i):
            i += 5
            a, i = _brace(s, i) if i < n and s[i] == "{" else (s[i], i + 1)
            b, i = _brace(s, i) if i < n and s[i] == "{" else (s[i], i + 1)
            out.append(f"({convert_tex(a)})/({convert_tex(b)})")
            continue
        if s.startswith("\\binom", i):
            i += 6
            a, i = _brace(s, i)
            b, i = _brace(s, i)
            out.append(f"C({convert_tex(a)},{convert_tex(b)})")
            continue
        if s.startswith(("\\mathbb", "\\mathcal", "\\mathrm", "\\text", "\\mathbf"), i):
            cmd_end = i + 1
            while cmd_end < n and s[cmd_end].isalpha():
                cmd_end += 1
            i = cmd_end
            if i < n and s[i] == "{":
                inner, i = _brace(s, i)
                t = convert_tex(inner)
            elif i < n:
                t = s[i]
                i += 1
            else:
                t = ""
            out.append("ℝ" if t == "R" else t)
            continue
        if s.startswith("\\overline", i):
            i += 9
            if i < n and s[i] == "{":
                inner, i = _brace(s, i)
                out.append(convert_tex(inner) + "\u0304")
            continue
        if s.startswith("\\bar", i) and (i + 4 >= n or not s[i + 4].isalpha()):
            i += 4
            if i < n and s[i] == "{":
                inner, i = _brace(s, i)
                out.append(convert_tex(inner) + "\u0304")
            elif i < n:
                out.append(s[i] + "\u0304")
                i += 1
            continue
        if s.startswith("\\lfloor", i):
            out.append("⌊")
            i += 7
            continue
        if s.startswith("\\rfloor", i):
            out.append("⌋")
            i += 7
            continue
        if s.startswith("\\|", i) or s.startswith("\\lVert", i) or s.startswith("\\rVert", i):
            out.append("‖")
            i += 2 if s.startswith("\\|", i) else 6
            continue
        if s.startswith("\\exp", i) and (i + 4 >= n or not s[i + 4].isalpha()):
            out.append("exp")
            i += 4
            continue
        if s.startswith("\\bmod", i):
            out.append(" mod ")
            i += 5
            continue
        if s.startswith("\\mod", i) and (i + 4 >= n or not s[i + 4].isalpha()):
            out.append(" mod ")
            i += 4
            continue
        if s[i] == "\\":
            j = i + 1
            if j < n and not s[j].isalpha():
                out.append(CMDS.get(s[j], s[j]))
                i = j + 1
                continue
            while j < n and s[j].isalpha():
                j += 1
            out.append(CMDS.get(s[i + 1 : j], s[i + 1 : j]))
            i = j
            if i < n and s[i] == " ":
                i += 1
            continue
        if s[i] in "^_":
            sup = s[i] == "^"
            i += 1
            if i < n and s[i] == "{":
                inner, i = _brace(s, i)
                piece = convert_tex(inner)
            elif i < n:
                piece = s[i]
                i += 1
            else:
                piece = ""
            out.append(("^" + piece) if sup else ("_" + piece))
            continue
        if s[i] == "{":
            inner, i = _brace(s, i)
            out.append(convert_tex(inner))
            continue
        if s[i] == "}":
            i += 1
            continue
        out.append(s[i])
        i += 1
    return "".join(out)


def math_html(blob: str) -> str:
    display = blob.startswith("\\[")
    inner = blob[2:-2].strip()
    rendered = convert_tex(inner)
    tag = "div" if display else "span"
    cls = "math-display" if display else "math"
    return f'<{tag} class="{cls}" dir="ltr">{rendered}</{tag}>'


def protect_math(text: str) -> tuple[str, list[str]]:
    blobs: list[str] = []

    def store(m: re.Match) -> str:
        blobs.append(math_html(m.group(0)))
        return f"§§MATH{len(blobs) - 1}§§"

    text = re.sub(r"\\\[(.+?)\\\]", store, text, flags=re.S)
    text = re.sub(r"\\\((.+?)\\\)", store, text, flags=re.S)
    return text, blobs


def restore_math(html: str, blobs: list[str]) -> str:
    return re.sub(r"§§MATH(\d+)§§", lambda m: blobs[int(m.group(1))], html)


def md_to_html(text: str) -> str:
    text, blobs = protect_math(text)
    html = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )
    return restore_math(html, blobs)


def add_heading_ids(html: str) -> tuple[str, list[tuple[str, int, str]]]:
    toc: list[tuple[str, int, str]] = []
    n = [0]

    def repl(m: re.Match) -> str:
        tag = m.group(1)
        inner = m.group(2)
        n[0] += 1
        i = n[0]
        plain = re.sub(r"<[^>]+>", "", inner)
        plain = re.sub(r"\s+", " ", plain).strip()
        toc.append((tag, i, plain))
        extra = ""
        if tag == "h1" and re.match(r"^(فصل|پیوست|منابع)", plain):
            extra = ' class="chapter"'
        return f"<{tag} id=\"s{i}\"{extra}>{inner}</{tag}>"

    html = re.sub(r"<(h[12])[^>]*>(.*?)</\1>", repl, html, flags=re.S)
    return html, toc


def toc_html(toc: list[tuple[str, int, str]]) -> str:
    items = []
    for tag, i, text in toc:
        cls = "h1" if tag == "h1" else "h2"
        items.append(f'<li class="{cls}"><a href="#s{i}">{text}</a></li>')
    return (
        '<nav class="toc"><h1>فهرست مطالب</h1><ol>\n'
        + "\n".join(items)
        + "\n</ol></nav>"
    )


def cover_html(logo_b64: str) -> str:
    return f"""
<section class="cover">
  <div class="bism">بسمه تعالی</div>
  <img class="logo" alt="آرم دانشگاه صنعتی اصفهان" src="data:image/png;base64,{logo_b64}"/>
  <div class="uni">دانشگاه صنعتی اصفهان</div>
  <div class="fac">دانشکده مهندسی برق و کامپیوتر<br/>گروه مهندسی کامپیوتر</div>
  <div class="rule"></div>
  <div class="kind">پروژهٔ کارشناسی</div>
  <div class="title">طبقه‌بندی مبتنی بر گراف با<br/>استفاده از لم نظم زمردی<br/>(Szemerdi)</div>
  <div class="card">
    <b>دانشجو:</b> سید محمد فاطمی<br/>
    <b>استاد راهنما:</b> دکتر زینب مالکی<br/>
    دانشیار، دانشکده مهندسی برق و کامپیوتر
  </div>
  <div class="year">اصفهان — ۱۴۰۵ هجری شمسی</div>
</section>
"""


def main() -> None:
    md = MD.read_text(encoding="utf-8")
    body = md_to_html(md)
    body, toc = add_heading_ids(body)
    logo_b64 = base64.b64encode(LOGO.read_bytes()).decode("ascii")
    html = f"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8"/>
<title>پروژه کارشناسی — سید محمد فاطمی</title>
<style>{vazir_faces()}
{CSS}</style>
</head>
<body>
{cover_html(logo_b64)}
{toc_html(toc)}
{body}
</body>
</html>
"""
    HTML.write_text(html, encoding="utf-8")
    tmp = PDF.with_name("REPORT_FA.new.pdf")
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
    print(f"wrote {PDF} ({PDF.stat().st_size} bytes) toc={len(toc)}")


if __name__ == "__main__":
    main()

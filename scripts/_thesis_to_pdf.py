"""BSc thesis PDF from THESIS_FA.md: cover (IUT logo), linked TOC, RTL body."""
from __future__ import annotations

import base64
import os
import re
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "THESIS_FA.md"
HTML = ROOT / "THESIS_FA.html"
PDF = ROOT / "THESIS_FA.pdf"
LOGO = ROOT / "assets" / "iut-logo.png"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")

NAVY = "#143a6b"
GOLD = "#c4a35a"
FONT_DIR = ROOT / "assets" / "fonts"

# B Nazanin is a licensed font; it is read from the user's font folder at
# build time and never copied into the repo. Vazirmatn stays as fallback.
BNAZANIN_CANDIDATES = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts" / "B-NAZANIN.TTF",
    Path(r"C:\Windows\Fonts\B-NAZANIN.TTF"),
    Path(r"C:\Windows\Fonts\BNAZANIN.TTF"),
    Path(r"C:\Windows\Fonts\B Nazanin.ttf"),
]


def find_bnazanin() -> Path | None:
    for p in BNAZANIN_CANDIDATES:
        if p.exists():
            return p
    return None


def font_faces() -> str:
    chunks = []
    bn = find_bnazanin()
    if bn is not None:
        b64 = base64.b64encode(bn.read_bytes()).decode("ascii")
        for weight in (400, 700):
            chunks.append(
                f"@font-face{{font-family:'B Nazanin';src:url(data:font/ttf;base64,{b64})"
                f" format('truetype');font-weight:{weight};font-style:normal;font-display:block;}}"
            )
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


# kept for callers that still import the old name
vazir_faces = font_faces


CSS = rf"""
* {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
@page {{ size: A4; margin: 22mm 20mm 24mm 20mm; }}
html {{ direction: rtl; }}
body {{
  font-family: "B Nazanin", "Times New Roman", Vazirmatn, Tahoma, serif;
  font-size: 13.5pt;
  line-height: 1.95;
  color: #1a1a1a;
  text-align: justify;
}}

.ltr {{
  direction: ltr; text-align: left; unicode-bidi: isolate;
  font-family: "Times New Roman", Cambria, Georgia, serif;
  font-size: 12pt; line-height: 1.6;
}}
.ltr p {{ margin: 0.5em 0; }}
.ltr h1 {{ text-align: left; font-family: "Times New Roman", Cambria, serif; }}

h1 {{
  font-size: 19pt; line-height: 1.45; margin: 1.15em 0 0.4em;
  page-break-after: avoid; color: {NAVY}; font-weight: 700;
}}
h1.chapter {{
  page-break-before: always;
  margin-top: 0;
  padding-top: 6mm;
  border-top: 1.35pt solid {NAVY};
}}
h2 {{
  font-size: 15.5pt; margin: 1em 0 0.3em; page-break-after: avoid; color: {NAVY};
}}
h3 {{ font-size: 14pt; margin: 0.8em 0 0.25em; page-break-after: avoid; color: #1e3a5f; }}
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
  direction: ltr; text-align: left; font-size: 10pt; unicode-bidi: isolate;
}}
p code, li code, td code {{ background: #f3f3f3; padding: 0 3px; border-radius: 2px; }}
pre {{ background: #f4f4f4; padding: 8px; white-space: pre-wrap; }}
table {{
  border-collapse: collapse; width: 100%; font-size: 9.6pt;
  line-height: 1.45; margin: 0.7em 0; word-break: break-word;
}}
table.wide {{ font-size: 8.6pt; line-height: 1.3; }}
thead {{ display: table-header-group; }}
tr {{ page-break-inside: avoid; }}
th, td {{ border: none; padding: 3px 4px; text-align: center; vertical-align: top; }}
th {{ background: {NAVY}; color: #fff; font-weight: 600; }}
td {{ background: #f7f7f7; }}
td .math, th .math {{ font-size: 0.95em; }}
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
img {{
  max-width: 100%; height: auto; display: block;
  margin: 0.9em auto 0.35em; page-break-inside: avoid;
}}
p img {{ margin-top: 0.5em; }}
p.caption {{
  text-align: center; font-size: 11.5pt; color: #333;
  margin: 0 0 1.1em;
}}

.cover {{
  page-break-after: always;
  height: 243mm;
  box-sizing: border-box;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  justify-content: flex-start;
  padding: 4mm 8mm 2mm;
}}
.cover .bism {{ font-size: 15pt; color: {NAVY}; margin: 0 0 5mm; font-weight: 700; }}
.cover img.logo {{ width: 40mm; height: auto; margin: 0 0 4mm 0; }}
.cover .rule {{
  width: 78%; height: 5mm; margin: 1mm 0 3mm;
}}
.cover .uni {{ font-size: 20pt; font-weight: 800; color: {NAVY}; margin: 0.1em 0; }}
.cover .fac {{ font-size: 14pt; margin: 0.1em 0 6mm; color: #243047; line-height: 1.6; }}
.cover .kind {{
  font-size: 14pt; font-weight: 700; color: #fff;
  background: {NAVY};
  padding: 0.3em 1.4em;
  letter-spacing: 0.04em;
  margin: 0 0 6mm;
}}
.cover .title {{
  font-size: 19pt; font-weight: 800; line-height: 1.55;
  color: {NAVY}; margin: 0 10mm 0.4em;
}}
.cover .sub {{ font-size: 12.5pt; color: #3a3a3a; margin: 0 14mm 7mm; line-height: 1.6; }}
.cover .card {{
  width: 78%;
  background: #fbfaf6;
  padding: 4mm 6mm;
  text-align: right;
  font-size: 13.5pt;
  line-height: 1.95;
  color: #1a1a1a;
}}
.cover .card b {{ color: {NAVY}; }}
.cover .year {{ font-size: 14pt; color: {NAVY}; margin-top: 7mm; font-weight: 700; }}

.toc {{ page-break-after: always; }}
.toc > h1 {{
  text-align: center; border-right: 0;
  padding-right: 0; margin-top: 0;
}}
.toc ol {{ padding-right: 0.4em; list-style: none; }}
.toc a {{ color: {NAVY}; text-decoration: none; }}
.toc li {{ line-height: 1.6; }}
.toc .h1 {{ font-weight: 700; margin: 0.4em 0 0.08em; font-size: 12.5pt; }}
.toc .h2 {{ font-size: 11.5pt; margin: 0.05em 1.1em 0.05em 0; color: #333; font-weight: 400; }}
"""


CMDS = {
    "varepsilon": "ε",
    "epsilon": "ϵ",
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
    "dots": "…",
    "vdots": "⋮",
    "quad": " ",
    "qquad": "  ",
    " ": " ",
    ",": " ",
    ";": " ",
    "ell": "ℓ",
    "Delta": "Δ",
    "delta": "δ",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "lambda": "λ",
    "mu": "μ",
    "forall": "∀",
    "exists": "∃",
    "mid": " | ",
    "setminus": "∖",
    "langle": "⟨",
    "rangle": "⟩",
    "bigl": "",
    "bigr": "",
    "Bigl": "",
    "Bigr": "",
    "big": "",
    "Big": "",
    "min": "min",
    "max": "max",
    "arg": "arg",
    "log": "log",
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
            out.append(f"<sup>{piece}</sup>" if sup else f"<sub>{piece}</sub>")
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


def plain_alt(text: str) -> str:
    """Image alt text must be plain: math there would inject HTML into the attribute."""

    def strip(m: re.Match) -> str:
        return re.sub(r"<[^>]+>", "", convert_tex(m.group(1)))

    text = re.sub(r"\\\((.+?)\\\)", strip, text)
    return text.replace('"', "'")


def sanitize_image_alts(text: str) -> str:
    return re.sub(
        r"!\[([^\]]*)\]\(",
        lambda m: "![" + plain_alt(m.group(1)) + "](",
        text,
    )


def md_to_html(text: str) -> str:
    text = sanitize_image_alts(text)
    text, blobs = protect_math(text)
    html = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )
    html = restore_math(html, blobs)
    html = inline_images(html)
    html = mark_captions(html)
    html = wrap_ltr_sections(html, ("Abstract", "منابع"), ltr_heading=("Abstract",))
    html = mark_wide_tables(html)
    return html


def mark_captions(html: str) -> str:
    """A paragraph whose whole content is one <em> is a figure caption."""
    return re.sub(
        r"<p><em>(.*?)</em></p>",
        lambda m: '<p class="caption">' + m.group(1) + "</p>",
        html,
        flags=re.S,
    )


def wrap_ltr_sections(html: str, titles: tuple[str, ...], ltr_heading: tuple[str, ...] = ()) -> str:
    """Everything after an <h1> in ``titles`` up to the next <h1>/<hr> becomes LTR."""
    for title in titles:
        pat = re.compile(
            r"(<h1[^>]*>\s*" + re.escape(title) + r"\s*</h1>)(.*?)(?=<h1|<hr\s*/?>|\Z)",
            flags=re.S,
        )
        if title in ltr_heading:
            html = pat.sub(
                lambda m: '<div class="ltr">' + m.group(1) + m.group(2) + "</div>", html
            )
        else:
            html = pat.sub(
                lambda m: m.group(1) + '<div class="ltr">' + m.group(2) + "</div>", html
            )
    return html


def mark_wide_tables(html: str) -> str:
    """Tables with 7+ columns get the compact style."""

    def repl(m: re.Match) -> str:
        table = m.group(0)
        head = re.search(r"<thead>.*?</thead>", table, flags=re.S)
        ncols = len(re.findall(r"<th", head.group(0))) if head else 0
        if ncols >= 7:
            return table.replace("<table>", '<table class="wide">', 1)
        return table

    return re.sub(r"<table>.*?</table>", repl, html, flags=re.S)


def inline_images(html: str) -> str:
    """Embed local figures as data URIs so print from file:// still sees them."""

    def repl(m: re.Match) -> str:
        src = m.group(1).replace("\\", "/")
        path = (ROOT / src).resolve() if not Path(src).is_absolute() else Path(src)
        if not path.exists():
            return m.group(0)
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return f'<img src="data:{mime};base64,{b64}"'

    return re.sub(r'<img src="([^"]+)"', repl, html)


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
        if tag == "h1" and re.match(r"^(فصل|پیوست|منابع|چکیده|Abstract|فهرست)", plain):
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
  <div class="uni">دانشگاه صنعتی اصفهان</div>
  <div class="rule"></div>
  <div class="kind">پروژۀ کارشناسی</div>
  <div class="title">ارزیابی تجربی لم منظمی سِمِرِدی<br/>در خوشه‌بندی مبتنی بر گراف</div>
  <div class="sub">پیاده‌سازی و ارزیابی مقالۀ منتشرشده در سال ۲۰۲۶</div>
  <div class="card">
    <b>دانشجو:</b> سید محمد فاطمی<br/>
    <b>استاد راهنما:</b> دکتر زینب مالکی<br/>
    دانشیار، دانشکده مهندسی برق و کامپیوتر
  </div>
  <div class="year">۱۴۰۵ هجری شمسی</div>
</section>
"""


def normalize_persian_for_pdf(text: str) -> str:
    # B Nazanin often fails on ه + combining hamza (U+0654); use ۀ (U+06C0) like the rest of the thesis.
    return text.replace("\u0647\u0654", "\u06c0")


def main() -> None:
    md = normalize_persian_for_pdf(MD.read_text(encoding="utf-8"))
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
    tmp = PDF.with_name("THESIS_FA.new.pdf")
    if tmp.exists():
        tmp.unlink()
    printer = ROOT / "scripts" / "_print_pdf.mjs"
    subprocess.run(
        ["node", str(printer), str(HTML.resolve()), str(tmp.resolve()), "--page-numbers"],
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

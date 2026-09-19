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
@page {{
  size: A4;
  margin: 25mm 25mm 28mm 25mm;
  @bottom-center {{
    content: none;
  }}
}}
@page unnumbered {{
  margin: 20mm 20mm 20mm 20mm;
  @bottom-center {{ content: none; }}
  @top-left {{ content: none; }}
}}
html {{ direction: rtl; }}
body {{
  font-family: "B Nazanin", "Times New Roman", Vazirmatn, Tahoma, serif;
  font-size: 13pt;
  line-height: 1.9;
  color: #1a1a1a;
  text-align: justify;
}}

.ltr {{
  direction: ltr; text-align: left; unicode-bidi: isolate;
  font-family: "Times New Roman", Cambria, Georgia, serif;
  font-size: 12pt; line-height: 1.6;
}}
.ltr p {{
  margin: 0.55em 0;
  padding-left: 2.4em;
  text-indent: -2.4em;
  text-align: left;
}}
.ltr h1 {{ text-align: left; font-family: "Times New Roman", Cambria, serif; }}

h1 {{
  font-size: 19pt; line-height: 1.45; margin: 1.15em 0 0.4em;
  page-break-after: avoid; color: {NAVY}; font-weight: 700;
}}
h1.chapter {{
  page-break-before: always;
  margin-top: 0;
  padding-top: 8mm;
}}
h2 {{
  font-size: 15.5pt; margin: 1em 0 0.3em; page-break-after: avoid; color: {NAVY};
}}
h3 {{ font-size: 14pt; margin: 0.8em 0 0.25em; page-break-after: avoid; color: #1e3a5f; }}
p {{ margin: 0.42em 0; }}
ul, ol {{ margin: 0.35em 0; padding-right: 1.35em; padding-left: 0; }}
li {{ margin: 0.12em 0; }}
hr {{
  display: none;
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
td:first-child .math {{ font-size: 1.05em; }}
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
a {{ color: {NAVY}; text-decoration: underline; }}
.ltr a {{
  color: {NAVY};
  text-decoration: underline;
  word-break: break-all;
}}
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
  page: unnumbered;
  page-break-after: always;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  justify-content: flex-start;
  padding: 6mm 16mm 10mm;
  color: #000;
}}
.cover img.logo {{
  width: 28mm;
  height: auto;
  margin: 0 0 3mm;
  display: block;
}}
.cover .uni {{ font-size: 10pt; font-weight: 700; margin: 0 0 6mm; color: #000; }}
.cover .fac {{ font-size: 14pt; font-weight: 400; margin: 0 0 14mm; color: #000; }}
.cover .title {{
  font-size: 16pt; font-weight: 700; line-height: 1.85;
  color: #000; margin: 0 8mm 12mm;
}}
.cover .kind {{ font-size: 16pt; font-weight: 400; margin: 0 0 14mm; color: #000; }}
.cover .author {{ font-size: 12pt; font-weight: 700; margin: 0 0 14mm; color: #000; }}
.cover .advisor-label {{ font-size: 14pt; font-weight: 400; margin: 0 0 4mm; color: #000; }}
.cover .advisor {{ font-size: 12pt; font-weight: 700; margin: 0 0 14mm; color: #000; }}
.cover .year {{ font-size: 10pt; font-weight: 700; margin: 0; padding-top: 0; color: #000; }}

.rights {{
  page: unnumbered;
  page-break-after: always;
  box-sizing: border-box;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: center;
  min-height: 220mm;
  padding: 0 22mm;
  color: #000;
}}
.rights p {{
  font-size: 14pt;
  font-weight: 700;
  line-height: 2.1;
  margin: 0;
}}

.toc {{
  page-break-before: always;
  page-break-after: always;
  font-size: 12pt;
  color: #000;
}}
.toc > h1 {{
  text-align: center;
  border-right: 0;
  padding-right: 0;
  margin: 0 0 8mm;
  font-size: 12pt;
  font-weight: 700;
  color: #000;
}}
.toc-head {{
  display: flex;
  justify-content: space-between;
  font-size: 10pt;
  font-weight: 700;
  margin: 0 0 4mm;
  color: #000;
}}
.toc ol {{ padding: 0; list-style: none; margin: 0; }}
.toc a {{ color: #000; text-decoration: none; }}
.toc li {{ margin: 0; padding: 0; }}
.toc .row {{
  display: flex;
  flex-direction: row;
  align-items: baseline;
  gap: 0.35em;
  line-height: 1.85;
  font-size: 12pt;
  font-weight: 400;
}}
.toc .h1 .row {{ font-weight: 700; font-size: 10pt; margin: 0.55em 0 0.1em; }}
.toc .h2 .row {{ font-size: 12pt; font-weight: 400; margin: 0; padding-right: 0; }}
.toc .toc-title {{ flex: 0 1 auto; max-width: 78%; }}
.toc .toc-dots {{
  flex: 1 1 auto;
  border-bottom: 1px dotted #222;
  height: 0.85em;
  min-width: 8mm;
}}
.toc .toc-page {{
  flex: 0 0 auto;
  min-width: 1.6em;
  text-align: left;
  direction: ltr;
  unicode-bidi: isolate;
  font-variant-numeric: tabular-nums;
}}
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
    html = re.sub(r"<hr\s*/?>", "", html)
    html = inline_images(html)
    html = mark_captions(html)
    html = wrap_ltr_sections(html, ("مراجع",), ltr_heading=())
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
        if tag == "h1" and re.match(
            r"^(فصل|پیوست|مراجع|واژه‌نامه|چکیده|فهرست)", plain
        ):
            extra = ' class="chapter"'
        return f"<{tag} id=\"s{i}\"{extra}>{inner}</{tag}>"

    html = re.sub(r"<(h[12])[^>]*>(.*?)</\1>", repl, html, flags=re.S)
    return html, toc


SKIP_TOC = {
    "فهرست نمادها",
}


def toc_html(toc: list[tuple[str, int, str]]) -> str:
    items = [
        '<li class="h1"><div class="row">'
        '<span class="toc-title">فهرست مطالب</span>'
        '<span class="toc-dots"></span>'
        '<span class="toc-page" data-self="toc"></span>'
        "</div></li>"
    ]
    for tag, i, text in toc:
        if text in SKIP_TOC:
            continue
        cls = "h1" if tag == "h1" else "h2"
        items.append(
            f'<li class="{cls}"><a class="row" href="#s{i}">'
            f'<span class="toc-title">{text}</span>'
            f'<span class="toc-dots"></span>'
            f'<span class="toc-page" data-target="s{i}"></span>'
            f"</a></li>"
        )
    return (
        '<nav class="toc" id="toc"><h1>فهرست مطالب</h1>'
        '<div class="toc-head"><span>عنوان</span><span>صفحه</span></div>\n'
        "<ol>\n"
        + "\n".join(items)
        + "\n</ol></nav>"
    )


def peel_leading_h1(html: str) -> tuple[str, str]:
    m = re.match(r"(<h1[\s\S]*?</h1>[\s\S]*?)(?=<h1|\Z)", html)
    if not m:
        return "", html
    return m.group(1), html[m.end() :]


def front_html(logo_b64: str) -> str:
    return f"""
<section class="cover">
  <img class="logo" alt="" src="data:image/png;base64,{logo_b64}"/>
  <div class="uni">دانشگاه صنعتی اصفهان</div>
  <div class="fac">دانشکده مهندسی برق و کامپیوتر</div>
  <div class="title">طبقه‌بندی مبتنی بر گراف با<br/>استفاده از لم نظم زمردی<br/>(Szemerdi)</div>
  <div class="kind">پروژه کارشناسی مهندسی کامپیوتر</div>
  <div class="author">سید محمد فاطمی</div>
  <div class="advisor-label">استاد راهنما</div>
  <div class="advisor">دکتر زینب مالکی</div>
  <div class="year">۱۴۰۵</div>
</section>
<section class="rights">
  <p>کلیه حقوق مادی مترتب بر نتایج مطالعات،<br/>
  ابتکارات و نوآوری‌های ناشی از تحقیق موضوع<br/>
  این پژوهش متعلق به دانشگاه صنعتی اصفهان است.</p>
</section>
"""


def normalize_persian_for_pdf(text: str) -> str:
    # B Nazanin often fails on ه + combining hamza (U+0654); use ۀ (U+06C0) like the rest of the thesis.
    return text.replace("\u0647\u0654", "\u06c0")


def _fa_digits(n: int) -> str:
    return str(n).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))


def _norm_txt(s: str) -> str:
    import unicodedata

    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", "", s).replace("\u200c", "")


def fill_toc_page_numbers(html: str, pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return html
    reader = PdfReader(str(pdf_path))
    pages = [_norm_txt(p.extract_text() or "") for p in reader.pages]
    toc_idxs = [
        i
        for i, t in enumerate(pages)
        if "فهرستمطالب" in t or "عنوانصفحه" in t
    ]
    toc_page = (toc_idxs[0] + 1) if toc_idxs else 2
    body_start = next(
        (i for i, t in enumerate(pages) if "مسئلهاینپروژه" in t),
        (toc_idxs[-1] + 1) if toc_idxs else 2,
    )
    after_toc = body_start
    html = re.sub(
        r'(<span class="toc-page" data-self="toc">)(</span>)',
        rf"\g<1>{_fa_digits(toc_page)}\2",
        html,
        count=1,
    )
    last = after_toc + 1  # 1-based

    def repl(m: re.Match) -> str:
        nonlocal last
        title = m.group(1)
        target = m.group(2)
        plain = re.sub(r"<[^>]+>", "", title)
        plain = re.sub(r"^[۰-۹]+-[۰-۹]+\s*", "", plain)
        needle = re.sub(r"[\u064b-\u065f]", "", _norm_txt(plain))[:16]
        page_no = last
        if needle:
            for i in range(last - 1, len(pages)):
                blob = re.sub(r"[\u064b-\u065f]", "", pages[i])
                if needle in blob:
                    page_no = i + 1
                    last = page_no
                    break
        return (
            f'<span class="toc-title">{title}</span>'
            f'<span class="toc-dots"></span>'
            f'<span class="toc-page" data-target="{target}">{_fa_digits(page_no)}</span>'
        )

    return re.sub(
        r'<span class="toc-title">(.*?)</span>\s*'
        r'<span class="toc-dots"></span>\s*'
        r'<span class="toc-page" data-target="([^"]+)"></span>',
        repl,
        html,
        flags=re.S,
    )


def stamp_fa_page_numbers(pdf_path: Path, skip: int = 2) -> None:
    """Draw Eastern Arabic page numbers; Chrome cannot print counter(page, farsi)."""
    from io import BytesIO

    from pypdf import PdfReader, PdfWriter
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    font = find_bnazanin()
    if font is None:
        raise FileNotFoundError("B Nazanin is required to stamp Persian page numbers")
    pdfmetrics.registerFont(TTFont("BNazanin", str(font)))
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    writer.append(reader)
    for i, page in enumerate(writer.pages):
        if i < skip:
            continue
        w = float(page.mediabox.width)
        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=(w, float(page.mediabox.height)))
        c.setFont("BNazanin", 11)
        c.setFillColorRGB(0.2, 0.2, 0.2)
        c.drawCentredString(w / 2.0, 32, _fa_digits(i + 1))
        c.save()
        stamp = PdfReader(buf)
        page.merge_page(stamp.pages[0])
    out = pdf_path.with_suffix(".stamped.pdf")
    with out.open("wb") as f:
        writer.write(f)
    out.replace(pdf_path)


def print_pdf(html_path: Path, pdf_path: Path) -> None:
    printer = ROOT / "scripts" / "_print_pdf.mjs"
    if pdf_path.exists():
        pdf_path.unlink()
    subprocess.run(
        ["node", str(printer), str(html_path.resolve()), str(pdf_path.resolve())],
        check=True,
        cwd=str(ROOT / "scripts"),
    )


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
{front_html(logo_b64)}
{toc_html(toc)}
{body}
</body>
</html>
"""
    HTML.write_text(html, encoding="utf-8")
    tmp = PDF.with_name("THESIS_FA.new.pdf")
    print_pdf(HTML, tmp)
    filled = fill_toc_page_numbers(html, tmp)
    if filled != html:
        HTML.write_text(filled, encoding="utf-8")
        print_pdf(HTML, tmp)
    stamp_fa_page_numbers(tmp)
    if PDF.exists():
        try:
            PDF.unlink()
        except OSError:
            pass
    tmp.replace(PDF)
    print(f"wrote {PDF} ({PDF.stat().st_size} bytes) toc={len(toc)}")


if __name__ == "__main__":
    main()

"""BSc defense slides (widescreen PPTX, RTL Persian)."""
from __future__ import annotations

import re
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "THESIS_FA_slides.pptx"
OUT_ALT = ROOT / "THESIS_FA_slides_new.pptx"
LOGO = ROOT / "assets" / "iut-logo.png"
FIG = ROOT / "assets" / "figures"

PAPER = RGBColor(0xEB, 0xE4, 0xD6)
PAPER2 = RGBColor(0xF7, 0xF1, 0xE6)
PANEL = RGBColor(0xFF, 0xFC, 0xF8)
INK = RGBColor(0x1B, 0x18, 0x16)
MUTED = RGBColor(0x6B, 0x64, 0x5C)
LINE = RGBColor(0xD4, 0xCB, 0xBA)
ACCENT = RGBColor(0xA3, 0x3B, 0x24)
NAVY = RGBColor(0x1A, 0x33, 0x4A)
NAVY_DEEP = RGBColor(0x0D, 0x1C, 0x2C)
SAGE = RGBColor(0x2C, 0x5E, 0x52)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SHELL = RGBColor(0xDE, 0xD5, 0xC4)
FONT_FA = "B Nazanin"
FONT_EN = "Calibri"
FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
ARABIC = re.compile(r"[\u0600-\u06FF]")
LATIN = re.compile(
    r"[A-Za-z\u00C0-\u024F][A-Za-z\u00C0-\u024F0-9+\-_.\u00B2\u00B3\u00B9]*"
)


def _clean_fa(text: str) -> str:
    return (
        text.replace("\u2066", "")
        .replace("\u2069", "")
        .replace("\u200e", "")
        .replace("\u200f", "")
        .replace("ۀ", "ه")
        .replace("ٔ", "")
        .replace("·", " | ")
    )


def _style_run(run, *, rtl: bool) -> None:
    font = FONT_FA if rtl else FONT_EN
    run.font.name = font
    rPr = run._r.get_or_add_rPr()
    rPr.set("rtl", "1" if rtl else "0")
    rPr.set("lang", "fa-IR" if rtl else "en-US")
    pairs = (("ea", FONT_FA if rtl else FONT_EN), ("cs", FONT_FA if rtl else FONT_EN), ("latin", FONT_EN))
    for tag, face in pairs:
        el = rPr.find(qn(f"a:{tag}"))
        if el is None:
            el = etree.SubElement(rPr, qn(f"a:{tag}"))
        el.set("typeface", face)


def _p(p, align="r") -> None:
    pPr = p._p.get_or_add_pPr()
    if align == "r":
        p.alignment = PP_ALIGN.RIGHT
        pPr.set("rtl", "1")
        pPr.set("algn", "r")
    elif align == "en-r":
        p.alignment = PP_ALIGN.RIGHT
        pPr.set("rtl", "0")
        pPr.set("algn", "r")
    elif align == "c":
        p.alignment = PP_ALIGN.CENTER
        pPr.set("rtl", "0")
        pPr.set("algn", "ctr")
    else:
        p.alignment = PP_ALIGN.LEFT
        pPr.set("rtl", "0")
        pPr.set("algn", "l")


def _wipe(p) -> None:
    el = p._p
    for child in list(el):
        if child.tag == qn("a:r"):
            el.remove(child)


def add_run(p, text, size=20, bold=False, color=INK, rtl=True) -> None:
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    _style_run(run, rtl=rtl)


def add_mixed(p, text, size=20, bold=False, color=INK) -> None:
    text = _clean_fa(text)
    if not ARABIC.search(text):
        add_run(p, text, size=size, bold=bold, color=color, rtl=False)
        return
    pos = 0
    for m in LATIN.finditer(text):
        left = text[m.start() - 1] if m.start() else " "
        right = text[m.end()] if m.end() < len(text) else " "
        glued = left in "-/" or right in "-/" or bool(ARABIC.match(left) or ARABIC.match(right))
        if glued:
            continue
        if m.start() > pos:
            add_run(p, text[pos:m.start()], size=size, bold=bold, color=color, rtl=True)
        add_run(p, m.group(), size=size, bold=bold, color=color, rtl=False)
        pos = m.end()
    if pos < len(text):
        add_run(p, text[pos:], size=size, bold=bold, color=color, rtl=True)


def box(slide, l, t, w, h, fill=None, line=None, rounded=False):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    sh = slide.shapes.add_shape(kind, l, t, w, h)
    if rounded:
        try:
            sh.adjustments[0] = 0.1
        except Exception:
            pass
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(0.75)
    return sh


def _spPr(shape):
    return shape._element.spPr


def fill_alpha(shape, pct: float) -> None:
    solid = _spPr(shape).find(qn("a:solidFill"))
    if solid is None:
        return
    srgb = solid.find(qn("a:srgbClr"))
    if srgb is None:
        return
    el = srgb.find(qn("a:alpha"))
    if el is None:
        el = etree.SubElement(srgb, qn("a:alpha"))
    el.set("val", str(int(pct * 1000)))


def shadow(
    shape,
    *,
    blur=0.32,
    dist=0.09,
    dir_deg=120,
    alpha=22,
    color="1A334A",
) -> None:
    spPr = _spPr(shape)
    for old in spPr.findall(qn("a:effectLst")):
        spPr.remove(old)
    lst = etree.SubElement(spPr, qn("a:effectLst"))
    shdw = etree.SubElement(lst, qn("a:outerShdw"))
    shdw.set("blurRad", str(int(blur * 914400)))
    shdw.set("dist", str(int(dist * 914400)))
    shdw.set("dir", str(int(dir_deg * 60000)))
    shdw.set("algn", "ctr")
    shdw.set("rotWithShape", "0")
    srgb = etree.SubElement(shdw, qn("a:srgbClr"))
    srgb.set("val", color)
    a = etree.SubElement(srgb, qn("a:alpha"))
    a.set("val", str(int(alpha * 1000)))


def gradient(shape, c1: RGBColor, c2: RGBColor, angle: float = 108) -> None:
    shape.fill.gradient()
    shape.fill.gradient_angle = angle
    stops = shape.fill.gradient_stops
    stops[0].color.rgb = c1
    stops[1].color.rgb = c2


def blob(slide, l, t, w, h, color, alpha=10):
    sh = slide.shapes.add_shape(MSO_SHAPE.OVAL, l, t, w, h)
    sh.line.fill.background()
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    fill_alpha(sh, alpha)
    return sh


def paper_bg(prs, slide, *, inverse=False):
    W, H = prs.slide_width, prs.slide_height
    bg = box(slide, 0, 0, W, H, PAPER)
    if inverse:
        gradient(bg, NAVY_DEEP, NAVY, 118)
    else:
        gradient(bg, PAPER, PAPER2, 108)
        blob(slide, Inches(9.8), Inches(-1.4), Inches(5.2), Inches(5.2), ACCENT, 8)
        blob(slide, Inches(-1.6), Inches(4.6), Inches(4.4), Inches(4.4), NAVY, 9)
    rail = box(slide, 0, 0, Inches(0.14), H, ACCENT if not inverse else ACCENT)
    return bg


def tf(slide, l, t, w, h):
    frame = slide.shapes.add_textbox(l, t, w, h).text_frame
    frame.word_wrap = True
    return frame


def write(frame, lines, size=20, bold=False, color=INK, space=8, align="r"):
    first = True
    for line in lines:
        p = frame.paragraphs[0] if first else frame.add_paragraph()
        first = False
        _wipe(p)
        if ARABIC.search(line):
            use = align
        elif align == "c":
            use = "c"
        elif align == "r":
            use = "en-r"
        else:
            use = "l"
        _p(p, use)
        p.space_after = Pt(space)
        add_mixed(p, line, size=size, bold=bold, color=color)


def hairline(slide, l, t, w):
    box(slide, l, t, w, Pt(1.1), LINE)


def new(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def chrome(prs, slide, title: str, page: int, kicker: str = ""):
    W, H = prs.slide_width, prs.slide_height
    paper_bg(prs, slide)
    bar = box(slide, Inches(0.45), Inches(0.28), Inches(12.45), Inches(0.92), PANEL, rounded=True)
    shadow(bar, blur=0.22, dist=0.05, alpha=16)
    kicker = kicker or "دفاع کارشناسی"
    f = tf(slide, Inches(0.7), Inches(0.32), W - Inches(1.6), Inches(0.28))
    write(f, [kicker], size=12, color=ACCENT, space=0)
    f = tf(slide, Inches(0.7), Inches(0.55), W - Inches(1.6), Inches(0.55))
    write(f, [title], size=26, bold=True, color=NAVY, space=0)
    f = tf(slide, Inches(0.55), H - Inches(0.38), Inches(8.5), Inches(0.28))
    write(f, ["سید محمد فاطمی  |  دکتر زینب مالکی"], size=11, color=MUTED, space=0)
    chip = box(slide, W - Inches(1.55), H - Inches(0.42), Inches(0.95), Inches(0.28), NAVY, rounded=True)
    f = tf(slide, W - Inches(1.55), H - Inches(0.44), Inches(0.95), Inches(0.28))
    write(f, [str(page).translate(FA_DIGITS)], size=11, color=WHITE, space=0, align="c")


def add_picture_fit(slide, path: Path, l, t, w, h):
    if not path.exists():
        f = tf(slide, l, t, w, Inches(0.5))
        write(f, [f"شکل پیدا نشد: {path.name}"], size=16, color=MUTED, space=0)
        return
    pic = slide.shapes.add_picture(str(path), l, t, width=w)
    if pic.height > h:
        ratio = h / pic.height
        pic.height = h
        pic.width = int(pic.width * ratio)
        pic.left = int(l + (w - pic.width) / 2)


def fig_slide(prs, slides, title, path, caption="", kicker="شکل"):
    sl = new(prs)
    slides.append(sl)
    chrome(prs, sl, title, len(slides), kicker=kicker)
    cap_h = 0.85 if caption else 0
    shell = box(sl, Inches(0.5), Inches(1.38), Inches(12.35), Inches(5.62 - cap_h), SHELL, rounded=True)
    shadow(shell, blur=0.4, dist=0.12, alpha=26)
    box(
        sl,
        Inches(0.62),
        Inches(1.5),
        Inches(12.11),
        Inches(5.38 - cap_h),
        PANEL,
        rounded=True,
    )
    add_picture_fit(
        sl,
        path,
        Inches(0.9),
        Inches(1.62),
        Inches(11.55),
        Inches(4.95 - cap_h),
    )
    if caption:
        f = tf(sl, Inches(0.55), Inches(6.25), Inches(12.2), Inches(0.7))
        write(f, [caption], size=18, color=INK, space=0)
    return sl


def trio(slide, cards, top=1.7, height=4.6):
    n = len(cards)
    gap = 0.22
    width = (12.3 - gap * (n - 1)) / n
    for i, (kick, title, body) in enumerate(cards):
        x = Inches(0.5 + (n - 1 - i) * (width + gap))
        card = box(slide, x, Inches(top), Inches(width), Inches(height), PANEL, rounded=True)
        shadow(card, blur=0.28, dist=0.08, alpha=18)
        box(slide, x, Inches(top), Inches(width), Inches(0.12), ACCENT if i == 0 else NAVY)
        f = tf(slide, x + Inches(0.25), Inches(top + 0.35), Inches(width - 0.5), Inches(0.4))
        write(f, [kick], size=14, color=ACCENT, space=0)
        f = tf(slide, x + Inches(0.25), Inches(top + 0.9), Inches(width - 0.5), Inches(1.3))
        write(f, [title], size=22 if n > 3 else 26, bold=True, color=NAVY, space=0)
        f = tf(slide, x + Inches(0.25), Inches(top + 2.3), Inches(width - 0.5), Inches(2.0))
        write(f, [body], size=18, color=INK, space=0)


def numbered(slide, items, top=1.38):
    n = max(len(items), 1)
    row_h = min(1.02, 5.55 / n)
    for i, item in enumerate(items):
        y = Inches(top) + Inches(i * row_h)
        h = Inches(row_h - 0.1)
        card = box(slide, Inches(0.5), y, Inches(12.35), h, PANEL, rounded=True)
        shadow(card, blur=0.2, dist=0.05, alpha=14)
        box(slide, Inches(12.68), y + Inches(0.12), Inches(0.1), h - Inches(0.24), ACCENT if i == 0 else NAVY)
        f = tf(slide, Inches(11.55), y + Inches(0.14), Inches(1.0), h - Inches(0.22))
        write(f, [f"{i+1:02d}".translate(FA_DIGITS)], size=16, bold=True, color=ACCENT, space=0)
        f = tf(slide, Inches(0.75), y + Inches(0.14), Inches(10.65), h - Inches(0.22))
        write(f, [item], size=18, color=INK, space=0)


def build() -> Path:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    W, H = prs.slide_width, prs.slide_height
    slides = []

    s = new(prs)
    slides.append(s)
    paper_bg(prs, s)
    plate = box(s, Inches(0.5), Inches(0.42), Inches(12.35), Inches(4.7), PANEL, rounded=True)
    shadow(plate, blur=0.45, dist=0.14, alpha=28)
    if LOGO.exists():
        s.shapes.add_picture(str(LOGO), Inches(11.55), Inches(0.62), height=Inches(0.95))
    f = tf(s, Inches(0.85), Inches(0.7), Inches(10.2), Inches(0.4))
    write(f, ["دانشگاه صنعتی اصفهان  |  دانشکده مهندسی برق و کامپیوتر"], size=15, color=ACCENT, space=0)
    f = tf(s, Inches(0.85), Inches(1.45), Inches(11.6), Inches(2.15))
    write(
        f,
        ["طبقه‌بندی مبتنی بر گراف", "با استفاده از لم نظم زمردی"],
        size=36,
        bold=True,
        color=NAVY,
        space=6,
    )
    f = tf(s, Inches(0.85), Inches(3.85), Inches(10), Inches(0.4))
    write(f, ["Szemerédi regularity lemma"], size=16, color=MUTED, space=0)
    box(s, Inches(0.85), Inches(4.45), Inches(2.4), Inches(0.08), ACCENT)
    meta = [
        ("دانشجو", "سید محمد فاطمی"),
        ("سال", "۱۴۰۵"),
        ("مقطع", "کارشناسی مهندسی کامپیوتر"),
        ("استاد راهنما", "دکتر زینب مالکی"),
    ]
    for i, (lab, val) in enumerate(meta):
        x = Inches(0.5 + (len(meta) - 1 - i) * 3.2)
        card = box(s, x, Inches(5.4), Inches(3.02), Inches(1.55), PANEL, rounded=True)
        shadow(card, blur=0.22, dist=0.06, alpha=18)
        f = tf(s, x + Inches(0.18), Inches(5.55), Inches(2.7), Inches(0.32))
        write(f, [lab], size=12, color=ACCENT, space=0)
        f = tf(s, x + Inches(0.18), Inches(5.9), Inches(2.7), Inches(0.75))
        write(f, [val], size=16, bold=True, color=INK, space=0)

    sl = new(prs)
    slides.append(sl)
    chrome(prs, sl, "مشکل، و راه لم", len(slides), kicker="افتتاح")
    numbered(
        sl,
        [
            "مشکل اصلی خوشه‌بندی مبتنی بر گراف، بار محاسباتی گراف شباهت است. هر نمونه یک رأس است و شباهت‌ها یال می‌شوند، و این گراف خیلی زود سنگین می‌شود.",
            "کارهای قبلی نشان داده‌اند لم نظم زمردی می‌تواند این بار را کم کند.",
            "با این لم، گراف اصلی افراز می‌شود و از روی آن یک گراف کوچک ساخته می‌شود. این گراف کوچک ساختار اصلی را نگه می‌دارد، ولی تعداد رأس‌هایش خیلی کمتر است.",
            "خوشه‌بندی روی گراف کوچک انجام می‌شود و برچسب‌ها به گراف اصلی برمی‌گردند. به این ترتیب حجم محاسبه به‌طور محسوس کم می‌شود.",
        ],
    )

    sl = new(prs)
    slides.append(sl)
    chrome(prs, sl, "مقاله چه نشان داد", len(slides), kicker="افتتاح")
    numbered(
        sl,
        [
            "پارامترهای این روش روی نتیجه اثر جدی دارند و کارهای قبلی به این اثر نپرداخته بودند.",
            "مقاله این اثر را با چهار الگوریتم شناخته‌شده و تعداد زیادی دادهٔ واقعی بررسی کرد.",
            "بازهٔ مناسب پارامترها هم دقت خوشه‌بندی را بالا می‌برد و هم محاسبه را سبک‌تر می‌کند.",
            "افراز لم از افراز معمولی k-means بهتر است. الگوریتم‌های نسبتاً قدیمی، با همین لم، از روش‌های جدیدتر جلو می‌زنند. یعنی لم از یک قضیهٔ نظری به یک ابزار عملی برای خوشه‌بندی آمده است.",
        ],
    )

    fig_slide(
        prs,
        slides,
        "هر نمونه یک رأس",
        FIG / "concept_vertices.png",
        "هر سطر یک نمونه است و سه نقطهٔ خاکستری، سه ویژگی آن نمونه‌اند. همان سطر، با همان رنگ، یک دایره در گراف می‌شود. هنوز خطی بین دایره‌ها نیست.",
        kicker="از جدول تا گراف",
    )

    fig_slide(
        prs,
        slides,
        "شباهت، وزن یال است",
        FIG / "concept_edges.png",
        "دو نمونهٔ شبیه، یال کلفت می‌گیرند. دو نمونهٔ دور، یال نازک یا هیچ. رنگ و ضخامت یعنی شدت شباهت، نه یک یال صفر و یک.",
        kicker="از جدول تا گراف",
    )

    fig_slide(
        prs,
        slides,
        "جدول واقعی همین مسیر را می‌رود",
        FIG / "fig_from_table_to_graph.png",
        "ویژگی‌ها اول هم‌مقیاس می‌شوند، بعد شباهت حساب می‌شود، بعد گراف ساخته می‌شود. این گراف، ورودی هر دو راه آزمایش است.",
        kicker="از جدول تا گراف",
    )

    fig_slide(
        prs,
        slides,
        "چرا گراف اصلی سنگین است",
        FIG / "concept_heavy_graph.png",
        "چپ گراف شلوغ است. وسط همان رأس‌ها در چهار گروه جدا قرار می‌گیرند. راست هر گروه یک رأس شده و ضخامت خط، شدت ارتباط دو گروه است.",
        kicker="انگیزه",
    )

    sl = new(prs)
    slides.append(sl)
    chrome(prs, sl, "لم در سه حرکت", len(slides), kicker="لم")
    moves = [
        ("حرکت ۱", "گروه‌بندی", "رأس‌ها به چند گروه تقریباً هم‌اندازه تقسیم می‌شوند.", FIG / "concept_partition.png"),
        ("حرکت ۲", "یک عدد به‌جای یال‌ها", "بین دو گروه فقط چگالی ارتباط می‌ماند.", FIG / "concept_density.png"),
        ("حرکت ۳", "گراف کوچک", "هر گروه یک رأس می‌شود و خوشه‌بندی همان‌جا انجام می‌شود.", FIG / "concept_reduced.png"),
    ]
    gap = 0.22
    width = (12.3 - gap * 2) / 3
    for i, (kick, title, body, path) in enumerate(moves):
        x = 0.5 + (2 - i) * (width + gap)
        card = box(sl, Inches(x), Inches(1.5), Inches(width), Inches(5.15), PANEL, rounded=True)
        shadow(card, blur=0.28, dist=0.08, alpha=18)
        box(sl, Inches(x), Inches(1.5), Inches(width), Inches(0.1), ACCENT if i == 0 else NAVY)
        f = tf(sl, Inches(x + 0.2), Inches(1.68), Inches(width - 0.4), Inches(0.28))
        write(f, [kick], size=13, color=ACCENT, space=0)
        f = tf(sl, Inches(x + 0.2), Inches(1.96), Inches(width - 0.4), Inches(0.55))
        write(f, [title], size=20, bold=True, color=NAVY, space=0)
        add_picture_fit(sl, path, Inches(x + 0.18), Inches(2.6), Inches(width - 0.36), Inches(2.55))
        f = tf(sl, Inches(x + 0.2), Inches(5.25), Inches(width - 0.4), Inches(1.2))
        write(f, [body], size=15, color=INK, space=0)

    fig_slide(
        prs,
        slides,
        "حرکت اول: گروه‌بندی",
        FIG / "concept_partition.png",
        "چهار گروه تقریباً هم‌اندازه. نقطه‌های خاکستری کنار تصویر، رأس‌هایی‌اند که در هیچ گروه تمیزی جا نگرفته‌اند و کنار گذاشته می‌شوند.",
        kicker="لم",
    )

    fig_slide(
        prs,
        slides,
        "گروه منظم یعنی ارتباط یکنواخت",
        FIG / "concept_regular.png",
        "بین این دو گروه، یال‌ها همه‌جا تقریباً مثل هم‌اند. هر تکه از گروه چپ، با هر تکه از گروه راست، همان قدر رابطه دارد.",
        kicker="لم",
    )

    fig_slide(
        prs,
        slides,
        "گروه نامنظم یعنی یک جا شلوغ و یک جا خالی",
        FIG / "concept_irregular.png",
        "بالای تصویر پر از یال است و پایین تقریباً خالی. اینجا یک عدد میانگین، رابطه را درست خلاصه نمی‌کند. لم این جفت را نامنظم حساب می‌کند.",
        kicker="لم",
    )

    fig_slide(
        prs,
        slides,
        "همین تفاوت، روی یک شکل",
        FIG / "fig_eps_regular_pair.png",
        "چپ منظم است و راست نامنظم. اپسیلون میزان مجاز همین تفاوت است: اگر اختلاف از آن حد کمتر باشد، جفت را منظم می‌گیریم.",
        kicker="لم",
    )

    sl = new(prs)
    slides.append(sl)
    chrome(prs, sl, "سه certificate", len(slides), kicker="افراز")
    cards = [
        ("Alon 1", "یال کم، یعنی منظم", FIG / "concept_alon1.png"),
        ("Alon 3", "تودهٔ همسایهٔ مشترک", FIG / "concept_alon3.png"),
        ("Alon 2", "درجهٔ پرت", FIG / "concept_alon2.png"),
    ]
    gap = 0.22
    width = (12.3 - gap * 2) / 3
    for i, (kick, title, path) in enumerate(cards):
        x = 0.5 + (2 - i) * (width + gap)
        card = box(sl, Inches(x), Inches(1.5), Inches(width), Inches(5.15), PANEL, rounded=True)
        shadow(card, blur=0.28, dist=0.08, alpha=18)
        box(sl, Inches(x), Inches(1.5), Inches(width), Inches(0.1), ACCENT if i == 0 else NAVY)
        f = tf(sl, Inches(x + 0.18), Inches(1.7), Inches(width - 0.36), Inches(0.32))
        write(f, [kick], size=14, color=ACCENT, space=0)
        f = tf(sl, Inches(x + 0.18), Inches(2.05), Inches(width - 0.36), Inches(0.7))
        write(f, [title], size=20, bold=True, color=NAVY, space=0)
        add_picture_fit(sl, path, Inches(x + 0.15), Inches(2.85), Inches(width - 0.3), Inches(3.4))

    sl = new(prs)
    slides.append(sl)
    chrome(prs, sl, "سه اصلاح مقاله", len(slides), kicker="افراز")
    numbered(
        sl,
        [
            "هر گروه حداکثر با یک گروه دیگر نامنظم شمرده می‌شود. بدون این محدودیت، تعداد گروه‌ها در هر دور انفجاری زیاد می‌شد.",
            "certificate با روش درجهٔ Fiorucci ساخته می‌شود، نه با جست‌وجوی کامل روش اصلی.",
            "وقتی اندازهٔ گروه‌ها به‌قدر کافی کوچک شد، حلقه متوقف می‌شود. خروجی تقریباً منظم است، نه منظمِ اثبات‌شده.",
        ],
    )

    fig_slide(
        prs,
        slides,
        "refinement یعنی بریدن گروه",
        FIG / "concept_refine.png",
        "چپ، تکهٔ قرمز certificate است و هنوز داخل همان گروه. راست، همان تکه جدا شده و بقیه گروه خودش مانده. نقطه‌های خاکستری خرده‌هایی‌اند که به مجموعهٔ استثنایی می‌روند.",
        kicker="افراز",
    )

    fig_slide(
        prs,
        slides,
        "حرکت دوم: یک عدد به‌جای همهٔ یال‌ها",
        FIG / "concept_density.png",
        "چپ، همهٔ یال‌های بین دو گروه است. راست، همان رابطه با یک میله. ضخامت میله یعنی چگالی: سهم یال‌های موجود از همهٔ یال‌های ممکن.",
        kicker="لم",
    )

    fig_slide(
        prs,
        slides,
        "حرکت سوم: گراف کوچک",
        FIG / "concept_reduced.png",
        "هر گروه با فلش به رأس هم‌رنگ خودش می‌رود. ضخامت یال بین رأس‌های بزرگ، چگالی همان دو گروه است. گراف کوچک معمولاً ۱۶ تا ۶۴ رأس دارد.",
        kicker="لم",
    )

    sl = new(prs)
    slides.append(sl)
    chrome(prs, sl, "دو راه، از یک گراف", len(slides), kicker="pipeline")
    paths = [
        ("راه مستقیم", "همان گراف را همان‌جا رنگ می‌کنیم. هر رنگ یک خوشه است.", FIG / "concept_path_direct.png"),
        ("راه لم", "اول دو رأس بزرگ. بعد همان رنگ به همان نقطه‌ها برمی‌گردد.", FIG / "concept_path_lemma.png"),
    ]
    for i, (title, body, path) in enumerate(paths):
        x = 0.45 + (1 - i) * 6.45
        card = box(sl, Inches(x), Inches(1.48), Inches(6.2), Inches(5.2), PANEL, rounded=True)
        shadow(card, blur=0.28, dist=0.08, alpha=18)
        box(sl, Inches(x), Inches(1.48), Inches(6.2), Inches(0.1), ACCENT if i == 0 else NAVY)
        f = tf(sl, Inches(x + 0.25), Inches(1.68), Inches(5.7), Inches(0.45))
        write(f, [title], size=22, bold=True, color=NAVY, space=0)
        add_picture_fit(sl, path, Inches(x + 0.35), Inches(2.2), Inches(5.5), Inches(3.15))
        f = tf(sl, Inches(x + 0.25), Inches(5.5), Inches(5.7), Inches(0.95))
        write(f, [body], size=16, color=INK, space=0)

    fig_slide(
        prs,
        slides,
        "برگرداندن برچسب",
        FIG / "concept_lift.png",
        "خوشه روی گراف کوچک به هر گروه یک رنگ می‌دهد. همهٔ نمونه‌های داخل آن گروه همان رنگ را می‌گیرند. خروجی نهایی باز هم برچسبِ تک‌تک نمونه‌هاست.",
        kicker="pipeline",
    )

    fig_slide(
        prs,
        slides,
        "مسیر کار این پروژه",
        FIG / "fig_pipeline.png",
        "ساخت گراف یک بار انجام می‌شود. شاخهٔ پایه همان‌جا می‌ماند. شاخهٔ لم از افراز و گراف کوچک رد می‌شود و در پایان برچسب را بالا می‌آورد.",
        kicker="pipeline",
    )

    sl = new(prs)
    slides.append(sl)
    chrome(prs, sl, "چهار الگوریتم، هر دو راه", len(slides), kicker="pipeline")
    trio(
        sl,
        [
            ("۱", "SPC", "خوشه‌بندی طیفی. گراف را با بردارهای ویژه می‌بُرد."),
            ("۲", "APC", "انتشار وابستگی. نمونه‌ها خودشان نمایندهٔ خوشه را انتخاب می‌کنند."),
            ("۳", "DSet", "مجموعهٔ غالب. یک تکهٔ خیلی متراکم را به‌عنوان خوشه برمی‌دارد."),
            ("۴", "SPRG", "خوشه‌بندی طیفی با شباهت جنگل تصادفی. شباهت از جنگل می‌آید، بعد برش طیفی."),
        ],
        top=1.55,
        height=4.7,
    )

    fig_slide(
        prs,
        slides,
        "گروه لم با خوشه یکی نیست",
        FIG / "concept_polish.png",
        "دایره، گروه لم است و همهٔ داخلش یک رنگ دارند. نقطهٔ درشت میان همسایه‌های قرمز نشسته، ولی چون عضو گروه آبی است آبی مانده. polish فقط رنگ خوشهٔ همان نقطه را قرمز می‌کند.",
        kicker="بعد از لم",
    )

    sl = new(prs)
    slides.append(sl)
    chrome(prs, sl, "پروتکل آزمایش", len(slides), kicker="آزمایش")
    numbered(
        sl,
        [
            "بیست مجموعهٔ داده و چهار الگوریتم: هشتاد خانه، که هفتاد و هشت تایش معتبر ماند.",
            "ده مجموعه با همهٔ نمونه‌ها اجرا شد و ده مجموعه با حدود چهارصد نمونهٔ طبقه‌ای.",
            "برای هر خانه، شبکهٔ تنظیمات مثل آزمایش مقاله انتخاب شد و بهترین مقدار گزارش شد.",
            "ادعای اصلی این ارائه، مقایسه با راه پایهٔ خودمان است، نه رونویسی جدول چاپ‌شدهٔ مقاله.",
        ],
    )

    sl = new(prs)
    slides.append(sl)
    chrome(prs, sl, "نتیجه روی گراف خودمان", len(slides), kicker="آزمایش")
    f = tf(sl, Inches(0.55), Inches(1.45), Inches(12.2), Inches(0.7))
    write(f, ["هفتاد و هشت مقایسه. هر بار همان الگوریتم، یک بار روی گراف اصلی و یک بار روی گراف کوچک."], size=18, color=MUTED, space=0)
    cards = [
        ("۶۰", "گراف کوچک بهتر", SAGE),
        ("۳", "برابر", NAVY),
        ("۱۵", "گراف اصلی بهتر", ACCENT),
    ]
    for i, (n, lab, col) in enumerate(cards):
        x = Inches(0.5 + (2 - i) * 4.2)
        card = box(sl, x, Inches(2.3), Inches(3.95), Inches(3.15), PANEL, rounded=True)
        shadow(card, blur=0.32, dist=0.1, alpha=20)
        box(sl, x, Inches(2.3), Inches(3.95), Inches(0.12), col)
        f = tf(sl, x, Inches(2.6), Inches(3.95), Inches(1.5))
        write(f, [n], size=60, bold=True, color=col, space=0, align="c")
        f = tf(sl, x, Inches(4.3), Inches(3.95), Inches(0.7))
        write(f, [lab], size=20, color=INK, space=0, align="c")

    fig_slide(
        prs,
        slides,
        "اختلاف روی مجموعه‌های کامل",
        FIG / "fig_delta_nmi_full.png",
        "میلهٔ رو به بالا یعنی گراف کوچک از راه پایه بهتر بوده است.",
        kicker="نتیجه",
    )

    sl = new(prs)
    slides.append(sl)
    chrome(prs, sl, "کجا فرق زیاد بود", len(slides), kicker="مثال")
    trio(
        sl,
        [
            ("شراب", "پایه ضعیف، لم قوی", "سه الگوریتم از حدود نیم به نزدیک ۰/۸۹ رسیدند."),
            ("آپاندیسیت", "تطابق کامل", "دو الگوریتم از پایهٔ متوسط به ۱ رسیدند."),
            ("اسکناس", "طیفی ضعیف بود", "خوشه‌بندی طیفی از حدود ۰/۱۹ به حدود ۰/۸۲ رسید."),
        ],
        height=4.5,
    )

    sl = new(prs)
    slides.append(sl)
    chrome(prs, sl, "جمع‌بندی", len(slides), kicker="پایان")
    trio(
        sl,
        [
            ("۱", "لم فشرده می‌کند", "گراف بزرگ به چند ده رأس کاهش پیدا می‌کند."),
            ("۲", "کیفیت معمولاً بهتر ماند", "در ۶۰ مقایسه گراف کوچک بهتر بود، در ۳ مقایسه برابر، در ۱۵ مقایسه بدتر."),
            ("۳", "الگوریتم قدیمی قوی‌تر شد", "همان روش‌های کلاسیک، بعد از کوچک کردن گراف، در بیشتر آزمایش‌ها دقیق‌تر شدند. بیشترین سود جایی بود که راه پایه ناپایدار بود."),
        ],
        height=4.5,
    )

    s = new(prs)
    slides.append(s)
    paper_bg(prs, s, inverse=True)
    blob(s, Inches(9.2), Inches(-1.2), Inches(5.6), Inches(5.6), ACCENT, 18)
    blob(s, Inches(-1.8), Inches(4.2), Inches(4.8), Inches(4.8), SAGE, 14)
    plate = box(s, Inches(1.35), Inches(2.05), Inches(10.65), Inches(3.35), PANEL, rounded=True)
    shadow(plate, blur=0.5, dist=0.16, alpha=32, color="000000")
    f = tf(s, Inches(1.7), Inches(2.4), Inches(10), Inches(1.3))
    write(f, ["پرسش و پاسخ"], size=44, bold=True, color=NAVY, space=0)
    box(s, Inches(1.7), Inches(3.85), Inches(2.2), Inches(0.09), ACCENT)
    f = tf(s, Inches(1.7), Inches(4.15), Inches(10), Inches(0.6))
    write(f, ["با سپاس از دکتر زینب مالکی"], size=20, color=MUTED, space=0)

    try:
        prs.save(OUT)
        return OUT
    except PermissionError:
        prs.save(OUT_ALT)
        return OUT_ALT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path} ({path.stat().st_size} bytes)")

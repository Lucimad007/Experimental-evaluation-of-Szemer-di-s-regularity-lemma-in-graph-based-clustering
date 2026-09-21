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
    write(f, ["سید محمد فاطمی  |  دانشگاه صنعتی اصفهان"], size=11, color=MUTED, space=0)
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


def fig_slide(prs, slides, title, path, page_note=""):
    sl = new(prs)
    slides.append(sl)
    chrome(prs, sl, title, len(slides), kicker="شکل")
    shell = box(sl, Inches(0.5), Inches(1.38), Inches(12.35), Inches(5.62), SHELL, rounded=True)
    shadow(shell, blur=0.4, dist=0.12, alpha=26)
    box(sl, Inches(0.62), Inches(1.5), Inches(12.11), Inches(5.38), PANEL, rounded=True)
    add_picture_fit(sl, path, Inches(0.9), Inches(1.68), Inches(11.55), Inches(5.02))
    if page_note:
        sl.notes_slide.notes_text_frame.text = page_note
    return sl


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
    rule = box(s, Inches(0.85), Inches(4.45), Inches(2.4), Inches(0.08), ACCENT)
    meta = [
        ("دانشجو", "سید محمد فاطمی"),
        ("استاد راهنما", "دکتر زینب مالکی"),
        ("سال", "۱۴۰۵"),
        ("مقطع", "کارشناسی مهندسی کامپیوتر"),
    ]
    for i, (lab, val) in enumerate(meta):
        x = Inches(0.5 + i * 3.2)
        card = box(s, x, Inches(5.4), Inches(3.02), Inches(1.55), PANEL, rounded=True)
        shadow(card, blur=0.22, dist=0.06, alpha=18)
        f = tf(s, x + Inches(0.18), Inches(5.55), Inches(2.7), Inches(0.32))
        write(f, [lab], size=12, color=ACCENT, space=0)
        f = tf(s, x + Inches(0.18), Inches(5.9), Inches(2.7), Inches(0.75))
        write(f, [val], size=16, bold=True, color=INK, space=0)

    def body(title, items, notes="", kicker=""):
        sl = new(prs)
        slides.append(sl)
        chrome(prs, sl, title, len(slides), kicker=kicker)
        numbered(sl, items)
        if notes:
            sl.notes_slide.notes_text_frame.text = notes
        return sl

    outline = [
        "مسئله: خوشه‌بندی روی گراف بزرگ",
        "لم نظم زمردی و گراف کاهش‌یافته R",
        "الگوریتم ۱ مقاله هو و همکاران",
        "پیاده‌سازی و پروتکل این پروژه",
        "نتایج: مسیر R در برابر همان الگوریتم روی G",
        "محدودیت‌ها و جمع‌بندی",
    ]
    sl = new(prs)
    slides.append(sl)
    chrome(prs, sl, "مسیر ارائه", len(slides), kicker="فهرست")
    for i, item in enumerate(outline):
        col, row = i % 2, i // 2
        x = Inches(0.55 + col * 6.25)
        y = Inches(1.55 + row * 1.7)
        card = box(sl, x, y, Inches(5.95), Inches(1.45), PANEL, rounded=True)
        shadow(card, blur=0.28, dist=0.08, alpha=20)
        box(sl, x + Inches(5.8), y, Inches(0.15), Inches(1.45), ACCENT if i < 2 else SAGE)
        f = tf(sl, x + Inches(0.25), y + Inches(0.22), Inches(1.1), Inches(0.4))
        write(f, [f"{i+1:02d}".translate(FA_DIGITS)], size=16, bold=True, color=ACCENT, space=0)
        f = tf(sl, x + Inches(0.25), y + Inches(0.62), Inches(5.45), Inches(0.65))
        write(f, [item], size=18, bold=True, color=NAVY, space=0)
    sl.notes_slide.notes_text_frame.text = "حدود ۱۵ تا ۲۰ دقیقه؛ شکل‌ها را می‌توان سریع رد کرد اگر وقت کم باشد."

    body(
        "مسئله",
        [
            "در خوشه‌بندی گرافی هر نمونه یک رأس است و شباهت، وزن یال.",
            "الگوریتم روی گراف G اجرا می‌شود، نه روی بردار خام ویژگی.",
            "ماتریس شباهت از مرتبه توان دوم n حافظه می‌گیرد؛ بسیاری از روش‌ها از مرتبه توان سوم n زمان می‌برند.",
            "برای USPS با حدود ۱۱۰۰۰ نمونه، S نزدیک به ۱۲۰ میلیون درایه دارد.",
        ],
        kicker="زمینه",
    )

    fig_slide(prs, slides, "از جدول ویژگی تا گراف شباهت", FIG / "fig_from_table_to_graph.png")

    body(
        "سؤال این پروژه",
        [
            "آیا می‌توان G را با گراف بسیار کوچک‌تر R عوض کرد؟",
            "هدف: خوشه‌بندی سریع‌تر روی R، با کیفیت حفظ‌شده یا بهتر.",
            "ابزار ساخت R: لم نظم زمردی.",
            "لم وجود افراز را می‌گوید؛ برتری تجربی خوشه‌بندی روی R را باید آزمایش کرد.",
        ],
        "تأکید کنید که سؤال دفاع همان مقایسه R با G روی گراف خودتان است، نه بازتولید رقم‌به‌رقم جداول مقاله.",
        kicker="سؤال",
    )

    body(
        "لم نظم زمردی به زبان ساده",
        [
            "در گراف به‌قدر کافی بزرگ، رأس‌ها به چند کلاس تقریباً هم‌اندازه تقسیم می‌شوند.",
            "یال میان تقریباً هر دو کلاس مثل گراف تصادفی با یک چگالی ثابت رفتار می‌کند.",
            "به چنین جفتی اپسیلون-منظم می‌گویند.",
            "پس برای ساختار کلی، همان عدد چگالی d کافی است؛ جزئیات یال‌به‌یال لازم نیست.",
        ],
        kicker="نظریه",
    )

    fig_slide(prs, slides, "جفت منظم در برابر جفت نامنظم", FIG / "fig_eps_regular_pair.png")

    body(
        "گراف کاهش‌یافته R",
        [
            "هر کلاس افراز یک رأس در R می‌شود.",
            "وزن یال میان دو رأس جدید برابر چگالی میان دو کلاس است.",
            "اگر افراز k کلاس داشته باشد، R فقط k رأس دارد.",
            "در این آزمایش‌ها k معمولاً بین ۱۶ و ۶۴ است؛ G صدها یا هزاران رأس دارد.",
        ],
        kicker="نظریه",
    )

    body(
        "الگوریتم ۱ مقاله هو و همکاران",
        [
            "افراز تقریبی Alon روی گراف، با اصلاح‌های عملی مقاله.",
            "ساخت R از چگالی کلاس‌ها.",
            "خوشه‌بندی روی R (نه روی G): SPC، APC، DSet، SPRG.",
            "کپی برچسب هر رأس R به اعضای کلاس، و تخصیص مجموعه استثنایی وی‌صفر.",
            "ادعا: روی بیست مجموعه، هم NMI بالاتر و هم زمان کمتر.",
        ],
        kicker="مقاله مرجع، ۲۰۲۶",
    )

    body(
        "کار این پروژه",
        [
            "پیاده‌سازی مستقل از شبه‌کد تا CSV نهایی.",
            "مستند کردن جاهایی که متن مقاله ساکت است.",
            "آزمایش روی بیست مجموعه و چهار الگوریتم.",
            "بستن سه میان‌بر غیرلمی: ترتیب فایل، گراف کامل، خوشه‌بندی روی G به‌جای R.",
            "سؤال اصلی دفاع: روی همان G که خودم ساختم، آیا R بهتر است؟",
        ],
        kicker="سهم دانشجو",
    )

    fig_slide(prs, slides, "مسیر آزمایش", FIG / "fig_pipeline.png")
    fig_slide(prs, slides, "روش پایه روی G در برابر روش تقویت‌شده روی R", FIG / "fig_orig_vs_enhanced.png")

    body(
        "افزوده و ممنوع",
        [
            "افزوده برای اجرای منصفانه: z-score، kNN، ترتیب گرافی رأس‌ها، NJW، پالایش.",
            "پالایش بعد از الگوریتم ۱ است و فقط در شاخه تقویت‌شده؛ در متن Hou نیست.",
            "ممنوع: افراز با ترتیب ردیف فایل UCI (نشت برچسب).",
            "ممنوع: گراف گاوسی کامل که آزمون Alon را بی‌معنی می‌کند.",
            "ممنوع: گزارش خوشه‌بندی روی G به‌عنوان خروجی روش لم.",
        ],
        "اگر داور بپرسد این همان آزمایش Hou است: هسته الگوریتم ۱ بله؛ پیش‌پردازش و پالایش خیر.",
        kicker="صداقت آزمایش",
    )

    fig_slide(prs, slides, "چرا ترتیب فایل خطرناک است", FIG / "fig_index_order_artifact.png")

    body(
        "پروتکل آزمایش",
        [
            "۲۰ مجموعه × ۴ الگوریتم = ۸۰ سلول؛ ۷۸ عدد معتبر.",
            "ده مجموعه کامل؛ ده مجموعه بزرگ طبقه‌ای به حدود ۴۰۰ نمونه کاهش یافت.",
            "معیار: NMI با برچسب واقعی؛ بهترین تنظیم شبکه گزارش می‌شود.",
            "سخت‌افزار: CPU شخصی، i5-13400، ۳۲ گیگابایت.",
            "هشت روش جدیدتر مقاله اجرا نشد؛ ستون Hou فقط عدد چاپ‌شده است.",
        ],
        kicker="روش",
    )

    sl = new(prs)
    slides.append(sl)
    chrome(prs, sl, "نتیجه اصلی", len(slides), kicker="یافته")
    f = tf(sl, Inches(0.55), Inches(1.4), Inches(12.2), Inches(0.4))
    write(f, ["روی همان گراف G این پروژه"], size=16, color=MUTED, space=0)
    cards = [
        ("۶۰", "بهتر از پایه", SAGE),
        ("۳", "برابر", NAVY),
        ("۱۵", "بدتر از پایه", ACCENT),
    ]
    for i, (n, lab, col) in enumerate(cards):
        x = Inches(0.5 + i * 4.2)
        card = box(sl, x, Inches(1.95), Inches(3.95), Inches(3.55), PANEL, rounded=True)
        shadow(card, blur=0.38, dist=0.11, alpha=24)
        box(sl, x, Inches(1.95), Inches(3.95), Inches(0.14), col)
        f = tf(sl, x, Inches(2.35), Inches(3.95), Inches(1.7))
        write(f, [n], size=64, bold=True, color=col, space=0, align="c")
        f = tf(sl, x, Inches(4.25), Inches(3.95), Inches(0.7))
        write(f, [lab], size=18, color=INK, space=0, align="c")
    note = box(sl, Inches(0.5), Inches(5.7), Inches(12.35), Inches(0.95), PANEL, rounded=True)
    shadow(note, blur=0.2, dist=0.05, alpha=14)
    f = tf(sl, Inches(0.75), Inches(5.88), Inches(11.9), Inches(0.65))
    write(
        f,
        ["حدود ۷۷٪ سلول‌های معتبر: مسیر R برنده است. مقایسه با ستون Hou جداست: ۲۲ بهتر، ۵۶ بدتر."],
        size=16,
        color=INK,
        space=0,
    )

    fig_slide(prs, slides, "لم منهای پایه روی ده مجموعه کامل", FIG / "fig_delta_nmi_full.png")

    body(
        "چند سلول روشن",
        [
            "Wine: APC، DSet و SPRG از حدود ۰/۵۷، ۰/۴۸ و ۰/۰۲ به نزدیک ۰/۸۹.",
            "SPC پایه روی Wine با ۰/۹۱ از Reg-SPC چاپ‌شده مقاله (۰/۷۶) بالاتر است.",
            "Appendicitis: APC و DSet از ۰/۶۸ و ۰/۴۲ به ۱/۰۰.",
            "Banknote: SPC از ۰/۱۹ به ۰/۸۲. Ecoli: هر چهار الگوریتم از پایه بهتر شدند.",
            "بیشترین سود جایی است که روش پایه ناپایدار بود، نه جایی که طیفی از قبل قوی بود.",
        ],
        kicker="جزئیات",
    )

    fig_slide(
        prs,
        slides,
        "عدد این پروژه در برابر عدد چاپ‌شده مقاله",
        FIG / "fig_ours_vs_hou.png",
        "نقاط بالای قطر از مقاله بهترند. مربع‌های طلایی بیشتر مجموعه‌های کاهش‌یافته‌اند. برای کیفیت پیاده‌سازی، مقایسه با پایه معتبرتر است.",
    )

    body(
        "محدودیت‌ها",
        [
            "بهترین پارامتر با دیدن برچسب واقعی انتخاب می‌شود (اوراکل). مقاله هم در آزمایش ۲ چنین می‌کند.",
            "z-score و kNN در متن مقاله نیستند. عنوان دقیق‌تر کار: الگوریتم ۱ روی گراف kNN استانداردشده.",
            "ده مجموعه کاهش اندازه دارند و به مجموعه کامل تعمیم‌پذیر نیستند.",
            "آزمایش ۱، جایگزینی افراز با k-means، مجموعه USPS کامل، و هشت روش جدیدتر اجرا نشد.",
        ],
        kicker="صداقت",
    )

    body(
        "جمع‌بندی",
        [
            "الگوریتم ۱ روی گراف kNN استانداردشده فشرده‌ساز عملی است.",
            "در اکثر این آزمایش‌ها کیفیت از همان الگوریتم روی G بهتر یا برابر ماند.",
            "جداول مقاله رقم‌به‌رقم بازتولید نشد؛ پروتکل در متن کامل نیست.",
            "می‌توان الگوریتم ۱ را با صداقت پیاده کرد و در بیشتر موارد بدون میان‌بر غیرلمی بهتر خوشه‌بندی گرفت.",
        ],
        "اگر یک سؤال ماند: سهم لم از سهم پالایش جدا گزارش شده است.",
        kicker="پایان",
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

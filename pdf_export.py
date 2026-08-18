import os
from fpdf import FPDF
from fpdf.enums import WrapMode

# Common locations for a Unicode-capable TTF font across Linux/macOS/Windows.
# DejaVu Sans is bundled with most Linux distros (incl. Ubuntu) and covers a
# very wide character range (Latin, Cyrillic, Greek, Hebrew, etc.). It does
# NOT cover Arabic shaping, but it renders Arabic codepoints without crashing.
_CANDIDATE_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
    "C:\\Windows\\Fonts\\DejaVuSans.ttf",
]


def _find_unicode_font() -> str | None:
    for path in _CANDIDATE_FONT_PATHS:
        if os.path.isfile(path):
            return path
    return None


def _ascii_safe(text: str) -> str:
    """Fallback: drop characters the base PDF font can't render, rather than crash."""
    return text.encode("latin-1", errors="ignore").decode("latin-1")


def build_research_pdf(topic: str, summary: str, sources: list, tools_used: list, full_report: str = "") -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    unicode_font_path = _find_unicode_font()
    font_name = "Helvetica"
    prep = _ascii_safe  # default: strip unsupported chars

    if unicode_font_path:
        pdf.add_font("Unicode", "", unicode_font_path)
        try:
            pdf.add_font("Unicode", "B", unicode_font_path)
        except Exception:
            pass
        font_name = "Unicode"
        prep = lambda t: t  # no stripping needed, font supports the full text

    def set_font(style="", size=11):
        try:
            pdf.set_font(font_name, style, size)
        except Exception:
            pdf.set_font(font_name, "", size)

    def write_cell(text, h=6, style="", size=11):
        """multi_cell with char-level wrapping so long unbroken strings
        (e.g. URLs with no spaces) never raise 'not enough horizontal space'.
        Also forces the cursor back to the left margin first — otherwise a
        prior multi_cell call can leave x near the page's right edge, which
        combined with WrapMode.CHAR causes an infinite loop trying to fit a
        character into ~0mm of width."""
        set_font(style, size)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, h, prep(text), wrapmode=WrapMode.CHAR, new_x="LMARGIN", new_y="NEXT")

    write_cell(topic, h=10, style="B", size=16)
    pdf.ln(2)

    set_font("B", 12)
    pdf.cell(0, 8, "Summary", new_x="LMARGIN", new_y="NEXT")
    write_cell(summary, h=6, style="", size=11)
    pdf.ln(4)

    if full_report and full_report.strip() != summary.strip():
        set_font("B", 12)
        pdf.cell(0, 8, "Full Report", new_x="LMARGIN", new_y="NEXT")
        write_cell(full_report, h=6, style="", size=11)
        pdf.ln(4)

    if sources:
        set_font("B", 12)
        pdf.cell(0, 8, "Sources", new_x="LMARGIN", new_y="NEXT")
        for s in sources:
            write_cell(f"- {s}", h=6, style="", size=10)
        pdf.ln(4)

    if tools_used:
        set_font("B", 12)
        pdf.cell(0, 8, "Tools used", new_x="LMARGIN", new_y="NEXT")
        write_cell(", ".join(tools_used), h=6, style="", size=10)

    if not unicode_font_path:
        pdf.ln(4)
        set_font("", 7)
        pdf.set_text_color(150, 150, 150)
        write_cell(
            "Note: some non-Latin characters (e.g. Arabic) were omitted from this "
            "PDF because no Unicode font was found on this system.",
            h=4, style="", size=7,
        )

    return bytes(pdf.output())
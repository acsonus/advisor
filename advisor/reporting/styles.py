"""Visual styles, colors, and layout metrics for PDF reports."""

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

# Colour palette
DARK_BG = colors.HexColor('#1A1A2E')
ACCENT = colors.HexColor('#16213E')
GREEN = colors.HexColor('#00B050')
RED = colors.HexColor('#FF0000')
YELLOW = colors.HexColor('#FFC000')
WHITE = colors.white
LIGHT_GREY = colors.HexColor('#D9D9D9')
MID_GREY = colors.HexColor('#595959')


def build_report_styles() -> dict[str, ParagraphStyle]:
    """Build typography and paragraph styles for the report."""
    base = getSampleStyleSheet()
    styles: dict[str, ParagraphStyle] = {}

    styles['title'] = ParagraphStyle(
        'ReportTitle',
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=WHITE,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    styles['subtitle'] = ParagraphStyle(
        'ReportSubtitle',
        fontName='Helvetica',
        fontSize=10,
        textColor=LIGHT_GREY,
        alignment=TA_CENTER,
        spaceAfter=16,
    )
    styles['section'] = ParagraphStyle(
        'SectionHeader',
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=ACCENT,
        spaceBefore=12,
        spaceAfter=4,
        borderPad=4,
    )
    styles['body'] = ParagraphStyle(
        'Body',
        fontName='Helvetica',
        fontSize=9,
        textColor=colors.black,
        spaceAfter=4,
    )
    styles['disclaimer'] = ParagraphStyle(
        'Disclaimer',
        fontName='Helvetica-Oblique',
        fontSize=7,
        textColor=MID_GREY,
        alignment=TA_CENTER,
        spaceBefore=8,
    )
    styles['signal_buy'] = ParagraphStyle('SigBuy', parent=styles['body'], textColor=GREEN, fontName='Helvetica-Bold')
    styles['signal_sell'] = ParagraphStyle('SigSell', parent=styles['body'], textColor=RED, fontName='Helvetica-Bold')
    styles['signal_hold'] = ParagraphStyle('SigHold', parent=styles['body'], textColor=MID_GREY)

    return styles


def get_signal_colour(signal: str) -> colors.Color:
    """Return matching ReportLab color for Buy/Sell/Hold signal."""
    s = str(signal).strip().lower()
    if s == 'buy':
        return GREEN
    if s == 'sell':
        return RED
    return MID_GREY


def render_sentiment_bar(score: float, width: int = 20) -> str:
    """ASCII progress bar for sentiment score (-1 … 1)."""
    filled = int((score + 1) / 2 * width)
    filled = max(0, min(width, filled))
    return '[' + '█' * filled + '░' * (width - filled) + f']  {score:+.3f}'

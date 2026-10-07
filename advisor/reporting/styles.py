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
    """
    Construct a centralized dictionary of ReportLab paragraph styles for PDF generation.

    Goal:
    -----
    Standardize visual presentation across PDF report sections (title, subtitles,
    section headers, body text, disclaimers, and colored signal tags) using
    consistent typography, leading, and padding.

    Execution Principle:
    --------------------
    1. Loads the base ReportLab stylesheet (`getSampleStyleSheet()`).
    2. Builds custom `ParagraphStyle` definitions:
       - `'title'`: Large bold Helvetica centered in white for dark banner headers.
       - `'subtitle'`: Light-grey metadata sub-headers for tickers and timestamps.
       - `'section'`: High-contrast header for table and analysis categories.
       - `'body'`: Compact body text suitable for table cells.
       - `'disclaimer'`: Small oblique text for regulatory and risk disclosures.
       - `'signal_buy'`, `'signal_sell'`, `'signal_hold'`: Bold, color-accented signal text.
    3. Returns a dictionary mapping style names to their `ParagraphStyle` instances.

    Returns:
    --------
    dict[str, ParagraphStyle]
        Style dictionary indexed by style name.
    """
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
    """
    Map directional signal text to its semantic ReportLab color.

    Goal:
    -----
    Provide immediate visual distinction for trade directions within PDF tables.

    Execution Principle:
    --------------------
    1. Normalizes the input string by stripping whitespace and lowercasing.
    2. Maps:
       - 'buy'  -> GREEN (`#00B050`)
       - 'sell' -> RED (`#FF0000`)
       - any other (e.g. 'hold') -> MID_GREY (`#595959`)

    Parameters:
    -----------
    signal : str
        Signal text ('Buy', 'Sell', 'Hold').

    Returns:
    --------
    colors.Color
        ReportLab color representation.
    """
    s = str(signal).strip().lower()
    if s == 'buy':
        return GREEN
    if s == 'sell':
        return RED
    return MID_GREY


def render_sentiment_bar(score: float, width: int = 20) -> str:
    """
    Render a text-based ASCII progress bar representing a sentiment score.

    Goal:
    -----
    Provide a compact, scannable visual indicator of sentiment magnitude and polarity
    inside table cells without requiring embedded graphic assets.

    Execution Principle:
    --------------------
    1. Maps `score` from interval `[-1.0, 1.0]` to fraction `[0.0, 1.0]` via `(score + 1) / 2`.
    2. Multiplies by total `width` characters to determine filled count.
    3. Clamps filled count within `[0, width]`.
    4. Concatenates filled block glyphs (`'█'`) and unfilled light shade glyphs (`'░'`).
    5. Appends the exact formatted signed numeric score: `[████░░░░]  +0.250`.

    Parameters:
    -----------
    score : float
        Sentiment score between -1.0 and 1.0.
    width : int, default 20
        Total character width of the progress bar track.

    Returns:
    --------
    str
        Formatted string containing the bar graphic and signed score.
    """
    filled = int((score + 1) / 2 * width)
    filled = max(0, min(width, filled))
    return '[' + '█' * filled + '░' * (width - filled) + f']  {score:+.3f}'

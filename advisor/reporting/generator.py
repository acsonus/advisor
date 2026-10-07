"""PDF report generation using ReportLab."""

from datetime import datetime
import os
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from advisor.reporting.styles import (
    ACCENT,
    DARK_BG,
    GREEN,
    LIGHT_GREY,
    MID_GREY,
    RED,
    WHITE,
    build_report_styles,
    render_sentiment_bar,
)


class PdfReportGenerator:
    """Builder for professional trading analysis PDF reports."""

    def __init__(self, ticker: str, budget_eur: float = 0.0):
        self.ticker = ticker
        self.budget_eur = budget_eur
        self.styles = build_report_styles()
        self.usable_width = A4[0] - 3.6 * cm

    def generate(
        self,
        ohlcv_df: pd.DataFrame,
        news_df: pd.DataFrame | None,
        output_path: str | None = None,
    ) -> str:
        """Build and save PDF report."""
        if output_path is None:
            date_str = datetime.now().strftime('%Y%m%d_%H%M')
            output_path = os.path.join(
                os.getcwd(),
                f'trading_report_{self.ticker}_{date_str}.pdf',
            )

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=1.8 * cm,
            rightMargin=1.8 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
            title=f'Trading Report — {self.ticker}',
            author='TradingStrategies / ProfessionalTrader',
        )

        story = []
        W = self.usable_width
        styles = self.styles

        # ── Header banner ───────────────────────────────────────────────────
        header_data = [[
            Paragraph('TRADING ANALYSIS REPORT', styles['title']),
        ]]
        header_table = Table(header_data, colWidths=[W])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), DARK_BG),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('ROUNDEDCORNERS', [6]),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 0.2 * cm))

        meta_text = (
            f'Ticker: <b>{self.ticker}</b> &nbsp;|&nbsp; '
            f'Date: <b>{datetime.now().strftime("%B %d, %Y")}</b> &nbsp;|&nbsp; '
            f'Budget: <b>€{self.budget_eur:,.0f}</b>'
        )
        story.append(Paragraph(meta_text, styles['subtitle']))
        story.append(HRFlowable(width=W, thickness=1, color=LIGHT_GREY))

        # ── 1. Market Data Summary ───────────────────────────────────────────
        story.append(Paragraph('1. Market Data Summary', styles['section']))

        if not ohlcv_df.empty:
            latest = ohlcv_df.iloc[-1]
            prev = ohlcv_df.iloc[-2] if len(ohlcv_df) > 1 else latest
            chg = latest['Close'] - prev['Close']
            chg_pct = (chg / prev['Close'] * 100) if prev['Close'] != 0 else 0.0

            summary_data = [
                ['Metric', 'Value'],
                ['Latest Close', f"${latest['Close']:.2f}"],
                ['Change (1D)', f"{'▲' if chg >= 0 else '▼'} ${abs(chg):.2f} ({chg_pct:+.2f}%)"],
                ['Period High', f"${ohlcv_df['Close'].max():.2f}"],
                ['Period Low', f"${ohlcv_df['Close'].min():.2f}"],
                ['Observations', str(len(ohlcv_df))],
            ]
            t = Table(summary_data, colWidths=[W * 0.45, W * 0.55])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), DARK_BG),
                ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
                ('GRID', (0, 0), (-1, -1), 0.5, MID_GREY),
                ('PADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(t)

        story.append(Spacer(1, 0.4 * cm))

        # ── 2. Strategy Signals ─────────────────────────────────────────────
        story.append(Paragraph('2. Strategy Signals (Last 10 Trading Days)', styles['section']))

        if not ohlcv_df.empty:
            signal_cols = ['Close', 'atr_signal', 'ma_rsi_signal', 'sentiment_signal', 'daily_sentiment']
            available = [c for c in signal_cols if c in ohlcv_df.columns]
            display_df = ohlcv_df[available].tail(10).copy()

            header_row = ['Date'] + [c.replace('_', ' ').title() for c in available]
            rows = [header_row]
            for date, row in display_df.iterrows():
                r = [str(date)[:10]]
                for col in available:
                    val = row[col]
                    if col == 'Close':
                        r.append(f'${val:.2f}')
                    elif col == 'daily_sentiment':
                        r.append(f'{val:+.3f}')
                    else:
                        r.append(str(val))
                rows.append(r)

            col_w = W / len(header_row)
            t = Table(rows, colWidths=[col_w] * len(header_row))

            cell_styles = [
                ('BACKGROUND', (0, 0), (-1, 0), DARK_BG),
                ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.4, LIGHT_GREY),
                ('PADDING', (0, 0), (-1, -1), 4),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F2F2F2')]),
            ]

            signal_col_indices = [i for i, c in enumerate(available) if 'signal' in c]
            for ri, (_, row) in enumerate(display_df.iterrows(), start=1):
                for ci in signal_col_indices:
                    col = available[ci]
                    val = str(row[col])
                    bg = GREEN if val == 'Buy' else (RED if val == 'Sell' else colors.white)
                    tc = WHITE if val in ('Buy', 'Sell') else colors.black
                    cell_styles += [
                        ('BACKGROUND', (ci + 1, ri), (ci + 1, ri), bg),
                        ('TEXTCOLOR', (ci + 1, ri), (ci + 1, ri), tc),
                        ('FONTNAME', (ci + 1, ri), (ci + 1, ri), 'Helvetica-Bold'),
                    ]

            t.setStyle(TableStyle(cell_styles))
            story.append(t)

        story.append(Spacer(1, 0.4 * cm))

        # ── 3. News Sentiment ───────────────────────────────────────────────
        story.append(Paragraph('3. News Sentiment Analysis', styles['section']))

        if news_df is not None and not news_df.empty:
            required = {'Title', 'Sentiment_Label', 'Sentiment_Score', 'Publication Date'}
            if required.issubset(news_df.columns):
                news_rows = [['Title', 'Date', 'Label', 'Score', 'Bar']]
                for _, row in news_df.head(10).iterrows():
                    title = str(row['Title'])[:60] + ('…' if len(str(row['Title'])) > 60 else '')
                    pub = str(row['Publication Date'])[:10]
                    label = str(row['Sentiment_Label'])
                    score = float(row['Sentiment_Score']) if row['Sentiment_Score'] is not None else 0.0
                    signed = score if label == 'Positive' else (-score if label == 'Negative' else 0.0)
                    bar = render_sentiment_bar(signed, width=10)
                    news_rows.append([title, pub, label, f'{score:.3f}', bar])

                col_widths = [W * 0.38, W * 0.11, W * 0.10, W * 0.08, W * 0.33]
                nt = Table(news_rows, colWidths=col_widths)
                news_styles = [
                    ('BACKGROUND', (0, 0), (-1, 0), DARK_BG),
                    ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 7),
                    ('GRID', (0, 0), (-1, -1), 0.4, LIGHT_GREY),
                    ('PADDING', (0, 0), (-1, -1), 4),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F2F2F2')]),
                ]
                for ri, row_data in enumerate(news_rows[1:], start=1):
                    label = row_data[2]
                    tc = GREEN if label == 'Positive' else (RED if label == 'Negative' else MID_GREY)
                    news_styles.append(('TEXTCOLOR', (2, ri), (2, ri), tc))
                    news_styles.append(('FONTNAME', (2, ri), (2, ri), 'Helvetica-Bold'))

                nt.setStyle(TableStyle(news_styles))
                story.append(nt)

                if 'Signed_Score' in news_df.columns:
                    overall = news_df['Signed_Score'].mean()
                    verdict = 'BULLISH' if overall > 0.2 else ('BEARISH' if overall < -0.2 else 'NEUTRAL')
                    v_colour = GREEN if overall > 0.2 else (RED if overall < -0.2 else MID_GREY)
                    story.append(Spacer(1, 0.2 * cm))
                    verdict_data = [[
                        Paragraph(
                            f'Overall Sentiment: <font color="#{v_colour.hexval()[2:]}"><b>{verdict}</b></font>'
                            f' &nbsp; Score: <b>{overall:+.3f}</b>',
                            styles['body'],
                        )
                    ]]
                    vt = Table(verdict_data, colWidths=[W])
                    vt.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#EEF4FF')),
                        ('PADDING', (0, 0), (-1, -1), 8),
                        ('ROUNDEDCORNERS', [4]),
                    ]))
                    story.append(vt)
            else:
                story.append(Paragraph('News data not available or incomplete.', styles['body']))
        else:
            story.append(Paragraph('No news data available.', styles['body']))

        story.append(Spacer(1, 0.4 * cm))

        # ── 4. Disclaimer ────────────────────────────────────────────────────
        story.append(HRFlowable(width=W, thickness=0.5, color=LIGHT_GREY))
        story.append(Paragraph(
            'DISCLAIMER: This report is generated for informational and educational purposes only. '
            'It does not constitute financial advice. Trading stocks involves substantial risk of '
            'capital loss. Always consult a licensed financial advisor before making investment decisions.',
            styles['disclaimer'],
        ))

        doc.build(story)
        return os.path.abspath(output_path)


def generate_report(
    ticker: str,
    ohlcv_df: pd.DataFrame,
    news_df: pd.DataFrame | None,
    budget_eur: float = 0.0,
    output_path: str | None = None,
) -> str:
    """Backward-compatible function wrapper for PDF report generation."""
    generator = PdfReportGenerator(ticker=ticker, budget_eur=budget_eur)
    return generator.generate(ohlcv_df=ohlcv_df, news_df=news_df, output_path=output_path)

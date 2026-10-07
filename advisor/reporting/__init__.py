"""Reporting and visualization package."""

from advisor.reporting.generator import PdfReportGenerator, generate_report
from advisor.reporting.styles import build_report_styles, get_signal_colour, render_sentiment_bar

__all__ = [
    "PdfReportGenerator",
    "generate_report",
    "build_report_styles",
    "get_signal_colour",
    "render_sentiment_bar",
]

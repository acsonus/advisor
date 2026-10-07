"""
sentiments.py — Backward-compatibility facade.

Re-exports `YahooSentiments` from the `advisor.sentiment` package.
"""

from advisor.sentiment import YahooSentiments

__all__ = ["YahooSentiments"]
"""
conftest.py — pytest configuration for the advisor test suite.
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: mark tests that require live network / Yahoo Finance market data",
    )

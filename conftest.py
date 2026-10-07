"""
conftest.py — pytest configuration for the advisor test suite.
"""


def pytest_configure(config):
    """
    Goal:
        Register custom pytest markers with the test runner to prevent unrecognised marker warnings.

    Execution Principle:
        1. Access the test configuration object (`config`).
        2. Append the definition for the `"live"` marker via `addinivalue_line`.
        3. Clarifies to the test runner that tests decorated with `@pytest.mark.live` require network/Yahoo Finance access.
    """
    config.addinivalue_line(
        "markers",
        "live: mark tests that require live network / Yahoo Finance market data",
    )

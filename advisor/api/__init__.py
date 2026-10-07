"""Flask Web API package."""

from advisor.api.app import create_app
from advisor.api.routes import strategies_bp

__all__ = ["create_app", "strategies_bp"]

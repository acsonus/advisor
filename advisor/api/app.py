"""Flask application factory."""

from flask import Flask
from advisor.api.routes import strategies_bp


def create_app(config: dict | None = None) -> Flask:
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__)
    if config:
        app.config.update(config)

    app.register_blueprint(strategies_bp)

    @app.route("/api/health", methods=["GET"])
    def health_check():
        return {"status": "healthy"}

    return app

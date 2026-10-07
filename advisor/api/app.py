"""Flask application factory."""

from flask import Flask
from advisor.api.routes import strategies_bp


def create_app(config: dict | None = None) -> Flask:
    """
    Construct, configure, and initialize a Flask application instance.

    Goal:
    -----
    Implement the Application Factory pattern, enabling multiple isolated instances
    of the web application to be created with distinct configurations for production,
    development, and automated testing environments.

    Execution Principle:
    --------------------
    1. Instantiates a new `Flask(__name__)` application instance.
    2. If a configuration dictionary is provided, updates `app.config` with it.
    3. Registers the `strategies_bp` blueprint hosting strategy execution and reporting routes.
    4. Registers a lightweight health-check endpoint (`GET /api/health`).
    5. Returns the fully configured Flask application object.

    Parameters:
    -----------
    config : dict, optional
        Key-value configuration dictionary (e.g. `{'TESTING': True}`).

    Returns:
    --------
    Flask
        Configured WSGI application instance.
    """
    app = Flask(__name__)
    if config:
        app.config.update(config)

    app.register_blueprint(strategies_bp)

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """
        Liveness and readiness check endpoint.

        Goal:
        -----
        Provide an instantaneous health status check for load balancers and container orchestrators.

        Execution Principle:
        --------------------
        Returns an HTTP 200 JSON object `{'status': 'healthy'}` with no external I/O.
        """
        return {"status": "healthy"}

    return app

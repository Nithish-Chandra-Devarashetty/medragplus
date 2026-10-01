"""Flask application factory and dev-server entry point.

Run from the project root:   python -m backend.app
"""
from __future__ import annotations

import logging
import os

from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from backend.config.settings import Settings
from backend.services.container import Services, build_services


def create_app(settings: Settings | None = None, services: Services | None = None) -> Flask:
    settings = settings or Settings.from_env()
    services = services or build_services(settings)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = settings.max_audio_mb * 1024 * 1024
    app.extensions["medrag"] = services
    CORS(app, resources={r"/api/*": {"origins": settings.cors_origins}})

    from backend.api import auth_routes, chat_routes, history_routes

    for bp in (auth_routes.bp, chat_routes.bp, history_routes.bp):
        app.register_blueprint(bp, url_prefix="/api")

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", components=services.components())

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException):
        return jsonify(error=exc.description), exc.code

    @app.errorhandler(Exception)
    def server_error(exc: Exception):
        app.logger.exception("Unhandled error")
        return jsonify(error="Internal server error"), 500

    if settings.preload_models:
        services.preload()
    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app()
    # No reloader: embedded Qdrant and llama.cpp must only be opened by one process.
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), threaded=True,
            use_reloader=False)


if __name__ == "__main__":
    main()

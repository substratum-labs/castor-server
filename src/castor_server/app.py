"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from castor_server.config import settings
from castor_server.store.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.DEBUG if settings.debug else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    await init_db()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="castor-server",
        description="Self-hosted REST API compatible with Anthropic Managed Agents",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Error handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logging.getLogger("castor_server").exception("Unhandled error")
        return JSONResponse(
            status_code=500,
            content={"error": {"type": "server_error", "message": str(exc)}},
        )

    # Routes
    from castor_server.api.agents import router as agents_router
    from castor_server.api.events import router as events_router
    from castor_server.api.sessions import router as sessions_router

    app.include_router(agents_router)
    app.include_router(sessions_router)
    app.include_router(events_router)

    # Health check
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app

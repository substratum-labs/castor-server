"""Async SQLAlchemy database setup."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from castor_server.config import settings

engine = create_async_engine(settings.database_url, echo=settings.debug)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


def set_session_factory(factory) -> None:
    """Override the global async_session factory (used by tests).

    Background tasks dispatched by session_manager call ``async_session()``
    directly to get a fresh session — they can't use the request's session
    because it closes when the request returns. This setter lets tests
    point that factory at their test engine.
    """
    global async_session
    async_session = factory


async def init_db() -> None:
    """Create all tables (or migrate schema if stale).

    For SQLite in dev mode: if the existing DB is missing columns that
    the current code expects, drop all tables and recreate. This avoids
    500 errors after ``pip install --upgrade castor-server`` when the
    schema has evolved (e.g. new ``agent_fn_factory`` column).

    For Postgres: use Alembic migrations (not yet implemented). The
    drop-and-recreate behavior is SQLite-only.
    """
    import logging

    from .db_models import Base

    logger = logging.getLogger("castor_server.database")

    async with engine.begin() as conn:
        # First try: create tables that don't exist yet
        await conn.run_sync(Base.metadata.create_all)

        # Check schema freshness (SQLite only): try inserting into
        # agents with all current columns. If it fails, the schema
        # is stale — drop and recreate.
        if "sqlite" in settings.database_url:
            try:
                await conn.execute(
                    __import__("sqlalchemy").text(
                        "SELECT agent_fn_factory FROM agents LIMIT 0"
                    )
                )
            except Exception:
                logger.warning(
                    "schema_stale — dropping and recreating SQLite tables "
                    "(this is normal after upgrading castor-server)"
                )
                await conn.run_sync(Base.metadata.drop_all)
                await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session

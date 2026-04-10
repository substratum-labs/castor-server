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
    """Create all tables."""
    from .db_models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session

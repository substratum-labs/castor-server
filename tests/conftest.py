"""Shared test fixtures."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from castor_server.app import create_app
from castor_server.store.database import get_session
from castor_server.store.db_models import Base

# Default to in-memory SQLite. Override with TEST_DATABASE_URL to use Postgres:
#   TEST_DATABASE_URL="postgresql+asyncpg://user:pw@host:port/db" uv run pytest
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "sqlite+aiosqlite:///:memory:")


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def db_engine():
    if TEST_DATABASE_URL.startswith("sqlite"):
        from sqlalchemy.pool import StaticPool

        engine = create_async_engine(
            TEST_DATABASE_URL,
            echo=False,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    else:
        # Postgres / other: use NullPool so connections are closed immediately
        # after use (avoids fixture teardown hangs from pooled connections held
        # by fire-and-forget tasks).
        from sqlalchemy.pool import NullPool

        engine = create_async_engine(TEST_DATABASE_URL, echo=False, poolclass=NullPool)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield engine

    # Drain any in-flight background tasks dispatched by session_manager
    # before tearing down the engine. Without this, fire-and-forget tasks
    # try to grab DB connections during dispose and hang on Postgres.
    from castor_server.core.session_manager import session_manager

    await session_manager.drain()
    await asyncio.sleep(0.05)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    except Exception:
        pass
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    session_factory = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def client(db_engine) -> AsyncGenerator[AsyncClient, None]:
    session_factory = async_sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )

    # Override the global async_session factory so background tasks
    # dispatched by session_manager use the test engine.
    from castor_server.store import database as db_module

    original_factory = db_module.async_session
    db_module.set_session_factory(session_factory)

    app = create_app()

    async def override_get_session():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    # Restore original factory after test
    db_module.set_session_factory(original_factory)

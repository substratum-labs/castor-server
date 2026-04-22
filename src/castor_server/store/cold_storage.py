"""SQLiteVec-backed cold storage for evicted agent context.

Implements ``castor.ColdStorageProtocol`` using sqlite-vec for vector
similarity search. Namespace is ``agent_id`` (not session_id) so
evicted context is shared across sessions of the same agent.

Schema:
    cold_entries(id INTEGER PK, agent_id TEXT INDEXED, source TEXT,
                 content_json TEXT, summary TEXT, embedding BLOB,
                 created_at TEXT, metadata_json TEXT)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("castor_server.cold_storage")


class SQLiteVecColdStorage:
    """Persistent cold storage using SQLite + sqlite-vec.

    Implements castor.ColdStorageProtocol.
    """

    def __init__(self, db_path: str = "./castor_cold.db") -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")

            # Load sqlite-vec extension
            import sqlite_vec

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)

            self._init_schema()
        return self._conn

    def _init_schema(self) -> None:
        conn = self._conn
        assert conn is not None
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cold_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'eviction',
                content_json TEXT NOT NULL,
                summary TEXT,
                created_at TEXT NOT NULL,
                metadata_json TEXT DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cold_agent
            ON cold_entries(agent_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cold_agent_source
            ON cold_entries(agent_id, source)
        """)
        # Virtual table for vector search
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS cold_embeddings
            USING vec0(entry_id INTEGER PRIMARY KEY, embedding float[384])
        """)
        conn.commit()

    async def store(
        self,
        agent_id: str,
        messages: list[Any],
        summary: str | None = None,
        source: str = "eviction",
    ) -> None:
        """Persist evicted messages to cold storage."""
        conn = self._get_conn()

        # Serialize messages
        serialized = []
        for msg in messages:
            if hasattr(msg, "model_dump"):
                serialized.append(msg.model_dump())
            elif isinstance(msg, dict):
                serialized.append(msg)
            else:
                serialized.append({"content": str(msg)})

        content_json = json.dumps(serialized, ensure_ascii=False)
        now = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

        cursor = conn.execute(
            "INSERT INTO cold_entries"
            " (agent_id, source, content_json, summary, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (agent_id, source, content_json, summary, now),
        )
        entry_id = cursor.lastrowid

        # Generate and store embedding
        embedding = self._embed(summary or content_json[:1000])
        conn.execute(
            "INSERT INTO cold_embeddings (entry_id, embedding) VALUES (?, ?)",
            (entry_id, _serialize_f32(embedding)),
        )
        conn.commit()

    async def search(
        self,
        agent_id: str,
        query: str,
        max_results: int = 5,
        filter: dict | None = None,
    ) -> list[Any]:
        """Retrieve relevant messages via vector similarity search.

        ``filter`` is a dict of field → value constraints. Supported keys:
        ``source`` (provenance filter, e.g. ``{"source": "eviction"}``).
        """
        conn = self._get_conn()
        query_embedding = self._embed(query)

        rows = conn.execute(
            """
            SELECT e.entry_id, distance
            FROM cold_embeddings e
            WHERE e.embedding MATCH ?
            AND k = ?
            ORDER BY distance
            """,
            (_serialize_f32(query_embedding), max_results * 3),
        ).fetchall()

        if not rows:
            return []

        entry_ids = [r[0] for r in rows]
        placeholders = ",".join("?" for _ in entry_ids)

        sql = f"""
            SELECT id, content_json, summary, source, created_at
            FROM cold_entries
            WHERE id IN ({placeholders}) AND agent_id = ?
        """
        params: list[Any] = list(entry_ids) + [agent_id]

        if filter and filter.get("source"):
            sql += " AND source = ?"
            params.append(filter["source"])

        results = conn.execute(sql, params).fetchall()

        output: list[Any] = []
        for row in results[:max_results]:
            entry = json.loads(row[1])
            output.append(
                {
                    "id": row[0],
                    "messages": entry,
                    "summary": row[2],
                    "source": row[3],
                    "created_at": row[4],
                }
            )

        return output

    async def read(self, agent_id: str, memory_id: str) -> dict[str, Any] | None:
        """Lookup a single entry by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, content_json, summary, source, created_at"
            " FROM cold_entries WHERE id = ? AND agent_id = ?",
            (memory_id, agent_id),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "messages": json.loads(row[1]),
            "summary": row[2],
            "source": row[3],
            "created_at": row[4],
        }

    async def delete(self, agent_id: str, memory_id: str) -> bool:
        """Delete a single entry by ID (irreversible)."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM cold_entries WHERE id = ? AND agent_id = ?",
            (memory_id, agent_id),
        )
        # Also remove embedding
        conn.execute(
            "DELETE FROM cold_embeddings WHERE entry_id = ?",
            (memory_id,),
        )
        conn.commit()
        return cursor.rowcount > 0

    async def store_explicit(
        self,
        agent_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store explicit memory entry (from mem_store syscall)."""
        conn = self._get_conn()
        now = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        content_json = json.dumps(
            [{"role": "memory", "content": content}], ensure_ascii=False
        )

        cursor = conn.execute(
            """INSERT INTO cold_entries
               (agent_id, source, content_json, summary, created_at, metadata_json)
               VALUES (?, 'explicit', ?, ?, ?, ?)""",
            (
                agent_id,
                content_json,
                content[:200],
                now,
                json.dumps(metadata or {}),
            ),
        )
        entry_id = cursor.lastrowid

        embedding = self._embed(content)
        conn.execute(
            "INSERT INTO cold_embeddings (entry_id, embedding) VALUES (?, ?)",
            (entry_id, _serialize_f32(embedding)),
        )
        conn.commit()

    async def list_entries(
        self,
        agent_id: str,
        *,
        filter: dict | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List cold storage entries for an agent (for HTTP API)."""
        conn = self._get_conn()
        sql = (
            "SELECT id, source, summary, created_at, metadata_json"
            " FROM cold_entries WHERE agent_id = ?"
        )
        params: list[Any] = [agent_id]

        if filter and filter.get("source"):
            sql += " AND source = ?"
            params.append(filter["source"])

        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(sql, params).fetchall()
        return [
            {
                "id": r[0],
                "source": r[1],
                "summary": r[2],
                "created_at": r[3],
                "metadata": json.loads(r[4] or "{}"),
            }
            for r in rows
        ]

    @staticmethod
    def _embed(text: str) -> list[float]:
        """Generate a simple hash-based embedding (384-dim).

        Production: replace with a real embedding model (e.g. via LiteLLM
        or sentence-transformers). This hash-based approach enables basic
        keyword matching via cosine similarity without an external dependency.
        """
        # Deterministic hash-based pseudo-embedding
        # Split text into overlapping trigrams and hash each
        dim = 384
        vec = [0.0] * dim
        text = text.lower()
        for i in range(len(text) - 2):
            trigram = text[i : i + 3]
            h = int(hashlib.md5(trigram.encode()).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0

        # L2 normalize
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


def _serialize_f32(vec: list[float]) -> bytes:
    """Serialize a float vector to bytes for sqlite-vec."""
    import struct

    return struct.pack(f"{len(vec)}f", *vec)

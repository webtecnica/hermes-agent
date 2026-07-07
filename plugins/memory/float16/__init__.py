"""Float16 semantic memory plugin — MemoryProvider interface.

Captures conversation turns as float16 vector embeddings for semantic search.
Uses sentence-transformers for embedding generation and stores vectors in a
local SQLite database with cosine similarity retrieval.

Config via config.yaml:
  memory:
    float16:
      model: "all-MiniLM-L6-v2"   # sentence-transformers model name
      top_k: 5                     # max results per recall
      min_query_len: 15            # minimum query length to trigger recall
      min_turn_len: 10             # minimum turn length to capture
      score_threshold: 0.25        # minimum cosine similarity score (0-1)

Storage: $HERMES_HOME/float16/memories.db (profile-scoped)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Default configuration
_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_TOP_K = 5
_DEFAULT_MIN_QUERY_LEN = 15
_DEFAULT_MIN_TURN_LEN = 10
_DEFAULT_SCORE_THRESHOLD = 0.25
_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension
_DB_TIMEOUT = 5.0  # SQLite busy timeout in seconds

# ---------------------------------------------------------------------------
# Lazy model loading (thread-safe singleton)
# ---------------------------------------------------------------------------

_model_lock = threading.Lock()
_model_instance: Any = None


def _load_model(model_name: str):
    """Load a sentence-transformers model (thread-safe, cached singleton)."""
    global _model_instance
    if _model_instance is not None:
        return _model_instance
    with _model_lock:
        if _model_instance is not None:
            return _model_instance
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading float16 embedding model: %s", model_name)
            _model_instance = SentenceTransformer(model_name)
            return _model_instance
        except Exception as e:
            logger.error("Failed to load embedding model %s: %s", model_name, e)
            raise


def _normalize(vector):
    """L2-normalize a numpy array in-place and return it."""
    norm = float(vector.dtype.type((vector ** 2).sum() ** 0.5))
    if norm > 0:
        vector /= norm
    return vector


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_plugin_config() -> Dict[str, Any]:
    """Read float16 provider config from memory.float16 in config.yaml."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        memory_config = config.get("memory", {})
        if not isinstance(memory_config, dict):
            return {}
        provider_config = memory_config.get("float16", {})
        if isinstance(provider_config, dict):
            return dict(provider_config)
    except Exception:
        pass
    return {}


def _get_config_value(
    config: Dict[str, Any], key: str, default
) -> Any:
    """Read a config value with type coercion."""
    value = config.get(key, default)
    if value is None:
        return default
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.lower() in {"true", "1", "yes", "y", "on"}
        return bool(value)
    if isinstance(default, int):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    if isinstance(default, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return value


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL DEFAULT '',
    role        TEXT NOT NULL DEFAULT 'user',
    content     TEXT NOT NULL,
    embedding   BLOB,
    created_at  REAL NOT NULL DEFAULT (julianday('now'))
);

CREATE INDEX IF NOT EXISTS idx_memories_session_id ON memories(session_id);
"""

_INSERT_SQL = "INSERT INTO memories (session_id, role, content, embedding, created_at) VALUES (?, ?, ?, ?, julianday('now'))"

_SELECT_ALL_SQL = "SELECT id, role, content, embedding FROM memories ORDER BY id"

_DELETE_SESSION_SQL = "DELETE FROM memories WHERE session_id = ?"

_VACUUM_SQL = "PRAGMA wal_checkpoint(TRUNCATE)"


class _Float16DB:
    """Thread-safe SQLite store for float16 embeddings."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    def open(self):
        with self._lock:
            if self._conn is not None:
                return
            self._conn = sqlite3.connect(self._db_path, timeout=_DB_TIMEOUT)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.commit()

    def close(self):
        with self._lock:
            if self._conn is None:
                return
            try:
                self._conn.execute(_VACUUM_SQL)
            except Exception:
                pass
            self._conn.close()
            self._conn = None

    def store(self, session_id: str, role: str, content: str, embedding_blob: bytes):
        with self._lock:
            if self._conn is None:
                return
            self._conn.execute(
                _INSERT_SQL,
                (session_id, role, content, embedding_blob),
            )
            self._conn.commit()

    def search(
        self, query_blob: bytes, top_k: int, score_threshold: float
    ) -> List[Dict[str, Any]]:
        """Return memories sorted by cosine similarity to query_blob."""
        import numpy as np

        query_vec = np.frombuffer(query_blob, dtype=np.float16)
        query_norm = float((query_vec ** 2).sum() ** 0.5)
        if query_norm > 0:
            query_vec = query_vec / query_norm

        results: List[Dict[str, Any]] = []

        with self._lock:
            if self._conn is None:
                return results
            rows = self._conn.execute(_SELECT_ALL_SQL).fetchall()

        for row_id, role, content, emb_blob in rows:
            if emb_blob is None:
                continue
            try:
                emb = np.frombuffer(emb_blob, dtype=np.float16)
                emb_norm = float((emb ** 2).sum() ** 0.5)
                if emb_norm > 0:
                    emb = emb / emb_norm
                score = float((query_vec * emb).sum())
            except Exception:
                continue

            if score >= score_threshold:
                results.append({
                    "id": row_id,
                    "role": role,
                    "content": content[:2000],
                    "score": round(score, 4),
                })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def clear_session(self, session_id: str):
        with self._lock:
            if self._conn is None:
                return
            self._conn.execute(_DELETE_SESSION_SQL, (session_id,))
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            if self._conn is None:
                return 0
            row = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
            return row[0] if row else 0


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

RECALL_SCHEMA = {
    "name": "float16_recall",
    "description": (
        "Search float16 semantic memory for past conversation turns relevant "
        "to the current query. Returns ranked results with similarity scores."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in semantic memory.",
            },
        },
        "required": ["query"],
    },
}

STATUS_SCHEMA = {
    "name": "float16_status",
    "description": "Check float16 semantic memory status — model loaded, memory count, database size.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

CLEAR_SCHEMA = {
    "name": "float16_clear",
    "description": "Clear all float16 memories for the current session.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------


class Float16MemoryProvider(MemoryProvider):
    """Float16 semantic memory provider — turn capture, embedding, and similarity search."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = dict(config) if config else _load_plugin_config()
        self._model_name = _get_config_value(self._config, "model", _DEFAULT_MODEL)
        self._top_k = _get_config_value(self._config, "top_k", _DEFAULT_TOP_K)
        self._min_query_len = _get_config_value(self._config, "min_query_len", _DEFAULT_MIN_QUERY_LEN)
        self._min_turn_len = _get_config_value(self._config, "min_turn_len", _DEFAULT_MIN_TURN_LEN)
        self._score_threshold = _get_config_value(self._config, "score_threshold", _DEFAULT_SCORE_THRESHOLD)

        self._db: Optional[_Float16DB] = None
        self._model = None
        self._hermes_home = ""
        self._session_id = ""
        self._model_lock = threading.Lock()
        self._embedding_error_reported = False

    @property
    def name(self) -> str:
        return "float16"

    def is_available(self) -> bool:
        """Check if numpy and sentence-transformers are importable."""
        try:
            import numpy  # noqa: F401
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def get_config_schema(self):
        return [
            {
                "key": "model",
                "description": "Sentence-transformers model name for embeddings",
                "default": _DEFAULT_MODEL,
            },
            {
                "key": "top_k",
                "description": "Max results per recall query",
                "default": str(_DEFAULT_TOP_K),
            },
            {
                "key": "score_threshold",
                "description": "Minimum cosine similarity score (0.0–1.0)",
                "default": str(_DEFAULT_SCORE_THRESHOLD),
            },
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        import numpy as np  # noqa: F401 — verify availability

        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home", "")
        if not self._hermes_home:
            try:
                from hermes_constants import get_hermes_home
                self._hermes_home = str(get_hermes_home())
            except Exception:
                self._hermes_home = os.path.join(
                    os.path.expanduser("~"), ".hermes"
                )

        db_dir = Path(self._hermes_home) / "float16"
        db_path = str(db_dir / "memories.db")
        self._db = _Float16DB(db_path)
        self._db.open()
        logger.info(
            "Float16 memory initialized (model=%s, db=%s, existing_memories=%d)",
            self._model_name, db_path, self._db.count(),
        )

    def system_prompt_block(self) -> str:
        if not self.is_available():
            return ""
        return (
            "# Float16 Semantic Memory\n"
            "Active. Past conversation turns are captured as float16 vector embeddings. "
            "Use float16_recall to search semantic memory, float16_status to check state, "
            "float16_clear to wipe session memories."
        )

    def _encode(self, text: str) -> Optional[bytes]:
        """Embed text and return float16 numpy blob, or None on failure."""
        import numpy as np

        try:
            model = _load_model(self._model_name)
        except Exception:
            if not self._embedding_error_reported:
                logger.warning(
                    "Float16 embedding model unavailable; "
                    "recall and capture disabled until model loads"
                )
                self._embedding_error_reported = True
            return None

        try:
            vec = model.encode(text, normalize_embeddings=True)
            vec_f16 = np.asarray(vec, dtype=np.float16)
            return vec_f16.tobytes()
        except Exception as e:
            logger.debug("Float16 encode failed: %s", e)
            return None

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Search memories for context relevant to the current query."""
        if not query or len(query.strip()) < self._min_query_len:
            return ""
        if self._db is None:
            return ""

        query_blob = self._encode(query.strip()[:5000])
        if query_blob is None:
            return ""

        results = self._db.search(query_blob, self._top_k, self._score_threshold)
        if not results:
            return ""

        lines = ["## Float16 Semantic Memory", ""]
        for r in results:
            score_pct = int(r["score"] * 100)
            lines.append(
                f"- [{score_pct}%] ({r['role']}) {r['content']}"
            )

        return "\n".join(lines)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """No-op: prefetch runs synchronously at turn start."""
        pass

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Capture the user turn as a float16 embedding."""
        sid = session_id or self._session_id

        if not user_content or len(user_content.strip()) < self._min_turn_len:
            return
        text = user_content.strip()[:5000]

        blob = self._encode(text)
        if blob is None:
            return

        if self._db is not None:
            self._db.store(sid, "user", text, blob)

        # Also store the assistant response if substantive
        if assistant_content and len(assistant_content.strip()) >= self._min_turn_len:
            asst_blob = self._encode(assistant_content.strip()[:5000])
            if asst_blob is not None and self._db is not None:
                self._db.store(sid, "assistant", assistant_content.strip()[:5000], asst_blob)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Flush and checkpoint on session end."""
        if self._db is not None:
            self._db.clear_session("")
            logger.debug("Float16 session end: memories flushed")

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RECALL_SCHEMA, STATUS_SCHEMA, CLEAR_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if tool_name == "float16_recall":
            return self._tool_recall(args)
        elif tool_name == "float16_status":
            return self._tool_status()
        elif tool_name == "float16_clear":
            return self._tool_clear()
        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None
        logger.debug("Float16 memory shut down")

    # -- Tool implementations -------------------------------------------------

    def _tool_recall(self, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return tool_error("query is required")

        if self._db is None:
            return tool_error("Float16 database not initialized")

        query_blob = self._encode(query.strip()[:5000])
        if query_blob is None:
            return tool_error(
                "Embedding model unavailable. "
                "Ensure sentence-transformers is installed and the model can load."
            )

        results = self._db.search(query_blob, self._top_k, self._score_threshold)
        if not results:
            return json.dumps({"result": "No relevant memories found."})

        return json.dumps({"result": results}, default=str)

    def _tool_status(self) -> str:
        status: Dict[str, Any] = {
            "model": self._model_name,
            "top_k": self._top_k,
            "score_threshold": self._score_threshold,
            "min_query_len": self._min_query_len,
            "min_turn_len": self._min_turn_len,
        }

        import numpy as np
        status["numpy_available"] = True

        try:
            import sentence_transformers  # noqa: F401
            status["sentence_transformers_available"] = True
        except ImportError:
            status["sentence_transformers_available"] = False

        try:
            _load_model(self._model_name)
            status["model_loaded"] = True
        except Exception:
            status["model_loaded"] = False

        if self._db is not None:
            status["memory_count"] = self._db.count()
            try:
                db_path = self._db._db_path
                size = Path(db_path).stat().st_size if Path(db_path).exists() else 0
                status["database_size_bytes"] = size
            except Exception:
                pass

        return json.dumps({"status": status}, default=str)

    def _tool_clear(self) -> str:
        if self._db is not None:
            self._db.clear_session(self._session_id)
        return json.dumps({"result": "Session memories cleared."})


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Register Float16 as a memory provider plugin."""
    ctx.register_memory_provider(Float16MemoryProvider())

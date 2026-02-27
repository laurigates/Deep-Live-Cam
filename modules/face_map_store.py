"""Thread-safe face mapping state store (Issue #59).

Replaces the module-level ``source_target_map``, ``simple_map``, and
``MAP_LOCK`` in ``modules.globals`` with a single object that owns its lock
and exposes a clear interface.

Usage::

    from modules.face_map_store import STORE

    STORE.add_blank()
    entries = STORE.get_entries()   # always a snapshot
    STORE.clear()
"""
from __future__ import annotations

import threading
from typing import Any


class FaceMapStore:
    """Encapsulates face-mapping state with a clear locking contract.

    Two separate data structures are managed:

    * ``_entries`` — the detailed map used for image/video processing
      (``source_target_map`` in the old globals).
    * ``_simple`` — the simplified map used for live/simple mode
      (``simple_map`` in the old globals).

    All public methods acquire a non-reentrant ``threading.Lock`` for the
    minimal duration needed, then return snapshots so callers never hold a
    reference to the live internal state.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[dict[str, Any]] = []
        self._simple: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Entry management
    # ------------------------------------------------------------------

    def add_blank(self) -> None:
        """Append a placeholder entry with no source or target face."""
        with self._lock:
            existing_ids = [m['id'] for m in self._entries]
            next_id = (max(existing_ids) + 1) if existing_ids else 0
            self._entries.append({'id': next_id})

    def set_entries(self, entries: list[dict[str, Any]]) -> None:
        """Replace all entries atomically."""
        with self._lock:
            self._entries = list(entries)

    def get_entries(self) -> list[dict[str, Any]]:
        """Return a snapshot of all entries (not the live list)."""
        with self._lock:
            return list(self._entries)

    def clear(self) -> None:
        """Remove all entries and reset the simple map."""
        with self._lock:
            self._entries.clear()
            self._simple = {}

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    def has_valid_map(self) -> bool:
        """Return ``True`` if any entry has both a source and target face."""
        with self._lock:
            return any(
                "source" in m and "target" in m
                for m in self._entries
            )

    def default_source_face(self) -> Any:
        """Return the first source face found, or ``None``."""
        with self._lock:
            return next(
                (m['source']['face'] for m in self._entries if "source" in m),
                None,
            )

    def simplify(self) -> None:
        """Build the simple map from paired entries.

        Paired entries are those that have both a ``source`` and ``target``
        key.  The result is stored internally and retrievable via
        :meth:`get_simple_map`.
        """
        with self._lock:
            paired = [
                m for m in self._entries
                if "source" in m and "target" in m
            ]
            self._simple = {
                'source_faces': [m['source']['face'] for m in paired],
                'target_embeddings': [
                    m['target']['face'].normed_embedding for m in paired
                ],
            }

    # ------------------------------------------------------------------
    # Simple map
    # ------------------------------------------------------------------

    def set_simple_map(
        self,
        source_faces: list[Any],
        target_embeddings: list[Any],
    ) -> None:
        """Update the simplified map used for live/simple-mode swapping."""
        with self._lock:
            self._simple = {
                'source_faces': source_faces,
                'target_embeddings': target_embeddings,
            }

    def get_simple_map(self) -> dict[str, Any]:
        """Return a shallow copy of the simplified map."""
        with self._lock:
            return dict(self._simple)


# Module-level singleton — import this instead of accessing globals.
STORE: FaceMapStore = FaceMapStore()

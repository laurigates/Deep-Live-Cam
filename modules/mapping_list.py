"""Unified face mapping list data model (Unified Face Mapping UI).

Pure data model with no Tkinter dependency.  Manages an ordered list of
source→target(pin) mapping entries and exposes derived state for the
processing pipeline.

Usage::

    from modules.mapping_list import MAPPING_LIST

    MAPPING_LIST.set_source(0, "/face.png", cv2_img, face_obj)
    MAPPING_LIST.add_entry()
    MAPPING_LIST.sync_to_store(STORE)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MappingEntry:
    """A single source→target(pin) mapping."""

    id: int
    source_path: str | None = None
    source_face: Any | None = None
    source_cv2: np.ndarray | None = None
    pin_path: str | None = None
    pin_face: Any | None = None
    pin_cv2: np.ndarray | None = None


class MappingList:
    """Ordered list of face mapping entries with change notification.

    Starts with a single empty entry (matching current single-source UI).
    """

    def __init__(self) -> None:
        self._entries: list[MappingEntry] = [MappingEntry(id=0)]
        self._next_id: int = 1
        self._on_change: list[Callable[[], None]] = []

    def _notify(self) -> None:
        for cb in self._on_change:
            cb()

    def on_change(self, callback: Callable[[], None]) -> None:
        self._on_change.append(callback)

    # ------------------------------------------------------------------
    # Entry management
    # ------------------------------------------------------------------

    def add_entry(self) -> MappingEntry:
        entry = MappingEntry(id=self._next_id)
        self._next_id += 1
        self._entries.append(entry)
        self._notify()
        return entry

    def remove_entry(self, entry_id: int) -> None:
        self._entries = [e for e in self._entries if e.id != entry_id]
        self._notify()

    def clear_all(self) -> None:
        self._entries.clear()
        self._notify()

    def get_entries(self) -> list[MappingEntry]:
        return list(self._entries)

    # ------------------------------------------------------------------
    # Source / pin management
    # ------------------------------------------------------------------

    def set_source(
        self,
        entry_id: int,
        path: str,
        cv2_img: np.ndarray,
        face: Any,
    ) -> None:
        for entry in self._entries:
            if entry.id == entry_id:
                entry.source_path = path
                entry.source_cv2 = cv2_img
                entry.source_face = face
                self._notify()
                return

    def set_pin(
        self,
        entry_id: int,
        path: str,
        cv2_img: np.ndarray,
        face: Any,
    ) -> None:
        for entry in self._entries:
            if entry.id == entry_id:
                entry.pin_path = path
                entry.pin_cv2 = cv2_img
                entry.pin_face = face
                self._notify()
                return

    def clear_pin(self, entry_id: int) -> None:
        for entry in self._entries:
            if entry.id == entry_id:
                entry.pin_path = None
                entry.pin_cv2 = None
                entry.pin_face = None
                self._notify()
                return

    # ------------------------------------------------------------------
    # Derived state
    # ------------------------------------------------------------------

    def effective_map_faces(self) -> bool:
        """True when >1 entry or any entry has a pin set."""
        if len(self._entries) > 1:
            return True
        return any(e.pin_path is not None for e in self._entries)

    def effective_source_path(self) -> str | None:
        """First entry's source_path (backward compat with globals.source_path)."""
        if self._entries:
            return self._entries[0].source_path
        return None

    def source_count(self) -> int:
        """Number of entries that have a source face."""
        return sum(1 for e in self._entries if e.source_face is not None)

    # ------------------------------------------------------------------
    # Sync to FaceMapStore
    # ------------------------------------------------------------------

    def sync_to_store(self, store: Any) -> None:
        """Convert entries to FaceMapStore format and update the store.

        Only entries with a source face are included.  If any entry has a pin,
        ``store.simplify()`` is called to build the simple map.
        """
        store_entries: list[dict[str, Any]] = []
        has_pins = False

        for entry in self._entries:
            if entry.source_face is None:
                continue
            store_entry: dict[str, Any] = {
                "id": entry.id,
                "source": {
                    "cv2": entry.source_cv2,
                    "face": entry.source_face,
                },
            }
            if entry.pin_face is not None:
                store_entry["target"] = {
                    "cv2": entry.pin_cv2,
                    "face": entry.pin_face,
                }
                has_pins = True
            store_entries.append(store_entry)

        store.set_entries(store_entries)
        if has_pins:
            store.simplify()

    # ------------------------------------------------------------------
    # Persistence (paths only — face objects are not serializable)
    # ------------------------------------------------------------------

    def to_dict(self) -> list[dict[str, Any]]:
        result = []
        for entry in self._entries:
            d: dict[str, str | None] = {
                "id": entry.id,
                "source_path": entry.source_path,
                "pin_path": entry.pin_path,
            }
            result.append(d)
        return result

    def restore_from_dict(self, data: list[dict]) -> None:
        """Restore entries from serialized data (paths only)."""
        self._entries.clear()
        max_id = -1
        for d in data:
            entry_id = d.get("id", 0)
            entry = MappingEntry(
                id=entry_id,
                source_path=d.get("source_path"),
                pin_path=d.get("pin_path"),
            )
            self._entries.append(entry)
            if entry_id > max_id:
                max_id = entry_id
        self._next_id = max_id + 1 if max_id >= 0 else 0

    def restore_from_source_path(self, source_path: str) -> None:
        """Migration: create single mapping from legacy source_path."""
        self._entries = [MappingEntry(id=0, source_path=source_path)]
        self._next_id = 1


# Module-level singleton
MAPPING_LIST: MappingList = MappingList()

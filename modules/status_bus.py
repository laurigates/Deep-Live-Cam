"""Event bus for status message delivery — Issue #56.

Decouples ``modules.core`` from ``modules.ui`` for status updates.
Core publishes events; UI (or any other consumer) subscribes.

Usage::

    # Producer (core, processors, etc.)
    from modules.status_bus import BUS
    BUS.publish("Processing frame 42/100", "core")

    # Consumer (GUI mode)
    from modules.status_bus import BUS
    BUS.subscribe(ui.update_status)

    # Consumer (headless / test)
    BUS.subscribe(lambda msg, caller: print(f"[{caller}] {msg}"))
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

StatusCallback = Callable[[str, str], None]

logger = logging.getLogger(__name__)


class StatusBus:
    """Simple observer bus for status message delivery.

    Thread-safe: ``subscribe``, ``unsubscribe``, ``publish``, and ``clear``
    can be called from any thread concurrently.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[StatusCallback] = []

    def subscribe(self, callback: StatusCallback) -> None:
        """Register *callback* to receive future published messages."""
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: StatusCallback) -> None:
        """Remove *callback*.  No-op if not subscribed."""
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s is not callback]

    def publish(self, message: str, caller: str) -> None:
        """Deliver *message* to all current subscribers.

        Iterates over a snapshot so that subscribers added/removed during
        delivery don't affect the current round.
        """
        with self._lock:
            snapshot = list(self._subscribers)
        for sub in snapshot:
            try:
                sub(message, caller)
            except Exception:  # intentionally broad — a misbehaving subscriber must not disrupt others
                logger.exception("StatusBus subscriber %r raised; continuing", sub)

    def clear(self) -> None:
        """Remove all subscribers."""
        with self._lock:
            self._subscribers.clear()


# Module-level singleton — import this in producers and consumers.
BUS: StatusBus = StatusBus()

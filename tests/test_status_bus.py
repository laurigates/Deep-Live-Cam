"""Tests for StatusBus — Issue #56.

Verifies the event bus that decouples core.py from ui.py for status updates.
"""

import threading


class TestStatusBusBasics:
    def test_subscribe_and_publish(self):
        from modules.status_bus import StatusBus

        bus = StatusBus()
        received = []
        bus.subscribe(lambda msg, caller: received.append((msg, caller)))
        bus.publish("hello", "core")
        assert received == [("hello", "core")]

    def test_multiple_subscribers(self):
        from modules.status_bus import StatusBus

        bus = StatusBus()
        a, b = [], []
        bus.subscribe(lambda m, c: a.append(m))
        bus.subscribe(lambda m, c: b.append(m))
        bus.publish("ping", "test")
        assert a == ["ping"]
        assert b == ["ping"]

    def test_unsubscribe_stops_delivery(self):
        from modules.status_bus import StatusBus

        bus = StatusBus()
        received = []

        def cb(m, c):
            received.append(m)

        bus.subscribe(cb)
        bus.unsubscribe(cb)
        bus.publish("should not arrive", "test")
        assert received == []

    def test_clear_removes_all_subscribers(self):
        from modules.status_bus import StatusBus

        bus = StatusBus()
        received = []
        bus.subscribe(lambda m, c: received.append(m))
        bus.subscribe(lambda m, c: received.append(m))
        bus.clear()
        bus.publish("nothing", "test")
        assert received == []

    def test_publish_with_no_subscribers_is_safe(self):
        from modules.status_bus import StatusBus

        bus = StatusBus()
        bus.publish("nobody listening", "test")  # must not raise

    def test_module_level_bus_singleton(self):
        from modules.status_bus import BUS, StatusBus

        assert isinstance(BUS, StatusBus)

    def test_failing_subscriber_is_logged_and_others_still_receive(self, caplog):
        """A raising subscriber must be logged and must not disrupt others (issue #104)."""
        import logging

        from modules.status_bus import StatusBus

        bus = StatusBus()
        received = []

        def bad_subscriber(msg, caller):
            raise RuntimeError("subscriber exploded")

        bus.subscribe(bad_subscriber)
        bus.subscribe(lambda msg, caller: received.append(msg))

        with caplog.at_level(logging.ERROR, logger="modules.status_bus"):
            bus.publish("msg", "test")

        assert received == ["msg"]
        assert any(r.levelname == "ERROR" for r in caplog.records)
        assert "subscriber" in caplog.text


class TestStatusBusThreadSafety:
    def test_concurrent_publishes_deliver_to_all_subscribers(self):
        from modules.status_bus import StatusBus

        bus = StatusBus()
        received = []
        lock = threading.Lock()

        def cb(msg, caller):
            with lock:
                received.append(msg)

        bus.subscribe(cb)
        threads = [threading.Thread(target=bus.publish, args=(f"msg{i}", "t")) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(received) == 50

    def test_subscribe_during_publish_is_safe(self):
        from modules.status_bus import StatusBus

        bus = StatusBus()
        errors = []

        def late_subscriber():
            try:
                bus.subscribe(lambda m, c: None)
            except Exception as e:
                errors.append(e)

        def publisher():
            try:
                for _ in range(100):
                    bus.publish("msg", "pub")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=publisher),
            threading.Thread(target=late_subscriber),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


class TestCoreUsesStatusBus:
    def test_update_status_publishes_to_bus(self):
        """core.update_status must publish to the global BUS."""
        import modules.globals
        from modules.status_bus import BUS

        received = []

        def cb(msg, caller):
            received.append((msg, caller))

        BUS.subscribe(cb)
        try:
            modules.globals.headless = True  # avoid UI calls
            from modules.core import update_status

            update_status("test message", "TEST")
        finally:
            BUS.unsubscribe(cb)

        assert any(msg == "test message" and caller == "TEST" for msg, caller in received)

    def test_core_does_not_import_ui_at_module_level(self):
        """modules.core must not import modules.ui at the top level."""
        import ast
        import inspect

        import modules.core

        source = inspect.getsource(modules.core)
        tree = ast.parse(source)

        top_level_ui_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Import | ast.ImportFrom)
            and (
                (
                    isinstance(node, ast.Import)
                    and any("ui" in alias.name for alias in node.names if alias.name == "modules.ui")
                )
                or (isinstance(node, ast.ImportFrom) and node.module == "modules.ui")
            )
            and isinstance(node.col_offset, int)
            and node.col_offset == 0  # top-level (not inside a function/class)
        ]
        assert top_level_ui_imports == [], (
            "modules.core imports modules.ui at module level — this forces UI initialization in headless mode"
        )

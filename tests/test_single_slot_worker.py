"""Tests for modules/single_slot_worker.py (Phase 7)."""

import threading
import time


def test_processes_new_input():
    from modules.single_slot_worker import single_slot_worker_loop

    input_holder = [{"data": "hello", "seq": 1}]
    output_holder = [None]
    lock = threading.Lock()
    stop_event = threading.Event()

    def process_fn(inp):
        return {"result": inp["data"].upper(), "seq": inp["seq"]}

    t = threading.Thread(
        target=single_slot_worker_loop,
        args=(input_holder, output_holder, lock, stop_event, process_fn),
        daemon=True,
    )
    t.start()

    # Wait for processing
    deadline = time.time() + 2.0
    while time.time() < deadline:
        with lock:
            out = output_holder[0]
        if out is not None:
            break
        time.sleep(0.01)

    stop_event.set()
    t.join(timeout=2.0)

    assert out is not None
    assert out["result"] == "HELLO"
    assert out["seq"] == 1


def test_skips_duplicate_seq():
    from modules.single_slot_worker import single_slot_worker_loop

    call_count = [0]

    def process_fn(inp):
        call_count[0] += 1
        return {"seq": inp["seq"]}

    input_holder = [{"seq": 5}]
    output_holder = [None]
    lock = threading.Lock()
    stop_event = threading.Event()

    t = threading.Thread(
        target=single_slot_worker_loop,
        args=(input_holder, output_holder, lock, stop_event, process_fn),
        daemon=True,
    )
    t.start()

    # Wait for the first processing
    deadline = time.time() + 2.0
    while time.time() < deadline:
        with lock:
            out = output_holder[0]
        if out is not None:
            break
        time.sleep(0.01)

    # Let it idle for a bit — should not reprocess same seq
    time.sleep(0.05)
    stop_event.set()
    t.join(timeout=2.0)

    assert call_count[0] == 1


def test_stops_when_stop_event_set():
    from modules.single_slot_worker import single_slot_worker_loop

    input_holder = [None]
    output_holder = [None]
    lock = threading.Lock()
    stop_event = threading.Event()

    def process_fn(inp):
        return {"seq": inp["seq"]}

    t = threading.Thread(
        target=single_slot_worker_loop,
        args=(input_holder, output_holder, lock, stop_event, process_fn),
        daemon=True,
    )
    t.start()

    # Stop immediately
    stop_event.set()
    t.join(timeout=2.0)
    assert not t.is_alive()


def test_process_fn_receives_correct_input():
    from modules.single_slot_worker import single_slot_worker_loop

    received = [None]

    def process_fn(inp):
        received[0] = inp
        return {"seq": inp["seq"]}

    input_holder = [{"seq": 42, "extra": "data"}]
    output_holder = [None]
    lock = threading.Lock()
    stop_event = threading.Event()

    t = threading.Thread(
        target=single_slot_worker_loop,
        args=(input_holder, output_holder, lock, stop_event, process_fn),
        daemon=True,
    )
    t.start()

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if received[0] is not None:
            break
        time.sleep(0.01)

    stop_event.set()
    t.join(timeout=2.0)

    assert received[0] is not None
    assert received[0]["seq"] == 42
    assert received[0]["extra"] == "data"


def test_concurrent_reads_and_writes():
    """Thread-safety: writer updates input while worker reads."""
    from modules.single_slot_worker import single_slot_worker_loop

    input_holder = [None]
    output_holder = [None]
    lock = threading.Lock()
    stop_event = threading.Event()

    def process_fn(inp):
        return {"value": inp["seq"] * 2, "seq": inp["seq"]}

    t = threading.Thread(
        target=single_slot_worker_loop,
        args=(input_holder, output_holder, lock, stop_event, process_fn),
        daemon=True,
    )
    t.start()

    # Rapidly update input with increasing seq
    for i in range(1, 20):
        with lock:
            input_holder[0] = {"seq": i}
        time.sleep(0.005)

    # Give worker time to process the latest
    time.sleep(0.1)
    stop_event.set()
    t.join(timeout=2.0)

    with lock:
        out = output_holder[0]
    # Output should reflect some processed seq > 0
    assert out is not None
    assert out["seq"] > 0
    assert out["value"] == out["seq"] * 2

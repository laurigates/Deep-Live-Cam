"""Reusable single-slot holder worker loop for async thread pipelines."""

import threading
import time
from typing import Callable


def single_slot_worker_loop(
    input_holder: list,
    output_holder: list,
    lock: threading.Lock,
    stop_event: threading.Event,
    process_fn: Callable[[dict], dict],
    idle_sleep: float = 0.005,
) -> None:
    """Run a worker loop that reads from a single-slot input holder and writes
    to a single-slot output holder.

    Uses a monotonic ``seq`` int in each input dict to detect new vs. stale
    slots.  The producer overwrites stale input; the consumer always reads
    the latest (never queues).

    *process_fn* receives the input dict and must return an output dict
    (typically including ``{'seq': inp['seq']}``) or ``None`` to skip.
    """
    last_processed_seq = -1

    while not stop_event.is_set():
        with lock:
            inp = input_holder[0]

        if inp is None:
            time.sleep(idle_sleep)
            continue

        seq = inp['seq']
        if seq == last_processed_seq:
            time.sleep(idle_sleep)
            continue

        result = process_fn(inp)
        last_processed_seq = seq

        if result is not None:
            with lock:
                output_holder[0] = result

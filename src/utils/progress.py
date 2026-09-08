"""Newline-based progress suitable for captured output streams."""

import time
from concurrent.futures import FIRST_COMPLETED, wait


def completed_with_progress(futures, stage):
    pending = set(futures)
    total = len(pending)
    completed = 0
    started = time.monotonic()
    print(f"[{stage}] pending tasks={total}; progress counts this invocation only", flush=True)
    while pending:
        ready, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
        for future in ready:
            yield future
            completed += 1
        print(f"[{stage}] processed={completed}/{total} in_flight_or_queued={len(pending)} "
              f"elapsed={time.monotonic() - started:.0f}s (processed does not mean accepted)",
              flush=True)

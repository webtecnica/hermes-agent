"""Bounded streaming output collector — O(max_bytes) memory for any producer.

Replaces the unbounded ``output_chunks: list[str]`` in
``tools/environments/base.py::_wait_for_process()``.

A verbose foreground subprocess can write gigabytes of output before the
configurable ``tool_output.max_bytes`` truncation in ``terminal_tool.py`` ever
runs.  By that point the gateway has already OOM'd or frozen.

This collector enforces the cap *while draining the subprocess*, not after
collecting the complete stream.  It keeps a bounded head and tail window of
the output plus an omitted-byte count.  Proxy operations (sudo-failure
scanning, ANSI stripping, redaction, plugin hooks) receive bounded data.
"""

from __future__ import annotations

import threading
from typing import List


class BoundedOutputCollector:
    """Collect output chunks with a bounded memory footprint.

    Keeps the first ~40% of the budget as a head and the last ~60% as a tail,
    discarding middle content that exceeds the cap.  The omission is reported
    as a ``[...N chars omitted...]`` marker in ``get_output()``.

    Thread-safe: ``append()`` acquires a lock so the drain thread and the poll
    loop's interrupt/timeout paths can safely coexist.
    """

    __slots__ = (
        "_lock",
        "_max_bytes",
        "_head_budget",
        "_tail_budget",
        "_head",
        "_head_len",
        "_tail",
        "_tail_len",
        "_omitted",
    )

    def __init__(self, max_bytes: int = 50_000) -> None:
        if max_bytes < 1:
            raise ValueError(f"max_bytes must be >= 1, got {max_bytes}")
        self._lock = threading.Lock()
        self._max_bytes = max_bytes
        # Same ratio as terminal_tool.py truncation: 40% head, 60% tail
        self._head_budget = int(max_bytes * 0.4)
        self._tail_budget = max_bytes - self._head_budget

        self._head: List[str] = []
        self._head_len = 0
        self._tail: List[str] = []
        self._tail_len = 0
        self._omitted = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, chunk: str) -> None:
        """Append a decoded output chunk, discarding middle if over budget.

        Thread-safe (acquires an internal lock).
        """
        if not chunk:
            return
        with self._lock:
            self._append_unlocked(chunk)

    def get_output(self) -> str:
        """Return the bounded output string (thread-safe read).

        The returned string is at most ``max_bytes`` plus the fixed overhead
        of the truncation notice.
        """
        with self._lock:
            return self._build_output_unlocked()

    @property
    def omitted(self) -> int:
        """Total characters discarded because the output exceeded max_bytes."""
        with self._lock:
            return self._omitted

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    @property
    def retained_bytes(self) -> int:
        """Total characters currently retained in head + tail."""
        with self._lock:
            return self._head_len + self._tail_len

    @property
    def total_bytes_seen(self) -> int:
        """Total characters ever appended (retained + omitted)."""
        with self._lock:
            return self._head_len + self._tail_len + self._omitted

    def __len__(self) -> int:
        return self.retained_bytes

    # ------------------------------------------------------------------
    # Internal helpers (caller must hold the lock)
    # ------------------------------------------------------------------

    def _append_unlocked(self, chunk: str) -> None:
        length = len(chunk)
        if length == 0:
            return

        # ---- Head ----
        if self._head_len < self._head_budget:
            space = self._head_budget - self._head_len
            if length <= space:
                self._head.append(chunk)
                self._head_len += length
                return
            # Chunk straddles the head boundary: take what fits, spill rest
            take = chunk[:space]
            self._head.append(take)
            self._head_len = self._head_budget
            remainder = chunk[space:]
            self._send_to_tail(remainder)
        else:
            # Head already full, everything goes to tail
            self._send_to_tail(chunk)

    def _send_to_tail(self, chunk: str) -> None:
        """Route a chunk into the tail buffer, discarding from the left if over budget."""
        length = len(chunk)
        if length == 0:
            return

        self._tail.append(chunk)
        self._tail_len += length

        # Compact tail from the left until it fits within budget
        self._compact_tail()

    def _compact_tail(self) -> None:
        """Drop bytes from the left of the tail until it fits within budget.

        This is O(1) because we operate on whole strings and pop from the
        front of the list, which is O(k) *in theory* for a list (Python has
        to shift remaining elements), but in practice the list rarely grows
        beyond a few dozen elements since each chunk is at most 4096 bytes
        from os.read().
        """
        while self._tail_len > self._tail_budget and self._tail:
            first = self._tail[0]
            excess = self._tail_len - self._tail_budget
            if not first:
                self._tail.pop(0)
                continue

            drop = min(len(first), excess)
            if drop >= len(first):
                # Drop the entire first segment
                self._tail_len -= len(first)
                self._omitted += len(first)
                self._tail.pop(0)
            else:
                # Partially trim: keep the *rightmost* part of this segment
                # (we preserve tail, so we discard from the left)
                kept = first[drop:]
                self._tail[0] = kept
                self._tail_len -= drop
                self._omitted += drop
                # After trimming the first segment the tail should fit
                # (excess was the amount over budget, and we just removed
                # at least `excess` bytes).  Re-check the while condition.
                break

    def _build_output_unlocked(self) -> str:
        head = "".join(self._head)
        tail = "".join(self._tail)

        if self._omitted > 0:
            notice = (
                f"\n\n... [OUTPUT TRUNCATED - {self._omitted} chars omitted] ...\n\n"
            )
            return head + notice + tail
        return head + tail

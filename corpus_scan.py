#!/usr/bin/env python3
"""The full-text scan, and the process pool that runs it off the GIL.

Why this is a module of its own
-------------------------------
``wookiee_chat.py`` is loaded by path (``spec_from_file_location``) from the
batch scripts, so it lives in ``sys.modules`` under a name -- ``wc`` -- that no
child process can import. A spawned worker has to be able to ``import`` the
function it is asked to run, so the scan lives here, in a plain importable
module with no dependencies beyond the standard library.

Why a process pool
------------------
Measured on the live 179k-question run: ``answer_questions.py`` sat at 89-99%
of *one* core on a ten-core machine and would not go faster, at any
``--concurrency``. The scan is ``bytes.find`` over a 365 MB buffer, and
CPython does not release the GIL for it -- one 275 ms memmem holds the whole
interpreter, event loop included. ``asyncio.to_thread`` does not help: it moves
the call off the event loop's stack while it still owns the GIL.

At 0.54 ``search_*`` calls per question and 13 q/s that is ~7 scans/second, or
~1.0 core-second of GIL per wall-clock second. The process was not waiting on
DeepSeek. It was waiting on itself.

So the buffer is written to a file once and every process ``mmap``s it. One
copy in the OS page cache, no pickling of 365 MB, and each worker holds its own
GIL. A scan is split across the workers on article boundaries, which also cuts
the latency of a single search from ~275 ms to ~275/N ms.
"""

from __future__ import annotations

import bisect
import mmap
import multiprocessing
import re
from array import array
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Offsets are stored as a native int64 array: `bisect` works on it directly,
# it round-trips to disk with `tobytes`/`frombytes`, and 171k of them cost
# 1.4 MB against ~5 MB for the equivalent list of Python ints.
OFFSET_TYPECODE = "q"


def scan_range(buf, starts, doc_lo: int, doc_hi: int, query: str,
               regex: bool, cap: int) -> tuple[dict[int, int], bool]:
    """Match counts per article, over articles ``[doc_lo, doc_hi)``.

    -> ({article index: matches}, whether the scan stopped at ``cap``).

    ``buf`` is the lowercased concatenation of every article and ``starts[i]``
    is where article ``i`` begins in it. The byte range is derived from the
    article range rather than passed in, so a shard boundary can never fall
    inside an article -- which is what lets the shards be merged without
    double-counting, and what keeps a match from straddling two of them.
    """
    if doc_lo >= doc_hi:
        return {}, False
    lo = starts[doc_lo]
    hi = starts[doc_hi] if doc_hi < len(starts) else len(buf)

    counts: dict[int, int] = {}
    hits = 0
    doc = doc_lo                  # matches arrive in order, so the search for
                                  # the owning article walks forward

    def own(pos: int) -> int:
        nonlocal doc
        if pos < starts[doc]:
            doc = bisect.bisect_right(starts, pos) - 1
        else:
            while doc + 1 < len(starts) and starts[doc + 1] <= pos:
                doc += 1
        return doc

    if regex:
        # Compiled here rather than passed in because a compiled pattern does
        # not pickle usefully. The caller has already validated it, so a
        # re.error at this point is a bug, not bad model input.
        pattern = re.compile(query.encode(), re.IGNORECASE)
        # `pos`/`endpos` rather than a slice: it does not copy, `^` still
        # anchors only at the real start of the buffer, and a `\b` at the
        # shard boundary still sees the byte before it -- which is the NUL
        # separator, exactly as an unsharded scan would see.
        for m in pattern.finditer(buf, lo, hi):
            i = own(m.start())
            counts[i] = counts.get(i, 0) + 1
            hits += 1
            if hits >= cap:
                break
    else:
        needle = query.lower().encode()
        if not needle:
            return {}, False
        pos = buf.find(needle, lo, hi)
        while pos >= 0:
            i = own(pos)
            counts[i] = counts.get(i, 0) + 1
            hits += 1
            if hits >= cap:
                break
            pos = buf.find(needle, pos + len(needle), hi)

    return counts, hits >= cap


def open_index(text_path: str | Path,
               starts_path: str | Path) -> tuple[mmap.mmap, array]:
    """-> (read-only mmap of the search buffer, article start offsets).

    The file handle is closed immediately: the mapping keeps the file alive,
    and a worker that never closes its descriptor would hold one per pool.
    """
    with open(text_path, "rb") as fh:
        buf = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    starts = array(OFFSET_TYPECODE)
    starts.frombytes(Path(starts_path).read_bytes())
    return buf, starts


# ---- worker side ------------------------------------------------------------
#
# Module globals rather than arguments: the mapping is established once per
# worker, at pool start-up, and every task after that is (range, query).

_BUF: mmap.mmap | None = None
_STARTS: array | None = None


def _worker_init(text_path: str, starts_path: str) -> None:
    global _BUF, _STARTS
    _BUF, _STARTS = open_index(text_path, starts_path)


def _worker_scan(doc_lo: int, doc_hi: int, query: str, regex: bool,
                 cap: int) -> tuple[dict[int, int], bool]:
    return scan_range(_BUF, _STARTS, doc_lo, doc_hi, query, regex, cap)


# ---- parent side ------------------------------------------------------------

def _context():
    """Start workers by forking a clean server, not by forking the parent.

    Never the platform default. On Linux that is ``fork``, and this pool is
    created by a script that is about to run an event loop with a connection
    pool in it -- forking a process with threads in it is the classic way to
    inherit a lock nobody will ever release. ``forkserver`` forks from a small
    single-threaded server instead, with this module preloaded, so each worker
    starts in milliseconds and inherits nothing.

    What this does *not* avoid, on any start method: the worker re-imports the
    main script to resolve the function it was handed. That is why the scripts
    that build a pool keep their work under ``if __name__ == "__main__"`` --
    without the guard a worker would start the whole run again. Measured cost
    of the re-import for answer_questions.py: 97 ms, once per worker.
    """
    try:
        ctx = multiprocessing.get_context("forkserver")
        ctx.set_forkserver_preload([__name__])
        return ctx
    except (ValueError, AttributeError):        # not available on this platform
        return multiprocessing.get_context("spawn")


class SearchPool:
    """Runs ``scan_range`` across processes, sharded on article boundaries.

    Deliberately a *synchronous* API. ``answer_questions.py`` already calls the
    tools through ``asyncio.to_thread``; blocking that thread on a pool result
    is a wait on IPC, which does release the GIL. Making this async would buy
    nothing and would put the fan-out in the tool layer, where it does not
    belong.
    """

    def __init__(self, text_path: Path, starts_path: Path, n_articles: int,
                 workers: int, shards: int | None = None):
        self.workers = max(1, workers)
        shards = self.workers if shards is None else max(1, shards)
        # One shard per worker: a single scan then uses the whole machine, and
        # under load the extra tasks simply queue, which costs latency rather
        # than throughput.
        self.bounds = _split(n_articles, shards)
        self._pool = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=_context(),
            initializer=_worker_init,
            initargs=(str(text_path), str(starts_path)),
        )

    def scan(self, query: str, regex: bool = False,
             cap: int = 500_000) -> tuple[dict[int, int], bool]:
        # The cap is divided across the shards rather than applied globally.
        # It only exists to bound the work on a pathological query, and the
        # one thing it feeds is the word "over" in front of a result count.
        per = max(1, cap // len(self.bounds))
        futures = [self._pool.submit(_worker_scan, lo, hi, query, regex, per)
                   for lo, hi in self.bounds]
        merged: dict[int, int] = {}
        capped = False
        for fut in futures:
            counts, hit_cap = fut.result()
            merged.update(counts)   # shards own disjoint articles: no collisions
            capped = capped or hit_cap
        return merged, capped

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> SearchPool:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _split(n: int, parts: int) -> list[tuple[int, int]]:
    """``n`` articles into ``parts`` contiguous, near-equal ranges."""
    parts = max(1, min(parts, n)) if n else 1
    step, rest = divmod(n, parts)
    bounds, lo = [], 0
    for i in range(parts):
        hi = lo + step + (1 if i < rest else 0)
        if hi > lo:
            bounds.append((lo, hi))
        lo = hi
    return bounds or [(0, n)]

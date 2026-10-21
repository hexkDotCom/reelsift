"""One heavy GPU user at a time, across every Reelsift process on this machine.

Analysis, the review app's models and music generation each need most of the GPU's memory. They
take this lock first. It is an operating-system file lock, so it is released when its process
ends, even after a crash. Rendering (hardware video decode and encode) doesn't take it.
"""

import json
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import platformdirs

from .tools import Cancelled

_held = threading.local()
_process_lock = threading.Lock()
"""Threads of one process share its file handle, so they queue here first."""


def lock_path() -> Path:
    folder = Path(os.environ.get("REELSIFT_LOCK_DIR") or platformdirs.user_runtime_path(
        "reelsift", appauthor=False, ensure_exists=False))  # fmt: skip
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "gpu.lock"


def _try_lock(f) -> bool:
    if sys.platform == "win32":
        import msvcrt

        try:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl

    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(f) -> None:
    if sys.platform == "win32":
        import msvcrt

        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def holder() -> str | None:
    """What holds the GPU now, as written by its holder (may be stale while it is released)."""
    try:
        data = json.loads(lock_path().with_suffix(".json").read_text(encoding="utf-8"))
        return f"{data['what']} (process {data['pid']})"
    except (OSError, ValueError, KeyError):
        return None


@contextmanager
def hold(
    what: str,
    cancelled: Callable[[], bool] = lambda: False,
    waiting: Callable[[str | None], None] | None = None,
) -> Iterator[None]:
    """Hold the GPU while the block runs. Re-entrant within a thread. While another user holds
    it, waiting(holder) is called about once a second; cancelled() ends the wait with Cancelled."""
    if getattr(_held, "depth", 0):
        _held.depth += 1
        try:
            yield
        finally:
            _held.depth -= 1
        return
    while not _process_lock.acquire(timeout=1.0):
        if cancelled():
            raise Cancelled()
        if waiting:
            waiting(f"{what} in this process")
    try:
        f = open(lock_path(), "a+b")
        try:
            while not _try_lock(f):
                if cancelled():
                    raise Cancelled()
                if waiting:
                    waiting(holder())
                time.sleep(1.0)
            note = lock_path().with_suffix(".json")
            note.write_text(json.dumps({"what": what, "pid": os.getpid()}), encoding="utf-8")
            _held.depth = 1
            try:
                yield
            finally:
                _held.depth = 0
                note.unlink(missing_ok=True)
                _unlock(f)
        finally:
            f.close()
    finally:
        _process_lock.release()

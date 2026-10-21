"""Resumable jobs: long work split into items, each recorded as done or failed in the database.

Running the same kind of job with the same parameters again picks up where the last run stopped:
finished items are skipped, failed and unfinished ones are retried. Interrupting with Ctrl+C
marks the job cancelled; nothing already finished is lost.
"""

import json
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from .project import Trip, now
from .tools import Cancelled

Progress = Callable[[int, int, str], None]


def is_out_of_memory(error: BaseException) -> bool:
    """GPU or system memory ran out: retrying the next item would fail the same way."""
    return (
        isinstance(error, MemoryError)
        or type(error).__name__ == "OutOfMemoryError"
        or "out of memory" in str(error).lower()
    )


@dataclass
class JobResult:
    job_id: int
    done: int = 0
    skipped: int = 0
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failed


class Job:
    def __init__(
        self, trip: Trip, kind: str, params: dict, items: Iterable[str], redo: Iterable[str] = ()
    ):
        """redo: items to run again even if an earlier run of this job finished them, because
        what they made is gone or damaged. Stages that work out their items from what is
        actually saved pass every item here."""
        self.trip = trip
        self.kind = kind
        self.params = json.dumps(params, sort_keys=True)
        db = trip.db
        row = db.execute(
            "SELECT id FROM jobs WHERE kind = ? AND params = ? AND state != 'done'"
            " ORDER BY id DESC LIMIT 1",
            (kind, self.params),
        ).fetchone()
        if row:
            self.id = row["id"]
            db.execute(
                "UPDATE jobs SET state = 'running', updated_at = ? WHERE id = ?", (now(), self.id)
            )
        else:
            self.id = db.execute(
                "INSERT INTO jobs (kind, params, state, created_at, updated_at)"
                " VALUES (?, ?, 'running', ?, ?)",
                (kind, self.params, now(), now()),
            ).lastrowid
        self.items = list(dict.fromkeys(items))
        db.executemany(
            "INSERT OR IGNORE INTO job_items (job_id, item, state) VALUES (?, ?, 'pending')",
            [(self.id, item) for item in self.items],
        )
        db.executemany(
            "UPDATE job_items SET state = 'pending' WHERE job_id = ? AND item = ?",
            [(self.id, item) for item in dict.fromkeys(redo)],
        )

    def _done_items(self) -> set[str]:
        rows = self.trip.db.execute(
            "SELECT item FROM job_items WHERE job_id = ? AND state = 'done'", (self.id,)
        )
        return {r["item"] for r in rows}

    def _finish(self, state: str, message: str) -> None:
        self.trip.db.execute(
            "UPDATE jobs SET state = ?, message = ?, updated_at = ? WHERE id = ?",
            (state, message, now(), self.id),
        )

    def run(self, work: Callable[[str], None], progress: Progress | None = None) -> JobResult:
        """Call work(item) for every unfinished item. work must be safe to repeat."""
        result = JobResult(self.id)
        finished = self._done_items()
        todo = [i for i in self.items if i not in finished]
        result.skipped = len(self.items) - len(todo)
        db = self.trip.db
        try:
            for n, item in enumerate(todo, 1):
                if progress:
                    progress(result.skipped + n, len(self.items), item)
                try:
                    work(item)
                except KeyboardInterrupt:
                    raise
                except Exception as e:  # one bad file must not stop the rest
                    if is_out_of_memory(e):
                        self._finish("failed", "ran out of memory")
                        raise
                    detail = "".join(traceback.format_exception_only(e)).strip()
                    result.failed[item] = detail
                    db.execute(
                        "UPDATE job_items SET state = 'failed', error = ? WHERE job_id = ? AND item = ?",
                        (detail, self.id, item),
                    )
                    continue
                result.done += 1
                db.execute(
                    "UPDATE job_items SET state = 'done', error = NULL WHERE job_id = ? AND item = ?",
                    (self.id, item),
                )
        except (KeyboardInterrupt, Cancelled):
            self._finish("cancelled", f"stopped after {result.done} of {len(todo)} items")
            raise
        if result.failed:
            self._finish("failed", f"{len(result.failed)} of {len(self.items)} items failed")
        else:
            self._finish("done", f"{len(self.items)} items")
        return result

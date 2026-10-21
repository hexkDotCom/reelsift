"""The trip folder: configuration, SQLite database, caches, and a single-writer lock.

Layout:
    reelsift.toml     trip configuration
    reelsift.db       metadata, analysis results, decisions
    reelsift.lock     held while a process writes to the trip
    cache/proxies/    generated low-resolution proxies
    debug/            generated debug pages

Original footage is only ever read.
"""

import json
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from . import config as config_mod

DB_NAME = "reelsift.db"
LOCK_NAME = "reelsift.lock"

# Each entry upgrades the schema by one version. Never edit an entry once released; append.
MIGRATIONS: list[str] = [
    """
    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

    CREATE TABLE assets (
        id INTEGER PRIMARY KEY,
        fingerprint TEXT NOT NULL UNIQUE,
        path TEXT NOT NULL UNIQUE,
        size INTEGER NOT NULL,
        mtime_ns INTEGER NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN ('video', 'proxy', 'photo')),
        device TEXT NOT NULL,
        probe_json TEXT NOT NULL,
        missing INTEGER NOT NULL DEFAULT 0,
        added_at TEXT NOT NULL
    );

    CREATE TABLE cameras (
        id INTEGER PRIMARY KEY,
        device TEXT NOT NULL,
        identity TEXT NOT NULL,
        label TEXT NOT NULL UNIQUE,
        UNIQUE (device, identity)
    );

    CREATE TABLE recordings (
        id INTEGER PRIMARY KEY,
        key TEXT NOT NULL UNIQUE,
        camera_id INTEGER NOT NULL REFERENCES cameras(id),
        kind TEXT NOT NULL CHECK (kind IN ('video', 'photo')),
        duration TEXT NOT NULL,
        fps TEXT,
        vfr INTEGER NOT NULL DEFAULT 0,
        width INTEGER,
        height INTEGER,
        rotation INTEGER NOT NULL DEFAULT 0,
        hdr INTEGER NOT NULL DEFAULT 0
    );

    CREATE TABLE chapters (
        recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
        idx INTEGER NOT NULL,
        asset_id INTEGER NOT NULL UNIQUE REFERENCES assets(id),
        start TEXT NOT NULL,
        duration TEXT NOT NULL,
        source_start TEXT NOT NULL,
        PRIMARY KEY (recording_id, idx)
    );

    CREATE TABLE proxies (
        recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
        chapter_idx INTEGER NOT NULL,
        origin TEXT NOT NULL CHECK (origin IN ('camera', 'generated')),
        path TEXT NOT NULL,
        asset_id INTEGER REFERENCES assets(id),
        offset TEXT NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        upright INTEGER NOT NULL,
        settings TEXT NOT NULL,
        PRIMARY KEY (recording_id, chapter_idx)
    );

    CREATE TABLE time_evidence (
        id INTEGER PRIMARY KEY,
        recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
        provenance TEXT NOT NULL,
        utc_at_zero TEXT NOT NULL,
        zone_offset_min INTEGER,
        confidence REAL NOT NULL,
        detail TEXT NOT NULL
    );

    CREATE TABLE clocks (
        recording_id INTEGER PRIMARY KEY REFERENCES recordings(id) ON DELETE CASCADE,
        utc_at_zero TEXT NOT NULL,
        drift_ppm REAL NOT NULL DEFAULT 0,
        provenance TEXT NOT NULL,
        confidence REAL NOT NULL,
        zone_offset_min INTEGER,
        base_utc_at_zero TEXT NOT NULL,
        correction_s REAL NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE anchors (
        id INTEGER PRIMARY KEY,
        recording_a INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
        time_a TEXT NOT NULL,
        recording_b INTEGER REFERENCES recordings(id) ON DELETE CASCADE,
        time_b TEXT,
        utc TEXT,
        note TEXT NOT NULL DEFAULT '',
        residual_s REAL,
        created_at TEXT NOT NULL,
        CHECK ((recording_b IS NULL) != (utc IS NULL)),
        CHECK ((recording_b IS NULL) = (time_b IS NULL))
    );

    CREATE TABLE jobs (
        id INTEGER PRIMARY KEY,
        kind TEXT NOT NULL,
        params TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN ('running', 'done', 'failed', 'cancelled')),
        message TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE job_items (
        job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
        item TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN ('pending', 'done', 'failed')),
        error TEXT,
        PRIMARY KEY (job_id, item)
    );

    CREATE TABLE issues (
        id INTEGER PRIMARY KEY,
        stage TEXT NOT NULL,
        severity TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'error')),
        message TEXT NOT NULL,
        path TEXT,
        recording_id INTEGER REFERENCES recordings(id) ON DELETE CASCADE,
        created_at TEXT NOT NULL
    );
    """,
    # 2: analysis results and edits. Analysis times are seconds of recording time as REAL;
    # cut points are snapped to source frames only when rendering.
    """
    CREATE TABLE stage_runs (
        recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
        stage TEXT NOT NULL,
        version TEXT NOT NULL,
        finished_at TEXT NOT NULL,
        PRIMARY KEY (recording_id, stage)
    );

    CREATE TABLE observations (
        recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
        second INTEGER NOT NULL,
        motion REAL, activity REAL, shake REAL, blur REAL, brightness REAL,
        dark REAL, bright REAL, scene REAL, loudness REAL, speech REAL,
        PRIMARY KEY (recording_id, second)
    );

    CREATE TABLE speech_segments (
        id INTEGER PRIMARY KEY,
        recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
        start REAL NOT NULL,
        end REAL NOT NULL,
        text TEXT NOT NULL,
        language TEXT NOT NULL,
        avg_logprob REAL NOT NULL,
        no_speech_prob REAL NOT NULL
    );

    CREATE TABLE words (
        recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
        start REAL NOT NULL,
        end REAL NOT NULL,
        word TEXT NOT NULL,
        probability REAL NOT NULL
    );
    CREATE INDEX words_by_time ON words (recording_id, start);

    CREATE TABLE embeddings (
        recording_id INTEGER PRIMARY KEY REFERENCES recordings(id) ON DELETE CASCADE,
        path TEXT NOT NULL,
        model TEXT NOT NULL,
        fps REAL NOT NULL,
        count INTEGER NOT NULL,
        dim INTEGER NOT NULL
    );

    CREATE TABLE candidates (
        id INTEGER PRIMARY KEY,
        recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
        start REAL NOT NULL,
        end REAL NOT NULL,
        core_start REAL NOT NULL,
        core_end REAL NOT NULL,
        min_start REAL NOT NULL,
        max_end REAL NOT NULL,
        source TEXT NOT NULL,
        version INTEGER NOT NULL
    );
    CREATE INDEX candidates_by_recording ON candidates (recording_id, start);

    CREATE TABLE edits (
        id INTEGER PRIMARY KEY,
        length INTEGER NOT NULL,
        revision INTEGER NOT NULL,
        items TEXT NOT NULL,
        settings TEXT NOT NULL,
        approved INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        UNIQUE (length, revision)
    );
    """,
    # 3: review decisions and preferences.
    """
    CREATE TABLE decisions (
        id INTEGER PRIMARY KEY,
        kind TEXT NOT NULL CHECK (kind IN ('drop', 'force', 'keep')),
        recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
        start REAL NOT NULL,
        end REAL NOT NULL,
        length INTEGER,
        replaces TEXT,
        edits_before TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL
    );

    CREATE TABLE preferences (
        id INTEGER PRIMARY KEY,
        text TEXT NOT NULL,
        weights TEXT NOT NULL,
        searches TEXT NOT NULL,
        exclusions TEXT NOT NULL,
        interpretation TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
    # 4: vision-language descriptions of fixed blocks, and cached text embeddings.
    """
    CREATE TABLE vlm_blocks (
        recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
        start REAL NOT NULL,
        end REAL NOT NULL,
        description TEXT,
        tags TEXT NOT NULL,
        interest REAL,
        quality TEXT NOT NULL,
        model TEXT NOT NULL,
        PRIMARY KEY (recording_id, start)
    );

    CREATE TABLE text_embeddings (
        text TEXT NOT NULL,
        model TEXT NOT NULL,
        embedding BLOB NOT NULL,
        PRIMARY KEY (text, model)
    );
    """,
    # 5: decisions record which earlier decisions they switched off or created, for exact undo.
    """
    ALTER TABLE decisions ADD COLUMN suppressed TEXT NOT NULL DEFAULT '[]';
    ALTER TABLE decisions ADD COLUMN created TEXT NOT NULL DEFAULT '[]';
    """,
    # 6: see _migration_6.
    lambda db: _migration_6(db),
    # 7: preferences per edit length, with a style and clip lengths. Rows without a length are
    # the shared preferences of older trips.
    """
    ALTER TABLE preferences ADD COLUMN length INTEGER;
    ALTER TABLE preferences ADD COLUMN settings TEXT NOT NULL DEFAULT '{}';
    """,
    # 8: generated music (immutable files in the trip's music/ folder) and the soundtrack of
    # each edit revision.
    """
    CREATE TABLE music_assets (
        id INTEGER PRIMARY KEY,
        path TEXT NOT NULL UNIQUE,
        sha256 TEXT NOT NULL,
        duration REAL NOT NULL,
        sample_rate INTEGER NOT NULL,
        channels INTEGER NOT NULL,
        length INTEGER NOT NULL,
        analysis TEXT NOT NULL,
        request TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE soundtracks (
        edit_id INTEGER PRIMARY KEY REFERENCES edits(id) ON DELETE CASCADE,
        source TEXT NOT NULL CHECK (source IN ('none', 'files', 'generated')),
        asset_id INTEGER REFERENCES music_assets(id),
        files TEXT NOT NULL DEFAULT '[]',
        mix TEXT NOT NULL,
        fit TEXT NOT NULL DEFAULT '{}',
        created_at TEXT NOT NULL,
        CHECK ((source = 'generated') = (asset_id IS NOT NULL))
    );
    """,
    # 9: per-second travel for rides: GPS speed (km/h) and the picture's travel value.
    """
    CREATE TABLE travel (
        recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
        second INTEGER NOT NULL,
        kmh REAL,
        optical REAL,
        PRIMARY KEY (recording_id, second)
    );
    """,
]

SCHEMA_VERSION = len(MIGRATIONS)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(path: Path, *, shared: bool = False) -> sqlite3.Connection:
    """Open the trip database. shared=True allows use from several threads; the caller must
    then serialize access itself."""
    db = sqlite3.connect(path, isolation_level=None, check_same_thread=not shared)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.execute("PRAGMA journal_mode = WAL")
    db.execute("PRAGMA busy_timeout = 5000")
    return db


@contextmanager
def transaction(db: sqlite3.Connection) -> Iterator[None]:
    """Group writes so they land together or not at all. Nested use joins the outer
    transaction, so a larger operation can wrap smaller ones.

    It takes the write lock first (BEGIN IMMEDIATE): what it reads can't change before it writes,
    and two writers on different connections wait for each other (busy_timeout) instead of one
    failing with "database is locked" when it turns from reading to writing."""
    if db.in_transaction:
        yield
        return
    db.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        db.execute("ROLLBACK")
        raise
    db.execute("COMMIT")


immediate = transaction
"""Spelled out where taking the write lock before reading is the point."""


def _migration_6(db: sqlite3.Connection) -> None:
    """Assets whose file changed into something unreadable are kept but marked; the
    recording-id high-water mark is seeded from every id still referenced; time evidence is
    read again so local wall-clock times saved by older versions are recognized as local."""
    from .ingest import refresh_time_evidence, reserve_recording_id

    db.execute("ALTER TABLE assets ADD COLUMN changed INTEGER NOT NULL DEFAULT 0")
    highest = db.execute("SELECT COALESCE(MAX(id), 0) FROM recordings").fetchone()[0]
    for row in db.execute("SELECT items FROM edits"):
        for item in json.loads(row["items"]):
            highest = max(highest, int(item["recording_id"]))
    reserve_recording_id(db, highest)
    refresh_time_evidence(db)


def migrate(db: sqlite3.Connection) -> int:
    """Bring the schema up to date and return the version it started from."""
    version = db.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"This trip was written by a newer Reelsift (schema {version}, this version knows "
            f"{SCHEMA_VERSION}). Upgrade Reelsift to open it."
        )
    for target in range(version, SCHEMA_VERSION):
        with transaction(db):
            step = MIGRATIONS[target]
            if callable(step):
                step(db)
            else:
                for statement in _statements(step):
                    db.execute(statement)
            db.execute(f"PRAGMA user_version = {target + 1}")
    return version


def _statements(script: str) -> list[str]:
    return [s.strip() for s in script.split(";") if s.strip()]


class TripLockedError(RuntimeError):
    pass


class TripNeedsUpgradeError(TripLockedError):
    """A read-only command opened a trip written by an older Reelsift."""


class TripLock:
    """An OS-level lock on reelsift.lock, released automatically if the process dies."""

    def __init__(self, path: Path):
        self.path = path
        self._file = None

    def acquire(self) -> None:
        self._file = open(self.path, "a+b")
        try:
            if sys.platform == "win32":
                import msvcrt

                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            self._file.close()
            self._file = None
            raise TripLockedError(
                f"Another Reelsift process is working on this trip ({self.path.parent})."
            ) from e

    def release(self) -> None:
        if self._file is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                self._file.seek(0)
                msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None


class Trip:
    """An open trip folder. Use as a context manager; writing requires write=True."""

    def __init__(self, root: Path, *, write: bool = False):
        self.root = root.resolve()
        if not (self.root / config_mod.TRIP_CONFIG).is_file():
            raise FileNotFoundError(f"{self.root} is not a Reelsift trip (no reelsift.toml)")
        self.write = write
        self._lock = TripLock(self.root / LOCK_NAME) if write else None
        self.db: sqlite3.Connection
        self.config = config_mod.load(self.root)

    @classmethod
    def create(cls, root: Path, footage: list[Path], name: str = "") -> "Trip":
        root.mkdir(parents=True, exist_ok=True)
        if (root / config_mod.TRIP_CONFIG).exists():
            raise FileExistsError(f"{root} already contains a trip")
        cfg = config_mod.Config(name=name or root.name, footage=[str(p.resolve()) for p in footage])
        config_mod.write_trip(root, cfg)
        return cls(root, write=True)

    def __enter__(self) -> "Trip":
        if self._lock:
            self._lock.acquire()
        try:
            self.db = connect(self.root / DB_NAME)
            if self.write:
                migrate(self.db)
            else:
                version = self.db.execute("PRAGMA user_version").fetchone()[0]
                if version != SCHEMA_VERSION:
                    self.db.close()
                    raise TripNeedsUpgradeError(
                        "This trip was saved by an older Reelsift. Run a command that changes "
                        "it once (such as reelsift ingest) to upgrade it."
                    )
        except BaseException:
            if self._lock:
                self._lock.release()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self.db.close()
        if self._lock:
            self._lock.release()

    def worker(self, *, shared: bool = False) -> "Trip":
        """A second handle on this open trip for another thread of the same process.

        It shares the lock already held by this handle and has its own database connection,
        because SQLite connections can't cross threads. Close it with close().
        """
        other = object.__new__(Trip)
        other.root, other.write, other._lock, other.config = (
            self.root,
            self.write,
            None,
            self.config,
        )
        other.db = connect(self.root / DB_NAME, shared=shared)
        return other

    def close(self) -> None:
        self.db.close()

    @property
    def proxy_dir(self) -> Path:
        path = self.root / "cache" / "proxies"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def debug_dir(self) -> Path:
        path = self.root / "debug"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def footage_paths(self) -> list[Path]:
        return [Path(p) for p in self.config.footage]

    def issue(
        self,
        stage: str,
        severity: str,
        message: str,
        *,
        path: Path | str | None = None,
        recording_id: int | None = None,
    ) -> None:
        self.db.execute(
            "INSERT INTO issues (stage, severity, message, path, recording_id, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (stage, severity, message, str(path) if path else None, recording_id, now()),
        )

    def clear_issues(self, stage: str) -> None:
        self.db.execute("DELETE FROM issues WHERE stage = ?", (stage,))

    def disk_usage(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())


def is_inside(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False


CACHES = {
    "render": "rendered clips, joined before the final mix",
    "stills": "photos framed for rendering",
    "thumbs": "thumbnails for the review app",
    "music": "beat analysis of music",
    "proxies": "proxies made by Reelsift (camera proxies are not touched)",
    "embeddings": "picture embeddings from analysis",
}


def folder_size(path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else 0


def clean(trip, proxies: bool = False, analysis: bool = False) -> dict[str, int]:
    """Delete caches that can be rebuilt; returns bytes freed per cache. Generated proxies and
    embeddings are kept unless asked for, because rebuilding them takes long."""
    import shutil

    from . import proxies as proxies_mod

    chosen = ["render", "stills", "thumbs", "music"]
    if proxies:
        chosen.append("proxies")
    if analysis:
        chosen.append("embeddings")
    freed = {}
    for name in chosen:
        folder = trip.root / "cache" / name
        freed[name] = folder_size(folder)
        shutil.rmtree(folder, ignore_errors=True)
    trip.db.execute("DELETE FROM jobs WHERE kind = 'render'")
    if proxies:
        trip.db.execute("DELETE FROM jobs WHERE kind = 'proxies'")
        proxies_mod.reconcile(trip)
    return freed

"""Ingest: find footage, identify each file, group files into recordings, pair camera proxies.

Re-running ingest is cheap: files whose path, size and modification time are unchanged are not
read again. A file that moved is recognized by its fingerprint and relinked.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path

from . import devices, photos
from .jobs import Job, Progress
from .media import SIDECAR_EXTENSIONS, VIDEO_EXTENSIONS, Probe, fingerprint, parse_probe, probe
from .project import Trip, is_inside, now, transaction
from .timebase import Provenance, to_db

STAGE = "ingest"


@dataclass
class IngestSummary:
    files: int = 0
    new: int = 0
    relinked: int = 0
    unchanged: int = 0
    failed: dict[str, str] = field(default_factory=dict)
    recordings: int = 0
    cameras: int = 0
    camera_proxies: int = 0
    missing: int = 0
    photos: int = 0


def scan(trip: Trip) -> tuple[list[Path], list[Path], list[Path]]:
    """Return (videos, camera proxy files, photos) under the trip's footage folders. Other files
    whose type Reelsift doesn't read are mentioned once, by extension."""
    from collections import Counter

    main, proxies, stills = [], [], []
    skipped: Counter[str] = Counter()
    for root in trip.footage_paths():
        if not root.exists():
            trip.issue(STAGE, "error", "Footage folder not found", path=root)
            continue
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for path in candidates:
            suffix = path.suffix.lower()
            if not path.is_file() or path.name.startswith(".") or is_inside(path, trip.root):
                continue
            if suffix not in VIDEO_EXTENSIONS | photos.PHOTO_EXTENSIONS:
                if suffix not in SIDECAR_EXTENSIONS:
                    skipped[suffix or "(none)"] += 1
                continue
            if suffix in photos.PHOTO_EXTENSIONS:
                stills.append(path)
            else:
                (proxies if devices.is_proxy_name(path) else main).append(path)
    if skipped:
        kinds = ", ".join(f"{n} {ext}" for ext, n in sorted(skipped.items()))
        trip.issue(STAGE, "info", f"Skipped files of types Reelsift doesn't read: {kinds}")
    return main, proxies, stills


def _asset_probe(row) -> Probe:
    return parse_probe(Path(row["path"]), json.loads(row["probe_json"]))


def upsert_asset(trip: Trip, path: Path, kind: str) -> tuple[int, str]:
    """Record a file and return (asset id, what happened: new, relinked or unchanged)."""
    db = trip.db
    stat = path.stat()
    at_path = db.execute("SELECT * FROM assets WHERE path = ?", (str(path),)).fetchone()
    if at_path and at_path["size"] == stat.st_size and at_path["mtime_ns"] == stat.st_mtime_ns:
        db.execute("UPDATE assets SET missing = 0, changed = 0 WHERE id = ?", (at_path["id"],))
        return at_path["id"], "unchanged"
    fp = fingerprint(path)
    replaced = at_path is not None and at_path["fingerprint"] != fp
    row = db.execute("SELECT * FROM assets WHERE fingerprint = ?", (fp,)).fetchone()
    if row is None:
        # Read the new content before forgetting anything: if it can't be read, the earlier
        # footage keeps its recording and decisions, marked as changed so nothing renders from
        # the file until it is restored or readable.
        try:
            raw, device = _read(path, kind)
        except Exception:
            if replaced:
                db.execute("UPDATE assets SET changed = 1 WHERE id = ?", (at_path["id"],))
            raise
    with transaction(db):
        if replaced:
            # The file at this path now has different content: what was known about the old
            # content (its recording, proxy, analysis and decisions) no longer applies.
            forget_asset(trip, at_path["id"])
            trip.issue(STAGE, "warning",
                       "File content changed since it was last read; it is treated as new footage",
                       path=path)  # fmt: skip
        if row:
            db.execute(
                "UPDATE assets SET path = ?, mtime_ns = ?, missing = 0, changed = 0 WHERE id = ?",
                (str(path), stat.st_mtime_ns, row["id"]),
            )
            return row["id"], "relinked" if row["path"] != str(path) else "unchanged"
        return _insert_asset(trip, path, stat, fp, kind, raw, device), "new"


def _read(path: Path, kind: str) -> tuple[dict, str]:
    """Probe a file: (raw probe data, device name)."""
    if kind == "photo":
        return photos.probe(path), "photo"
    info = probe(path)
    if info.main_video is None:
        raise ValueError("no video stream")
    return info.raw, devices.detect(path, info).name


def _insert_asset(trip: Trip, path: Path, stat, fp: str, kind: str, raw: dict, device: str) -> int:
    db = trip.db
    return db.execute(
        "INSERT INTO assets (fingerprint, path, size, mtime_ns, kind, device, probe_json, added_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            fp,
            str(path),
            stat.st_size,
            stat.st_mtime_ns,
            kind,
            device,
            json.dumps(raw),
            now(),
        ),
    ).lastrowid


def new_recording_id(trip: Trip) -> int:
    """A recording id never used before in this trip. SQLite would reuse the id of a deleted
    recording, and old edit revisions would then point at different footage."""
    db = trip.db
    highest = db.execute("SELECT COALESCE(MAX(id), 0) FROM recordings").fetchone()[0]
    new_id = max(reserved_recording_id(db), highest) + 1
    reserve_recording_id(db, new_id)
    return new_id


def reserved_recording_id(db) -> int:
    row = db.execute("SELECT value FROM meta WHERE key = 'last_recording_id'").fetchone()
    return int(row["value"]) if row else 0


def reserve_recording_id(db, recording_id: int) -> None:
    """Remember that ids up to recording_id are taken, even after those recordings are gone."""
    db.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_recording_id', ?)",
        (str(max(reserved_recording_id(db), recording_id)),),
    )


def forget_asset(trip: Trip, asset_id: int) -> None:
    """Remove an asset and everything derived from it: the recording it belongs to (with its
    analysis, candidates and decisions, by cascade) or the proxy rows that use it."""
    db = trip.db
    with transaction(db):
        for r in db.execute(
            "SELECT recording_id FROM chapters WHERE asset_id = ?", (asset_id,)
        ).fetchall():
            reserve_recording_id(db, r["recording_id"])
            db.execute("DELETE FROM recordings WHERE id = ?", (r["recording_id"],))
        db.execute("DELETE FROM proxies WHERE asset_id = ?", (asset_id,))
        db.execute("DELETE FROM assets WHERE id = ?", (asset_id,))


def camera_folder(trip: Trip, path: Path) -> str:
    """The folder a file was copied into, as a path from its footage root, such as
    "Anna/DJI_001". Two cameras copied into Anna/DJI_001 and Ben/DJI_001 stay two cameras;
    files directly in a footage root use the root's own name.

    With several footage roots, the path starts with the root's name (or its full path, when
    two roots have the same name), so DCIM in one root and DCIM in another stay apart. A file
    inside nested roots belongs to the innermost one."""
    roots = [r.resolve() for r in trip.footage_paths()]
    folder = path.parent.resolve()
    for root in sorted(roots, key=lambda r: len(r.parts), reverse=True):
        try:
            rel = folder.relative_to(root)
        except ValueError:
            continue
        if len(roots) == 1:
            prefix = ""
        elif [r.name for r in roots].count(root.name) == 1:
            prefix = root.name
        else:
            prefix = root.as_posix()
        if not rel.parts:
            return prefix or root.name
        return f"{prefix}/{rel.as_posix()}" if prefix else rel.as_posix()
    return path.parent.name


def _camera(
    trip: Trip, profile: devices.DeviceProfile, identity: str, folder: str, label: str | None = None
) -> int:
    db = trip.db
    row = db.execute(
        "SELECT id FROM cameras WHERE device = ? AND identity = ?", (profile.name, identity)
    ).fetchone()
    if row:
        return row["id"]
    base = f"{label or profile.label} ({folder})"
    label, n = base, 2
    while db.execute("SELECT 1 FROM cameras WHERE label = ?", (label,)).fetchone():
        label, n = f"{base} {n}", n + 1
    return db.execute(
        "INSERT INTO cameras (device, identity, label) VALUES (?, ?, ?)",
        (profile.name, identity, label),
    ).lastrowid


def _duration(info: Probe) -> Fraction:
    video = info.main_video
    return video.duration if video and video.duration else info.duration


def _group(trip: Trip, row) -> tuple[tuple, int]:
    """(device, camera identity, chapter group) and the file's chapter index."""
    path, info = Path(row["path"]), _asset_probe(row)
    profile = devices.by_name(row["device"])
    key = profile.chapter_key(path, info)
    identity = profile.camera_identity(path, info, camera_folder(trip, path))
    return (profile.name, identity, key.group if key else f"file:{row['fingerprint']}"), (
        key.index if key else 0
    )


DETECTION_VERSION = 2
"""Raised when device profiles are added or recognize more files, so files read by an older
Reelsift are looked at again (from their saved probe data, without reading them)."""


def redetect_devices(trip: Trip) -> bool:
    """Match files ingested by an older Reelsift against the current device profiles. A file now
    read as a different device leaves its recording, which is made again by build_recordings
    (losing its analysis and review decisions). Photos that name a phone join its camera."""
    db = trip.db
    row = db.execute("SELECT value FROM meta WHERE key = 'detection_version'").fetchone()
    if row and int(row["value"]) >= DETECTION_VERSION:
        return False
    changed = 0
    for asset in db.execute(
        "SELECT id, path, device, probe_json FROM assets WHERE kind = 'video'"
    ).fetchall():
        info = _asset_probe(asset)
        if info.main_video is None:
            continue
        device = devices.detect(Path(asset["path"]), info).name
        if device == asset["device"]:
            continue
        with transaction(db):
            for r in db.execute(
                "SELECT recording_id FROM chapters WHERE asset_id = ?", (asset["id"],)
            ).fetchall():
                reserve_recording_id(db, r["recording_id"])
                db.execute("DELETE FROM recordings WHERE id = ?", (r["recording_id"],))
            db.execute("UPDATE assets SET device = ? WHERE id = ?", (device, asset["id"]))
        changed += 1
    model = devices.by_name("model")
    for photo in db.execute(
        "SELECT r.id, a.path, a.probe_json FROM recordings r JOIN cameras k ON k.id = r.camera_id"
        " JOIN chapters c ON c.recording_id = r.id JOIN assets a ON a.id = c.asset_id"
        " WHERE r.kind = 'photo' AND k.device = ?",
        (PHOTO_PROFILE.name,),
    ).fetchall():
        name = photos.camera_name(json.loads(photo["probe_json"])["photo"])
        if name == "Unknown camera":
            continue
        folder = camera_folder(trip, Path(photo["path"]))
        camera_id = _camera(trip, model, f"{name}|folder:{folder}", folder, label=name)
        db.execute("UPDATE recordings SET camera_id = ? WHERE id = ?", (camera_id, photo["id"]))
    db.execute("DELETE FROM cameras WHERE id NOT IN (SELECT camera_id FROM recordings)")
    # Rules that read time can have changed for files whose device didn't (a zone from a local
    # time in the file name).
    refresh_time_evidence(db)
    db.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('detection_version', ?)",
        (str(DETECTION_VERSION),),
    )
    if changed:
        trip.issue(STAGE, "warning",
                   f"{changed} files are now recognized as a particular camera; their recordings "
                   "were made again, so their analysis must be redone and their review decisions "
                   "are gone")  # fmt: skip
    return True


def _camera_labels(db) -> dict[int, str]:
    """The label of each file's camera, by asset id."""
    return {r["asset_id"]: r["label"] for r in db.execute(
        "SELECT c.asset_id, k.label FROM chapters c JOIN recordings r ON r.id = c.recording_id"
        " JOIN cameras k ON k.id = r.camera_id")}  # fmt: skip


def _carry_camera_settings(trip: Trip, before: dict[int, str]) -> None:
    """Settings in reelsift.toml are kept by camera label: when files move to a camera with a new
    label, its settings follow them."""
    from . import config as config_mod

    moved: dict[str, set[str]] = defaultdict(set)
    for asset, label in _camera_labels(trip.db).items():
        if before.get(asset) not in (None, label):
            moved[label].add(before[asset])
    cameras, changed = trip.config.cameras, False
    for label, old in sorted(moved.items()):
        found = {o: cameras[o] for o in sorted(old) if o in cameras}
        if label in cameras or not found:
            continue
        if len({(c.zone, c.creation_time) for c in found.values()}) > 1:
            trip.issue(STAGE, "warning",
                       f"Cameras {', '.join(found)} are now one camera, {label}, but have "
                       "different settings; set them again for it")  # fmt: skip
            continue
        settings = next(iter(found.values()))
        cameras[label] = config_mod.CameraConfig(settings.zone, settings.creation_time)
        trip.issue(STAGE, "info", f"Settings of {', '.join(found)} now apply to {label}")
        changed = True
    if changed:
        config_mod.write_trip(trip.root, trip.config)


def build_recordings(trip: Trip, _depth: int = 0) -> int:
    """Turn video assets that are not part of a recording yet into recordings. A chapter or
    segment copied in later joins the recording it continues."""
    db = trip.db
    rows = db.execute(
        "SELECT a.* FROM assets a LEFT JOIN chapters c ON c.asset_id = a.id"
        " WHERE a.kind = 'video' AND c.asset_id IS NULL ORDER BY a.path"
    ).fetchall()
    groups: dict[tuple, list[tuple[int, object, int | None]]] = defaultdict(list)
    for row in rows:
        group, index = _group(trip, row)
        groups[group].append((index, row, None))
    # Files already in recordings that belong to the same chapter groups (only names are
    # compared first: reading a camera's identity can take longer).
    wanted = {(g[0], g[2]) for g in groups if not g[2].startswith("file:")}
    if wanted:
        for row in db.execute(
            "SELECT a.*, c.recording_id AS recording FROM assets a"
            " JOIN chapters c ON c.asset_id = a.id WHERE a.kind = 'video'"
        ).fetchall():
            path, profile = Path(row["path"]), devices.by_name(row["device"])
            key = profile.chapter_key(path, _asset_probe(row))
            if key is None or (profile.name, key.group) not in wanted:
                continue
            group, index = _group(trip, row)
            if group in groups:
                groups[group].append((index, row, row["recording"]))

    created, rebuilt = 0, False
    for (device, identity, _), members in groups.items():
        profile = devices.by_name(device)
        members.sort(key=lambda m: m[0])
        indexes = [m[0] for m in members]
        gaps = [i for i in range(indexes[0], indexes[-1]) if i not in indexes]
        if gaps and not profile.continuous:
            trip.issue(STAGE, "warning",
                       f"Recording has missing chapters (found parts {indexes}); the parts on "
                       "either side of a gap are separate recordings",
                       path=Path(members[0][1]["path"]))  # fmt: skip
        # Chapters split into runs without gaps (and segments into continuous runs, for cameras
        # that record in segments like dashcams); each run is one recording.
        runs = profile.split_runs([(i, Path(r["path"]), _asset_probe(r)) for i, r, _ in members])
        for run in runs:
            chosen = [members[k] for k in run]
            if all(m[2] is not None for m in chosen):
                continue  # nothing new in this run
            with transaction(db):
                if all(m[2] is None for m in chosen):
                    _create_recording(trip, device, identity, [(i, r) for i, r, _ in chosen])
                    created += 1
                    continue
                if _extend_recording(trip, chosen):
                    continue
                # A new chapter in the middle, or one that joins two recordings: they are made
                # again as one, which loses their analysis and review decisions.
                for recording in sorted({m[2] for m in chosen if m[2] is not None}):
                    reserve_recording_id(db, recording)
                    db.execute("DELETE FROM recordings WHERE id = ?", (recording,))
                _create_recording(trip, device, identity, [(i, r) for i, r, _ in chosen])
                trip.issue(STAGE, "warning",
                           "A chapter was added inside an existing recording; it was made again, "
                           "so its analysis must be redone and its review decisions are gone",
                           path=Path(chosen[0][1]["path"]))  # fmt: skip
            created += 1
            rebuilt = True
    if rebuilt and _depth < 2:
        # Chapters of a remade recording that fell outside its new run.
        created += build_recordings(trip, _depth + 1)
    return created


def _extend_recording(trip: Trip, chosen: list) -> bool:
    """Add new chapters at the end of the one recording that the rest of the run already is.
    Its id, review decisions and the analysis of its earlier part stay valid; analysis is redone
    to cover the new part. False when the run isn't that."""
    db = trip.db
    recordings = {m[2] for m in chosen if m[2] is not None}
    if len(recordings) != 1:
        return False
    (recording,) = recordings
    known = [m for m in chosen if m[2] is not None]
    if chosen[: len(known)] != known:
        return False
    chapters = db.execute(
        "SELECT asset_id, idx, start, duration FROM chapters WHERE recording_id = ? ORDER BY idx",
        (recording,),
    ).fetchall()
    if [c["asset_id"] for c in chapters] != [m[1]["id"] for m in known]:
        return False
    start = Fraction(chapters[-1]["start"]) + Fraction(chapters[-1]["duration"])
    idx = chapters[-1]["idx"] + 1
    for _, row, _ in chosen[len(known) :]:
        info = _asset_probe(row)
        duration = _duration(info)
        db.execute(
            "INSERT INTO chapters (recording_id, idx, asset_id, start, duration, source_start)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (recording, idx, row["id"], to_db(start), to_db(duration),
             to_db(info.main_video.start if info.main_video else Fraction(0))),
        )  # fmt: skip
        start, idx = start + duration, idx + 1
    db.execute("UPDATE recordings SET duration = ? WHERE id = ?", (to_db(start), recording))
    db.execute("DELETE FROM stage_runs WHERE recording_id = ?", (recording,))
    device, chapters, mtime_ns = _recording_chapters(db, recording)
    _store_time_evidence(db, recording, recording_evidence(device, chapters), mtime_ns, start)
    trip.issue(STAGE, "info",
               "New chapters continue an existing recording; they were added to it and its "
               "analysis will be redone", path=Path(chosen[len(known)][1]["path"]),
               recording_id=recording)  # fmt: skip
    return True


def recording_evidence(device: str, chapters: list[tuple[Path, Probe, Fraction]]) -> list:
    """A recording's time evidence from all its chapters, as (path, probe, start in the
    recording): each kind from the first chapter that has it, moved back to the recording's
    start. A GoPro may find GPS only in a later chapter."""
    profile = devices.by_name(device)
    found: dict = {}
    for n, (path, info, start) in enumerate(chapters):
        for ev in profile.time_evidence(path, info):
            if ev.provenance in found:
                continue
            if n:
                ev = replace(ev, utc_at_zero=ev.utc_at_zero - timedelta(seconds=float(start)),
                             detail=f"{ev.detail} (in chapter {n + 1}, {path.name})")  # fmt: skip
            found[ev.provenance] = ev
    return list(found.values())


def _recording_chapters(db, recording_id: int) -> tuple[str, list, int]:
    """(device, chapters for recording_evidence, modified time of the last file)."""
    rows = db.execute(
        "SELECT a.path, a.device, a.probe_json, a.mtime_ns, c.start FROM chapters c"
        " JOIN assets a ON a.id = c.asset_id WHERE c.recording_id = ? ORDER BY c.idx",
        (recording_id,),
    ).fetchall()
    chapters = [(Path(r["path"]), _asset_probe(r), Fraction(r["start"])) for r in rows]
    return rows[0]["device"], chapters, rows[-1]["mtime_ns"]


def _create_recording(trip: Trip, device: str, identity: str, members: list) -> None:
    db = trip.db
    members.sort(key=lambda m: m[0])
    first = members[0][1]
    first_path = Path(first["path"])
    profile = devices.by_name(device)
    infos = [_asset_probe(m[1]) for m in members]
    camera_id = _camera(trip, profile, identity, camera_folder(trip, first_path),
                        label=profile.camera_label(first_path, infos[0]))  # fmt: skip
    video = infos[0].main_video
    durations = [_duration(i) for i in infos]
    fps = video.avg_fps if video and not video.vfr else None
    width, height = video.upright_size if video else (None, None)
    recording_id = db.execute(
        "INSERT INTO recordings (id, key, camera_id, kind, duration, fps, vfr, width, height,"
        " rotation, hdr) VALUES (?, ?, ?, 'video', ?, ?, ?, ?, ?, ?, ?)",
        (
            new_recording_id(trip),
            first["fingerprint"],
            camera_id,
            to_db(sum(durations, Fraction(0))),
            to_db(fps) if fps else None,
            int(bool(video and video.vfr)),
            width,
            height,
            video.rotation if video else 0,
            int(bool(video and video.hdr)),
        ),
    ).lastrowid
    start = Fraction(0)
    starts = []
    for idx, ((_, row), info, duration) in enumerate(zip(members, infos, durations, strict=True)):
        starts.append(start)
        db.execute(
            "INSERT INTO chapters (recording_id, idx, asset_id, start, duration, source_start)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                recording_id,
                idx,
                row["id"],
                to_db(start),
                to_db(duration),
                to_db(info.main_video.start if info.main_video else Fraction(0)),
            ),
        )
        start += duration
    last = members[-1][1]
    chapters = [(Path(m[1]["path"]), i, s) for m, i, s in zip(members, infos, starts, strict=True)]
    evidence = recording_evidence(device, chapters)
    _store_time_evidence(db, recording_id, evidence, last["mtime_ns"], start)


PHOTO_PROFILE = devices.DeviceProfile()
PHOTO_PROFILE.name = "photo"
PHOTO_PROFILE.label = "Photos"


def build_photos(trip: Trip) -> int:
    """Each photo not yet in a recording becomes a recording of kind photo, with no duration."""
    db = trip.db
    rows = db.execute(
        "SELECT a.* FROM assets a LEFT JOIN chapters c ON c.asset_id = a.id"
        " WHERE a.kind = 'photo' AND c.asset_id IS NULL ORDER BY a.path"
    ).fetchall()
    for row in rows:
        info = json.loads(row["probe_json"])["photo"]
        path = Path(row["path"])
        name = photos.camera_name(info)
        with transaction(db):
            folder = camera_folder(trip, path)
            # A photo that names its make and model belongs to the same camera as that phone's
            # videos (see devices/model.py); others are a camera of their own.
            if name != "Unknown camera":
                camera_id = _camera(trip, devices.by_name("model"), f"{name}|folder:{folder}",
                                    folder, label=name)  # fmt: skip
            else:
                camera_id = _camera(trip, PHOTO_PROFILE, f"{name}|folder:{folder}", folder,
                                    label=f"{name} photos")  # fmt: skip
            recording_id = db.execute(
                "INSERT INTO recordings (id, key, camera_id, kind, duration, fps, vfr, width,"
                " height, rotation, hdr) VALUES (?, ?, ?, 'photo', '0/1', NULL, 0, ?, ?, 0, 0)",
                (new_recording_id(trip), row["fingerprint"], camera_id, info["width"],
                 info["height"]),
            ).lastrowid  # fmt: skip
            db.execute(
                "INSERT INTO chapters (recording_id, idx, asset_id, start, duration, source_start)"
                " VALUES (?, 0, ?, '0/1', '0/1', '0/1')",
                (recording_id, row["id"]),
            )
            _store_time_evidence(db, recording_id, photos.time_evidence(info), row["mtime_ns"], 0)
    return len(rows)


def refresh_time_evidence(db) -> None:
    """Read every recording's time evidence again from its stored probe data, so evidence saved
    by an older Reelsift follows the current rules (no files are read)."""
    for rec in db.execute("SELECT id, kind, duration FROM recordings").fetchall():
        if not db.execute("SELECT 1 FROM chapters WHERE recording_id = ?", (rec["id"],)).fetchone():
            continue
        if rec["kind"] == "photo":
            row = db.execute(
                "SELECT a.probe_json, a.mtime_ns FROM chapters c JOIN assets a"
                " ON a.id = c.asset_id WHERE c.recording_id = ?", (rec["id"],)
            ).fetchone()  # fmt: skip
            evidence = photos.time_evidence(json.loads(row["probe_json"])["photo"])
            mtime_ns = row["mtime_ns"]
        else:
            device, chapters, mtime_ns = _recording_chapters(db, rec["id"])
            evidence = recording_evidence(device, chapters)
        _store_time_evidence(db, rec["id"], evidence, mtime_ns, Fraction(rec["duration"]))


def _store_time_evidence(db, recording_id, evidence, mtime_ns, duration) -> None:
    db.execute("DELETE FROM time_evidence WHERE recording_id = ?", (recording_id,))
    if not evidence:
        # Last resort: cameras usually set the modified time when a recording ends, but copying
        # files can change it, so this is marked unreliable.
        end = datetime.fromtimestamp(mtime_ns / 1e9, UTC)
        evidence = [
            devices.TimeEvidence(
                Provenance.FILE_MODIFIED,
                end - timedelta(seconds=float(duration)),
                None,
                0.1,
                f"no time in the file; assumed the recording ended at its modified time {end:%Y-%m-%d %H:%M:%S} UTC",
            )
        ]
    for ev in evidence:
        db.execute(
            "INSERT INTO time_evidence (recording_id, provenance, utc_at_zero, zone_offset_min,"
            " confidence, detail) VALUES (?, ?, ?, ?, ?, ?)",
            (
                recording_id,
                ev.provenance.value,
                ev.utc_at_zero.isoformat(),
                ev.zone_offset_min,
                ev.confidence,
                ev.detail,
            ),
        )


def proxy_mismatch(source: Probe, proxy: Probe) -> str | None:
    """Why a camera proxy can't stand in for its source, or None if it can."""
    s, p = source.main_video, proxy.main_video
    if s is None or p is None:
        return "missing video stream"
    s_dur, p_dur = _duration(source), _duration(proxy)
    frame = 1 / (s.avg_fps or Fraction(25))
    if abs(s_dur - p_dur) > frame:
        return f"duration differs ({float(s_dur):.3f} s vs {float(p_dur):.3f} s)"
    if s.frames and p.frames and s.frames != p.frames:
        return f"frame count differs ({s.frames} vs {p.frames})"
    if s.rotation != p.rotation:
        return f"rotation differs ({s.rotation} vs {p.rotation})"
    if s.width and p.width and abs(s.width / s.height - p.width / p.height) > 0.01:
        return "aspect ratio differs"
    return None


def relink_camera_proxies(trip: Trip, proxy_files: list[Path]) -> None:
    """Camera proxies that moved with their footage are found again by content, and their
    proxy rows follow the asset's new path."""
    for path in proxy_files:
        try:
            upsert_asset(trip, path, "proxy")
        except Exception as e:  # an unreadable proxy is reported when pairing
            trip.issue(STAGE, "info", f"Camera proxy unreadable ({e})", path=path)
    trip.db.execute(
        "UPDATE proxies SET path = (SELECT a.path FROM assets a WHERE a.id = proxies.asset_id)"
        " WHERE origin = 'camera' AND asset_id IS NOT NULL"
    )


def pair_camera_proxies(trip: Trip) -> int:
    db = trip.db
    rows = db.execute(
        "SELECT c.recording_id, c.idx, c.source_start, a.path, a.device, a.probe_json"
        " FROM chapters c JOIN assets a ON a.id = c.asset_id"
        " LEFT JOIN proxies p ON p.recording_id = c.recording_id AND p.chapter_idx = c.idx"
        " WHERE p.recording_id IS NULL"
    ).fetchall()
    paired = 0
    for row in rows:
        path = Path(row["path"])
        candidate = devices.by_name(row["device"]).proxy_for(path)
        if candidate is None:
            continue
        source = parse_probe(path, json.loads(row["probe_json"]))
        try:
            asset_id, _ = upsert_asset(trip, candidate, "proxy")
        except Exception as e:
            trip.issue(
                STAGE,
                "warning",
                f"Camera proxy unreadable ({e}); one will be generated",
                path=candidate,
                recording_id=row["recording_id"],
            )
            continue
        proxy = _asset_probe(
            db.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone()
        )
        problem = proxy_mismatch(source, proxy)
        if problem:
            trip.issue(
                STAGE,
                "warning",
                f"Camera proxy doesn't match its source: {problem}; one will be generated",
                path=candidate,
                recording_id=row["recording_id"],
            )
            continue
        p = proxy.main_video
        offset = Fraction(row["source_start"]) - p.start
        db.execute(
            "INSERT INTO proxies (recording_id, chapter_idx, origin, path, asset_id, offset, width,"
            " height, upright, settings) VALUES (?, ?, 'camera', ?, ?, ?, ?, ?, ?, ?)",
            (
                row["recording_id"],
                row["idx"],
                str(candidate),
                asset_id,
                to_db(offset),
                p.width,
                p.height,
                int(p.rotation == 0),
                json.dumps({"origin": "camera"}),
            ),
        )
        paired += 1
    return paired


def ingest(trip: Trip, progress: Progress | None = None) -> IngestSummary:
    summary = IngestSummary()
    db = trip.db
    trip.clear_issues(STAGE)
    main, proxy_files, still_files = scan(trip)
    summary.files = len(main) + len(still_files)
    db.execute("UPDATE assets SET missing = 1 WHERE kind IN ('video', 'photo')")
    stills = set(map(str, still_files))

    def work(item: str) -> None:
        _, what = upsert_asset(trip, Path(item), "photo" if item in stills else "video")
        setattr(summary, what, getattr(summary, what) + 1)

    roots = sorted(str(p) for p in trip.footage_paths())
    items = [str(p) for p in main] + sorted(stills)
    # Every file is looked at again: a file finished by an earlier run may have changed since.
    # Unchanged files cost only a stat, so this is cheap.
    result = Job(trip, "ingest", {"footage": roots}, items, redo=items).run(work, progress)
    for item, error in result.failed.items():
        trip.issue(STAGE, "error", f"Could not read file: {error}", path=item)
    summary.failed = result.failed
    # A file that failed to read is still there; whether it changed is recorded on its asset.
    db.executemany("UPDATE assets SET missing = 0 WHERE path = ?", [(p,) for p in items])

    labels = _camera_labels(db)
    redetected = redetect_devices(trip)
    summary.recordings = build_recordings(trip)
    summary.photos = build_photos(trip)
    if redetected:
        _carry_camera_settings(trip, labels)
    relink_camera_proxies(trip, proxy_files)
    from .proxies import reconcile

    reconcile(trip)
    summary.camera_proxies = pair_camera_proxies(trip)
    summary.cameras = db.execute("SELECT COUNT(*) FROM cameras").fetchone()[0]

    missing = db.execute(
        "SELECT path FROM assets WHERE kind IN ('video', 'photo') AND missing = 1"
    ).fetchall()
    summary.missing = len(missing)
    for row in missing:
        trip.issue(STAGE, "warning", "File is no longer in the footage folders", path=row["path"])

    paired = {r["path"] for r in db.execute("SELECT path FROM proxies WHERE origin = 'camera'")}
    for path in proxy_files:
        if str(path) not in paired:
            trip.issue(
                STAGE, "info", "Camera proxy without a matching full-resolution file", path=path
            )
    return summary

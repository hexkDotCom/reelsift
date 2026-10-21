"""Review: decisions on edits, and re-selecting around them. Used by the web app; no models.

Every edit length is its own edit, with its own preferences (a description, sliders, a style
and clip lengths) and its own decisions. Decision semantics:
- drop: removes the interval from one edit, or from every edit when no length is given. It is
  never selected there again, and no automatically sized clip may reach into it (until undone).
  It switches off earlier keeps and adds of that footage, keeping what they kept outside it; a
  later keep or add overrides the drop for its own interval.
- force: puts the interval into one edit.
- keep: like force, for a clip already in the edit; it is locked so re-selection keeps it.
  Trimming a clip is a keep of the new interval that replaces the old one.

Every change is one operation: it runs in one database transaction, and every edit revision it
saves records the operation and the decisions active at that moment. Undo switches back exactly
the decisions a decision switched off or created and restores the edits it applied to;
restoring a revision brings back that edit and its own decisions. Nothing is reshuffled behind
your back: a drop removes clips and offers replacements; only "select" and "refill" choose new
clips.
"""

import itertools
import json
import math
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace

import numpy as np

from . import edits as edits_mod
from . import moments as moments_mod
from . import prefs as prefs_mod
from .project import Trip, now, transaction
from .scoring import DEFAULT_MODE, MODES, Features, features, load_prompt_bank, score
from .select import (
    GAP,
    ClipLength,
    Forbidden,
    Item,
    Selection,
    clip_interval,
    clip_length,
    merge,
    select,
)
from .tripdata import TripData, load, utc_label


class BrokenEdits(RuntimeError):
    """A change would have broken the rules edits must keep; it was rolled back."""

    def __init__(self, problems: list[str]):
        super().__init__(
            "This change would break the edits, so it wasn't saved: " + "; ".join(problems[:3])
        )
        self.problems = problems


def _subtract(spans: list[tuple[float, float]], a: float, b: float) -> list[tuple[float, float]]:
    """The parts of spans outside [a, b)."""
    out = []
    for s, e in spans:
        if b <= s or e <= a:
            out.append((s, e))
            continue
        if a - s > 1e-3:
            out.append((s, a))
        if e - b > 1e-3:
            out.append((b, e))
    return out


MIN_INTERVAL = 0.2
"""Shortest interval a decision may cover, in seconds."""
TRIM_MIN = 0.5
"""What is left of a kept clip after a drop or trim is kept only if at least this long."""
INDEPENDENT = "independent_edits"
"""Meta key set once the decisions of an older trip were converted to one copy per edit."""


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


class Review:
    def __init__(
        self,
        trip: Trip,
        text_encoder: Callable[[list[str]], np.ndarray] | None = None,
        language_model: Callable[[str], str] | None = None,
    ):
        """text_encoder embeds search texts (the analysis embedding model's text side);
        language_model answers a prompt, for turning preference text into sliders and drafting
        descriptions. Both are optional and should load lazily; without them, searches stay
        unavailable and preferences use keywords."""
        self.trip = trip
        self.lock = threading.RLock()
        self.text_encoder = text_encoder
        self.language_model = language_model
        self._op: str | None = None
        self.data: TripData
        self.feats: Features
        self._separate_old_decisions()
        self.prefs: dict[int, prefs_mod.Preferences] = {}
        self.own_prefs: dict[int, bool] = {}
        self.prefs_version: dict[int, int] = {n: 0 for n in self.lengths}
        """Counts changes to each edit's preferences, so work done outside the lock can tell
        whether they changed meanwhile."""
        for n in self.lengths:
            prefs, own = prefs_mod.latest(trip.db, n)
            prefs = prefs or prefs_mod.Preferences()
            prefs.style = prefs.style or prefs_mod.default_style(n, self.lengths)
            self.prefs[n], self.own_prefs[n] = prefs, own
        self.weights = {n: p.weights(MODES[DEFAULT_MODE]) for n, p in self.prefs.items()}
        self.searches = self._all_searches(compute=False)
        self.reload()

    def _separate_old_decisions(self) -> None:
        """Older trips nested their edits: a keep or add applied to its edit and every longer
        one. Give each of those edits its own copy, once, so the edits stay as they were.

        History keeps working: decisions are renumbered in their original order with each
        copy just before its original, which lists the copies as decisions it created. Undo
        therefore takes back the original together with its copies, a drop that switched off
        the original switches off its copies too, and every saved revision's list of active
        decisions includes the copies."""
        db = self.trip.db
        if db.execute("SELECT 1 FROM meta WHERE key = ?", (INDEPENDENT,)).fetchone():
            return
        rows = db.execute("SELECT * FROM decisions ORDER BY id").fetchall()
        new_rows: list[dict] = []
        mapping: dict[int, list[int]] = {}
        copies_of: dict[int, list[int]] = {}
        for d in rows:
            d = dict(d)
            copies = []
            if d["kind"] in ("keep", "force"):
                first = d["length"] if d["length"] is not None else self.lengths[0]
                d["length"] = first
                for n in (m for m in self.lengths if m > first):
                    copy = {**d, "length": n, "suppressed": "[]", "created": "[]"}
                    copy["id"] = len(new_rows) + 1
                    new_rows.append(copy)
                    copies.append(copy["id"])
            old = d["id"]
            d["id"] = len(new_rows) + 1
            new_rows.append(d)
            mapping[old] = [d["id"], *copies]
            copies_of[d["id"]] = copies

        def translate(ids) -> list[int]:
            return [n for i in ids for n in mapping.get(i, [])]

        with transaction(db):
            for d in new_rows:
                if d["id"] in copies_of:  # an original row, not a copy
                    d["suppressed"] = json.dumps(translate(json.loads(d["suppressed"] or "[]")))
                    created = translate(json.loads(d["created"] or "[]"))
                    d["created"] = json.dumps(created + copies_of[d["id"]])
            db.execute("DELETE FROM decisions")
            columns = list(new_rows[0]) if new_rows else []
            for d in new_rows:
                db.execute(
                    f"INSERT INTO decisions ({', '.join(columns)})"
                    f" VALUES ({', '.join('?' for _ in columns)})",
                    [d[c] for c in columns],
                )
            for e in db.execute("SELECT id, settings FROM edits").fetchall():
                settings = json.loads(e["settings"])
                if "decisions" in settings:
                    settings["decisions"] = translate(settings["decisions"])
                    db.execute(
                        "UPDATE edits SET settings = ? WHERE id = ?",
                        (json.dumps(settings), e["id"]),
                    )
            db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, '1')", (INDEPENDENT,))

    # --- data -------------------------------------------------------------------------------

    def reload(self) -> None:
        with self.lock:
            self.data = load(self.trip)
            self.bank = load_prompt_bank(self.trip)
            self.feats = features(self.data, self.bank, self.searches)
            self.moments = moments_mod.index(self.data, self.feats.embedding)
            self.rescore()

    def rescore(self) -> None:
        self.scores: dict[int, np.ndarray] = {}
        self.contributions: dict[int, np.ndarray] = {}
        for n in self.lengths:
            self.scores[n], self.contributions[n] = score(self.feats, self.weights[n])

    @property
    def lengths(self) -> list[int]:
        return sorted(self.trip.config.lengths)

    def style(self, length: int) -> prefs_mod.Style:
        return prefs_mod.STYLES[self.prefs[length].style]

    def clips(self, length: int) -> ClipLength:
        """How long the clips of this edit are: set by hand, or automatic for its length."""
        p = self.prefs[length]
        return clip_length(length, self.style(length).speech, p.clip)

    @contextmanager
    def operation(self) -> Iterator[str]:
        """One user-visible change: a single transaction, and one id on every revision saved.

        Before the change is committed, the edits are checked against the rules every change
        must keep (see problems()); a change that would break them is rolled back."""
        with self.lock, transaction(self.trip.db):
            outer = self._op is not None
            if not outer:
                self._op = uuid.uuid4().hex[:12]
            try:
                yield self._op
                if not outer:
                    found = self.problems()
                    if found:
                        raise BrokenEdits(found)
            finally:
                if not outer:
                    self._op = None

    def problems(self) -> list[str]:
        """Rules every saved edit must keep; returns what is broken, if anything.

        - every clip is a valid interval inside its recording;
        - clips of one edit never overlap;
        - no clip Reelsift sized itself (unlocked) reaches into footage dropped from its edit;
        - a kept clip (locked) reaches into dropped footage only where a later keep or add in
          its edit brought that footage back."""
        found: list[str] = []
        for n in self.lengths:
            edit = edits_mod.latest(self.trip, n)
            if edit is None:
                continue
            forbidden, still_dropped = self.forbidden(n), self.still_dropped(n)
            items = [i for i in edit.items if i.recording_id in self.data.recordings]
            for i in items:
                duration = self.data.recordings[i.recording_id].duration
                if not (0 <= i.start < i.end <= duration + 1e-3):
                    found.append(f"{n} s edit: clip {i.start}-{i.end} is outside recording "
                                 f"{i.recording_id}")  # fmt: skip
                spans = forbidden if not i.locked else still_dropped
                for a, b in spans.get(i.recording_id, []):
                    if overlap(i.start, i.end, a, b) > 1e-3:
                        found.append(f"{n} s edit: clip {i.start}-{i.end} reaches into "
                                     f"dropped footage {a}-{b}")  # fmt: skip
            ordered = sorted(items, key=lambda i: (i.recording_id, i.start))
            for a, b in itertools.pairwise(ordered):
                if a.recording_id == b.recording_id and b.start < a.end - 1e-3:
                    found.append(
                        f"{n} s edit: clips {a.start}-{a.end} and {b.start}-{b.end} overlap"
                    )
        found += self._ride_problems()
        return found

    def _ride_problems(self) -> list[str]:
        from . import ride as ride_mod

        edit = edits_mod.latest(self.trip, ride_mod.RIDE)
        if edit is None:
            return []
        found = []
        drops = ride_mod.global_drops(self.trip)
        items = [i for i in edit.items if i.recording_id in self.data.recordings]
        for i in items:
            duration = self.data.recordings[i.recording_id].duration
            if not (0 <= i.start < i.end <= duration + 1e-3) or not i.speed > 0:
                found.append(f"ride: stretch {i.start}-{i.end} of recording {i.recording_id} "
                             "is not a valid piece of footage")  # fmt: skip
            for a, b in drops.get(i.recording_id, []):
                if overlap(i.start, i.end, a, b) > 1e-3:
                    found.append(f"ride: stretch {i.start}-{i.end} reaches into dropped "
                                 f"footage {a}-{b}")  # fmt: skip
        ordered = sorted(items, key=lambda i: (i.recording_id, i.start))
        for a, b in itertools.pairwise(ordered):
            if a.recording_id == b.recording_id and b.start < a.end - 1e-3:
                found.append(f"ride: stretches {a.start}-{a.end} and {b.start}-{b.end} overlap")
        return found

    def decisions(self, length: int | None = None) -> list:
        """Active decisions; with a length, those that apply to that edit (including drops
        from every edit)."""
        rows = self.trip.db.execute(
            "SELECT * FROM decisions WHERE active = 1 ORDER BY id"
        ).fetchall()
        if length is None:
            return rows
        return [d for d in rows if d["length"] is None or d["length"] == length]

    def forbidden(self, length: int) -> Forbidden:
        """Footage dropped from this edit."""
        out: Forbidden = {}
        for d in self.decisions(length):
            if d["kind"] == "drop":
                out.setdefault(d["recording_id"], []).append((d["start"], d["end"]))
        return out

    def still_dropped(self, length: int) -> Forbidden:
        """Footage dropped from this edit that no later keep or add in it has brought back."""
        decisions = self.decisions(length)
        out: Forbidden = {}
        for k, d in enumerate(decisions):
            if d["kind"] != "drop":
                continue
            spans = [(d["start"], d["end"])]
            for later in decisions[k + 1 :]:
                if later["kind"] != "drop" and later["recording_id"] == d["recording_id"]:
                    spans = _subtract(spans, later["start"], later["end"])
            if spans:
                out.setdefault(d["recording_id"], []).extend(spans)
        return out

    def allowed(self, length: int) -> np.ndarray:
        c = self.data.candidates
        mask = np.ones(len(c), bool)
        for rec, spans in self.forbidden(length).items():
            for a, b in spans:
                mask &= ~((c.recording == rec) & (c.core_end > a) & (c.core_start < b))
        for name in self.prefs[length].exclusions:
            if name in self.feats.names:
                mask &= self.feats.z[:, self.feats.names.index(name)] < prefs_mod.EXCLUDE_Z
        return mask

    # --- preferences ------------------------------------------------------------------------

    def _bank_model(self) -> str:
        from .scoring import prompt_bank_model

        return prompt_bank_model(self.trip) or ""

    def _all_searches(self, compute: bool) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for p in self.prefs.values():
            out.update(self._embed_searches(p, compute=compute))
        return out

    def _embed_searches(
        self, prefs, compute: bool = True, vectors: dict[str, np.ndarray] | None = None
    ) -> dict[str, np.ndarray]:
        """Embeddings for the preference searches: from the cache, then from `vectors` computed
        beforehand, else computed now if allowed. New ones are cached."""
        model = self._bank_model()
        vectors = vectors or {}
        out = {}
        missing = []
        for s in prefs.searches:
            row = self.trip.db.execute(
                "SELECT embedding FROM text_embeddings WHERE text = ? AND model = ?",
                (s["text"], model),
            ).fetchone()
            if row:
                out[prefs_mod.search_feature(s["text"])] = np.frombuffer(row[0], np.float32)
            elif s["text"] in vectors:
                v = np.asarray(vectors[s["text"]], np.float32)
                self.trip.db.execute(
                    "INSERT OR REPLACE INTO text_embeddings (text, model, embedding) VALUES (?, ?, ?)",
                    (s["text"], model, v.tobytes()),
                )
                out[prefs_mod.search_feature(s["text"])] = v
            else:
                missing.append(s["text"])
        if missing and compute and self.text_encoder is not None:
            vectors = self.text_encoder(missing)
            for text, v in zip(missing, vectors, strict=True):
                v = np.asarray(v, np.float32)
                self.trip.db.execute(
                    "INSERT OR REPLACE INTO text_embeddings (text, model, embedding) VALUES (?, ?, ?)",
                    (text, model, v.tobytes()),
                )
                out[prefs_mod.search_feature(text)] = v
        return out

    def preferences(self, length: int) -> dict:
        p = self.prefs[length]
        auto = clip_length(length, self.style(length).speech)
        return {
            **p.to_json(),
            "length": length,
            "version": self.prefs_version[length],
            "own": self.own_prefs[length],
            "styles": [{"name": s.name, "label": s.label} for s in prefs_mod.STYLES.values()],
            "auto_clip": [round(auto.shortest, 1), round(auto.longest, 1)],
            "vocabulary": [
                {"name": n, "label": prefs_mod.LABELS.get(n, n)} for n in prefs_mod.vocabulary()
            ],
            "searches_available": self.text_encoder is not None,
            "drafting_available": self.language_model is not None,
        }

    def set_text(self, length: int, text: str) -> prefs_mod.Preferences:
        """Interpret wishes in plain language for one edit and re-select it.

        Reading the text and embedding new searches use models, which can take long the first
        time; both happen before the review lock is taken, so the app stays responsive."""
        prefs = prefs_mod.compile_text(text, self.language_model)
        with self.lock:
            prefs.style, prefs.clip = self.prefs[length].style, self.prefs[length].clip
        vectors = self.embed_new_searches(prefs)
        return self.apply(length, prefs, vectors)

    def embed_new_searches(self, prefs) -> dict[str, np.ndarray]:
        """Text embeddings for searches not cached yet, without touching the database."""
        if self.text_encoder is None:
            return {}
        with self.lock:
            cached = set(self._embed_searches(prefs, compute=False))
        missing = [s["text"] for s in prefs.searches
                   if prefs_mod.search_feature(s["text"]) not in cached]  # fmt: skip
        if not missing:
            return {}
        return dict(zip(missing, self.text_encoder(missing), strict=True))

    def apply(
        self,
        length: int,
        prefs: "prefs_mod.Preferences",
        vectors: dict[str, np.ndarray] | None = None,
    ) -> "prefs_mod.Preferences":
        """Use these preferences for one edit: re-score it, then re-select it, keeping its
        decisions. If saving fails, both the database and the in-memory scoring stay as they
        were."""
        if length not in self.lengths:
            raise ValueError(f"no {length} s edit")
        with self.lock:
            previous = (dict(self.prefs), dict(self.own_prefs), dict(self.weights),
                        self.searches, self.feats, self.scores, self.contributions,
                        dict(self.prefs_version))  # fmt: skip
            try:
                with self.operation():
                    prefs.style = prefs.style or prefs_mod.default_style(length, self.lengths)
                    prefs.clean()
                    searches = self._embed_searches(prefs, vectors=vectors)
                    unavailable = [
                        s for s in prefs.searches
                        if prefs_mod.search_feature(s["text"]) not in searches
                    ]  # fmt: skip
                    prefs.searches = [s for s in prefs.searches if s not in unavailable]
                    prefs.clean()
                    prefs.interpretation = prefs_mod.describe(prefs) + [
                        f'can\'t search for "{s["text"]}" without the embedding model'
                        for s in unavailable
                    ]
                    self.prefs = {**self.prefs, length: prefs}
                    self.own_prefs = {**self.own_prefs, length: True}
                    self.weights = {**self.weights, length: prefs.weights(MODES[DEFAULT_MODE])}
                    self.searches = {**self.searches, **searches}
                    self.feats = features(self.data, self.bank, self.searches)
                    self.rescore()
                    prefs_mod.save(self.trip.db, length, prefs)
                    self.select(length)
                    self.prefs_version = {
                        **self.prefs_version,
                        length: self.prefs_version[length] + 1,
                    }
            except BaseException:
                (self.prefs, self.own_prefs, self.weights, self.searches, self.feats,
                 self.scores, self.contributions, self.prefs_version) = previous  # fmt: skip
                raise
            return prefs

    # --- items ------------------------------------------------------------------------------

    def _candidate_for(self, recording_id: int, start: float, end: float) -> int:
        c = self.data.candidates
        same = np.flatnonzero(c.recording == recording_id)
        if not len(same):
            return -1
        hits = np.maximum(
            0, np.minimum(c.core_end[same], end) - np.maximum(c.core_start[same], start)
        )
        return int(c.id[same[int(np.argmax(hits))]])

    def _item(
        self, length: int, recording_id: int, start: float, end: float, source: str = "manual",
        crop_x: float = 0.5,
    ) -> Item:  # fmt: skip
        cid = self._candidate_for(recording_id, start, end)
        idx = np.flatnonzero(self.data.candidates.id == cid)
        return Item(
            recording_id=recording_id,
            start=round(start, 3),
            end=round(end, 3),
            candidate_id=cid,
            score=round(float(self.scores[length][idx[0]]), 4) if len(idx) else 0.0,
            utc=self.data.utc(recording_id, start),
            source=source,
            locked=True,
            crop_x=crop_x,
        )

    def _crop_for(self, length: int, recording_id: int, start: float, end: float) -> float:
        """The crop position a clip covering this interval has in the current edit."""
        current = edits_mod.latest(self.trip, length)
        best, best_overlap = 0.5, 0.0
        for i in current.items if current else []:
            o = overlap(i.start, i.end, start, end) if i.recording_id == recording_id else 0.0
            if o > best_overlap:
                best, best_overlap = i.crop_x, o
        return best

    def forced(self, length: int) -> list[Item]:
        """The kept and added clips of this edit, without footage dropped after them (a later
        drop always wins, however the decisions came to be active together)."""
        decisions = self.decisions(length)
        out = []
        for k, d in enumerate(decisions):
            if d["kind"] not in ("force", "keep"):
                continue
            spans = [(d["start"], d["end"])]
            for later in decisions[k + 1 :]:
                if later["kind"] == "drop" and later["recording_id"] == d["recording_id"]:
                    spans = _subtract(spans, later["start"], later["end"])
            for lo, hi in spans:
                if hi - lo >= TRIM_MIN or (lo, hi) == (d["start"], d["end"]):
                    crop = self._crop_for(length, d["recording_id"], lo, hi)
                    out.append(self._item(length, d["recording_id"], lo, hi, "kept", crop))
        return merge(out)

    # --- edits ------------------------------------------------------------------------------

    def edit(self, length: int) -> edits_mod.StoredEdit:
        from .ride import RIDE

        with self.lock:
            current = edits_mod.latest(self.trip, length)
            if current is None and length == RIDE:
                raise KeyError("no ride yet")
            if current is None:
                self.select(length)
                current = edits_mod.latest(self.trip, length)
            known = [i for i in current.items if i.recording_id in self.data.recordings]
            if len(known) != len(current.items):
                # Footage that was replaced or removed can't be shown or rendered any more.
                current = self._save(length, known, "removed footage that is gone")
            return current

    def _save(
        self,
        length: int,
        items: list[Item],
        note: str,
        soundtrack_from: int | None = None,
        ride_from: dict | None = None,
    ) -> edits_mod.StoredEdit:
        """ride_from: for the ride, the settings of the revision the items come from (restore,
        undo); by default the latest revision's."""
        from .ride import RIDE

        with self.operation() as op:
            if length == RIDE:  # a ride keeps its order and settings
                source = ride_from
                if source is None:
                    latest = edits_mod.latest(self.trip, RIDE)
                    source = latest.settings if latest else {}
                settings = {"note": note, "op": op, "ride": source.get("ride"),
                            "report": source.get("report")}  # fmt: skip
                extra = {} if soundtrack_from is None else {"soundtrack_from": soundtrack_from}
                return edits_mod.save(self.trip, Selection(RIDE, items, 0.0), settings, **extra)
            items = sorted(items, key=_order)
            total = sum(i.duration for i in items)
            sel = Selection(length, items, max(0.0, length - total))
            settings = {
                "note": note,
                "op": op,
                "decisions": [d["id"] for d in self.decisions()],
                "preferences": self.prefs[length].to_json(),
            }
            extra = {} if soundtrack_from is None else {"soundtrack_from": soundtrack_from}
            return edits_mod.save(self.trip, sel, settings, **extra)

    def _choose(self, length: int, forced: list[Item]) -> list[Item]:
        """Clips for this edit around the given ones, chosen with its own preferences."""
        clips = self.clips(length)
        style = self.style(length)
        sel = select(
            self.data, self.scores[length], self.feats.embedding, length,
            allowed=self.allowed(length), forced=forced, moments=self.moments,
            forbidden=self.forbidden(length), clips=clips, coverage=style.coverage,
            coverage_buckets=max(4, int(length / clips.target / style.parts_per_clip)),
        )  # fmt: skip
        return sel.items

    def select(self, length: int) -> edits_mod.StoredEdit:
        """Choose this edit afresh, keeping its decisions but nothing else from earlier
        revisions."""
        with self.operation():
            items = self._snap(length, self._choose(length, self.forced(length)))
            return self._save(length, items, "select")

    def select_all(self) -> dict[int, edits_mod.StoredEdit]:
        with self.operation():
            return {n: self.select(n) for n in self.lengths}

    # --- music ------------------------------------------------------------------------------

    def music(self) -> list:
        """The trip's music tracks with their beats (analyzed once, then cached)."""
        if not self.trip.config.music:
            return []
        key = tuple(self.trip.config.music)
        if getattr(self, "_music_key", None) != key:
            from . import music as music_mod

            self._tracks = music_mod.load(self.trip.root, self.trip.config.music)
            self._music_key = key
        return self._tracks

    def soundtrack(self, length: int):
        """What plays under the current revision of an edit (soundtrack.Resolved), or None when
        its generated track is missing."""
        from . import soundtrack

        edit = edits_mod.latest(self.trip, length)
        if edit is None:
            st = soundtrack.Soundtrack(0, "files" if self.trip.config.music else "none", None,
                                       list(self.trip.config.music))  # fmt: skip
        else:
            st = soundtrack.get(self.trip, edit.id)
        key = (st.source, st.asset_id, tuple(st.files))
        cache = self.__dict__.setdefault("_soundtracks", {})
        if key not in cache:
            try:
                cache[key] = soundtrack.resolve(self.trip, edit.id) if edit else (
                    soundtrack.Resolved(self.music(), st.mix))  # fmt: skip
            except RuntimeError:
                cache[key] = None
        return cache[key]

    def beats(self, length: int) -> np.ndarray:
        """Beat times over an edit, from its music; none when the music has no clear beat."""
        from .music import WEAK_BEAT, beat_grid

        resolved = self.soundtrack(length)
        if resolved is None or not resolved.tracks:
            return np.zeros(0)
        if resolved.generated:
            track = resolved.tracks[0]
            return track.beats if track.confidence >= WEAK_BEAT else np.zeros(0)
        return beat_grid(resolved.tracks, length * 1.5, WEAK_BEAT)

    def _snap(self, length: int, items: list[Item]) -> list[Item]:
        """Snap the edit's cuts to the music's beats."""
        beats = self.beats(length)
        if not len(beats):
            return items
        from .select import snap_to_beats

        return snap_to_beats(sorted(items, key=_order), beats, self.data, self.forbidden(length))

    def fit_cuts(self, length: int) -> edits_mod.StoredEdit:
        """Move the edit's cuts onto its music's beats, as a new revision. It always starts from
        the edit as it was before any fitting, so fitting again (after choosing another track)
        doesn't add up."""
        from . import soundtrack

        with self.operation():
            edit = self.edit(length)
            st = soundtrack.get(self.trip, edit.id)
            items = self._unfitted(edit.items, st.fit.get("unfitted"))
            if not len(self.beats(length)):
                raise ValueError("This edit's music has no clear beat to fit the cuts to.")
            new = self._save(length, self._snap(length, items), "fit to music")
            st = soundtrack.get(self.trip, new.id)
            # The fingerprint of the fitted picture: once it changes (a trim, another
            # selection), the edit from before fitting no longer describes it.
            st.fit = {**st.fit, "fitted_cuts": True, "timing": soundtrack.timing(new.items),
                      "unfitted": [asdict(i) for i in sorted(items, key=_order)]}  # fmt: skip
            soundtrack.put(self.trip, st)
            return new

    @staticmethod
    def _unfitted(current: list[Item], saved: list[dict] | None) -> list[Item]:
        """The edit to fit from: the clips' ends from before the last fit, with everything else
        (locks, crops) as the clips are now; a clip locked since keeps its current ends."""
        if not saved:
            return list(current)
        base = edits_mod.items_from_json(json.dumps(saved))
        now = {(i.recording_id, round(i.start, 3)): i for i in current}
        if len(base) != len(current) or any(
            (b.recording_id, round(b.start, 3)) not in now for b in base
        ):
            return list(current)
        out = []
        for b in base:
            c = now[(b.recording_id, round(b.start, 3))]
            out.append(c if c.locked else replace(b, crop_x=c.crop_x, locked=c.locked))
        return out

    def use_track(self, length: int, asset_id: int) -> None:
        """Play a generated track under the current revision (the picture stays as it is)."""
        from . import musicgen

        with self.operation():
            musicgen.use(self.trip, self.edit(length).id, asset_id)
            musicgen.chosen(self.trip, length)

    def set_source(self, length: int, source: str) -> None:
        """No music, the trip's music files, or the edit's latest generated track."""
        from . import musicgen, soundtrack

        with self.operation():
            edit = self.edit(length)
            st = soundtrack.get(self.trip, edit.id)
            musicgen.chosen(self.trip, length)
            if source == "generated":
                made = soundtrack.assets(self.trip, length)
                if not made:
                    raise ValueError("No music has been generated for this edit yet.")
                musicgen.use(self.trip, edit.id, made[-1].id)
                return
            st.source, st.asset_id, st.fit = source, None, {}
            st.files = list(self.trip.config.music) if source == "files" else []
            soundtrack.put(self.trip, st)

    def set_mix(self, length: int, **levels) -> None:
        """Change the current revision's mix levels (dB); None leaves a level as it is."""
        from . import soundtrack

        with self.operation():
            st = soundtrack.get(self.trip, self.edit(length).id)
            st.mix = replace(st.mix, **{k: float(v) for k, v in levels.items() if v is not None})
            soundtrack.put(self.trip, st)

    def refill(self, length: int) -> edits_mod.StoredEdit:
        """Fill up to the target length, keeping every clip currently in the edit."""
        with self.operation():
            current = [replace(i, locked=True) for i in self.edit(length).items]
            sel = self._choose(length, merge(current + self.forced(length)))
            kept = [i for i in self.edit(length).items if i.locked] + self.forced(length)
            items = [replace(i, locked=_holds_kept(i, kept)) for i in sel]
            return self._save(length, self._snap(length, items), "refill")

    def set_crop(self, length: int, recording_id: int, start: float, end: float, x: float):
        if not math.isfinite(x):
            raise ValueError("the crop position must be a number from 0 to 1")
        with self.operation():
            items = list(self.edit(length).items)
            for n, i in enumerate(items):
                if (
                    i.recording_id == recording_id
                    and abs(i.start - start) < 1e-3
                    and abs(i.end - end) < 1e-3
                ):
                    items[n] = replace(i, crop_x=round(min(1.0, max(0.0, x)), 3))
                    return self._save(length, items, "crop")
            raise KeyError((recording_id, start, end))

    def approve(self, length: int) -> None:
        with self.operation():
            edit = self.edit(length)
            self.trip.db.execute("UPDATE edits SET approved = 1 WHERE id = ?", (edit.id,))

    def revisions(self, length: int) -> list[dict]:
        rows = self.trip.db.execute(
            "SELECT id, revision, approved, created_at, settings, items FROM edits WHERE length = ?"
            " ORDER BY revision DESC",
            (length,),
        )
        return [
            {
                "id": r["id"],
                "revision": r["revision"],
                "approved": bool(r["approved"]),
                "created_at": r["created_at"],
                "note": json.loads(r["settings"]).get("note", ""),
                "clips": len(json.loads(r["items"])),
            }
            for r in rows
        ]

    def restore(self, length: int, edit_id: int) -> edits_mod.StoredEdit:
        """Bring back a revision of this edit, with this edit's own decisions as they were then.
        Footage dropped from every edit since stays out."""
        with self.operation():
            row = self.trip.db.execute(
                "SELECT * FROM edits WHERE id = ? AND length = ?", (edit_id, length)
            ).fetchone()
            if row is None:
                raise KeyError(edit_id)
            settings = json.loads(row["settings"])
            if "decisions" in settings:
                active = set(settings["decisions"])
                for d in self.trip.db.execute(
                    "SELECT id FROM decisions WHERE length = ?", (length,)
                ).fetchall():
                    self.trip.db.execute(
                        "UPDATE decisions SET active = ? WHERE id = ?",
                        (int(d["id"] in active), d["id"]),
                    )
            items = edits_mod.items_from_json(row["items"])
            return self._save(length, self._without_dropped(length, items), f"restore {edit_id}",
                              soundtrack_from=edit_id, ride_from=settings)  # fmt: skip

    def _without_dropped(self, length: int, items: list[Item]) -> list[Item]:
        """Cut footage dropped from this edit out of its clips (kept clips keep what a later
        keep or add brought back). A ride loses only the dropped footage itself."""
        from . import ride as ride_mod

        if length == ride_mod.RIDE:
            return ride_mod.subtract_items(items, ride_mod.global_drops(self.trip))
        for locked, spans in ((False, self.forbidden(length)), (True, self.still_dropped(length))):
            for rec, intervals in spans.items():
                for a, b in intervals:
                    keep = [i for i in items if i.locked != locked]
                    cut = self._cut_out(length, [i for i in items if i.locked == locked],
                                        rec, a, b)  # fmt: skip
                    items = keep + cut
        return items

    # --- decisions --------------------------------------------------------------------------

    def decide(
        self,
        kind: str,
        recording_id: int,
        start: float,
        end: float,
        length: int | None = None,
        replaces: tuple[float, float] | None = None,
    ) -> dict[int, edits_mod.StoredEdit]:
        """Record a decision and apply it. A keep or add needs the edit's length; a drop
        without one applies to every edit. Returns the edits that changed."""
        if kind not in ("drop", "force", "keep"):
            raise ValueError(kind)
        if recording_id not in self.data.recordings:
            raise KeyError(recording_id)
        values = [start, end, *(replaces or [])]
        if not all(isinstance(v, int | float) and math.isfinite(v) for v in values):
            raise ValueError("times must be finite numbers")
        if replaces is not None and (len(replaces) != 2 or replaces[1] <= replaces[0]):
            raise ValueError("replaces must be a start and an end")
        if kind != "drop" and length is None:
            raise ValueError("a keep or add needs the edit's length")
        if length is not None and length not in self.lengths:
            raise ValueError(f"no {length} s edit")
        if kind == "drop" and replaces is not None:
            raise ValueError("a drop can't replace a clip")
        duration = self.data.recordings[recording_id].duration
        start, end = max(0.0, float(start)), min(duration, float(end))
        if end - start < MIN_INTERVAL:
            raise ValueError("the interval is empty or outside the recording")
        db = self.trip.db
        targets = self.lengths if length is None else [length]
        from . import ride as ride_mod

        with self.operation():
            ride_edit = edits_mod.latest(self.trip, ride_mod.RIDE)
            before = {
                n: (e.id if (e := edits_mod.latest(self.trip, n)) else None) for n in self.lengths
            }
            if ride_edit is not None and length is None:
                before[ride_mod.RIDE] = ride_edit.id
            suppressed: list[int] = []
            created: list[int] = []
            positive = "kind IN ('keep', 'force') AND active = 1 AND recording_id = ?"
            if kind == "drop":
                # A drop overrides earlier keeps and adds of the same footage in the edits it
                # applies to. What they kept outside the dropped interval stays kept.
                for d in db.execute(
                    f"SELECT * FROM decisions WHERE {positive} AND start < ? AND end > ?",
                    (recording_id, end, start),
                ).fetchall():
                    if d["length"] not in targets:
                        continue
                    suppressed.append(d["id"])
                    for lo, hi in _subtract([(d["start"], d["end"])], start, end):
                        if hi - lo >= TRIM_MIN:
                            created.append(self._insert_decision(
                                d["kind"], recording_id, lo, hi, d["length"], None, before
                            ))  # fmt: skip
            elif replaces:
                # A trim replaces the decision for exactly this clip. Any other decision of this
                # edit that the clip was made of (when keeps or adds merged) is cut down to the
                # trimmed interval.
                old_start, old_end = replaces
                for d in db.execute(
                    f"SELECT * FROM decisions WHERE {positive} AND length = ? AND start < ?"
                    " AND end > ?",
                    (recording_id, length, old_end, old_start),
                ).fetchall():
                    exact = abs(d["start"] - old_start) < 1e-3 and abs(d["end"] - old_end) < 1e-3
                    lo, hi = max(d["start"], start), min(d["end"], end)
                    if not exact and lo - d["start"] < 1e-3 and d["end"] - hi < 1e-3:
                        continue  # already fits inside the trimmed clip
                    suppressed.append(d["id"])
                    if not exact and hi - lo >= TRIM_MIN:
                        created.append(self._insert_decision(
                            d["kind"], recording_id, lo, hi, d["length"], None, before
                        ))  # fmt: skip
            for d in suppressed:
                db.execute("UPDATE decisions SET active = 0 WHERE id = ?", (d,))
            self._insert_decision(
                kind, recording_id, start, end, length, replaces, before, suppressed, created
            )
            changed = {}
            for n in targets:
                items = list(self.edit(n).items)
                if kind == "drop":
                    new = self._cut_out(n, items, recording_id, start, end)
                else:
                    new = self._place(n, items, recording_id, start, end, replaces,
                                      self.still_dropped(n).get(recording_id, []))  # fmt: skip
                if new != items:
                    changed[n] = self._save(n, new, kind)
            if ride_edit is not None and length is None and kind == "drop":
                new = ride_mod.subtract_items(ride_edit.items, {recording_id: [(start, end)]})
                if new != ride_edit.items:
                    changed[ride_mod.RIDE] = self._save(ride_mod.RIDE, new, "drop")
            return changed

    def _cut_out(
        self, length: int, items: list[Item], rec: int, start: float, end: float
    ) -> list[Item]:
        """An edit without footage [start, end). A clip touching it is cut; what is left of a
        kept clip stays (if not too short), while what is left of a chosen clip goes and
        refilling can use the time."""
        out = []
        for i in items:
            if i.recording_id != rec or overlap(i.start, i.end, start, end) <= 0:
                out.append(i)
                continue
            if not i.locked:
                continue
            for lo, hi in _subtract([(i.start, i.end)], start, end):
                if hi - lo >= TRIM_MIN:
                    piece = self._item(length, rec, lo, hi, i.source, i.crop_x)
                    out.append(replace(piece, locked=True))
        return out

    def _insert_decision(
        self, kind, recording_id, start, end, length, replaces, before,
        suppressed=(), created=(),
    ) -> int:  # fmt: skip
        return self.trip.db.execute(
            "INSERT INTO decisions (kind, recording_id, start, end, length, replaces, edits_before,"
            " suppressed, created, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (kind, recording_id, start, end, length,
             json.dumps(list(replaces)) if replaces else None, json.dumps(before),
             json.dumps(list(suppressed)), json.dumps(list(created)), now()),
        ).lastrowid  # fmt: skip

    def _place(
        self, length: int, items: list[Item], rec: int, start: float, end: float, replaces,
        dropped: list[tuple[float, float]] = (),
    ) -> list[Item]:  # fmt: skip
        """Put [start, end) into an edit, locked. A trim replaces its clip; otherwise every clip it
        touches is merged into one clip covering all of them, so nothing is lost; keeping a clip
        that already covers it just locks it. A clip that is only near (not overlapping) is
        joined only if the footage between them isn't dropped."""
        margin = 0.0 if replaces else GAP
        crop = 0.5
        rest: list[Item] = []
        touching: list[Item] = []
        for i in items:
            same = i.recording_id == rec
            if (
                same
                and replaces
                and abs(i.start - replaces[0]) < 1e-3
                and abs(i.end - replaces[1]) < 1e-3
            ):
                crop = i.crop_x
                continue
            if same and overlap(i.start, i.end, start, end) > 0:
                touching.append(i)
                continue
            if same and overlap(i.start, i.end, start - margin, end + margin) > 0:
                gap = (i.end, start) if i.end <= start else (end, i.start)
                if not any(overlap(*gap, a, b) > 0 for a, b in dropped):
                    touching.append(i)
                    continue
            rest.append(i)
        if (
            not replaces
            and len(touching) == 1
            and touching[0].start <= start
            and end <= touching[0].end
        ):
            placed = replace(touching[0], locked=True)
        else:
            lo = min([start] + [i.start for i in touching])
            hi = max([end] + [i.end for i in touching])
            if touching:
                crop = touching[0].crop_x
            placed = self._item(length, rec, lo, hi, "kept", crop)
        return sorted([*rest, placed], key=_order)

    def undo(self) -> dict[int, edits_mod.StoredEdit]:
        """Undo the last keep, drop or add. The picture goes back; music chosen since stays (it
        isn't a decision), with its end fitted to the picture again."""
        db = self.trip.db
        with self.operation():
            last = db.execute(
                "SELECT * FROM decisions WHERE active = 1 ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if last is None:
                return {}
            db.execute("UPDATE decisions SET active = 0 WHERE id = ?", (last["id"],))
            for d in json.loads(last["created"] or "[]"):
                db.execute("UPDATE decisions SET active = 0 WHERE id = ?", (d,))
            for d in json.loads(last["suppressed"] or "[]"):
                db.execute("UPDATE decisions SET active = 1 WHERE id = ?", (d,))
            # Only the edits the decision (and what it created) applied to go back, and only
            # those still configured; footage dropped from them since stays out.
            created = json.loads(last["created"] or "[]")
            lengths = {last["length"]} | {
                r["length"]
                for r in db.execute(
                    f"SELECT length FROM decisions WHERE id IN ({','.join('?' * len(created))})",
                    created,
                ).fetchall()
            }  # fmt: skip
            targets = self.lengths if None in lengths else [n for n in self.lengths if n in lengths]
            from .ride import RIDE

            if None in lengths:
                targets = [*targets, RIDE]  # a drop from every edit reached the ride too
            restored = {}
            for length, edit_id in json.loads(last["edits_before"]).items():
                n = int(length)
                if edit_id is None or n not in targets:
                    continue
                row = db.execute("SELECT items, settings FROM edits WHERE id = ?",
                                 (edit_id,)).fetchone()  # fmt: skip
                if row:
                    items = self._without_dropped(n, edits_mod.items_from_json(row["items"]))
                    ride_from = json.loads(row["settings"]) if n == RIDE else None
                    restored[n] = self._save(n, items, "undo", ride_from=ride_from)
            return restored

    # --- browsing ---------------------------------------------------------------------------

    def suggestions(self, length: int, recording_id: int, t: float, n: int = 6) -> list[int]:
        """Candidate indexes to fill a gap near (recording, t), best first."""
        with self.lock:
            c = self.data.candidates
            mask = self.allowed(length)
            for i in self.edit(length).items:
                mask &= ~(
                    (c.recording == i.recording_id)
                    & (c.end > i.start - GAP)
                    & (c.start < i.end + GAP)
                )
            near = self.data.utc(recording_id, t)
            utc = np.array(
                [
                    self.data.utc(int(r), float(s)) or np.nan
                    for r, s in zip(c.recording, c.core_start, strict=True)
                ]
            )
            distance = np.abs(utc - near) / 600 if near is not None else np.zeros(len(c))
            rank = self.scores[length] - np.nan_to_num(distance, nan=3.0)
            rank[~mask] = -np.inf
            best = np.argsort(-rank)[:n]
            return [int(i) for i in best if np.isfinite(rank[i])]

    def top_candidates(
        self, length: int, offset: int = 0, limit: int = 60, recording_id: int | None = None
    ) -> list[int]:
        with self.lock:
            rank = self.scores[length].copy()
            rank[~self.allowed(length)] = -np.inf
            if recording_id is not None:
                rank[self.data.candidates.recording != recording_id] = -np.inf
            order = np.argsort(-rank)
            order = order[np.isfinite(rank[order])]
            return [int(i) for i in order[offset : offset + limit]]

    def reasons(self, candidate_id: int, length: int, top: int = 3) -> list[dict]:
        idx = np.flatnonzero(self.data.candidates.id == candidate_id)
        if not len(idx):
            return []
        row = self.contributions[length][idx[0]]
        order = np.argsort(-np.abs(row))[:top]
        return [
            {"name": self.feats.names[j], "value": round(float(row[j]), 2)}
            for j in order
            if abs(row[j]) > 0.05
        ]

    def candidate_clip(self, index: int, length: int) -> tuple[float, float]:
        return clip_interval(self.data, index, self.clips(length), self.forbidden(length))

    def label(self, recording_id: int, t: float) -> str:
        return utc_label(self.data, recording_id, t)


def _order(i: Item):
    return (i.utc if i.utc is not None else float("inf"), i.recording_id, i.start)


def _holds_kept(item: Item, kept: list[Item]) -> bool:
    """Whether a clip contains footage of a kept clip (it may have merged with others)."""
    return any(
        k.recording_id == item.recording_id and overlap(item.start, item.end, k.start, k.end) > 1e-3
        for k in kept
    )


def item_dict(review: Review, item: Item, length: int) -> dict:
    from .ride import RIDE

    d = asdict(item)
    rec = review.data.recordings.get(item.recording_id)
    d.update(
        key=f"{item.recording_id}:{item.start:.3f}:{item.end:.3f}",
        duration=round(item.duration, 3),
        source_duration=round(item.source_duration, 3),
        camera=rec.camera if rec else "",
        time=review.label(item.recording_id, item.start),
        reasons=[] if length == RIDE else review.reasons(item.candidate_id, length),
    )
    return d

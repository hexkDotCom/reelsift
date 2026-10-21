"""HTTP API and static files for the review app.

The server binds to 127.0.0.1 only, rejects requests whose Host or Origin isn't this server, and
serves media only for recordings and outputs the trip knows about, never arbitrary paths.
"""

import asyncio
import contextlib
import logging
import threading
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import edits as edits_mod
from ..project import Trip
from ..render import Output
from ..review import Review, item_dict
from . import media

Seconds = Annotated[float, Field(allow_inf_nan=False)]

STATIC = Path(__file__).parent / "static"


class DecisionIn(BaseModel):
    kind: str
    recording_id: int
    start: Seconds
    end: Seconds
    length: int | None = None
    replaces: list[Seconds] | None = Field(None, min_length=2, max_length=2)
    board: int | None = None
    """The edit length shown when the decision was made, and that edit's id: if the edit has
    changed since, the request is refused rather than applied to clips the user didn't see."""
    expected_edit_id: int | None = None


class PreferencesText(BaseModel):
    text: str


class PreferencesIn(BaseModel):
    text: str = ""
    adjustments: dict[str, float] = {}
    searches: list[dict] = []
    exclusions: list[str] = []
    style: str = ""
    clip: list[float] | None = None
    """[shortest, longest] clip length in seconds, or None for automatic."""
    expected_version: int | None = None
    """The version of the preferences the form showed; a newer one is not overwritten."""


class CropIn(BaseModel):
    recording_id: int
    start: Seconds
    end: Seconds
    crop_x: Seconds
    expected_edit_id: int | None = None


class RenderIn(BaseModel):
    length: int
    vertical: bool = False
    expected_edit_id: int | None = None
    """The edit revision shown when the user asked; refused if it is no longer the latest."""
    preview: bool = False
    """A small, quick render to hear the music with the picture."""


class RideIn(BaseModel):
    target: Annotated[float, Field(gt=0, le=24 * 3600, allow_inf_nan=False)] | None = None
    speed: str = "8"
    camera: str | None = None
    keep_unknown: bool = True
    expected_edit_id: int | None = None


class RideFixIn(BaseModel):
    kind: str
    recording_id: int
    start: Seconds
    end: Seconds
    expected_edit_id: int | None = None


class GenerateIn(BaseModel):
    genre: str = Field("", max_length=100)
    mood: str = Field("", max_length=200)
    bpm: int | None = Field(None, ge=40, le=220)
    candidates: int = Field(2, ge=1, le=4)
    expected_edit_id: int | None = None


class TrackIn(BaseModel):
    asset_id: int
    expected_edit_id: int | None = None


class SourceIn(BaseModel):
    source: str
    expected_edit_id: int | None = None


class MixIn(BaseModel):
    music: Annotated[float, Field(ge=-60, le=12, allow_inf_nan=False)] | None = None
    background: Annotated[float, Field(ge=-60, le=12, allow_inf_nan=False)] | None = None
    duck: Annotated[float, Field(ge=-60, le=0, allow_inf_nan=False)] | None = None
    sound: Annotated[float, Field(ge=-60, le=12, allow_inf_nan=False)] | None = None
    reset: bool = False
    expected_edit_id: int | None = None


class MusicJobs:
    """Music generation in the background, one at a time, with its progress for the UI. It
    doesn't hold the review lock while it works; its results are written through its own
    connection, and only put under the edit if the edit is still the one it was made for."""

    def __init__(self, trip: Trip, before_gpu):
        self.trip = trip
        self.before_gpu = before_gpu
        self.state: dict = {"running": False}
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self, length: int, candidates: int, on_done, started=None) -> None:
        with self.lock:
            if self.state.get("running"):
                raise HTTPException(409, "Music is already being generated")
            self.state = {"running": True, "length": length, "messages": [], "error": None,
                          "made": [], "assigned": None, "note": ""}  # fmt: skip
        self.cancel.clear()
        self.thread = threading.Thread(target=self._run,
                                       args=(length, candidates, on_done, started),
                                       daemon=True)  # fmt: skip
        self.thread.start()

    def stop(self) -> None:
        from ..tools import cancel

        while self.thread is not None and self.thread.is_alive():
            cancel(self.cancel)
            self.thread.join(1.0)

    def _run(self, length: int, candidates: int, on_done, started) -> None:
        from .. import musicgen
        from ..tools import Cancelled, cancellable

        worker = None
        messages = self.state["messages"]
        try:
            worker = self.trip.worker()
            with cancellable(self.cancel):
                report = musicgen.generate(
                    worker, length, candidates=candidates, say=messages.append,
                    before_gpu=self.before_gpu, started=started,
                )  # fmt: skip
            self.state.update(made=report.made, assigned=report.assigned, note=report.note)
        except Cancelled:
            self.state.update(error="cancelled")
        except Exception as e:  # reported to the UI
            self.state.update(error=str(e))
        finally:
            self.state["running"] = False
            if worker is not None:
                worker.close()
            on_done()


class Renders:
    """Background renders, one at a time, with progress for the UI."""

    def __init__(self, trip: Trip):
        self.trip = trip
        self.state: dict = {"running": False}
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None

    def start(
        self, length: int, vertical: bool, items, revision: int, name: str, music,
        preview: bool = False, tag: str = "",
    ) -> None:  # fmt: skip
        with self.lock:
            if self.state.get("running"):
                raise HTTPException(409, "A render is already running")
            self.state = {"running": True, "length": length, "done": 0, "total": 0,
                          "output": None, "error": None}  # fmt: skip
        self.cancel.clear()
        self.thread = threading.Thread(
            target=self._run,
            args=(length, vertical, items, revision, name, music, preview, tag),
            daemon=True,
        )  # fmt: skip
        self.thread.start()

    def stop(self) -> None:
        """Cancel a running render and wait until it has stopped. Its FFmpeg processes are
        killed and it can't start new ones; the wait has no time limit, because the trip lock is
        released afterwards and nothing may write to the trip then."""
        from ..tools import cancel

        while self.thread is not None and self.thread.is_alive():
            cancel(self.cancel)
            self.thread.join(1.0)

    def _run(self, length, vertical, items, revision, name, music, preview, tag) -> None:
        from ..render import render
        from ..tools import Cancelled, cancellable

        worker = None
        try:
            worker = self.trip.worker()
            out = Output(1080, 1920) if vertical else Output()
            suffix = "-vertical" if vertical else ""
            folder = self.trip.root / "output"
            if preview:  # small and quick; its clips are kept, so a new mix only redoes the sound
                out = Output(360, 640, quality=30) if vertical else Output(640, 360, quality=30)
                folder = self.trip.root / "output" / "previews"
            dest = folder / f"{name}-{length}s-r{revision}{tag}{suffix}.mp4"

            def progress(n, total, _item):
                if self.cancel.is_set():
                    raise Cancelled()
                self.state.update(done=n, total=total)

            with cancellable(self.cancel):
                render(worker, items, dest, out, progress, music=music)
            self.state.update(running=False, output=dest.name, preview=preview)
        except Cancelled:
            self.state.update(running=False, error="render cancelled")
        except Exception as e:  # reported to the UI
            self.state.update(running=False, error=str(e))
        finally:
            if worker is not None:
                worker.close()


def quiet_disconnects(loop: asyncio.AbstractEventLoop) -> None:
    """A browser closed while a video was loading resets its connection; on Windows asyncio
    reports that as an error with a traceback, though nothing went wrong. Other errors are
    reported as before."""
    previous = loop.get_exception_handler()

    def handler(loop, context) -> None:
        if isinstance(context.get("exception"), (ConnectionResetError, ConnectionAbortedError)):
            return
        if previous is not None:
            previous(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(handler)


def quiet_shutdown(record: logging.LogRecord) -> bool:
    """A logging filter for the server: connections still open when the app stops (a paused
    video) are closed then, which is expected; it isn't reported as an error."""
    if "timeout graceful shutdown exceeded" in record.getMessage():
        return False
    return not (record.exc_info and record.exc_info[0] is not None
                and issubclass(record.exc_info[0], asyncio.CancelledError))  # fmt: skip


def create_app(trip: Trip, review: Review, port: int) -> FastAPI:
    renders = Renders(trip)
    unload = getattr(review.text_encoder, "unload", None)
    music_jobs = MusicJobs(trip, unload)

    @contextlib.asynccontextmanager
    async def lifespan(_app):
        quiet_disconnects(asyncio.get_running_loop())
        yield
        # The trip lock is released after this: nothing may still be writing then.
        renders.stop()
        music_jobs.stop()

    app = FastAPI(title="Reelsift", docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=lifespan)  # fmt: skip
    app.state.renders = renders
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    allowed_origins = {f"http://{h}" for h in allowed_hosts}
    db = review.trip

    from ..review import BrokenEdits

    @app.exception_handler(BrokenEdits)
    async def broken_edits(_request: Request, error: BrokenEdits):
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.middleware("http")
    async def same_origin_only(request: Request, call_next):
        host = request.headers.get("host", "")
        if host not in allowed_hosts and host != "testserver":
            return JSONResponse({"detail": "forbidden host"}, status_code=403)
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD") and origin and origin not in allowed_origins:
            return JSONResponse({"detail": "forbidden origin"}, status_code=403)
        response = await call_next(request)
        if not request.url.path.startswith(("/media/", "/api/")):
            # The app's own files: always checked, so a new Reelsift's app is used at once.
            response.headers["Cache-Control"] = "no-cache"
        return response

    def edit_payload(length: int) -> dict:
        edit = review.edit(length)
        items = []
        for item in edit.items:
            d = item_dict(review, item, length)
            spans = media.preview_spans(db, item.recording_id, item.start, item.end)
            d["spans"] = spans
            d["preview"] = spans[0] if spans else None
            rec = review.data.recordings[item.recording_id]
            d["photo"] = rec.kind == "photo"
            d["aspect"] = round(rec.width / rec.height, 4) if rec.width and rec.height else 16 / 9
            d["thumb"] = (
                f"/media/thumb/{item.recording_id}/{round((item.start + item.end) / 2 * 1000)}"
            )
            items.append(d)
        return {
            "length": length,
            "revision": edit.revision,
            "edit_id": edit.id,
            "approved": edit.approved,
            "duration": round(sum(i.duration for i in edit.items), 2),
            "items": items,
            "music": music_payload(length, edit.id),
        }

    def music_payload(length: int, edit_id: int) -> dict:
        from .. import soundtrack

        st = soundtrack.get(db, edit_id)
        settings = trip.config.music_for(length)
        tracks = []
        for a in soundtrack.assets(db, length):
            tracks.append({
                "id": a.id, "bpm": round(a.analysis.get("bpm", 0)), "duration": round(a.duration),
                "hint": a.request.get("hint"), "caption": a.request.get("caption", ""),
                "created_at": a.created_at, "missing": not a.path.is_file(),
                "url": f"/media/music/{a.id}",
            })  # fmt: skip
        return {
            "source": st.source,
            "asset_id": st.asset_id,
            "files": [Path(f).name for f in (st.files if st.source == "files" else [])],
            "mix": {
                "music": round(st.mix.music, 1),
                "background": round(st.mix.background, 1),
                "duck": round(st.mix.duck, 1),
                "sound": round(st.mix.sound, 1),
            },
            "fitted_cuts": bool(st.fit.get("fitted_cuts")),
            "ends_early": bool(st.fit.get("ends_early")),
            "genre": settings.genre,
            "mood": settings.mood,
            "bpm": settings.bpm,
            "tracks": tracks,
        }

    def candidate_payload(index: int, length: int) -> dict:
        c = review.data.candidates
        rec = int(c.recording[index])
        start, end = review.candidate_clip(index, length)
        spans = media.preview_spans(db, rec, start, end)
        return {
            "candidate_id": int(c.id[index]),
            "recording_id": rec,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 2),
            "score": round(float(review.scores[length][index]), 3),
            "camera": review.data.recordings[rec].camera,
            "time": review.label(rec, start),
            "source": str(c.source[index]),
            "photo": review.data.recordings[rec].kind == "photo",
            "reasons": review.reasons(int(c.id[index]), length),
            "thumb": f"/media/thumb/{rec}/{round((start + end) / 2 * 1000)}",
            "spans": spans,
            "preview": spans[0] if spans else None,
        }  # fmt: skip

    from ..ride import RIDE

    def check_length(length: int) -> None:
        if length == RIDE:
            if edits_mod.latest(db, RIDE) is None:
                raise HTTPException(404, "no ride yet")
            return
        if length not in review.lengths:
            raise HTTPException(404, f"no {length} s edit in this trip")

    def check_current(length: int | None, expected: int | None) -> None:
        """Refuse a change made against an edit revision that is no longer the latest."""
        if length is None or expected is None:
            return
        if review.edit(length).id != expected:
            raise HTTPException(409, "This edit changed in the meantime; it has been reloaded.")

    @app.get("/api/trip")
    def trip_info():
        with review.lock:
            return {
                "name": trip.config.name,
                "lengths": review.lengths,
                "recordings": len(review.data.recordings),
                "candidates": len(review.data.candidates),
                "cameras": sorted({r.camera for r in review.data.recordings.values()}),
                "recording_list": [
                    {
                        "id": r.id,
                        "camera": r.camera,
                        "time": review.label(r.id, 0.0),
                        "duration": round(r.duration, 1),
                    }
                    for r in sorted(
                        review.data.recordings.values(), key=lambda r: (r.utc_start or 0, r.id)
                    )
                ],
                "decisions": len(review.decisions()),
                "ride": edits_mod.latest(db, RIDE) is not None,
                "music": [
                    {"name": t.path.name, "bpm": round(t.bpm), "duration": round(t.duration)}
                    for t in review.music()
                ],
            }

    @app.get("/api/edits/{length}")
    def get_edit(length: int):
        check_length(length)
        with review.lock:
            return edit_payload(length)

    @app.post("/api/edits/{length}/select")
    def reselect(length: int, expected_edit_id: int | None = None):
        check_length(length)
        with review.lock:
            check_current(length, expected_edit_id)
            review.select(length)
            return edit_payload(length)

    @app.post("/api/edits/{length}/refill")
    def refill(length: int, expected_edit_id: int | None = None):
        check_length(length)
        with review.lock:
            check_current(length, expected_edit_id)
            review.refill(length)
            return edit_payload(length)

    @app.post("/api/edits/{length}/approve")
    def approve(length: int, expected_edit_id: int | None = None):
        check_length(length)
        with review.lock:
            check_current(length, expected_edit_id)
            review.approve(length)
            return edit_payload(length)

    @app.get("/api/edits/{length}/revisions")
    def revisions(length: int):
        check_length(length)
        with review.lock:
            return review.revisions(length)

    @app.post("/api/edits/{length}/restore/{edit_id}")
    def restore(length: int, edit_id: int):
        check_length(length)
        with review.lock:
            try:
                review.restore(length, edit_id)
            except KeyError:
                raise HTTPException(404, "no such revision") from None
            return edit_payload(length)

    @app.post("/api/decisions")
    def decide(d: DecisionIn):
        if d.kind not in ("drop", "force", "keep"):
            raise HTTPException(400, "kind must be drop, force or keep")
        if d.length is not None:
            check_length(d.length)
        if d.board is not None:
            check_length(d.board)
        with review.lock:
            check_current(d.board, d.expected_edit_id)
            try:
                changed = review.decide(
                    d.kind, d.recording_id, d.start, d.end, d.length,
                    tuple(d.replaces) if d.replaces else None,
                )  # fmt: skip
            except (ValueError, KeyError) as e:
                raise HTTPException(400, str(e)) from None
            return {"changed": sorted(changed)}

    @app.post("/api/edits/{length}/crop")
    def set_crop(length: int, body: CropIn):
        """Where the 9:16 crop sits for one clip; saved as a new revision."""
        check_length(length)
        with review.lock:
            check_current(length, body.expected_edit_id)
            try:
                review.set_crop(length, body.recording_id, body.start, body.end, body.crop_x)
            except ValueError as e:
                raise HTTPException(400, str(e)) from None
            except KeyError:
                raise HTTPException(404, "no such clip in this edit") from None
            return edit_payload(length)

    def apply_prefs(length: int, prefs, vectors, version: int) -> dict:
        """Save preferences worked out from `version` of this edit's; if they changed while a
        model was busy (another tab, another click), refuse rather than overwrite them."""
        with review.lock:
            if review.prefs_version[length] != version:
                raise HTTPException(
                    409, "This edit's description changed in the meantime; it has been reloaded."
                )
            try:
                review.apply(length, prefs, vectors)
            except ValueError as e:
                raise HTTPException(400, str(e)) from None
            return review.preferences(length)

    @app.get("/api/preferences/{length}")
    def get_preferences(length: int):
        check_length(length)
        with review.lock:
            return review.preferences(length)

    @app.post("/api/preferences/{length}/text")
    def set_preferences_text(length: int, body: PreferencesText):
        """Read a description into sliders for one edit and re-select it."""
        from ..prefs import compile_text

        check_length(length)
        text = body.text.strip()[:2000]
        with review.lock:
            version = review.prefs_version[length]
            style, clip = review.prefs[length].style, review.prefs[length].clip
        # Models (possibly loading for the first time) run without holding the review lock.
        prefs = compile_text(text, review.language_model)
        prefs.style, prefs.clip = style, clip
        return apply_prefs(length, prefs, review.embed_new_searches(prefs), version)

    @app.put("/api/preferences/{length}")
    def set_preferences(length: int, body: PreferencesIn):
        """Sliders, searches, style and clip lengths for one edit, as set in the app."""
        from ..prefs import Preferences

        check_length(length)
        with review.lock:
            current = review.prefs[length]
            version = review.prefs_version[length]
            if body.expected_version is not None and body.expected_version != version:
                raise HTTPException(
                    409, "This edit's settings were changed elsewhere; they have been reloaded."
                )
        prefs = Preferences(
            text=body.text or current.text,
            adjustments=body.adjustments,
            searches=body.searches,
            exclusions=body.exclusions,
            engine="sliders",
            style=body.style or current.style,
            clip=body.clip,
            drafted=current.drafted and (body.text or current.text) == current.text,
        )
        try:
            prefs.clean()
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        # A new search may load the picture model: that runs without holding the review lock.
        return apply_prefs(length, prefs, review.embed_new_searches(prefs), version)

    @app.post("/api/preferences/{length}/draft")
    def draft_preferences(length: int):
        """Write this edit's description again from the trip's footage, for its style."""
        from .. import drafts

        check_length(length)
        with review.lock:
            current = review.prefs[length]
            version = review.prefs_version[length]
            trip_summary = drafts.summary(review.trip, review.data)
        # The language model may load for the first time; that runs without the review lock.
        edits = {length: current.style}
        texts = drafts.write(trip_summary, edits, review.language_model)
        prefs = drafts.to_preferences(texts, edits, review.language_model, {length: current})
        prefs = prefs[length]
        return apply_prefs(length, prefs, review.embed_new_searches(prefs), version)

    @app.post("/api/undo")
    def undo():
        with review.lock:
            return {"changed": sorted(review.undo())}

    @app.get("/api/suggestions/{length}")
    def suggestions(length: int, recording_id: int, t: float):
        check_length(length)
        with review.lock:
            return [
                candidate_payload(i, length) for i in review.suggestions(length, recording_id, t)
            ]

    @app.get("/api/candidates/{length}")
    def candidates(length: int, offset: int = 0, limit: int = 40, recording_id: int | None = None):
        check_length(length)
        limit = max(1, min(limit, 200))
        with review.lock:
            indexes = review.top_candidates(length, offset, limit, recording_id)
            return [candidate_payload(i, length) for i in indexes]

    @app.post("/api/render")
    def start_render(r: RenderIn):
        from .. import soundtrack

        check_length(r.length)
        with review.lock:
            check_current(r.length, r.expected_edit_id)
            edit = review.edit(r.length)
            st = soundtrack.get(db, edit.id)
            try:
                music = soundtrack.resolve(db, edit.id)
            except RuntimeError as e:
                raise HTTPException(409, str(e)) from None
        import re

        name = re.sub(r"[^A-Za-z0-9]+", "-", trip.config.name).strip("-").lower() or "trip"
        if r.length == RIDE:
            name += "-ride"
        renders.start(r.length, r.vertical, edit.items, edit.revision, name, music, r.preview,
                      soundtrack.tag(st))  # fmt: skip
        return renders.state

    # --- the ride ------------------------------------------------------------------------------

    ride_job: dict = {"running": False}

    def ride_payload() -> dict:
        from .. import ride as ride_mod

        edit = edits_mod.latest(db, RIDE)
        settings = ride_mod.settings_of(db)
        out = {
            "exists": edit is not None,
            "settings": settings.to_json(),
            "cameras": [{"label": c, "seconds": round(s)} for c, s in ride_mod.cameras(db)],
            "job": dict(ride_job),
        }
        if edit is None:
            return out
        out.update(edit_payload(RIDE))
        out["report"] = edit.settings.get("report")
        stops = []
        camera = (out.get("report") or {}).get("camera") or settings.camera
        known = [c for c, _ in ride_mod.cameras(db)]
        recs = [r for r, _, _ in ride_mod.recordings_of(db, camera)] if camera in known else []
        if recs:
            from .. import travel

            for rec, (states, _) in ride_mod.states(db, recs).items():
                for a, b in travel.runs(states, travel.STOPPED):
                    spans = media.preview_spans(db, rec, float(a), float(b))
                    stops.append({"recording_id": rec, "start": float(a), "end": float(b),
                                  "spans": spans, "preview": spans[0] if spans else None,
                                  "time": review.label(rec, float(a)),
                                  "thumb": f"/media/thumb/{rec}/{round((a + b) / 2 * 1000)}"})  # fmt: skip
        out["stops"] = sorted(stops, key=lambda s: (recs.index(s["recording_id"]), s["start"]))
        return out

    def claim_ride_job(message: str) -> dict:
        """Start a ride build, under the review lock: one at a time, and remembering what it is
        built from, so a build that finishes after the ride (or a drop from every edit) changed
        isn't saved over the change."""
        from .. import ride as ride_mod

        if ride_job.get("running"):
            raise HTTPException(409, "The ride is already being built")
        current = edits_mod.latest(db, RIDE)
        token = uuid.uuid4().hex
        ride_job.clear()
        ride_job.update(running=True, error=None, message=message, token=token)
        return {"token": token, "base": current.id if current else None,
                "drops": ride_mod.global_drops(db)}  # fmt: skip

    def run_ride(settings, note: str, claim: dict) -> None:
        from .. import ride as ride_mod
        from ..tools import Cancelled

        worker = None
        try:
            worker = trip.worker()
            items, report = ride_mod.build(worker, settings,
                                           say=lambda m: ride_job.update(message=m))  # fmt: skip
            with review.lock:
                current = edits_mod.latest(db, RIDE)
                if ride_job.get("token") != claim["token"]:
                    return
                if (current.id if current else None) != claim["base"] or (
                        ride_mod.global_drops(db) != claim["drops"]):  # fmt: skip
                    ride_job.update(running=False, error="The ride changed while it was being "
                                    "built; build it again.", message="")  # fmt: skip
                    return
                ride_mod.save(db, items, settings, report, note)
            ride_job.update(running=False, error=None, message="")
        except (ride_mod.RideError, Cancelled) as e:
            ride_job.update(running=False, error=str(e))
        except Exception as e:  # reported to the UI
            ride_job.update(running=False, error=f"{type(e).__name__}: {e}")
        finally:
            worker.close()

    @app.get("/api/ride")
    def get_ride():
        with review.lock:
            return ride_payload()

    @app.post("/api/ride/build")
    def build_ride(r: RideIn):
        from .. import ride as ride_mod

        with review.lock:
            current = edits_mod.latest(db, RIDE)
            if r.expected_edit_id is not None and current is not None and (
                    current.id != r.expected_edit_id):  # fmt: skip
                raise HTTPException(409, "The ride changed in the meantime; it has been reloaded.")
            settings = ride_mod.settings_of(db)
            settings.target, settings.speed = r.target, r.speed
            settings.camera, settings.keep_unknown = r.camera, r.keep_unknown
            try:
                settings.validate()
            except ride_mod.RideError as e:
                raise HTTPException(422, str(e)) from None
            claim = claim_ride_job("Starting")
        threading.Thread(target=run_ride, args=(settings, "ride", claim), daemon=True).start()
        with review.lock:
            return ride_payload()

    @app.post("/api/ride/fix")
    def fix_ride(f: RideFixIn):
        """Drop a stretch from the ride, or keep a stop in it at normal speed."""
        from .. import ride as ride_mod

        if f.kind not in ("drop", "keep") or not f.end > f.start:
            raise HTTPException(422, "kind must be drop or keep, with end after start")
        with review.lock:
            if edits_mod.latest(db, RIDE) is not None:
                check_current(RIDE, f.expected_edit_id)
            settings = ride_mod.settings_of(db)
            settings.overrides.append({"kind": f.kind, "recording_id": f.recording_id,
                                       "start": f.start, "end": f.end})  # fmt: skip
            claim = claim_ride_job("Rebuilding")
        run_ride(settings, f"{f.kind} {f.start:.1f}-{f.end:.1f}", claim)
        if ride_job.get("error"):
            raise HTTPException(422, ride_job["error"])
        with review.lock:
            return ride_payload()

    # --- music ---------------------------------------------------------------------------------

    def music_done() -> None:
        with review.lock:
            review.__dict__.pop("_soundtracks", None)

    @app.get("/api/music")
    def music_status():
        from .. import musicgen

        profile = musicgen.installed()
        return {
            "installed": profile is not None,
            "profile": profile.name if profile else None,
            "max_duration": profile.max_duration if profile else None,
            "genres": sorted(musicgen.GENRES),
            "job": music_jobs.state,
        }

    @app.post("/api/edits/{length}/music/generate")
    def generate_music(length: int, g: GenerateIn):
        from .. import config as config_mod
        from .. import musicgen
        from ..config import MusicSettings

        check_length(length)
        if music_jobs.state.get("running"):
            raise HTTPException(409, "Music is already being generated")
        if musicgen.installed() is None:
            raise HTTPException(409, "Install the music model first: reelsift music install")
        with review.lock:
            check_current(length, g.expected_edit_id)
            own = trip.config.music_lengths.get(length, MusicSettings())
            given = MusicSettings(genre=g.genre.strip() or None, mood=g.mood.strip(), bpm=g.bpm)
            trip.config.music_lengths[length] = given.over(own)
            config_mod.write_trip(trip.root, trip.config)
            try:  # accepted for the revision shown, before the job starts
                started = musicgen.begin(db, length, g.expected_edit_id)
            except ValueError as e:
                raise HTTPException(409, str(e)) from None
        music_jobs.start(length, g.candidates, music_done, started)
        return music_jobs.state

    @app.get("/api/music/job")
    def music_job():
        return music_jobs.state

    @app.post("/api/music/job/cancel")
    def cancel_music_job():
        music_jobs.stop()
        return music_jobs.state

    @app.post("/api/edits/{length}/music/use")
    def use_track(length: int, t: TrackIn):
        check_length(length)
        with review.lock:
            check_current(length, t.expected_edit_id)
            try:
                review.use_track(length, t.asset_id)
            except KeyError:
                raise HTTPException(404, "no such track") from None
            review.__dict__.pop("_soundtracks", None)
            return edit_payload(length)

    @app.post("/api/edits/{length}/music/source")
    def music_source(length: int, s: SourceIn):
        check_length(length)
        if s.source not in ("none", "files", "generated"):
            raise HTTPException(422, "source must be none, files or generated")
        with review.lock:
            check_current(length, s.expected_edit_id)
            try:
                review.set_source(length, s.source)
            except ValueError as e:
                raise HTTPException(409, str(e)) from None
            review.__dict__.pop("_soundtracks", None)
            return edit_payload(length)

    @app.post("/api/edits/{length}/music/mix")
    def music_mix(length: int, m: MixIn):
        from ..ride import RIDE_MIX
        from ..soundtrack import Mix

        check_length(length)
        with review.lock:
            check_current(length, m.expected_edit_id)
            if m.reset:
                d = RIDE_MIX if length == RIDE else Mix()
                review.set_mix(length, music=d.music, background=d.background, duck=d.duck,
                               sound=d.sound)  # fmt: skip
            else:
                review.set_mix(length, music=m.music, background=m.background, duck=m.duck,
                               sound=m.sound)  # fmt: skip
            return edit_payload(length)

    @app.post("/api/edits/{length}/music/fit")
    def fit_cuts(length: int, expected_edit_id: int | None = None):
        check_length(length)
        with review.lock:
            check_current(length, expected_edit_id)
            try:
                review.fit_cuts(length)
            except ValueError as e:
                raise HTTPException(409, str(e)) from None
            return edit_payload(length)

    @app.get("/media/music/{asset_id}")
    def music_file(asset_id: int):
        from .. import soundtrack

        a = soundtrack.asset(db, asset_id)
        if a is None or not a.path.is_file():
            raise HTTPException(404)
        return FileResponse(a.path, media_type="audio/flac")

    @app.post("/api/export")
    def export_edit(r: RenderIn):
        from ..export import export

        check_length(r.length)
        with review.lock:
            check_current(r.length, r.expected_edit_id)
            edit = review.edit(r.length)
            import re

            slug = re.sub(r"[^A-Za-z0-9]+", "-", trip.config.name).strip("-").lower() or "trip"
            name = f"{slug}-{r.length}s-r{edit.revision}" + ("-vertical" if r.vertical else "")
            out = Output(1080, 1920) if r.vertical else Output()
            from .. import soundtrack

            try:
                music = soundtrack.resolve(db, edit.id)
            except RuntimeError as e:
                raise HTTPException(409, str(e)) from None
            paths = export(db, edit.items, name, trip.root / "output", out, music)
        return {"files": [p.name for p in paths]}

    @app.get("/api/render")
    def render_state():
        return renders.state

    @app.get("/media/thumb/{recording_id}/{ms}")
    def thumb(recording_id: int, ms: int):
        with review.lock:
            if recording_id not in review.data.recordings:
                raise HTTPException(404)
            _, make = media.thumbnail_job(db, recording_id, ms / 1000)
        if make is None:
            raise HTTPException(404)
        path = make()  # FFmpeg runs outside the lock, so thumbnails load in parallel
        return FileResponse(path, media_type="image/jpeg",
                            headers={"Cache-Control": "max-age=86400"})  # fmt: skip

    @app.get("/media/proxy/{recording_id}/{chapter}")
    def proxy(recording_id: int, chapter: int):
        with review.lock:
            path = media.proxy_path(db, recording_id, chapter)
        if path is None or not path.is_file():
            raise HTTPException(404)
        kind = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "video/mp4"
        return FileResponse(path, media_type=kind)

    @app.get("/media/output/{name}")
    def output(name: str, preview: bool = False, download: bool = False):
        """A rendered video plays in the browser unless download is asked for; timelines and
        sound files always download."""
        folder = (trip.root / "output" / ("previews" if preview else "")).resolve()
        path = (folder / name).resolve()
        kinds = {".mp4": "video/mp4", ".fcpxml": "application/xml", ".otio": "application/json",
                 ".wav": "audio/wav"}  # fmt: skip
        if path.parent != folder or not path.is_file() or path.suffix not in kinds:
            raise HTTPException(404)
        saved = {"filename": path.name} if download or path.suffix != ".mp4" else {}
        return FileResponse(path, media_type=kinds[path.suffix], **saved)

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app

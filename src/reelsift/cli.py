"""Command line interface."""

import sys
import webbrowser
from fractions import Fraction
from pathlib import Path
from typing import Annotated

import typer

from . import __version__, doctor, models
from .project import Trip, TripLockedError

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Turn hours of personal footage into short highlight edits.",
)
anchor_app = typer.Typer(no_args_is_help=True, help="Manual time anchors for clock sync.")
models_app = typer.Typer(no_args_is_help=True, help="Model presets and downloads.")
music_app = typer.Typer(no_args_is_help=True, help="Music under the edits.")
app.add_typer(anchor_app, name="anchor")
app.add_typer(models_app, name="models")
app.add_typer(music_app, name="music")

TripArg = Annotated[Path, typer.Argument(help="Trip folder.", show_default=False)]


def fail(message: str) -> typer.Exit:
    typer.secho(message, fg="red", err=True)
    return typer.Exit(1)


def progress(n: int, total: int, item: str) -> None:
    name = Path(item).name if ("/" in item or "\\" in item) else item
    sys.stderr.write(f"\r  [{n}/{total}] {name[:70]:<70}")
    sys.stderr.flush()
    if n == total:
        sys.stderr.write("\n")


def open_trip(path: Path, write: bool = True) -> Trip:
    try:
        return Trip(path, write=write)
    except FileNotFoundError as e:
        raise fail(str(e)) from None


def run_with(trip: Trip, fn):
    try:
        with trip:
            return fn(trip)
    except TripLockedError as e:
        raise fail(str(e)) from None
    except KeyboardInterrupt:
        sys.stderr.write("\n")
        raise fail("Stopped. Run the same command again to continue where it left off.") from None


def _version(value: bool) -> None:
    if value:
        typer.echo(f"reelsift {__version__}")
        raise typer.Exit()


@app.callback()
def main_options(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show the version.")
    ] = False,
) -> None:
    # Output saved to a file or read by another program is UTF-8, as on a terminal.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


@app.command()
def init(
    trip: TripArg,
    footage: Annotated[
        list[Path],
        typer.Option("--footage", "-f", help="Folder or file with footage; repeat for several."),
    ],
    name: Annotated[str, typer.Option(help="Name shown in the app.")] = "",
    length: Annotated[
        list[int] | None,
        typer.Option(
            "--length",
            "-l",
            help="Edit lengths in seconds, repeated (default 30, 180 and 600); "
            "later: lengths under [trip] in reelsift.toml.",
        ),
    ] = None,
) -> None:
    """Create a trip folder that reads footage from the given folders."""
    from . import config as config_mod

    missing = [p for p in footage if not p.exists()]
    if missing:
        raise fail(f"Not found: {', '.join(map(str, missing))}")
    if length and (any(n <= 0 for n in length) or len(set(length)) != len(length)):
        raise fail("--length: give each edit length once, in seconds, such as --length 60.")
    try:
        created = Trip.create(trip, footage, name)
    except FileExistsError as e:
        raise fail(str(e)) from None

    def lengths(t: Trip) -> None:
        if length:
            t.config.lengths = sorted(length)
            config_mod.write_trip(t.root, t.config)

    run_with(created, lengths)
    typer.echo(f"Created trip {trip}. Next: reelsift prepare {trip}")


def _ingest(t: Trip) -> None:
    from .ingest import ingest

    typer.echo("Reading footage")
    s = ingest(t, progress)
    typer.echo(
        f"  {s.files} files: {s.new} new, {s.relinked} moved, {s.unchanged} unchanged"
        + (f", {len(s.failed)} unreadable" if s.failed else "")
    )
    typer.echo(
        f"  {s.recordings} new recordings; {s.cameras} camera(s); "
        f"{s.camera_proxies} camera proxies paired"
        + (f"; {s.missing} files missing" if s.missing else "")
    )


def _proxies(t: Trip) -> None:
    from .proxies import generate, pending, reconcile

    reconcile(t)  # files deleted or made with older settings are due again
    todo = pending(t)
    if not todo:
        typer.echo("Proxies: all recordings have one")
        return
    typer.echo(f"Making {len(todo)} proxies")
    result = generate(t, progress)
    if result.failed:
        typer.secho(f"  {len(result.failed)} failed; see reelsift status", fg="yellow")


def _sync(t: Trip) -> None:
    from .sync import sync

    s = sync(t)
    parts = ", ".join(f"{n} {p}" for p, n in sorted(s.by_provenance.items()))
    typer.echo(
        f"Clock sync: {s.recordings} recordings ({parts or 'none placed'})"
        + (f"; {s.without_time} without time" if s.without_time else "")
        + (f"; {s.anchors} anchors, largest residual {s.max_residual_s:.2f} s" if s.anchors else "")
    )


def _timeline(t: Trip) -> Path:
    from .timeline import write

    path = write(t)
    typer.echo(f"Timeline: {path}")
    return path


@app.command()
def ingest(trip: TripArg) -> None:
    """Find and identify footage; group files into recordings; pair camera proxies."""
    run_with(open_trip(trip), _ingest)


@app.command()
def proxies(trip: TripArg) -> None:
    """Make proxies for recordings whose camera didn't provide a usable one."""
    run_with(open_trip(trip), _proxies)


@app.command()
def sync(trip: TripArg) -> None:
    """Place every recording on one UTC timeline, using anchors where given."""
    run_with(open_trip(trip), _sync)


@app.command()
def timeline(
    trip: TripArg,
    open_: Annotated[bool, typer.Option("--open", help="Open the page in the browser.")] = False,
) -> None:
    """Write a debug page showing all recordings on the timeline."""
    path = run_with(open_trip(trip, write=False), _timeline)
    if open_:
        webbrowser.open(path.as_uri())


@app.command()
def prepare(
    trip: TripArg,
    open_: Annotated[
        bool, typer.Option("--open", help="Open the timeline in the browser.")
    ] = False,
) -> None:
    """Ingest, make proxies, sync clocks and write the timeline, in one go."""

    def steps(t: Trip) -> Path:
        _ingest(t)
        _proxies(t)
        _sync(t)
        return _timeline(t)

    path = run_with(open_trip(trip), steps)
    if open_:
        webbrowser.open(path.as_uri())


PresetOpt = Annotated[
    str | None, typer.Option("--preset", help="Model preset; defaults to the trip's setting.")
]
LengthOpt = Annotated[
    list[int] | None,
    typer.Option("--length", "-l", help="Edit length in seconds; repeat for several."),
]


def _factories(t: Trip, preset_name: str | None):
    from .ml import ml_available

    if not ml_available():
        raise fail("Analysis needs the models: install with  uv sync --extra ml")
    card = doctor.gpu()
    preset = models.resolve(preset_name or t.config.preset, card.vram_gb if card else None)
    missing = [r for r in models.repos(preset) if not models.is_cached(r)]
    if missing:
        typer.echo(f"Downloading models for the {preset.name} preset: {', '.join(missing)}")

    def embedder():
        from .ml.siglip import SiglipEmbedder

        return SiglipEmbedder(preset.embedding)

    def transcriber():
        from .ml.whisper import WhisperTranscriber

        return WhisperTranscriber(preset.speech)

    def describer():
        from .ml.qwen import QwenVL

        return QwenVL(preset.vision, batch=preset.vlm_batch)

    embedder.model_name = preset.embedding
    transcriber.model_name = preset.speech

    return preset, embedder, transcriber, (preset.vision, describer) if preset.vision else None


def _analyze(t: Trip, preset_name: str | None) -> None:
    from . import candidates, gpu
    from .analysis import analyze as run_analysis
    from .jobs import is_out_of_memory

    preset, embedder, transcriber, describer = _factories(t, preset_name)
    typer.echo(f"Analyzing with the {preset.name} preset")
    try:
        with gpu.hold(f"analysis ({t.root.name})",
                      waiting=lambda h: typer.echo(f"Waiting for the GPU: {h or 'busy'}")):  # fmt: skip
            s = run_analysis(t, embedder, transcriber, progress, say=typer.echo,
                             describer=describer)  # fmt: skip
    except Exception as e:
        if not is_out_of_memory(e):
            raise
        smaller = {"large": "medium", "medium": "small", "small": "cpu"}.get(preset.name)
        hint = f"  reelsift analyze {t.root} --preset {smaller}" if smaller else ""
        raise fail(
            f"Ran out of memory with the {preset.name} preset. Close other programs that use the "
            "GPU, or use a smaller preset; finished recordings are kept."
            + (f"\n{hint}" if hint else "")
        ) from None
    made = candidates.generate(t)
    typer.echo(
        f"  analyzed: {s.speech} for speech, {s.visual} for pictures, {s.described} described;"
        f" {made} candidates"
        + (f"; {s.failed} failed" if s.failed else "")
        + (f"; {s.skipped_without_proxy} skipped (no proxy)" if s.skipped_without_proxy else "")
    )
    from .drafts import missing

    # Edits without a description of their own get a draft; one you wrote is never replaced.
    _draft(t, missing(t), describer, embedder)


def _draft(t: Trip, lengths: list[int], describer, embedder) -> None:
    """Draft the descriptions of these edit lengths from the footage, and save them."""
    from . import drafts
    from . import prefs as prefs_mod
    from .tripdata import load

    if not lengths:
        return
    current = {n: prefs_mod.latest(t.db, n) for n in lengths}
    styles = {
        n: (p.style if p and own and p.style else prefs_mod.default_style(n, t.config.lengths))
        for n, (p, own) in current.items()
    }
    keep = {n: p for n, (p, own) in current.items() if p is not None and own}
    from . import gpu

    complete, model = None, None
    with gpu.hold(f"drafting descriptions ({t.root.name})",
                  waiting=lambda h: typer.echo(f"Waiting for the GPU: {h or 'busy'}")):  # fmt: skip
        if describer is not None:
            name, factory = describer
            typer.echo(f"Drafting descriptions for the edits with {name}")
            model = factory()
            complete = model.complete
        else:
            typer.echo("Drafting descriptions for the edits from templates (no vision model)")
        drafted = drafts.draft(t, load(t), styles, complete, keep)
        from .ml.runtime import free_gpu

        if model is not None:
            complete = model = None  # release the language model before anything else loads
            free_gpu()
        # Searches are compared with the stored analysis, so they are embedded with the model
        # that made it, whichever preset drafted the text.
        from .scoring import prompt_bank_model

        encoder, bank = None, prompt_bank_model(t)
        if embedder is not None and bank and any(p.searches for p in drafted.values()):
            from .ml.siglip import SiglipEmbedder

            encoder = SiglipEmbedder(bank).encode_text
        drafts.save_drafts(t, drafted, encoder)
        encoder = None
        free_gpu()
    for n in sorted(drafted):
        label = prefs_mod.STYLES[drafted[n].style].label
        typer.echo(f"  {n} s, {label.lower()}: {drafted[n].text}")


def _select(t: Trip, lengths: list[int] | None, mode: str) -> None:
    """With the default mode, choose like the review app does, keeping each edit's decisions
    and preferences. Other modes ignore decisions and preferences; they are for comparing
    scoring."""
    from .edits import build, save

    if mode == "combined":
        from .review import Review

        unknown = sorted(set(lengths or []) - set(t.config.lengths))
        if unknown:
            raise fail(
                f"No {', '.join(f'{n} s' for n in unknown)} edit in this trip; its lengths are "
                f"{', '.join(f'{n} s' for n in t.config.lengths)} (change them under [trip] "
                "lengths in reelsift.toml, then run select)."
            )
        review = Review(t)
        if not len(review.data.candidates):
            raise fail("No candidates yet; run reelsift analyze first.")
        chosen = sorted(lengths) if lengths else review.lengths
        with review.operation():
            for length in chosen:
                review.select(length)
        for length in chosen:
            edit = review.edit(length)
            duration = sum(i.duration for i in edit.items)
            typer.echo(f"  {length} s edit: {len(edit.items)} clips, {duration:.1f} s"
                       f" (revision {edit.revision})")  # fmt: skip
            if duration < 0.8 * length:
                typer.secho(f"    Only {duration:.0f} s of the footage suits this edit, so it is "
                            f"shorter than {length} s.", fg="yellow")  # fmt: skip
        return
    for sel in build(t, lengths or t.config.lengths, mode):
        stored_one = save(t, sel, {"mode": mode})
        note = f", {sel.shortfall:.0f} s short" if sel.shortfall > 0.5 else ""
        typer.echo(
            f"  {sel.length} s edit: {len(sel.items)} clips, {sel.duration:.1f} s"
            f"{note} (revision {stored_one.revision})"
        )


def _render(t: Trip, lengths: list[int] | None, vertical: bool) -> list[Path]:
    from . import soundtrack
    from .edits import latest
    from .render import Output, render

    out_dir = t.root / "output"
    made = []
    for length in lengths or t.config.lengths:
        edit = latest(t, length, approved_only=False)
        if edit is None:
            typer.secho(f"  No {length} s edit yet; run reelsift select first.", fg="yellow")
            continue
        if not edit.items:
            typer.secho(
                f"  The {length} s edit is empty: no footage was long enough or allowed to use."
                " See reelsift status.",
                fg="yellow",
            )
            continue
        output = Output(1080, 1920) if vertical else Output()
        suffix = "-vertical" if vertical else ""
        st = soundtrack.get(t, edit.id)
        try:
            music = soundtrack.resolve(t, edit.id)
        except RuntimeError as e:
            raise fail(str(e)) from None
        name = f"{_slug(t.config.name)}-{length}s-r{edit.revision}{soundtrack.tag(st)}{suffix}.mp4"
        typer.echo(f"Rendering the {length} s edit ({len(edit.items)} clips)")
        made.append(render(t, edit.items, out_dir / name, output, progress, music=music))
        typer.echo(f"  {made[-1]}")
    return made


def _slug(name: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower() or "trip"


@app.command()
def analyze(trip: TripArg, preset: PresetOpt = None) -> None:
    """Analyze pictures, motion, sound and speech of every recording (slow, uses the GPU)."""
    run_with(open_trip(trip), lambda t: _analyze(t, preset))


@app.command()
def draft(trip: TripArg, length: LengthOpt = None, preset: PresetOpt = None) -> None:
    """Write the edits' descriptions again from the footage, replacing the current ones (a
    previous description stays in the history). Run select afterwards to use them."""

    def run(t: Trip) -> None:
        from .ml import ml_available

        lengths = sorted(length) if length else sorted(t.config.lengths)
        unknown = sorted(set(lengths) - set(t.config.lengths))
        if unknown:
            raise fail(f"No {', '.join(f'{n} s' for n in unknown)} edit in this trip.")
        describer = embedder = None
        if ml_available():
            _, embedder, _, describer = _factories(t, preset)
        _draft(t, lengths, describer, embedder)

    run_with(open_trip(trip), run)


@app.command("select")
def select_cmd(
    trip: TripArg,
    length: LengthOpt = None,
    mode: Annotated[str, typer.Option(help="Scoring: combined, motion or embedding.")] = "combined",
) -> None:
    """Choose clips for each edit length (fast; no models)."""
    run_with(open_trip(trip), lambda t: _select(t, length, mode))


@app.command("render")
def render_cmd(
    trip: TripArg,
    length: LengthOpt = None,
    ride: Annotated[
        bool, typer.Option("--ride", help="The ride (alone, unless lengths are given too).")
    ] = False,
    vertical: Annotated[bool, typer.Option(help="Render 9:16 instead of 16:9.")] = False,
) -> None:
    """Render the latest edit for each length from the original files."""

    def run(t: Trip) -> None:
        if not ride or length:
            _render(t, length, vertical)
        if ride:
            _render_ride(t, _latest_ride(t), vertical)

    run_with(open_trip(trip), run)


@app.command()
def make(
    trip: TripArg,
    preset: PresetOpt = None,
    length: LengthOpt = None,
    vertical: Annotated[bool, typer.Option(help="Also render every video 9:16.")] = False,
    ride: Annotated[
        bool, typer.Option("--ride", help="Also make and render the ride (see reelsift ride).")
    ] = False,
    only_ride: Annotated[
        bool,
        typer.Option(
            "--only-ride",
            help="Only the ride: no analysis and no highlight edits (a dashcam, a commute).",
        ),
    ] = False,
    ride_length: Annotated[
        str | None,
        typer.Option(help="How long the ride should last, such as 10:00 or 600 (implies --ride)."),
    ] = None,
    ride_speed: Annotated[
        float | None, typer.Option(help="A fixed speed for the ride instead (implies --ride).")
    ] = None,
    music: Annotated[
        str | None,
        typer.Option(
            "--music",
            help="Generate music in this genre (synthwave, techno, lo-fi...) for every video; "
            "needs reelsift music install.",
        ),
    ] = None,
    mood: Annotated[
        str | None, typer.Option(help="With --music: such as dark, sunny, epic.")
    ] = None,
) -> None:
    """Do everything, with no review: prepare, analyze, select and render every edit length (and
    the ride, music and 9:16 versions when asked). The videos are written to the trip's output
    folder and listed at the end."""
    from .ride import RIDE

    with_ride = ride or only_ride or ride_length is not None or ride_speed is not None
    if music:
        from . import musicgen

        if musicgen.installed() is None:
            raise fail("--music needs the music model; install it first:  reelsift music install")

    def steps(t: Trip) -> None:
        _ingest(t)
        _proxies(t)
        _sync(t)
        _timeline(t)
        lengths = [] if only_ride else sorted(length or t.config.lengths)
        if lengths:
            _analyze(t, preset)
            _select(t, lengths, "combined")
        if with_ride:
            _make_ride(t, ride_length, ride_speed)
        if music:
            from .config import MusicSettings

            given = MusicSettings(genre=music, mood=mood)
            keys = [*lengths, *([RIDE] if with_ride else [])]
            made_before = [n for n in keys if _has_generated(t, n, given)]
            for n in made_before:
                typer.echo(f"The {_label(n)} already plays music made as {music}"
                           f"{f', {mood}' if mood else ''}; kept (new music: reelsift music "
                           "generate).")  # fmt: skip
            todo = [n for n in keys if n not in made_before]
            if todo:
                _generate_music(t, todo, given, per_edit=True)
        made = []
        for v in (False, True) if vertical else (False,):
            if lengths:
                made += _render(t, lengths, v)
            if with_ride:
                made.append(_render_ride(t, _latest_ride(t), v))
        typer.echo("Done. The videos:" if made else "Done; nothing to render.")
        for path in made:
            typer.echo(f"  {path}")

    run_with(open_trip(trip), steps)


def _review_models(t: Trip):
    """Lazy loaders for the review app: SigLIP's text side for searches, Qwen for preferences.

    Each loads on first use and stays loaded. Without the ml extra both are None; searches are
    then unavailable and preferences use keywords."""
    import threading

    from .ml import ml_available

    if not ml_available():
        return None, None
    from .scoring import prompt_bank_model

    embed_model = prompt_bank_model(t)
    if embed_model is None:
        return None, None
    card = doctor.gpu()
    preset = models.resolve(t.config.preset, card.vram_gb if card else None)
    from . import gpu

    loaded: dict = {}
    lock = threading.Lock()
    # Always the GPU first, then this lock: unload() is called while holding the GPU.

    def text_encoder(texts: list[str]):
        with gpu.hold("the review app's search model"), lock:
            if "embed" not in loaded:
                from .ml.siglip import SiglipEmbedder

                typer.echo(f"Loading {embed_model} for searches")
                loaded["embed"] = SiglipEmbedder(embed_model)
            return loaded["embed"].encode_text(texts)

    def language_model(prompt: str) -> str:
        with gpu.hold("the review app's language model"), lock:
            if "llm" not in loaded:
                from .ml.qwen import QwenVL

                typer.echo(f"Loading {preset.vision} to read preferences")
                loaded["llm"] = QwenVL(preset.vision)
            return loaded["llm"].complete(prompt)

    def unload() -> None:
        """Free the GPU for something else (music generation); models load again when needed."""
        from .ml.runtime import free_gpu

        with lock:
            loaded.clear()
        free_gpu()

    text_encoder.unload = unload  # type: ignore[attr-defined]
    return text_encoder, (language_model if preset.vision and card else None)


@app.command()
def clean(
    trip: TripArg,
    proxies: Annotated[bool, typer.Option(help="Also delete proxies Reelsift made.")] = False,
    analysis: Annotated[
        bool, typer.Option(help="Also delete embeddings (needs analyze again).")
    ] = False,
) -> None:
    """Free disk space by deleting caches that can be rebuilt. Footage, decisions, edits and
    rendered output are never touched."""
    from .project import CACHES, folder_size
    from .project import clean as clean_caches

    def run_clean(t: Trip) -> None:
        typer.echo("Cache use before cleaning:")
        for name, what in CACHES.items():
            typer.echo(f"  {folder_size(t.root / 'cache' / name) / 2**30:7.2f} GB  {name}: {what}")
        freed = clean_caches(t, proxies=proxies, analysis=analysis)
        total = sum(freed.values()) / 2**30
        typer.echo(f"Freed {total:.2f} GB ({', '.join(freed)}).")
        if analysis:
            typer.echo("Run  reelsift analyze  before selecting again.")

    run_with(open_trip(trip), run_clean)


@app.command("review")
def review_cmd(
    trip: TripArg,
    port: Annotated[int, typer.Option(help="Port on 127.0.0.1.")] = 8765,
    browser: Annotated[bool, typer.Option(help="Open the app in the browser.")] = True,
) -> None:
    """Open the review app: check each edit, keep, drop or add clips, approve, render."""

    def serve(t: Trip) -> None:
        import threading

        import uvicorn

        from .review import Review
        from .web.app import create_app, quiet_shutdown

        text_encoder, language_model = _review_models(t)  # each holds the GPU while it works
        review = Review(t.worker(shared=True), text_encoder, language_model)
        from .edits import latest
        from .ride import RIDE

        if not len(review.data.candidates) and latest(t, RIDE) is None:
            raise fail("Nothing to review yet; run reelsift analyze first (or make a ride: "
                       "reelsift ride).")  # fmt: skip
        review.music()  # analyze music beats now, not while a request holds the review lock
        url = f"http://127.0.0.1:{port}/"
        typer.echo(f"Review app: {url}   (Ctrl+C to stop)")
        if browser:
            threading.Timer(1.0, webbrowser.open, [url]).start()
        # Ctrl+C waits at most a second for open connections (a paused video, or a browser
        # closed without saying so, would otherwise keep the app from stopping), then closes them.
        import logging

        logging.getLogger("uvicorn.error").addFilter(quiet_shutdown)
        try:
            uvicorn.run(create_app(t, review, port), host="127.0.0.1", port=port,
                        log_level="warning", timeout_graceful_shutdown=1)  # fmt: skip
        except KeyboardInterrupt:
            pass
        # A change still being saved (its thread outlives the server) finishes first; then the
        # app's connection to the trip is closed, so nothing writes once the trip is released.
        if not review.lock.acquire(timeout=0.2):
            typer.echo("Saving the last change…")
            while not review.lock.acquire(timeout=0.2):
                pass
        try:
            review.trip.close()
        finally:
            review.lock.release()
        typer.echo("Review app stopped.")

    run_with(open_trip(trip), serve)


@app.command("export")
def export_cmd(
    trip: TripArg,
    length: LengthOpt = None,
    vertical: Annotated[bool, typer.Option(help="A 9:16 timeline instead of 16:9.")] = False,
) -> None:
    """Write the latest edit as FCPXML and OpenTimelineIO, for DaVinci Resolve and other editors
    (with its music as a WAV on its own track)."""
    from . import soundtrack
    from .edits import latest
    from .export import export
    from .render import Output

    def run_export(t: Trip) -> None:
        for n in length or t.config.lengths:
            edit = latest(t, n)
            if edit is None:
                typer.secho(f"  No {n} s edit yet; run reelsift select first.", fg="yellow")
                continue
            name = f"{_slug(t.config.name)}-{n}s-r{edit.revision}" + (
                "-vertical" if vertical else ""
            )
            out = Output(1080, 1920) if vertical else Output()
            try:
                music = soundtrack.resolve(t, edit.id)
            except RuntimeError as e:
                raise fail(str(e)) from None
            for path in export(t, edit.items, name, t.root / "output", out, music):
                typer.echo(f"  {path}")
        typer.echo("Import the .fcpxml (File > Import > Timeline in DaVinci Resolve) or the .otio.")

    run_with(open_trip(trip, write=False), run_export)


RideOpt = Annotated[bool, typer.Option("--ride", help="The trip's ride (see reelsift ride).")]


def _keys(t: Trip, lengths: list[int] | None, ride: bool) -> list[int]:
    """Edit keys: the lengths asked for (all without), and the ride with --ride (alone if no
    length was asked for)."""
    from .ride import RIDE

    if ride:
        return [*(lengths or []), RIDE]
    return list(lengths or t.config.lengths)


def _label(n: int) -> str:
    from .ride import RIDE

    return "ride" if n == RIDE else f"{n} s edit"


def _latest_soundtracks(t: Trip, lengths: list[int] | None = None):
    """The soundtrack of each edit's current revision (lengths may include the ride's key)."""
    from . import soundtrack
    from .edits import latest

    for n in lengths or t.config.lengths:
        edit = latest(t, n)
        if edit is not None:
            yield n, edit, soundtrack.get(t, edit.id)


@music_app.command("set")
def music_set(
    trip: TripArg,
    files: Annotated[list[Path], typer.Argument(help="Music files, played in this order.")],
) -> None:
    """Play these music files under every edit; cuts will land on its beats after the next
    selection."""
    from . import config as config_mod
    from . import music, soundtrack

    def apply(t: Trip) -> None:
        missing = [f for f in files if not f.is_file()]
        if missing:
            raise fail(f"Not found: {', '.join(map(str, missing))}")
        try:
            tracks = music.load(t.root, [str(f.resolve()) for f in files])
        except RuntimeError as e:
            raise fail(f"Can't read the music: {e}") from None
        for tr in tracks:
            typer.echo(f"  {tr.path.name}: {tr.duration:.0f} s, about {tr.bpm:.0f} BPM, "
                       f"{len(tr.beats)} beats")  # fmt: skip
        t.config.music = [str(f.resolve()) for f in files]
        t.config.music_settings.source = "files"
        for s in t.config.music_lengths.values():
            s.source = None  # every edit, including those not selected yet
        config_mod.write_trip(t.root, t.config)
        from . import musicgen
        from .ride import RIDE

        for n, _, st in _latest_soundtracks(t, [*t.config.lengths, RIDE]):
            st.source, st.asset_id, st.files, st.fit = "files", None, list(t.config.music), {}
            soundtrack.put(t, st)
            musicgen.chosen(t, n)
        typer.echo("Music set for every edit. Run  reelsift select  (or Select again in the app) "
                   "so cuts land on the beat.")  # fmt: skip

    run_with(open_trip(trip), apply)


@music_app.command("clear")
def music_clear(trip: TripArg, length: LengthOpt = None, ride: RideOpt = False) -> None:
    """No music under the edits (or only these lengths); they keep only their own sound.
    Generated tracks are kept and can be used again."""
    from . import config as config_mod
    from . import soundtrack
    from .config import MusicSettings

    def apply(t: Trip) -> None:
        from .ride import RIDE

        keys = _keys(t, length, ride)
        if not length and not ride:
            keys = [*keys, RIDE]  # every edit, the ride too
            t.config.music = []
            t.config.music_settings.source = "none"
            for s in t.config.music_lengths.values():
                s.source = None
        else:
            for n in keys:
                t.config.music_lengths.setdefault(n, MusicSettings()).source = "none"
        config_mod.write_trip(t.root, t.config)
        from . import musicgen

        for n, _, st in _latest_soundtracks(t, keys):
            st.source, st.asset_id, st.files, st.fit = "none", None, [], {}
            soundtrack.put(t, st)
            musicgen.chosen(t, n)
            typer.echo(f"  {_label(n)}: no music")

    run_with(open_trip(trip), apply)


@music_app.command("install")
def music_install(
    profile: Annotated[
        str | None, typer.Option(help="xl or standard; chosen from the GPU.")
    ] = None,
) -> None:
    """Install the music model (ACE-Step 1.5, about 40 GB with its environment) for
    generating music."""
    from . import musicgen

    card = doctor.gpu()
    chosen = musicgen.PROFILES.get(profile) if profile else musicgen.recommend(
        card.vram_gb if card else None)  # fmt: skip
    if chosen is None:
        raise fail("Generating music needs an NVIDIA GPU with at least 14 GB of memory."
                   if not profile else f"Unknown profile {profile!r}; choose xl or standard.")  # fmt: skip
    if not chosen.measured:
        typer.secho(f"The {chosen.name} profile hasn't been measured yet.", fg="yellow")
    try:
        musicgen.install(chosen, typer.echo)
    except musicgen.MusicUnavailable as e:
        raise fail(str(e)) from None


@music_app.command("generate")
def music_generate(
    trip: TripArg,
    length: LengthOpt = None,
    all_lengths: Annotated[bool, typer.Option("--all", help="Every edit length.")] = False,
    ride: RideOpt = False,
    genre: Annotated[str | None, typer.Option(help="Such as techno, synthwave, lo-fi.")] = None,
    mood: Annotated[str | None, typer.Option(help="Such as dark, sunny, epic.")] = None,
    bpm: Annotated[
        int | None, typer.Option(help="Tempo; chosen from the genre if not set.")
    ] = None,
    candidates: Annotated[int | None, typer.Option(help="Tracks to make (default 2).")] = None,
    seed: Annotated[int | None, typer.Option(help="For repeatable results.")] = None,
) -> None:
    """Generate music for edits: several tracks each, the best-suited one put under the edit.
    Genre, mood and tempo are saved in reelsift.toml for next time."""
    from .config import MusicSettings
    from .ride import RIDE

    if not length and not all_lengths and not ride:
        raise fail("Choose the edits: --length 30 (repeatable), --ride or --all.")

    def run(t: Trip) -> None:
        chosen = sorted(length) if length else sorted(t.config.lengths) if all_lengths else []
        lengths = [*chosen, RIDE] if ride else chosen
        given = MusicSettings(genre=genre, mood=mood, bpm=bpm)
        _generate_music(t, lengths, given, bool(length or ride), candidates, seed)

    run_with(open_trip(trip), run)


def _generate_music(
    t: Trip,
    keys: list[int],
    given,
    per_edit: bool,
    candidates: int | None = None,
    seed: int | None = None,
) -> None:
    """Generate music for these edits and put the best-suited track under each. The genre, mood
    and tempo given are saved for these edits (per_edit) or as the trip's default."""
    from . import config as config_mod
    from . import musicgen, soundtrack
    from .config import MusicSettings

    if given.to_toml():
        for n in keys if per_edit else []:
            t.config.music_lengths[n] = given.over(t.config.music_lengths.get(n, MusicSettings()))
        if not per_edit:
            t.config.music_settings = given.over(t.config.music_settings)
        t.config.validate()
        config_mod.write_trip(t.root, t.config)
    try:
        reports = musicgen.generate_many(t, keys, candidates=candidates, seed=seed,
                                         say=lambda m: typer.echo(f"  {m}"))  # fmt: skip
    except (musicgen.MusicUnavailable, ValueError, RuntimeError) as e:
        raise fail(str(e)) from None
    for n, report in reports.items():
        if report.assigned is not None:
            a = soundtrack.asset(t, report.assigned)
            typer.echo(f"The {_label(n)} now plays track {a.id} ({a.analysis['bpm']:.0f} BPM). "
                       "Others: reelsift music list")  # fmt: skip
        elif report.note:
            typer.secho(f"The {_label(n)}: {report.note}", fg="yellow")


@music_app.command("list")
def music_list(trip: TripArg, length: LengthOpt = None, ride: RideOpt = False) -> None:
    """Generated tracks for each edit; * marks the one playing."""
    from . import soundtrack

    def show(t: Trip) -> None:
        keys = _keys(t, length, ride)
        playing = {n: st for n, _, st in _latest_soundtracks(t, keys)}
        for n in keys:
            st = playing.get(n)
            what = {"none": "no music", "files": "the trip's music files"}.get(
                st.source if st else "none", "a generated track")  # fmt: skip
            typer.echo(f"{_label(n)}: {what}")
            for a in soundtrack.assets(t, n):
                mark = "*" if st and st.asset_id == a.id else " "
                r = a.request
                typer.echo(f" {mark} {a.id:4d}  {a.analysis['bpm']:5.1f} BPM  "
                           f"{a.duration:5.0f} s  fit {r.get('hint', 0):+.2f}  {r['caption'][:60]}")  # fmt: skip

    run_with(open_trip(trip, write=False), show)


@music_app.command("use")
def music_use(
    trip: TripArg,
    length: Annotated[str, typer.Argument(help="Edit length in seconds, or ride.")],
    track: int,
) -> None:
    """Play a generated track (see music list) under an edit."""
    from . import musicgen
    from .edits import latest
    from .ride import RIDE

    def apply(t: Trip) -> None:
        nonlocal length
        key = RIDE if length.strip().lower() == "ride" else int(length)
        edit = latest(t, key)
        if edit is None:
            raise fail(f"No {_label(key)} yet.")
        try:
            musicgen.use(t, edit.id, track)
        except KeyError:
            raise fail(f"No generated track {track}; see reelsift music list.") from None
        musicgen.chosen(t, key)
        typer.echo(f"The {_label(key)} now plays track {track}.")

    run_with(open_trip(trip), apply)


@music_app.command("fit")
def music_fit(trip: TripArg, length: LengthOpt = None) -> None:
    """Move each edit's cuts onto its music's beats (as a new revision; never into a word)."""
    from .review import Review

    def apply(t: Trip) -> None:
        review = Review(t)
        for n in sorted(length or t.config.lengths):
            try:
                edit = review.fit_cuts(n)
            except ValueError as e:
                typer.secho(f"  {n} s edit: {e}", fg="yellow")
                continue
            typer.echo(f"  {n} s edit: cuts fitted (revision {edit.revision})")

    run_with(open_trip(trip), apply)


@music_app.command("mix")
def music_mix(
    trip: TripArg,
    length: LengthOpt = None,
    ride: RideOpt = False,
    music: Annotated[float | None, typer.Option(help="Music level, dB (default -6).")] = None,
    background: Annotated[
        float | None,
        typer.Option(help="Clips' sound outside speech while music plays, dB (default -9)."),
    ] = None,
    duck: Annotated[
        float | None,
        typer.Option(
            help="How far music drops while people speak, dB (default -12; 0 = not at all)."
        ),
    ] = None,
    sound: Annotated[
        float | None,
        typer.Option(help="The clips' own sound as a whole, dB (default 0; -60 = off)."),
    ] = None,
    reset: Annotated[bool, typer.Option(help="Back to the default levels.")] = False,
) -> None:
    """How loud the music is against the clips' own sound. Saved for the edits' current
    revisions and as the trip's default; render again to hear it."""
    from dataclasses import replace

    from . import config as config_mod
    from . import soundtrack
    from .config import MusicSettings

    def apply(t: Trip) -> None:
        given = MusicSettings(music=music, background=background, duck=duck, sound=sound)
        given.validate("mix")
        keys = _keys(t, length, ride)
        for n, _, st in _latest_soundtracks(t, keys):
            from .ride import RIDE, RIDE_MIX

            default = RIDE_MIX if n == RIDE else soundtrack.Mix()
            st.mix = default if reset else replace(
                st.mix, **{k: v for k, v in given.to_toml().items()})  # fmt: skip
            soundtrack.put(t, st)
            m = st.mix
            typer.echo(f"  {_label(n)}: music {_db(m.music)}, clip sound {_db(m.sound)}, under "
                       f"music {_db(m.background)}, drop under speech {m.duck:+.1f} dB")  # fmt: skip
        target = [t.config.music_lengths.setdefault(n, MusicSettings()) for n in keys] if (
            length or ride) else [t.config.music_settings]  # fmt: skip
        for s in target:
            for k in ("music", "background", "duck", "sound"):
                setattr(s, k, None if reset else (getattr(given, k) if getattr(given, k)
                                                  is not None else getattr(s, k)))  # fmt: skip
        config_mod.write_trip(t.root, t.config)

    run_with(open_trip(trip), apply)


def _db(value: float) -> str:
    return "off" if value <= -60 else f"{value:+.1f} dB"


@music_app.command("prune")
def music_prune(trip: TripArg) -> None:
    """Delete generated tracks that no edit revision uses."""

    def apply(t: Trip) -> None:
        from . import musicgen

        count, freed = musicgen.prune(t)
        typer.echo(f"Deleted {count} unused tracks ({freed / 2**20:.0f} MB).")

    run_with(open_trip(trip), apply)


@app.command("ride")
def ride_cmd(
    trip: TripArg,
    length: Annotated[
        str | None,
        typer.Option(
            "--length",
            "-l",
            help="How long the ride should last, as seconds or minutes:seconds "
            "(the speed is fitted to it).",
        ),
    ] = None,
    speed: Annotated[
        float | None,
        typer.Option(help="A fixed speed instead, such as 8 (the ride lasts what it lasts)."),
    ] = None,
    camera: Annotated[
        str | None, typer.Option(help="Camera label; default: the one with the most footage.")
    ] = None,
    cut_unknown: Annotated[
        bool, typer.Option(help="Also cut footage whose movement isn't clear (default: keep it).")
    ] = False,
    render_it: Annotated[bool, typer.Option("--render", help="Render it afterwards.")] = False,
    vertical: Annotated[bool, typer.Option(help="With --render: 9:16.")] = False,
) -> None:
    """The whole trip from one camera in time order, stops (red lights, waiting) cut out and the
    rest sped up: to a target length, or at a fixed speed."""

    def run(t: Trip) -> None:
        stored = _make_ride(t, length, speed, camera, cut_unknown)
        if render_it:
            _render_ride(t, stored, vertical)

    run_with(open_trip(trip), run)


def _make_ride(
    t: Trip,
    length: str | None,
    speed: float | None,
    camera: str | None = None,
    cut_unknown: bool = False,
):
    """Build the ride and save it as a new revision; what isn't given is the trip's setting."""
    from . import config as config_mod
    from . import ride as ride_mod

    settings = ride_mod.settings_of(t)
    if length is not None:
        try:
            parts = [float(p) for p in length.split(":")]
        except ValueError:
            raise fail(
                f"Length {length!r}: give seconds (600) or minutes:seconds (10:00)."
            ) from None
        seconds = 0.0
        for p in parts:
            seconds = seconds * 60 + p
        settings.target, settings.speed = seconds, "fit"
    if speed is not None:
        settings.speed = f"{speed:g}"
    if camera is not None:
        settings.camera = camera
    if cut_unknown:
        settings.keep_unknown = False
    if settings.speed == "fit" and settings.target is None:
        settings.speed = "8"
    try:
        items, report = ride_mod.build(t, settings, say=typer.echo)
    except ride_mod.RideError as e:
        raise fail(str(e)) from None
    t.config.ride = {k: v for k, v in settings.to_json().items() if k != "overrides"}
    config_mod.write_trip(t.root, t.config)
    _show_ride(report)
    latest = _latest_ride(t, required=False)
    if latest is not None and latest.items == items and (
            latest.settings.get("ride") == settings.to_json()):  # fmt: skip
        # Built again from the same settings: the same ride, music and all (no new revision).
        typer.echo(f"Ride unchanged (revision {latest.revision}).")
        return latest
    stored = ride_mod.save(t, items, settings, report, "ride")
    typer.echo(f"Ride saved (revision {stored.revision}).")
    return stored


def _has_generated(t: Trip, n: int, given) -> bool:
    """Whether this edit already plays music generated with this genre and mood (asked for the
    same way before)."""
    from . import soundtrack
    from .edits import latest

    edit = latest(t, n)
    st = soundtrack.get(t, edit.id) if edit is not None else None
    if st is None or st.source != "generated" or st.asset_id is None:
        return False
    asset = soundtrack.asset(t, st.asset_id)
    if asset is None or not asset.path.is_file():
        return False
    asked = asset.request  # what the playing track was made for, not what is set now
    return asked.get("genre") == given.genre and (
        given.mood is None or asked.get("mood") == given.mood
    )


def _latest_ride(t: Trip, required: bool = True):
    from .edits import latest
    from .ride import RIDE

    stored = latest(t, RIDE)
    if stored is None and required:
        raise fail("No ride yet; make one first:  reelsift ride TRIP")
    return stored


def _show_ride(report) -> None:
    def mins(s: float) -> str:
        return f"{s / 60:.1f} min" if s >= 90 else f"{s:.0f} s"

    sources = ", ".join(f"{n} from {k}" for k, n in report.sources.items())
    typer.echo(f"  {report.camera}: {report.recordings} recordings ({sources}), "
               f"{mins(report.footage)} of footage")  # fmt: skip
    typer.echo(f"  travelling {mins(report.moving)}, stops cut {mins(report.stopped)}, "
               f"uncertain {mins(report.unknown)}"
               + (f", kept at normal speed {mins(report.kept_stops)}" if report.kept_stops else "")
               + (f", dropped by hand {mins(report.dropped)}" if report.dropped else ""))  # fmt: skip
    typer.echo(f"  speed {report.speed:.1f}x: the ride lasts {mins(report.duration)}")
    typer.echo("  other speeds: " + ", ".join(f"{s:g}x {mins(d)}" for s, d in report.options()))
    for note in report.notes:
        typer.secho(f"  {note}", fg="yellow")


def _render_ride(t: Trip, stored, vertical: bool) -> Path:
    from . import soundtrack
    from .render import Output, render
    from .ride import RIDE

    st = soundtrack.get(t, stored.id)
    try:
        music = soundtrack.resolve(t, stored.id)
    except RuntimeError as e:
        raise fail(str(e)) from None
    out = Output(1080, 1920) if vertical else Output()
    name = (f"{_slug(t.config.name)}-ride-r{stored.revision}{soundtrack.tag(st)}"
            f"{'-vertical' if vertical else ''}.mp4")  # fmt: skip
    typer.echo(f"Rendering the ride ({len(stored.items)} stretches)")
    path = render(t, stored.items, t.root / "output" / name, out, progress, music=music)
    typer.echo(f"  {path}")
    assert stored.length == RIDE
    return path


@app.command("eval")
def eval_cmd(trip: TripArg, length: LengthOpt = None) -> None:
    """Compare scoring modes against the moments marked in the trip's labels.toml."""
    from .edits import build
    from .evaluate import evaluate, load_labels
    from .scoring import MODES
    from .tripdata import load

    def show(t: Trip) -> None:
        moments = load_labels(t)
        if not moments:
            raise fail(
                "No labels.toml in the trip folder (see reelsift/evaluate.py for the format)."
            )
        data = load(t)
        typer.echo(
            f"{sum(m.want for m in moments)} wanted and {sum(not m.want for m in moments)}"
            " unwanted moments"
        )
        for mode in MODES:
            for sel in build(t, length or t.config.lengths, mode, data):
                r = evaluate(data, sel, moments)
                typer.echo(
                    f"  {mode:<9} {r.length:>5} s: found {r.found}/{r.wanted} ({r.recall:.0%}),"
                    f" unwanted {r.unwanted_included}, in candidates {r.candidate_recall:.0%}"
                )

    run_with(open_trip(trip, write=False), show)


@app.command()
def status(trip: TripArg) -> None:
    """Summarize the trip and list open issues."""

    def show(t: Trip) -> None:
        db = t.db
        count = lambda sql: db.execute(sql).fetchone()[0]  # noqa: E731
        typer.echo(f"{t.config.name}  ({t.root})")
        typer.echo(f"  footage: {', '.join(t.config.footage)}")
        typer.echo(f"  lengths: {', '.join(f'{n} s' for n in t.config.lengths)}")
        typer.echo(
            f"  recordings: {count('SELECT COUNT(*) FROM recordings')}, cameras: "
            f"{count('SELECT COUNT(*) FROM cameras')}, proxies: "
            f"{count('SELECT COUNT(*) FROM proxies')}, placed on timeline: "
            f"{count('SELECT COUNT(*) FROM clocks')}"
        )
        typer.echo(f"  disk used by the trip folder: {t.disk_usage() / 2**30:.2f} GB")
        jobs = db.execute("SELECT kind, state, message FROM jobs WHERE state != 'done' ORDER BY id")
        for j in jobs:
            typer.echo(f"  unfinished job: {j['kind']} ({j['state']}: {j['message']})")
        issues = db.execute(
            "SELECT severity, stage, message, path FROM issues ORDER BY id"
        ).fetchall()
        if issues:
            typer.echo("Issues:")
            for i in issues:
                where = f"  [{Path(i['path']).name}]" if i["path"] else ""
                typer.echo(f"  {i['severity']:<7} {i['stage']:<6} {i['message']}{where}")

    run_with(open_trip(trip, write=False), show)


@anchor_app.command("add")
def anchor_add(
    trip: TripArg,
    recording: Annotated[str, typer.Argument(help="Recording id, or a file name of it.")],
    at: Annotated[str, typer.Argument(help="Position in it: seconds, m:ss or h:mm:ss.")],
    other: Annotated[
        str | None, typer.Option("--same-as", help="Other recording showing the same instant.")
    ] = None,
    other_at: Annotated[
        str | None, typer.Option("--other-at", help="Position in the other recording.")
    ] = None,
    utc: Annotated[
        str | None,
        typer.Option(help="Real time of the moment, with zone, e.g. 2026-10-11T10:05:00+02:00."),
    ] = None,
    note: Annotated[str, typer.Option(help="What is visible at that moment.")] = "",
) -> None:
    """Pin a moment: either to a real time (--utc) or to the same moment in another recording."""
    from .sync import add_anchor

    def add(t: Trip) -> None:
        try:
            anchor_id = add_anchor(t, recording, at, b=other, time_b=other_at, utc=utc, note=note)
        except ValueError as e:
            raise fail(str(e)) from None
        typer.echo(f"Added anchor {anchor_id}.")
        _sync(t)

    run_with(open_trip(trip), add)


@anchor_app.command("list")
def anchor_list(trip: TripArg) -> None:
    """List anchors and how well they agree."""

    def show(t: Trip) -> None:
        rows = t.db.execute("SELECT * FROM anchors ORDER BY id").fetchall()
        if not rows:
            typer.echo("No anchors.")
        for r in rows:
            target = (
                f"recording {r['recording_b']} at {float(Fraction(r['time_b'])):g} s"
                if r["recording_b"]
                else f"UTC {r['utc']}"
            )
            residual = "" if r["residual_s"] is None else f", residual {r['residual_s']:.2f} s"
            typer.echo(
                f"{r['id']}: recording {r['recording_a']} at {float(Fraction(r['time_a'])):g} s"
                f" = {target}"
                f"{residual}{'  ' + r['note'] if r['note'] else ''}"
            )

    run_with(open_trip(trip, write=False), show)


@anchor_app.command("remove")
def anchor_remove(trip: TripArg, anchor_id: int) -> None:
    """Delete an anchor and re-sync."""

    def remove(t: Trip) -> None:
        if not t.db.execute("DELETE FROM anchors WHERE id = ?", (anchor_id,)).rowcount:
            raise fail(f"No anchor {anchor_id}.")
        _sync(t)

    run_with(open_trip(trip), remove)


@app.command("doctor")
def doctor_cmd(
    trip: Annotated[
        Path | None, typer.Option(help="Check free space where this trip lives.")
    ] = None,
    preset: Annotated[str, typer.Option(help="Model preset to check.")] = "auto",
) -> None:
    """Check this machine: Python, FFmpeg, GPU, models and disk space."""
    marks = {
        doctor.OK: ("ok", "green"),
        doctor.WARN: ("warn", "yellow"),
        doctor.FAIL: ("FAIL", "red"),
    }
    failed = False
    for c in doctor.checks(trip, preset):
        mark, color = marks[c.status]
        typer.secho(f"{mark:>4}  {c.name}: {c.detail}", fg=color)
        if c.fix:
            typer.echo(f"      {c.fix}")
        failed |= c.status == doctor.FAIL
    if failed:
        raise typer.Exit(1)


@models_app.command("list")
def models_list() -> None:
    """Show the presets and which models are downloaded."""
    card = doctor.gpu()
    recommended = models.recommend(card.vram_gb if card else None)
    for p in models.PRESETS.values():
        star = " (recommended here)" if p.name == recommended.name else ""
        typer.echo(f"{p.name}{star}: needs about {p.min_vram_gb:g} GB of GPU memory. {p.note}")
        for repo in models.repos(p):
            typer.echo(
                f"    {'downloaded' if models.is_cached(repo) else 'not downloaded':<15} {repo}"
            )


@models_app.command("pull")
def models_pull(
    preset: Annotated[str, typer.Option(help="auto, large, medium, small or cpu.")] = "auto",
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask before downloading.")] = False,
) -> None:
    """Download the models for a preset, after showing their licenses and sizes."""
    card = doctor.gpu()
    try:
        chosen = models.resolve(preset, card.vram_gb if card else None)
    except ValueError as e:
        raise fail(str(e)) from None
    infos = [models.describe(r) for r in models.repos(chosen)]
    todo = [i for i in infos if not i.cached]
    typer.echo(f"Preset {chosen.name}:")
    for i in infos:
        state = "downloaded" if i.cached else f"{i.size_bytes / 2**30:.1f} GB to download"
        typer.echo(f"  {i.repo}  license: {i.license}  ({state})")
    if not todo:
        typer.echo("Everything is already downloaded.")
        return
    if not yes and not typer.confirm("Download now?"):
        raise typer.Exit(1)
    for i in todo:
        typer.echo(f"Downloading {i.repo}")
        models.download(i.repo)
    typer.echo("Done. Reelsift can now run offline.")


def main() -> None:
    app()

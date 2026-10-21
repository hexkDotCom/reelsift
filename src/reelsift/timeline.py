"""A self-contained HTML page showing every recording on one timeline, for checking sync."""

import html
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

from .project import Trip
from .sync import clock
from .timebase import Provenance, format_zone, from_db, zone

COLORS = {
    Provenance.MANUAL: "var(--manual)",
    Provenance.GPS: "var(--good)",
    Provenance.METADATA_WITH_ZONE: "var(--good)",
    Provenance.METADATA_ZONE_FROM_FILENAME: "var(--good)",
    Provenance.METADATA_ZONE_FROM_CONFIG: "var(--good)",
    Provenance.METADATA_ZONE_FROM_CAMERA: "var(--good)",
    Provenance.METADATA_UNKNOWN_ZONE: "var(--unsure)",
    Provenance.METADATA_LOCAL_UNKNOWN_ZONE: "var(--unsure)",
    Provenance.FILE_MODIFIED: "var(--bad)",
}

ROW_H, LABEL_W, WIDTH, AXIS_H = 34, 210, 1100, 26


@dataclass
class Bar:
    recording_id: int
    camera: str
    start: datetime
    end: datetime
    files: list[str]
    provenance: Provenance
    confidence: float
    zone_offset_min: int | None
    proxy: str
    flags: list[str]


def _bars(trip: Trip) -> list[Bar]:
    db = trip.db
    files = defaultdict(list)
    for r in db.execute(
        "SELECT c.recording_id, a.path FROM chapters c JOIN assets a ON a.id = c.asset_id"
        " ORDER BY c.recording_id, c.idx"
    ):
        files[r["recording_id"]].append(Path(r["path"]).name)
    proxy = {
        r["recording_id"]: r["origin"]
        for r in db.execute("SELECT recording_id, origin FROM proxies")
    }
    bars = []
    for rec in db.execute(
        "SELECT r.*, cam.label FROM recordings r JOIN cameras cam ON cam.id = r.camera_id"
    ):
        clk = clock(trip, rec["id"])
        if clk is None:
            continue
        duration = from_db(rec["duration"])
        flags = [f for f, on in (("HDR", rec["hdr"]), ("VFR", rec["vfr"])) if on]
        bars.append(
            Bar(
                rec["id"],
                rec["label"],
                clk.to_utc(Fraction(0)),
                clk.to_utc(duration),
                files[rec["id"]],
                clk.provenance,
                clk.confidence,
                clk.zone_offset_min,
                proxy.get(rec["id"], "none"),
                flags,
            )
        )
    return sorted(bars, key=lambda b: b.start)


def _fmt_duration(seconds: float) -> str:
    seconds = round(seconds)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _day_svg(bars: list[Bar], cameras: list[str], tz, anchors: list[tuple]) -> str:
    start = min(b.start for b in bars).astimezone(tz)
    end = max(b.end for b in bars).astimezone(tz)
    lo = start.replace(minute=0, second=0, microsecond=0)
    hi = end.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    span = (hi - lo).total_seconds()
    plot_w = WIDTH - LABEL_W - 10

    def x(t: datetime) -> float:
        return LABEL_W + (t.astimezone(tz) - lo).total_seconds() / span * plot_w

    height = AXIS_H + ROW_H * len(cameras) + 8
    parts = [
        f'<svg viewBox="0 0 {WIDTH} {height}" role="img" aria-label="Recordings on a timeline">'
    ]
    tick = lo
    step = timedelta(hours=1) if span <= 12 * 3600 else timedelta(hours=2)
    while tick <= hi:
        tx = x(tick)
        parts.append(
            f'<line x1="{tx:.1f}" y1="{AXIS_H - 6}" x2="{tx:.1f}" y2="{height}" class="grid"/>'
        )
        parts.append(f'<text x="{tx:.1f}" y="{AXIS_H - 10}" class="tick">{tick:%H:%M}</text>')
        tick += step
    for i, cam in enumerate(cameras):
        y = AXIS_H + i * ROW_H
        parts.append(
            f'<text x="8" y="{y + ROW_H / 2 + 4:.1f}" class="cam">{html.escape(cam)}</text>'
        )
    row = {cam: i for i, cam in enumerate(cameras)}
    for b in bars:
        y = AXIS_H + row[b.camera] * ROW_H + 6
        x0, x1 = x(b.start), x(b.end)
        tip = (
            f"#{b.recording_id} {', '.join(b.files)}\n"
            f"{b.start.astimezone(tz):%H:%M:%S} to {b.end.astimezone(tz):%H:%M:%S}"
            f" ({_fmt_duration((b.end - b.start).total_seconds())})\n"
            f"time: {b.provenance.value}, confidence {b.confidence:.2f}, "
            f"zone {format_zone(b.zone_offset_min)}\nproxy: {b.proxy}"
            + (f"\n{' '.join(b.flags)}" if b.flags else "")
        )
        parts.append(
            f'<rect x="{x0:.1f}" y="{y}" width="{max(x1 - x0, 2):.1f}" height="{ROW_H - 12}" rx="3"'
            f' fill="{COLORS[b.provenance]}"><title>{html.escape(tip)}</title></rect>'
        )
    for when, cams, label in anchors:
        ax = x(when)
        ys = [AXIS_H + row[c] * ROW_H + ROW_H / 2 for c in cams if c in row]
        if ys:
            parts.append(
                f'<line x1="{ax:.1f}" y1="{min(ys) - 10:.1f}" x2="{ax:.1f}" y2="{max(ys) + 10:.1f}"'
                f' class="anchor"><title>{html.escape(label)}</title></line>'
            )
    parts.append("</svg>")
    return "".join(parts)


STYLE = """
:root { --bg:#fbfbf9; --fg:#1d1d1b; --muted:#6b6b66; --line:#e4e3de; --good:#3d7fbf;
  --unsure:#d49a2a; --bad:#c4473a; --manual:#3f9a6b; color-scheme: light dark; }
@media (prefers-color-scheme: dark) { :root { --bg:#191918; --fg:#ecebe6; --muted:#9a9993;
  --line:#33332f; --good:#5b9ad6; --unsure:#e0ad48; --bad:#de6a5c; --manual:#5bb886; } }
body { margin:0; background:var(--bg); color:var(--fg);
  font:14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:1140px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:16px; margin:28px 0 8px; }
.sub { color:var(--muted); margin:0 0 16px; }
.legend span { display:inline-flex; align-items:center; gap:6px; margin-right:16px; color:var(--muted); }
.legend i { width:12px; height:12px; border-radius:3px; display:inline-block; }
.scroll { overflow-x:auto; }
svg { width:100%; min-width:720px; height:auto; }
svg .grid { stroke:var(--line); } svg .tick { fill:var(--muted); font-size:11px; text-anchor:middle; }
svg .cam { fill:var(--fg); font-size:12px; }
svg .anchor { stroke:var(--fg); stroke-width:2; stroke-dasharray:3 2; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th, td { text-align:left; padding:5px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--muted); font-weight:500; }
td.num { font-variant-numeric:tabular-nums; }
.sev-error { color:var(--bad); } .sev-warning { color:var(--unsure); } .sev-info { color:var(--muted); }
"""


def render(trip: Trip) -> str:
    bars = _bars(trip)
    zones = Counter(b.zone_offset_min for b in bars if b.zone_offset_min is not None)
    display_zone = zones.most_common(1)[0][0] if zones else 0
    tz = zone(display_zone)
    cameras = sorted({b.camera for b in bars})

    rec_camera = {b.recording_id: b.camera for b in bars}
    anchors = []
    for a in trip.db.execute("SELECT * FROM anchors"):
        clk = clock(trip, a["recording_a"])
        if clk is None:
            continue
        cams = [rec_camera.get(a["recording_a"])]
        if a["recording_b"]:
            cams.append(rec_camera.get(a["recording_b"]))
        residual = a["residual_s"]
        label = f"Anchor {a['id']}" + (
            f", residual {residual:.2f} s" if residual is not None else ""
        )
        anchors.append((clk.to_utc(from_db(a["time_a"])), [c for c in cams if c], label))

    days = defaultdict(list)
    for b in bars:
        days[b.start.astimezone(tz).date()].append(b)

    out = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>{html.escape(trip.config.name or 'Trip')} timeline</title>",
        f"<style>{STYLE}</style></head><body><main>",
        f"<h1>{html.escape(trip.config.name or 'Trip')}</h1>",
        f"<p class='sub'>{len(bars)} recordings from {len(cameras)} camera(s). "
        f"Times shown in {format_zone(display_zone)}. Generated {datetime.now():%Y-%m-%d %H:%M}.</p>",
        "<p class='legend'><span><i style='background:var(--good)'></i>time and zone known</span>"
        "<span><i style='background:var(--manual)'></i>set by anchors</span>"
        "<span><i style='background:var(--unsure)'></i>zone unknown</span>"
        "<span><i style='background:var(--bad)'></i>unreliable</span></p>",
    ]
    for day in sorted(days):
        out.append(f"<h2>{day:%A, %d %B %Y}</h2><div class='scroll'>")
        day_anchors = [a for a in anchors if a[0].astimezone(tz).date() == day]
        out.append(_day_svg(days[day], cameras, tz, day_anchors))
        out.append("</div>")

    out.append(
        "<h2>Recordings</h2><div class='scroll'><table><tr><th>#</th><th>Camera</th>"
        "<th>Files</th><th>Start</th><th>Length</th><th>Time from</th><th>Zone</th>"
        "<th>Proxy</th><th>Notes</th></tr>"
    )
    for b in bars:
        out.append(
            f"<tr><td class='num'>{b.recording_id}</td><td>{html.escape(b.camera)}</td>"
            f"<td>{html.escape(', '.join(b.files))}</td>"
            f"<td class='num'>{b.start.astimezone(tz):%Y-%m-%d %H:%M:%S}</td>"
            f"<td class='num'>{_fmt_duration((b.end - b.start).total_seconds())}</td>"
            f"<td>{b.provenance.value} ({b.confidence:.2f})</td>"
            f"<td>{format_zone(b.zone_offset_min)}</td><td>{b.proxy}</td>"
            f"<td>{' '.join(b.flags)}</td></tr>"
        )
    out.append("</table></div>")

    issues = trip.db.execute(
        "SELECT * FROM issues ORDER BY CASE severity WHEN 'error' THEN 0 WHEN 'warning' THEN 1"
        " ELSE 2 END, id"
    ).fetchall()
    if issues:
        out.append(
            "<h2>Issues</h2><table><tr><th>Severity</th><th>Stage</th><th>Message</th>"
            "<th>Where</th></tr>"
        )
        for i in issues:
            where = i["path"] or (f"recording {i['recording_id']}" if i["recording_id"] else "")
            out.append(
                f"<tr><td class='sev-{i['severity']}'>{i['severity']}</td><td>{i['stage']}</td>"
                f"<td>{html.escape(i['message'])}</td><td>{html.escape(str(where))}</td></tr>"
            )
        out.append("</table>")
    out.append("</main></body></html>")
    return "".join(out)


def write(trip: Trip) -> Path:
    path = trip.debug_dir / "timeline.html"
    path.write_text(render(trip), encoding="utf-8")
    return path

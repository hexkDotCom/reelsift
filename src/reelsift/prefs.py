"""Preferences: what you want more and less of, as adjustments to scoring weights.

Text like "more scenery and the kids, less road and fewer stops" becomes slider values over a fixed
vocabulary of features, plus free-text visual searches for anything outside it ("the red bridge"),
which are answered from stored frame embeddings. Turning text into sliders uses the language model
when it is available and a keyword matcher otherwise; either way the result is shown as sliders
you can adjust, and nothing is excluded without being listed.

Every edit length has its own preferences, with a style (a dynamic preview, highlights, or a
story) that shapes how clips are spread over the trip, and optionally its own clip lengths.
"""

import json
import re
from dataclasses import asdict, dataclass, field

from .project import now
from .scoring import PROMPTS, SIGNALS
from .vlm import TAGS

STEP = 0.6
"""Weight added per slider step."""
SEARCH_STEP = 0.8
EXCLUDE_Z = 1.0
"""A candidate is excluded when an excluded feature is this many deviations above typical."""

LABELS = {
    "motion": "camera moving fast",
    "activity": "movement in the picture",
    "shake": "shaky camera",
    "sharpness": "sharp picture",
    "exposure_problem": "too dark or too bright",
    "speech": "people talking",
    "loudness": "loud sound",
    "scene": "scene changes",
    "interest": "interesting, according to the AI",
    "vlm_quality": "picture problems the AI noticed",
    **{name: text.removeprefix("a ").removeprefix("an ") for name, text in PROMPTS.items()},
    **{f"tag:{t}": t.replace("_", " ") for t in TAGS},
}


def vocabulary() -> list[str]:
    return list(SIGNALS) + ["interest", "vlm_quality"] + list(PROMPTS) + [f"tag:{t}" for t in TAGS]


@dataclass(frozen=True)
class Style:
    name: str
    label: str
    brief: str
    """What an edit in this style is, for the language model that drafts its description."""
    template: str
    """The description used when no language model is available (keywords only)."""
    coverage: float
    """Penalty for picking again from a part of the trip already covered: high spreads the edit
    evenly from start to finish, low lets it gather wherever the best moments are."""
    parts_per_clip: float
    """How finely the trip is divided for coverage: one part per this many clips."""
    speech: str
    """How long speech clips may run: "short", "normal" or "long"."""


STYLES = {
    "preview": Style(
        "preview", "Dynamic preview",
        "a fast, energetic preview: the most striking, action-packed and visually strong moments, "
        "movement and speed, short cuts, no long talking",
        "lots of action, fast, scenery, faces, less talking, fewer stops",
        coverage=0.15, parts_per_clip=3, speech="short",
    ),
    "highlights": Style(
        "highlights", "Highlights",
        "the best moments of the whole outing, balanced between places, people and action",
        "scenery, people, landmarks, action",
        coverage=0.5, parts_per_clip=3, speech="normal",
    ),
    "story": Style(
        "story", "Story",
        "a story of the outing from start to finish: setting off, the journey, the places, the "
        "people and their conversations, the funny and the hard moments, and how it ended",
        "people, conversations, laughter, landmarks, food",
        coverage=1.5, parts_per_clip=1.5, speech="long",
    ),
}  # fmt: skip


def default_style(length: int, lengths: list[int]) -> str:
    """The shortest edit is a preview and the longest a story; any in between are highlights."""
    ordered = sorted(lengths)
    if len(ordered) > 1 and length == ordered[0]:
        return "preview"
    if len(ordered) > 1 and length == ordered[-1]:
        return "story"
    return "highlights"


CLIP_RANGE = (0.5, 120.0)
"""Smallest and largest clip length, in seconds, that can be set by hand."""


@dataclass
class Preferences:
    text: str = ""
    adjustments: dict[str, float] = field(default_factory=dict)
    """Slider value per feature, from -3 (much less) to +3 (much more)."""
    searches: list[dict] = field(default_factory=list)
    """{"text": what to look for, "value": slider from -3 to +3}."""
    exclusions: list[str] = field(default_factory=list)
    """Feature names, or "search:<text>", whose strong matches are left out entirely."""
    interpretation: list[str] = field(default_factory=list)
    engine: str = "none"
    style: str = ""
    """One of STYLES; empty means the default for the edit's length."""
    clip: list[float] | None = None
    """[shortest, longest] clip length in seconds, or None for automatic."""
    drafted: bool = False
    """Written by the language model from the trip's own descriptions, not by a person."""

    def weights(self, base: dict[str, float]) -> dict[str, float]:
        out = dict(base)
        for name, value in self.adjustments.items():
            out[name] = out.get(name, 0.0) + STEP * value
        for s in self.searches:
            out[search_feature(s["text"])] = SEARCH_STEP * s["value"]
        return out

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "Preferences":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})

    def clean(self) -> "Preferences":
        known = set(vocabulary())
        self.adjustments = {
            k: float(max(-3, min(3, v))) for k, v in self.adjustments.items() if k in known and v
        }
        seen, searches = set(), []
        excluded = set(self.exclusions)
        for s in self.searches:
            text = str(s.get("text", "")).strip()[:120]
            # A search at zero is dropped, unless it is there to leave something out.
            if text and text.lower() not in seen and (
                    s.get("value") or search_feature(text) in excluded):  # fmt: skip
                seen.add(text.lower())
                searches.append({"text": text, "value": float(max(-3, min(3, s["value"])))})
        self.searches = searches
        allowed = known | {search_feature(s["text"]) for s in self.searches}
        self.exclusions = [e for e in dict.fromkeys(self.exclusions) if e in allowed]
        if self.style and self.style not in STYLES:
            raise ValueError(f"unknown style {self.style!r}; choose from {', '.join(STYLES)}")
        if self.clip is not None:
            try:
                lo, hi = (float(x) for x in self.clip)
            except (TypeError, ValueError):
                raise ValueError("clip lengths must be two numbers: shortest and longest") from None
            if not (CLIP_RANGE[0] <= lo <= hi <= CLIP_RANGE[1]):
                raise ValueError(
                    f"clip lengths must be from {CLIP_RANGE[0]:g} to {CLIP_RANGE[1]:g} s, "
                    "with the shortest no longer than the longest"
                )
            self.clip = [round(lo, 2), round(hi, 2)]
        return self


def search_feature(text: str) -> str:
    return f"search:{text.strip().lower()}"


def describe(p: Preferences) -> list[str]:
    lines = []
    for name, value in sorted(p.adjustments.items(), key=lambda kv: -abs(kv[1])):
        word = "more" if value > 0 else "less"
        lines.append(f"{word} {LABELS.get(name, name)} ({value:+g})")
    for s in p.searches:
        word = "more" if s["value"] > 0 else "less"
        lines.append(f'{word} of what looks like "{s["text"]}" ({s["value"]:+g})')
    for e in p.exclusions:
        label = e.removeprefix("search:") if e.startswith("search:") else LABELS.get(e, e)
        lines.append(f"leave out: {label}")
    return lines


# --- keyword matcher --------------------------------------------------------------------------

KEYWORDS: dict[str, list[str]] = {
    "scenery": [
        "scenery",
        "scenic",
        "view",
        "views",
        "landscape",
        "landscapes",
        "mountain",
        "mountains",
        "nature",
        "panorama",
        "vista",
        "vistas",
        "hills",
    ],
    "landmark": [
        "landmark",
        "landmarks",
        "building",
        "buildings",
        "monument",
        "church",
        "temple",
        "castle",
        "architecture",
        "sights",
    ],
    "people": ["people", "friends", "us", "family", "group", "everyone", "person"],
    "faces": ["face", "faces", "closeup", "close-up", "portrait", "portraits", "selfie", "selfies"],
    "kids": ["kid", "kids", "child", "children", "son", "daughter", "baby", "toddler"],
    "action": [
        "action",
        "exciting",
        "fast",
        "speed",
        "jump",
        "jumps",
        "descent",
        "descents",
        "downhill",
        "tricks",
        "adrenaline",
    ],
    "water": [
        "water",
        "lake",
        "lakes",
        "river",
        "sea",
        "ocean",
        "beach",
        "waterfall",
        "pool",
        "swimming",
    ],
    "food": [
        "food",
        "eat",
        "eating",
        "lunch",
        "dinner",
        "breakfast",
        "meal",
        "meals",
        "cafe",
        "restaurant",
        "drinks",
        "coffee",
    ],
    "animal": [
        "animal",
        "animals",
        "dog",
        "dogs",
        "cat",
        "cats",
        "horse",
        "horses",
        "bird",
        "birds",
        "cow",
        "cows",
        "wildlife",
    ],
    "sky": ["sunset", "sunsets", "sunrise", "sky", "clouds"],
    "city": ["city", "town", "street", "streets", "village", "downtown"],
    "road": ["road", "roads", "asphalt", "highway", "pavement"],
    "speech": [
        "talk",
        "talking",
        "conversation",
        "conversations",
        "chat",
        "chatting",
        "speech",
        "voices",
        "stories",
    ],
    "tag:laughter": ["laugh", "laughing", "laughter", "fun", "jokes", "funny"],
    "tag:stop": [
        "stop",
        "stops",
        "break",
        "breaks",
        "pause",
        "pauses",
        "waiting",
        "rest",
        "resting",
    ],
    "shake": ["shaky", "shake", "shaking", "bumpy"],
    "exposure_problem": ["dark"],
    "tag:walking": ["walk", "walking", "hike", "hiking"],
    "tag:riding": ["ride", "riding", "cycling", "bike", "biking"],
    "tag:trail": ["trail", "trails", "path", "paths", "singletrack"],
    "tag:indoors": ["indoors", "inside"],
    "tag:crowd": ["crowd", "crowds"],
    "tag:celebration": ["party", "celebration", "birthday", "wedding"],
    "tag:sport": ["sport", "sports", "game", "match"],
}
NEGATIVE = {"less", "fewer", "no", "not", "without", "skip", "avoid", "don't", "dont", "minus",
            "exclude", "drop", "remove", "cut", "hate", "boring", "nothing"}  # fmt: skip
STRONG = {"lots", "much", "many", "plenty", "all", "loads", "mostly", "only"}
WEAK = {"bit", "little", "slightly", "some", "few", "occasional"}
STOP = {"more", "of", "the", "a", "an", "and", "or", "with", "to", "in", "on", "at", "for", "me",
        "i", "want", "would", "like", "please", "show", "keep", "include", "lot", "some", "bit",
        "be", "is", "are", "it", "that", "this", "those", "these", "my", "our", "we", "there",
        "also", "too", "very", "really", "just", "where", "when", "moments", "moment", "parts",
        "part", "shots", "shot", "clips", "clip", "footage", "scenes", "scene", "stuff", "things",
        "time", "times", "less", "fewer", "much", "lots", "many", "plenty", "loads",
        "little", "slightly", "few", "no", "not", "without", "skip", "avoid", "don't", "dont",
        "mostly", "only", "all", "any", "but", "even", "so", "us"}  # fmt: skip
WORD = re.compile(r"[\w'-]+", re.UNICODE)


def compile_rules(text: str) -> Preferences:
    prefs = Preferences(text=text, engine="keywords")
    for clause in re.split(r"[,;.\n]|\bbut\b|\band\b|\bplus\b", text, flags=re.IGNORECASE):
        words = [w.lower() for w in WORD.findall(clause)]
        if not words:
            continue
        sign = -1 if any(w in NEGATIVE for w in words) else 1
        strength = (
            3 if any(w in STRONG for w in words) else 1 if any(w in WEAK for w in words) else 2
        )
        matched = set()
        for feature, keys in KEYWORDS.items():
            hits = [w for w in words if w in keys]
            if hits:
                prefs.adjustments[feature] = sign * strength
                matched.update(hits)
        leftover = [w for w in words if w not in matched and w not in STOP and len(w) > 2]
        if leftover and not matched:
            prefs.searches.append({"text": " ".join(leftover), "value": sign * strength})
    prefs.clean()
    prefs.interpretation = describe(prefs)
    return prefs


# --- language model ---------------------------------------------------------------------------


def llm_prompt(text: str) -> str:
    vocab = "\n".join(f"- {name}: {LABELS[name]}" for name in vocabulary())
    return f"""You turn a person's wishes for a highlight video of their footage into settings.

Features you can raise or lower:
{vocab}

Their wishes (any language):
\"\"\"{text}\"\"\"

Reply with only a JSON object:
{{"more": {{"<feature>": <1, 2 or 3>}},
  "less": {{"<feature>": <1, 2 or 3>}},
  "search_more": ["<short English description of a visible thing that no feature covers>"],
  "search_less": ["<same, for things they want less of>"],
  "exclude": ["<feature, or a search text, they want left out completely>"]}}
Rules:
- Set only the features their words clearly ask for, usually two to six. Leave out every feature
  they don't mention; don't guess at what they might also like or dislike.
- "Include", "show", "highlight" and "keep" mean more; "skip", "omit", "avoid" and "less" mean
  less.
- Use search_more or search_less only for specific things no feature covers, such as "the red
  bridge" or "my blue jacket": a few plain words each, at most three of each, never a feature
  name.
- Use exclude only when they clearly say never, none or not at all."""


MAX_SEARCHES = 6
"""Most visual searches read from one description."""


def compile_llm(text: str, complete) -> Preferences | None:
    """complete(prompt) -> model answer. Returns None when the answer can't be used."""
    from .ml.qwen import parse_json

    data = parse_json(complete(llm_prompt(text)))
    if data is None:
        return None
    prefs = Preferences(text=text, engine="language model")
    try:
        for name, value in dict(data.get("more", {})).items():
            prefs.adjustments[name] = abs(float(value))
        for name, value in dict(data.get("less", {})).items():
            prefs.adjustments[name] = -abs(float(value))
        prefs.searches = [{"text": str(t), "value": 2.0} for t in data.get("search_more", [])]
        prefs.searches += [{"text": str(t), "value": -2.0} for t in data.get("search_less", [])]
        excludes = [str(e) for e in data.get("exclude", [])]
    except (TypeError, ValueError, AttributeError):
        return None
    known = set(vocabulary())
    # A feature name given as a search ("tag:road") is that feature, or nothing.
    searches = []
    for s in prefs.searches:
        if s["text"] in known:
            prefs.adjustments.setdefault(s["text"], s["value"])
        elif ":" not in s["text"]:
            searches.append(s)
    prefs.searches = searches[:MAX_SEARCHES]
    search_texts = {s["text"].lower(): s["text"] for s in prefs.searches}
    for e in excludes:
        if e in known:
            prefs.exclusions.append(e)
        elif e.lower() in search_texts:
            prefs.exclusions.append(search_feature(e))
        elif ":" not in e:
            prefs.searches.append({"text": e, "value": -3.0})
            prefs.exclusions.append(search_feature(e))
    prefs.clean()
    prefs.interpretation = describe(prefs)
    return prefs


def compile_text(text: str, complete=None) -> Preferences:
    """Use the language model when given, falling back to keywords if its answer is unusable."""
    if complete is not None:
        try:
            result = compile_llm(text, complete)
        except Exception:  # a model failure must not lose the request
            result = None
        if result is not None:
            return result
    return compile_rules(text)


def dumps(p: Preferences) -> dict:
    return {
        "text": p.text,
        "weights": json.dumps(p.adjustments),
        "searches": json.dumps(p.searches),
        "exclusions": json.dumps(p.exclusions),
        "interpretation": json.dumps({"lines": p.interpretation, "engine": p.engine}),
        "settings": json.dumps({"style": p.style, "clip": p.clip, "drafted": p.drafted}),
    }


def loads(row) -> Preferences:
    interp = json.loads(row["interpretation"])
    settings = json.loads(row["settings"]) if "settings" in row.keys() else {}
    return Preferences(
        text=row["text"],
        adjustments=json.loads(row["weights"]),
        searches=json.loads(row["searches"]),
        exclusions=json.loads(row["exclusions"]),
        interpretation=interp.get("lines", []),
        engine=interp.get("engine", "none"),
        style=settings.get("style", ""),
        clip=settings.get("clip"),
        drafted=bool(settings.get("drafted", False)),
    )


def save(db, length: int, p: Preferences) -> None:
    """Store the preferences of one edit length (the latest row per length is the current one)."""
    row = dumps(p)
    db.execute(
        "INSERT INTO preferences (length, text, weights, searches, exclusions, interpretation,"
        " settings, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (length, row["text"], row["weights"], row["searches"], row["exclusions"],
         row["interpretation"], row["settings"], now()),
    )  # fmt: skip


def latest(db, length: int) -> tuple[Preferences | None, bool]:
    """The current preferences of an edit length, and whether they are its own. Trips from
    before per-length preferences have one shared set, which every length starts from."""
    row = db.execute(
        "SELECT * FROM preferences WHERE length = ? ORDER BY id DESC LIMIT 1", (length,)
    ).fetchone()
    if row is not None:
        return loads(row), True
    row = db.execute(
        "SELECT * FROM preferences WHERE length IS NULL ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return (loads(row) if row is not None else None), False

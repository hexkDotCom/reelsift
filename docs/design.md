# How Reelsift works

This describes the design. What exists today is listed in the [README](../README.md); what is planned is in [roadmap.md](roadmap.md). The reasons behind individual details are in [design-notes.md](design-notes.md).

## What it does

1. You point Reelsift at the footage from one outing: videos and photos from one or more cameras and phones.
2. It analyzes everything once on your machine and stores the results in a trip folder.
3. It drafts a description for each edit from what it saw, in the edit's style: by default a 30 s dynamic preview, 3 min of highlights and a 10 min story. You can rewrite any of them in plain language.
4. It proposes the edits, any list of lengths you configure, each chosen independently with its own description.
5. In a local web app you review each edit: drop what you don't want, pull in moments it missed, trim, and change its description, style or clip lengths at any time. A new selection takes about a second, without analyzing the footage again.
6. It renders MP4s from the full-resolution originals, or exports a timeline for a video editor.

GPS is never required. Many cameras have none, have it switched off or lose the fix, so the paths without GPS are the main ones.

## Principles

| Topic | Decision | Why |
|---|---|---|
| Analysis vs selection | Analysis (slow, GPU) and selection (fast, CPU) are separate; selection never calls a model | Changing preferences, lengths or decisions must take seconds |
| Cut placement | Code decides where cuts go; models only score and describe short, fixed stretches | Video models place timestamps badly in long footage |
| Models | Local only, behind small interfaces, with presets picked from the GPU's memory; every model stage is optional | Works offline, costs nothing per trip, and a weaker machine still produces an edit |
| Storage | One folder per trip: SQLite for metadata, decisions and edits; files for proxies, embeddings and caches | Analysis runs once; everything else re-reads stored results |
| Footage | Originals are only ever read; files are identified by content, so moved footage is found again | Your footage is never at risk |
| Review UI | A FastAPI server on 127.0.0.1 with a dependency-free frontend bundled in the package | Opens in any browser, works offline, nothing to build |
| Output | SDR Rec.709 H.264 MP4 at one frame rate per edit; HDR footage is tone-mapped | One well-defined output is easier to get right |
| Editor export | FCPXML and OpenTimelineIO with basic cuts and each clip's own sound | Effects, crops and color don't carry over reliably between editors |

## Analyze once, re-select in seconds

- **Analysis** looks at pixels and sound: proxies, embeddings, speech, motion, picture quality and scene descriptions. It is slow, uses the GPU and runs once per trip. Every stored result records the model and settings that made it, so changing one invalidates only what depends on it.
- **Selection** works only on stored results: weighting, ranking, picking, refilling. It runs on the CPU.

During review you can rewrite your preferences, move sliders, change lengths or add a free-text visual search ("the red bridge"). A search compares your words with the stored picture embeddings, so it finds things nobody anticipated without running the vision model again. Your keep, drop and add decisions and your trims are applied on top of every new selection.

Turning preference text into slider settings is one short text-only model call. If no model is available, or the answer is unusable, a keyword matcher and the sliders still work.

Each edit length has its own preferences: a description, sliders, searches, a style and optionally its own clip lengths. The style decides how the edit is spread over the trip (a story covers it evenly from start to finish; a preview gathers the strongest moments wherever they are) and how long speech may run. Drafts are written once after analysis by the vision model's language side, from a summary of its own scene descriptions; without a model, each style has a fixed template.

## Data model

- **Asset:** one file as it sits on disk, identified by its size and samples of its content, with its probed stream metadata (time base, start offset, frame rate, rotation, color, audio).
- **Recording:** one continuous capture. A camera that splits long recordings into chapters gives an ordered list of chapter assets, checked for gaps. A photo is a recording with no duration.
- **Recording time:** a position in a recording, counted from its first video frame. Exact fractions are used where frames matter (chapter offsets, proxy offsets, clocks); analysis uses seconds. Intervals are half-open `[start, end)`.
- **Clock transform:** per recording, how recording time maps to real (UTC) time, where that answer came from (GPS, metadata with a zone, metadata plus a zone set per camera, file names, photo EXIF, manual anchors), and how confident it is. Everything is stored in UTC, so trips across time zones work.
- **Observations:** per-second signals and embeddings. They don't depend on the clock, so fixing the time sync never re-runs the models.
- **Candidates:** proposed clips, each with a scored core and handles on both sides that a cut may move into. They are rebuilt cheaply from observations.
- **Moments:** candidates from different cameras that show the same thing at the same time.
- **Decisions:** keep, drop and add, stored as recording intervals separately from edits, so they survive every new selection.
- **Edit revisions:** an ordered list of clips for one length, saved after every change, so any earlier revision can be restored.

## Pipeline

1. **Ingest.** Probe every file and match it to a device profile. Group chapters into recordings; report missing chapters instead of guessing. Pair camera-made proxies after checking their duration and orientation, and generate proxies for everything else.
2. **Clock sync.** Build each recording's clock from the best evidence available. Manual anchors ("this moment was at 10:05", or "these two cameras show the same instant") correct whole cameras, solved by least squares. Recordings with uncertain time stay usable.
3. **Speech.** Transcription with word timings, so cuts can avoid splitting words.
4. **Observations.** Per second: picture embeddings; camera motion and scene activity from optical flow (or camera sensors, where a parser exists); sharpness and exposure; loudness and speech.
5. **Scene descriptions.** A vision-language model describes fixed 6-second blocks and answers with tags from a fixed list and an interest score, validated against a schema. Anything not described, or answered badly, is stored as unknown, never as "no".
6. **Candidates.** Overlapping windows plus speech phrases, each with a core and handles. Photos become candidates too; bursts count once.
7. **Scoring.** Signals are normalized per trip, then weighted. Each candidate's score breakdown is kept and shown in the review app.
8. **Selection.** Each edit is chosen on its own, with its own scores, clip lengths and decisions. Repeated moments, near-identical shots and over-covered parts of the trip are penalized, the last one as strongly as the edit's style asks. Edits stay in time order.
9. **Review.** See below.
10. **Render.** Each clip is cut frame-accurately from the original, converted to the output frame rate, size and color, and the clips are joined. The sound mix keeps the clips' own sound where people speak, ducks optional music under speech and normalizes loudness. With music, cuts snap to beats during selection, so what you approve is what renders.
11. **Export.** FCPXML 1.10 and OpenTimelineIO timelines pointing at the original files and their timecode.

## Review

- A storyboard for each length, with thumbnails, previews and the reasons each clip was picked.
- **Drop** removes footage from this edit, or from every edit, and keeps it out of later selections there. It can be undone.
- **Add** puts footage into this edit.
- **Keep** locks a clip so later selections never remove it. **Trim** is a keep of the new interval.
- After a drop, nearby replacements are suggested; nothing reshuffles until you accept one, press Refill or press Select again.
- A browser of all candidates, filterable by recording, to pull in moments the edit missed.
- Undo, and a list of revisions that can be restored.
- Every change is one database transaction, checked against the rules edits must keep (valid intervals, no overlaps, nothing reaching into dropped footage) before it is saved.

## Built for other people's machines and footage

- **Device profiles.** Each camera or phone is a small profile: how to recognize its files, group chapters, pair proxies and read its time. Adding a device means adding a profile, a few seconds of sample footage and tests.
- **Hardware.** `reelsift doctor` checks the setup and says what to fix. Model presets follow the GPU's memory, and every model stage can be left out.
- **Privacy.** Footage never leaves the machine. No analytics or telemetry. The only network use is downloading models, which shows each license first.
- **Compatibility.** Database migrations keep older trip folders working.
- **Two interfaces.** The web app for most people, and a CLI that runs every stage for scripting.

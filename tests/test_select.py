"""Selection on hand-built trip data, without media."""

import itertools

import numpy as np

from reelsift.candidates import phrases, windows
from reelsift.moments import index as moment_index
from reelsift.scoring import normalize
from reelsift.select import Item, avoid_words, clip_length, select
from reelsift.tripdata import Candidates, Recording, TripData


def trip_data(n_recordings: int = 3, duration: float = 120.0) -> TripData:
    recordings, cands = {}, []
    cid = 0
    for r in range(1, n_recordings + 1):
        recordings[r] = Recording(r, 1, "cam", "video", duration, 1_000_000.0 + r * 1000, 0.8, 120)
        for p in windows(duration):
            cid += 1
            cands.append((cid, r, p.start, p.end, p.core_start, p.core_end,
                          max(0, p.start - 3), min(duration, p.end + 3), p.source))  # fmt: skip
    cols = list(zip(*cands, strict=True))
    c = Candidates(*(np.array(col) for col in cols[:8]), source=np.array(cols[8], dtype=object))
    return TripData(recordings=recordings, observations={}, embeddings={}, candidates=c)


def test_windows_overlap_and_reach_the_end():
    ws = windows(11.0)
    assert ws[0].start == 0 and ws[-1].end == 11.0
    assert all(b.start - a.start == 2.0 for a, b in itertools.pairwise(ws[:-1]))
    assert windows(3.0)[0].end == 3.0


def test_phrases_split_at_pauses():
    words = [(1.0, 1.4), (1.5, 2.2), (4.0, 4.5), (4.6, 5.3)]
    ps = phrases(words, 10.0)
    assert [(p.core_start, p.core_end) for p in ps] == [(1.0, 2.2), (4.0, 5.3)]


def test_clip_length_grows_with_edit_length():
    a, b, c = clip_length(30), clip_length(180), clip_length(600)
    assert a.target < b.target < c.target
    assert 1.5 <= a.target <= 2.5 and 6 <= c.target <= 8


def test_select_reaches_length_and_respects_mask():
    data = trip_data()
    n = len(data.candidates)
    scores = np.linspace(0, 1, n)
    emb = np.zeros((n, 4), np.float32)
    allowed = np.ones(n, bool)
    allowed[data.candidates.recording == 3] = False
    sel = select(data, scores, emb, 60, allowed=allowed)
    assert 55 <= sel.duration <= 64
    assert all(i.recording_id != 3 for i in sel.items)


def test_forced_items_are_kept_and_count_toward_length():
    data = trip_data()
    n = len(data.candidates)
    forced = Item(2, 10.0, 25.0, int(data.candidates.id[0]), 0.0, None, "window", locked=True)
    sel = select(data, np.zeros(n), np.zeros((n, 4), np.float32), 30, forced=[forced])
    assert forced in sel.items
    assert 27 <= sel.duration <= 33


def test_coverage_spreads_picks_over_recordings():
    data = trip_data(n_recordings=4)
    n = len(data.candidates)
    scores = np.where(data.candidates.recording == 1, 1.0, 0.8)
    sel = select(data, scores, np.zeros((n, 4), np.float32), 60)
    assert len({i.recording_id for i in sel.items}) >= 3


def test_cut_points_move_out_of_words():
    words = np.array([[1.0, 2.0], [3.0, 3.5]])
    assert avoid_words(words, 1.2, 0.0, 5.0) == 1.0
    assert avoid_words(words, 1.8, 0.0, 5.0) == 2.0
    assert avoid_words(words, 2.5, 0.0, 5.0) == 2.5
    assert avoid_words(words, 1.2, 1.1, 5.0) == 2.0


def test_normalize_is_robust_and_neutral_for_unknown():
    raw = np.array([[1.0], [2.0], [3.0], [4.0], [1000.0], [np.nan]])
    z = normalize(raw)
    assert z[5, 0] == 0
    assert z[4, 0] == 3  # the outlier is clipped
    assert abs(z[2, 0]) < 1e-9  # the median is neutral


def test_clip_lengths_set_by_hand_are_used():
    data = trip_data(n_recordings=4, duration=200.0)
    n = len(data.candidates)
    scores = np.random.default_rng(1).random(n)
    emb = np.zeros((n, 4), np.float32)
    auto = select(data, scores, emb, 90)
    long_clips = select(data, scores, emb, 90, clips=clip_length(90, clip=[5.0, 6.0]))
    assert (
        np.mean([i.duration for i in long_clips.items])
        > np.mean([i.duration for i in auto.items]) + 1
    )
    assert all(i.duration <= 6.0 + 1e-6 for i in long_clips.items)
    assert 85 <= long_clips.duration <= 96


def test_speech_may_run_longer_in_a_story_than_in_a_preview():
    assert clip_length(30, "short").longest_speech == clip_length(30).longest
    assert clip_length(30, "long").longest_speech >= 15
    assert clip_length(600, "normal").longest_speech == 15


def test_more_coverage_spreads_an_edit_over_the_trip():
    data = trip_data(n_recordings=6)
    n = len(data.candidates)
    scores = np.where(data.candidates.recording <= 2, 1.0, 0.0)
    emb = np.zeros((n, 4), np.float32)

    def recordings(coverage):
        sel = select(data, scores, emb, 60, coverage=coverage, coverage_buckets=6)
        return len({i.recording_id for i in sel.items})

    assert recordings(3.0) > recordings(0.0)


def test_second_camera_angle_of_a_moment_is_avoided():
    data = trip_data(n_recordings=2, duration=60.0)
    # Two cameras filming the same minute.
    data.recordings[2].camera_id = 2
    data.recordings[2].utc_start = data.recordings[1].utc_start
    n = len(data.candidates)
    emb = np.zeros((n, 4), np.float32)
    emb[:, 0] = 1.0  # everything looks alike
    moments = moment_index(data, emb)
    c = data.candidates
    a = int(np.flatnonzero((c.recording == 1) & (c.start == 10.0))[0])
    same = moments.same_moment(a)
    assert same[(c.recording == 2) & (c.start == 10.0)].all()
    assert not same[c.recording == 1].any(), "never the same camera"
    assert not same[(c.recording == 2) & (c.start >= 30.0)].any(), "not chained across time"
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 1, n) + np.where(c.start == 10.0, 3.0, 0.0)
    sel = select(data, scores, np.zeros((n, 4), np.float32), 8, moments=moments)
    angles = [i.recording_id for i in sel.items if abs(i.start - 10.0) < 3]
    assert len(angles) == 1


def test_moments_need_two_cameras():
    data = trip_data(n_recordings=2, duration=30.0)
    n = len(data.candidates)
    emb = np.ones((n, 4), np.float32) / 2
    moments = moment_index(data, emb)
    assert not any(moments.same_moment(i).any() for i in range(n))

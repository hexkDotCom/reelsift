import pytest

from reelsift.jobs import Job
from reelsift.project import Trip


@pytest.fixture
def trip(tmp_path):
    with Trip.create(tmp_path / "trip", [tmp_path]) as t:
        yield t


def test_failed_items_are_retried_and_done_items_skipped(trip):
    calls = []
    broken = {"b"}

    def work(item):
        calls.append(item)
        if item in broken:
            raise ValueError("bad file")

    first = Job(trip, "test", {"x": 1}, ["a", "b", "c"]).run(work)
    assert (first.done, set(first.failed)) == (2, {"b"})
    broken.clear()
    second = Job(trip, "test", {"x": 1}, ["a", "b", "c"]).run(work)
    assert second.job_id == first.job_id
    assert (second.done, second.skipped, second.ok) == (1, 2, True)
    assert calls == ["a", "b", "c", "b"]


def test_interrupt_marks_cancelled_and_resumes(trip):
    seen = []

    def stop_at_b(item):
        if item == "b":
            raise KeyboardInterrupt
        seen.append(item)

    with pytest.raises(KeyboardInterrupt):
        Job(trip, "test", {}, ["a", "b", "c"]).run(stop_at_b)
    assert trip.db.execute("SELECT state FROM jobs").fetchone()[0] == "cancelled"
    result = Job(trip, "test", {}, ["a", "b", "c"]).run(seen.append)
    assert seen == ["a", "b", "c"]
    assert result.skipped == 1


def test_finished_job_is_not_resumed(trip):
    first = Job(trip, "test", {}, ["a"]).run(lambda item: None)
    second = Job(trip, "test", {}, ["a"]).run(lambda item: None)
    assert second.job_id != first.job_id
    assert second.done == 1

import pytest

from reelsift.project import SCHEMA_VERSION, Trip, TripLockedError, connect, migrate


def test_create_makes_config_and_current_schema(tmp_path):
    with Trip.create(tmp_path / "trip", [tmp_path]) as trip:
        version = trip.db.execute("PRAGMA user_version").fetchone()[0]
    assert version == SCHEMA_VERSION
    assert (tmp_path / "trip" / "reelsift.toml").is_file()


def test_migrate_is_idempotent(tmp_path):
    db = connect(tmp_path / "x.db")
    assert migrate(db) == 0
    assert migrate(db) == SCHEMA_VERSION


def test_newer_schema_is_refused(tmp_path):
    db = connect(tmp_path / "x.db")
    db.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    with pytest.raises(RuntimeError, match="newer Reelsift"):
        migrate(db)


def test_second_writer_is_locked_out(tmp_path):
    root = tmp_path / "trip"
    with Trip.create(root, [tmp_path]):
        with pytest.raises(TripLockedError):
            with Trip(root, write=True):
                pass
        with Trip(root, write=False) as reader:
            assert reader.db.execute("SELECT COUNT(*) FROM recordings").fetchone()[0] == 0
    with Trip(root, write=True):
        pass


def test_opening_a_non_trip_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        Trip(tmp_path)


def test_creating_over_an_existing_trip_fails(tmp_path):
    with Trip.create(tmp_path / "trip", [tmp_path]):
        pass
    with pytest.raises(FileExistsError):
        Trip.create(tmp_path / "trip", [tmp_path])

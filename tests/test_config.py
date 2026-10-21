import pytest

from reelsift import config


def test_defaults_without_files(tmp_path):
    cfg = config.load(tmp_path, user_path=tmp_path / "missing.toml")
    assert cfg.lengths == [30, 180, 600]
    assert cfg.preset == "auto"


def test_trip_overrides_user_defaults(tmp_path):
    user = tmp_path / "user.toml"
    user.write_text('[defaults]\nlengths = [60, 300]\npreset = "medium"\n')
    (tmp_path / config.TRIP_CONFIG).write_text(
        '[trip]\nname = "Lake"\nlengths = [20, 120, 900, 1200]\n'
        '[cameras."Phone"]\nzone = "+05:30"\ncreation_time = "local"\n'
    )
    cfg = config.load(tmp_path, user_path=user)
    assert cfg.name == "Lake"
    assert cfg.lengths == [20, 120, 900, 1200]
    assert cfg.preset == "medium"
    assert cfg.camera("Phone").zone_offset_min == 330
    assert cfg.camera("Other").zone is None


def test_write_then_load_round_trips(tmp_path):
    cfg = config.Config(name="Walk", footage=["D:/a"], lengths=[45, 200])
    cfg.cameras["Cam"] = config.CameraConfig(zone="-03:00")
    config.write_trip(tmp_path, cfg)
    loaded = config.load(tmp_path, user_path=tmp_path / "none.toml")
    assert (loaded.name, loaded.footage, loaded.lengths) == ("Walk", ["D:/a"], [45, 200])
    assert loaded.camera("Cam").zone == "-03:00"


@pytest.mark.parametrize("bad", ["lengths = []", "lengths = [30, 30]", "lengths = [-5]"])
def test_invalid_lengths_are_rejected(tmp_path, bad):
    (tmp_path / config.TRIP_CONFIG).write_text(f"[trip]\n{bad}\n")
    with pytest.raises(ValueError):
        config.load(tmp_path, user_path=tmp_path / "none.toml")

from typer.testing import CliRunner

from reelsift.cli import app

from .conftest import needs_ffmpeg

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "reelsift" in result.output


@needs_ffmpeg
def test_init_prepare_status_and_anchors(tmp_path, footage):
    trip = str(tmp_path / "trip")
    assert runner.invoke(app, ["init", trip, "--footage", str(footage)]).exit_code == 0
    result = runner.invoke(app, ["prepare", trip])
    assert result.exit_code == 0, result.output
    assert "4 new recordings" in result.output
    assert (tmp_path / "trip" / "debug" / "timeline.html").is_file()

    result = runner.invoke(
        app,
        [
            "anchor",
            "add",
            trip,
            "DJI_20261011100000_0002_D",
            "0:01",
            "--same-as",
            "PXL_20261011_081000.mp4",
            "--other-at",
            "0.5",
        ],
    )
    assert result.exit_code == 0, result.output
    listed = runner.invoke(app, ["anchor", "list", trip])
    assert "recording" in listed.output and "residual 0.00 s" in listed.output
    assert runner.invoke(app, ["anchor", "remove", trip, "1"]).exit_code == 0

    status = runner.invoke(app, ["status", trip])
    assert status.exit_code == 0
    assert "recordings: 4" in status.output


def test_init_rejects_missing_footage(tmp_path):
    result = runner.invoke(app, ["init", str(tmp_path / "t"), "--footage", str(tmp_path / "nope")])
    assert result.exit_code == 1


@needs_ffmpeg
def test_anchor_needs_a_target(tmp_path, footage):
    trip = str(tmp_path / "trip")
    runner.invoke(app, ["init", trip, "--footage", str(footage)])
    result = runner.invoke(app, ["anchor", "add", trip, "1", "0"])
    assert result.exit_code == 1

from __future__ import annotations

from gpr_layer_audit import cli


def test_folder_analysis_safely_skips_incompatible_auto_plate(
    tmp_path, synthetic_acquisition, monkeypatch, capsys
):
    _, plate, _ = synthetic_acquisition
    raw = bytearray(plate.read_bytes())
    raw[512] = 6
    plate.write_bytes(raw)
    selected: dict[str, object] = {}

    def fake_run(args, road_path, plate_path=None, *, survey_id=None):
        selected.update(road=road_path, plate=plate_path, survey_id=survey_id)
        return 0

    monkeypatch.setattr(cli, "_run_analysis", fake_run)
    args = cli.build_parser().parse_args(
        ["analyze-folder", str(tmp_path), "--output", str(tmp_path / "output")]
    )

    assert cli._analyze_folder(args) == 0
    assert selected["plate"] is None
    assert "Skipping proposed calibration" in capsys.readouterr().out

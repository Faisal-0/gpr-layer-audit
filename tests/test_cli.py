from __future__ import annotations

from gpr_layer_audit import cli


def test_folder_analysis_uses_gain_mismatched_plate_for_waveform_only(
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
    assert selected["plate"] == plate
    assert "Waveform timing/ringdown will be used" in capsys.readouterr().out


def test_quick_design_cli_parses_individual_layer_thicknesses(tmp_path):
    args = cli.build_parser().parse_args(
        [
            "analyze",
            str(tmp_path / "road.DZT"),
            "--output",
            str(tmp_path / "output"),
            "--asphalt-thickness",
            "2in",
            "--base-thickness",
            "4in",
            "--dielectric",
            "7",
        ]
    )

    options = cli._analysis_options(args)

    assert [item.thickness_mm for item in options.layer_designs] == [50.8, 101.6, None]
    assert all(item.dielectric == 7.0 for item in options.layer_designs)

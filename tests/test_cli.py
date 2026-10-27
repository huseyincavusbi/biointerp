import pytest

from biointerp import cli


def test_list_models(capsys):
    assert cli.main(["list", "models"]) == 0
    out = capsys.readouterr().out
    assert "esm2_8m" in out
    assert "esmc_600m" in out


def test_list_experiments_and_datasets(capsys):
    assert cli.main(["list", "experiments"]) == 0
    assert "probing_classifier" in capsys.readouterr().out

    assert cli.main(["list", "datasets"]) == 0
    assert "swissprot" in capsys.readouterr().out


def test_list_rejects_unknown_kind():
    with pytest.raises(SystemExit) as exc:
        cli.main(["list", "bogus"])
    assert exc.value.code == 2


def test_run_dry_run(capsys):
    assert cli.main(["run", "probing_classifier", "--dry-run"]) == 0
    assert "biointerp.experiments.probing_classifier" in capsys.readouterr().out


def test_run_dry_run_with_set_override(capsys):
    assert cli.main(["run", "probing_classifier", "--dry-run", "--set", "n=10"]) == 0
    assert "--n 10" in capsys.readouterr().out


def test_invalid_set_value_exits():
    with pytest.raises(SystemExit):
        cli.main(["run", "probing_classifier", "--dry-run", "--set", "=5"])


def test_grid_dry_run(tmp_path, capsys):
    grid = tmp_path / "grid.yaml"
    grid.write_text(
        "runs:\n"
        "  - experiment: biointerp/experiments/extract.py\n"
        "    model_key: esm2_8m\n"
        "    args: ['--n', '5']\n"
    )
    assert cli.main(["grid", str(grid), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "biointerp.experiments.extract" in out
    assert "--n 5" in out


def test_missing_command_exits():
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2


def test_unknown_command_exits():
    with pytest.raises(SystemExit) as exc:
        cli.main(["nope"])
    assert exc.value.code == 2

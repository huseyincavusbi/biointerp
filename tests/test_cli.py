from biointerp.cli import main


def test_list_models(capsys):
    assert main(["list", "models"]) == 0
    assert "esm2_650m" in capsys.readouterr().out


def test_run_dry_run(capsys):
    assert (
        main(["run", "probing_classifier", "--model", "esm2_650m", "--set", "n=10", "--dry-run"])
        == 0
    )
    out = capsys.readouterr().out
    assert "--n 10" in out

from biointerp.runner import build_argv, load_config, run


def test_defaults_resolve():
    cfg = load_config()
    assert cfg["experiment"] == "probing_classifier"
    assert cfg["model"] == "esm2_650m"
    assert cfg["dataset"] == "swissprot"
    assert cfg["model_cfg"]["hf_id"] == "facebook/esm2_t33_650M_UR50D"


def test_model_key_flag_toggle():
    cfg = load_config("probing_classifier", "esm2_650m", "swissprot")
    argv = build_argv(cfg, {"n": "100"})
    assert argv[:3] == ["--n", "2000", "--model-key"]
    ladder = load_config("ladder_scaling", "esm2_8m")
    assert "--model-key" not in build_argv(ladder)


def test_dry_run_builds_command():
    result = run("probing_classifier", model="esm2_650m", overrides={"n": "10"}, dry_run=True)
    assert result["dry_run"] is True
    assert result["cmd"][-2:] == ["--n", "10"]

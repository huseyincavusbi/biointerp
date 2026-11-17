import json

from biointerp import cli
from biointerp.repro import check_contract, config_hash, metadata, missing_keys


def test_config_hash_is_deterministic_and_content_sensitive(tmp_path):
    first = tmp_path / "a.txt"
    first.write_text("one")
    second = tmp_path / "b.txt"
    second.write_text("two")

    assert config_hash([first]) == config_hash([first])
    assert config_hash([first]) != config_hash([second])
    assert len(config_hash([first])) == 16


def test_metadata_contains_python_and_packages():
    meta = metadata()
    assert meta["python"]
    assert "torch" in meta["packages"]
    assert len(meta["config_hash"]) == 16


def test_missing_keys_reports_missing_file_and_keys(tmp_path):
    path = tmp_path / "result.json"
    assert missing_keys(path, ["rows"]) == ["<file missing>"]
    path.write_text(json.dumps({"rows": []}))
    assert missing_keys(path, ["rows", "meta"]) == ["meta"]


def test_check_contract_accepts_valid_result(tmp_path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"rows": [], "meta": {}}))
    spec = {"results": [{"path": str(path), "required": ["rows", "meta"]}]}
    assert check_contract(spec) == []


def test_cli_repro_check(tmp_path):
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"rows": [], "meta": {}}))
    config = tmp_path / "repro.yaml"
    config.write_text(f"results:\n  - path: {result}\n    required: [rows, meta]\n")
    output = tmp_path / "metadata.json"

    rc = cli.main(
        [
            "repro",
            "--config",
            str(config),
            "--out",
            str(output),
            "--check",
        ]
    )

    assert rc == 0
    assert output.exists()

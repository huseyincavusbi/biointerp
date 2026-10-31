import json

import pandas as pd

from biointerp.experiments import carbonx_pilot as pilot


def test_load_variants_skips_non_variants(tmp_path):
    path = tmp_path / "vep.csv"
    pd.DataFrame(
        [
            {
                "ref_forward_sequence": "ACGTACGT",
                "alt_forward_sequence": "ACGTACGA",
                "label": 1,
                "chromosome": "1",
            },
            {
                "ref_forward_sequence": "ACGTACGT",
                "alt_forward_sequence": "ACGTACGT",
                "label": 0,
                "chromosome": "2",
            },
        ]
    ).to_csv(path, index=False)

    variants, _ = pilot.load_variants("clinvar_vep", str(path), 10, 0)

    assert len(variants) == 1
    assert variants[0]["offset"] == 7
    assert variants[0]["ref"] == "ACGTACGT"
    assert variants[0]["alt"] == "ACGTACGA"
    assert variants[0]["label"] == 1


def test_load_variants_from_manifest_order(tmp_path):
    path = tmp_path / "vep.csv"
    pd.DataFrame(
        [
            {
                "ref_forward_sequence": "ACGTACGT",
                "alt_forward_sequence": "ACGTACGA",
                "label": 1,
                "chromosome": "1",
                "position": 10,
                "ref": "T",
                "alt": "A",
            },
            {
                "ref_forward_sequence": "ACGTACGT",
                "alt_forward_sequence": "ACGAACGT",
                "label": 0,
                "chromosome": "2",
                "position": 20,
                "ref": "T",
                "alt": "A",
            },
        ]
    ).to_csv(path, index=False)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "variants": [
                    {
                        "chrom": "2",
                        "position": 20,
                        "ref": "T",
                        "alt": "A",
                        "label": 0,
                        "split": "test",
                    },
                    {
                        "chrom": "1",
                        "position": 10,
                        "ref": "T",
                        "alt": "A",
                        "label": 1,
                        "split": "train",
                    },
                ]
            }
        )
    )

    variants, _ = pilot.load_variants("clinvar_vep", str(path), 50, 0, manifest=str(manifest))

    assert [variant["chrom"] for variant in variants] == ["2", "1"]
    assert [variant["label"] for variant in variants] == [0, 1]


def test_read_done(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(
        json.dumps({"variant": 2, "condition": "dna_only"})
        + "\n"
        + json.dumps({"variant": 0, "condition": "correct_gene"})
        + "\n"
    )
    assert pilot._read_done(path) == {
        (2, "dna_only"),
        (0, "correct_gene"),
    }

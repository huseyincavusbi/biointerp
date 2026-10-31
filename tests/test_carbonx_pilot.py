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

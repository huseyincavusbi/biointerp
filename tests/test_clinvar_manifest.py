import pandas as pd

from biointerp.experiments import clinvar_manifest as manifest


def _row(chrom, pos, label, ref=None, alt=None):
    ref = ref or "ACGTACGT"
    if alt is None:
        ref_base = ref[3]
        alt_base = "A" if ref_base != "A" else "C"
        alt = ref[:3] + alt_base + ref[4:]
    return {
        "chromosome": str(chrom),
        "position": pos,
        "ref": ref[3],
        "alt": alt[3],
        "label": label,
        "ref_forward_sequence": ref,
        "alt_forward_sequence": alt,
    }


def test_build_manifest_deduplicates_and_removes_test_overlap():
    train = pd.DataFrame(
        [
            _row(1, 10, 1),
            _row(1, 10, 1),
            _row(1, 20, 0),
            _row(2, 30, 1),
            _row(2, 40, 0),
        ]
    )
    test = pd.DataFrame([_row(1, 20, 0)])
    cfg = {"ref_column": "ref_forward_sequence"}

    result = manifest.build_manifest(train, test, cfg, n=4, test_frac=0.5, seed=0)

    assert result["counts"]["train_raw"] == 5
    assert result["counts"]["train_valid_unique"] == 3
    assert result["counts"]["excluded_test_overlap"] == 1
    assert result["counts"]["sampled"] == 3
    assert {row["chrom"] for row in result["variants"]} == {"1", "2"}
    assert result["split"]["train"] + result["split"]["test"] == 3


def test_build_manifest_rejects_non_snv():
    ref = "ACGTACGT"
    train = pd.DataFrame(
        [
            _row(1, 10, 1, ref=ref),
            _row(1, 20, 1, ref=ref, alt=ref),
        ]
    )
    test = pd.DataFrame([_row(1, 99, 0)])
    result = manifest.build_manifest(
        train, test, {"ref_column": "ref_forward_sequence"}, 10, 0.2, 0
    )
    assert result["counts"]["train_valid_unique"] == 1

import numpy as np

from biointerp.replication import (
    analyze,
    extract_features,
    layer_profile,
    load_reference,
    load_split,
    neuron_layer_histogram,
)


class FakeAdapter:
    name = "fake"
    n_layers = 2
    intermediate = 3

    def extract(self, sequence: str, position: int):
        rng = np.random.default_rng(position)
        features = rng.normal(size=self.n_layers * self.intermediate)
        probabilities = np.zeros(20)
        probabilities[0] = 0.9
        probabilities[1] = 0.1
        return features, probabilities

    def decode(self, probabilities: np.ndarray) -> str:
        return "A" if int(np.argmax(probabilities)) == 0 else "C"


def test_layer_profile_finds_signal_layer():
    rng = np.random.default_rng(0)
    n, layers, width = 200, 2, 5
    y = rng.integers(0, 2, n)
    x = rng.normal(size=(n, layers * width))
    x[:, width:] += y[:, None] * 2.0
    train = np.ones(n, dtype=bool)

    profile = layer_profile(x, y, train, n_layers=layers, intermediate=width)

    assert profile.shape == (layers,)
    assert profile[1] > profile[0]
    assert profile[0] < 0.7


def test_neuron_layer_histogram_maps_indices_to_depth():
    histogram = neuron_layer_histogram(np.array([0, 3, 5]), n_layers=3, intermediate=2, bins=10)
    assert histogram.sum() == 1.0
    assert histogram[0] > 0
    assert histogram[5] > 0
    assert histogram[9] > 0


def test_load_split_reads_assignments(tmp_path):
    path = tmp_path / "assignments.tsv"
    path.write_text(
        "id\tcluster\tsplit\nseq0\tc0\ttrain\nseq1\tc1\ttest\nseq2\tc2\texcluded_assay\n"
    )
    keep, is_test = load_split(str(path), ["seq0", "seq1", "seq2"])
    assert keep.tolist() == [True, True, False]
    assert is_test.tolist() == [False, True, False]


def test_extract_features_uses_reference_positions():
    reference = {
        "position": np.array([2, 4]),
        "true_token": np.array(["A", "A"]),
    }
    extracted = extract_features(reference, ["ACDEFG", "ACDEFG"], FakeAdapter())
    assert extracted["cett"].shape == (2, 6)
    assert extracted["correct"].tolist() == [True, True]
    assert extracted["confidence"].tolist() == [0.9, 0.9]


def test_analyze_returns_both_probes_and_comparison():
    rng = np.random.default_rng(0)
    n = 60
    reference = {
        "cett": rng.normal(size=(n, 6)),
        "correct": np.arange(n) % 2 == 0,
        "protein": np.arange(n),
        "n_layers": 2,
        "intermediate": 3,
        "model": "reference",
    }
    extracted = {
        "cett": rng.normal(size=(n, 6)),
        "correct": np.arange(n) % 2 == 1,
        "confidence": np.full(n, 0.5),
        "pred_aa": np.array(["A"] * n),
    }
    keep = np.ones(n, dtype=bool)
    is_test = np.arange(n) >= 40

    result = analyze(
        reference,
        extracted,
        keep,
        is_test,
        FakeAdapter(),
        top_k=2,
        bootstrap=50,
        seed=0,
    )

    assert 0.0 <= result["adapter"]["auroc_probe"] <= 1.0
    assert 0.0 <= result["reference"]["auroc_probe"] <= 1.0
    assert "score_spearman" in result["comparison"]
    assert result["comparison"]["unit_overlap"] is None


def test_load_reference_truncates(tmp_path):
    path = tmp_path / "reference.npz"
    n = 10
    np.savez(
        path,
        cett=np.zeros((n, 6), dtype=np.float16),
        correct=np.arange(n) % 2 == 0,
        position=np.arange(n) + 1,
        true_aa=np.array(["A"] * n),
        protein=np.arange(n),
        n_layers=2,
        intermediate_dim=3,
        model="fake",
    )
    reference = load_reference(str(path), n=5)
    assert len(reference["position"]) == 5
    assert reference["cett"].shape == (5, 6)

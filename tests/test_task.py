from _fake_models import MASK_ID, FakeEsmTokenizer

from biointerp import task
from biointerp.task import _read_fasta, fetch_swissprot, make_masked_batch


def test_read_fasta_multiline(tmp_path):
    path = tmp_path / "seqs.fasta"
    path.write_text(">a\nACDE\nFGHI\n\n>b\nKLMN\n")
    assert _read_fasta(path) == ["ACDEFGHI", "KLMN"]


def test_masked_batch_shapes_and_alignment():
    tokenizer = FakeEsmTokenizer()
    sequences = ["ACDEFGHIKL", "ACDEFG"]
    tokens, positions, true_aa = make_masked_batch(sequences, tokenizer, seed=0)

    assert tokens["input_ids"].shape == (2, 12)
    assert tokens["attention_mask"].shape == (2, 12)
    assert positions.shape == (2, 1)
    assert len(true_aa) == 2

    for i, seq in enumerate(sequences):
        pos = int(positions[i, 0])
        assert tokens["input_ids"][i, pos] == MASK_ID
        assert true_aa[i][0] == seq[pos - 1]
        assert int(tokens["attention_mask"][i].sum()) == len(seq) + 2
        if len(seq) < len(sequences[0]):
            assert int(tokens["attention_mask"][i, -1]) == 0


def test_masked_batch_is_seed_deterministic():
    tokenizer = FakeEsmTokenizer()
    seq = "ACDEFGHIKLMNPQRSTVWY" * 2
    first = make_masked_batch([seq], tokenizer, seed=3)
    second = make_masked_batch([seq], tokenizer, seed=3)
    assert first[0]["input_ids"].equal(second[0]["input_ids"])
    assert first[1].equal(second[1])
    assert first[2] == second[2]

    other = make_masked_batch([seq], tokenizer, masks_per_seq=3, seed=4)
    first_three = make_masked_batch([seq], tokenizer, masks_per_seq=3, seed=3)
    assert not first_three[1].equal(other[1])


def test_masked_batch_multiple_masks_per_sequence():
    tokenizer = FakeEsmTokenizer()
    tokens, positions, true_aa = make_masked_batch(
        ["ACDEFGHIKLMNPQRSTVWY"], tokenizer, masks_per_seq=3, seed=1
    )
    assert positions.shape == (1, 3)
    assert len(set(positions[0].tolist())) == 3
    assert positions[0].tolist() == sorted(positions[0].tolist())
    for pos, aa in zip(positions[0].tolist(), true_aa[0]):
        assert tokens["input_ids"][0, pos] == MASK_ID
        assert aa == "ACDEFGHIKLMNPQRSTVWY"[pos - 1]


def test_masked_batch_clamps_masks_to_sequence_length():
    tokenizer = FakeEsmTokenizer()
    _, positions, true_aa = make_masked_batch(["ACD"], tokenizer, masks_per_seq=10, seed=0)
    assert positions.shape == (1, 1)
    assert len(true_aa[0]) == 1


class _FakeResponse:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)


def test_fetch_swissprot_streams_and_reservoirs(monkeypatch, tmp_path):
    monkeypatch.setattr(task, "DATA_DIR", tmp_path)
    lines = [">a", "MKTAA", ">b", "MKTAB", ">c", "MKTAC"]
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return _FakeResponse(lines)

    monkeypatch.setattr("requests.get", fake_get)
    seqs = fetch_swissprot(2, min_len=5, max_len=5, seed=1, pool=3, cache=False)

    assert len(seqs) == 2
    assert set(seqs) <= {"MKTAA", "MKTAB", "MKTAC"}
    assert len(calls) == 1
    assert not list(tmp_path.iterdir())


def test_fetch_swissprot_uses_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(task, "DATA_DIR", tmp_path)
    cache = tmp_path / "swissprot_n2_len5-10_seed3.fasta"
    cache.write_text(">a\nMKTAA\n>b\nMKTAB\n")

    def fail_get(*args, **kwargs):
        raise AssertionError("network should not be used when cache exists")

    monkeypatch.setattr("requests.get", fail_get)
    assert fetch_swissprot(2, min_len=5, max_len=10, seed=3) == ["MKTAA", "MKTAB"]


def test_fetch_swissprot_returns_empty_when_nothing_eligible(monkeypatch, tmp_path):
    monkeypatch.setattr(task, "DATA_DIR", tmp_path)
    monkeypatch.setattr("requests.get", lambda *a, **k: _FakeResponse([">a", "MKTAA"]))
    assert fetch_swissprot(1, min_len=99, max_len=100, seed=0, cache=False) == []

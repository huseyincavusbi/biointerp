import random

from biointerp.experiments.carbonx_perturb import perturb_sequence


def test_perturb_sequence_preserves_variant_position():
    sequence = "AACCGGTTACGA"
    result = perturb_sequence(sequence, 0, 6, "random", random.Random(0), keep={3})
    assert result is not None
    assert result[3] == sequence[3]
    assert result != sequence


def test_gc_matched_preserves_gc_count():
    sequence = "AACCGGTTACGA"
    result = perturb_sequence(sequence, 0, len(sequence), "gc_matched", random.Random(0))
    assert result is not None
    assert sum(base in "GC" for base in result) == sum(base in "GC" for base in sequence)


def test_shuffle_preserves_composition():
    sequence = "AACCGGTTACGA"
    result = perturb_sequence(sequence, 0, len(sequence), "shuffle", random.Random(1))
    assert result is not None
    assert sorted(result) == sorted(sequence)

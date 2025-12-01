import numpy as np
from tx.long_sequence import get_long_training_sequence


def test_output_is_complex_and_nonzero():
    """Provjerava da generisana LTF sekvenca nije nula i da je kompleksna."""
    seq = get_long_training_sequence(step=1)
    assert np.iscomplexobj(seq)
    assert np.any(np.abs(seq) > 1e-12)


def test_length_step_1():
    """Provjerava da LTF za step=1 ima tačan broj uzoraka."""
    seq = get_long_training_sequence(step=1)
    assert len(seq) == 96


def test_length_step_half():
    """Provjerava da LTF za step=0.5 ima tačan broj uzoraka."""
    seq = get_long_training_sequence(step=0.5)
    assert len(seq) == 192


def test_cyclic_prefix_correct():
    """Provjerava da je CP pravilno postavljen."""
    seq = get_long_training_sequence(step=1)

    cp = seq[:32]
    symbol = seq[32:96]

    # CP mora biti posljednjih 32 uzorka glavnog simbola
    assert np.allclose(cp, symbol[-32:])


def test_energy_positive():
    """Provjerava da LTF ima pozitivnu energiju."""
    seq = get_long_training_sequence(step=1)
    symbol = seq[32:96]

    energy = np.sum(np.abs(symbol)**2)

    assert energy > 0
    assert not np.isnan(energy)

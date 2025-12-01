import numpy as np
from tx.short_sequence import get_short_training_sequence


def test_length_step_1():
    """Provjerava da STS ima dužinu 160 uzoraka za step=1."""
    seq = get_short_training_sequence(step=1)
    assert len(seq) == 160


def test_length_step_half():
    """Provjerava da STS ima dužinu 320 uzoraka za step=0.5."""
    seq = get_short_training_sequence(step=0.5)
    assert len(seq) == 320


def test_energy_positive():
    """Provjerava da STS ima nenultu energiju."""
    seq = get_short_training_sequence(step=1)
    assert np.sum(np.abs(seq)**2) > 0


def test_first_sample_idft_definition():
    """Provjerava da prvi STS uzorak odgovara IDFT definiciji."""
    # Ponovimo ručno prvi sample (t = 0)
    Positive = np.array([
        0,0,0,0,   -1+1j,0,0,0,   -1-1j,0,0,0,   1+1j,0,0,0,
        1+1j,0,0,0, 1+1j,0,0,0,  -1-1j,0,0,0,  -1+1j,0,0,0
    ], dtype=complex)

    Negative = np.array([
        0,0,0,0,   0,0,0,0,   1+1j,0,0,0,   -1-1j,0,0,0,
        1+1j,0,0,0,  -1+1j,0,0,0,  1+1j,0,0,0,  -1-1j,0,0,0
    ], dtype=complex)

    Total = np.sqrt(13/6) * np.concatenate((Negative, Positive))

    m = np.arange(-32, 32)
    E0 = np.exp(1j * 2 * np.pi * 0 * m / 64)  # = 1

    expected = np.dot(Total, E0)

    seq = get_short_training_sequence(step=1)

    assert np.isclose(seq[0], expected)


def test_periodicity_matches_16_sample_block():
    """Provjerava da STS ima period 16 uzoraka kao u literaturi."""
    seq = get_short_training_sequence(step=1)

    # 160 uzoraka → 10 ponavljanja bloka od 16 uzoraka
    block = seq[:16]

    for i in range(1, 10):
        assert np.allclose(seq[i*16:(i+1)*16], block)

import numpy as np
from tx.short_sequence import get_short_training_sequence
import pytest

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
        0,0,0,0,   -1-1j,0,0,0,   -1-1j,0,0,0,   1+1j,0,0,0,
        1+1j,0,0,0, 1+1j,0,0,0,  1+1j,0,0,0,  0,0,0,0
    ], dtype=complex)

    Negative = np.array([
        0,0,0,0,   0,0,0,0,   1+1j,0,0,0,   -1-1j,0,0,0,
        1+1j,0,0,0,  -1-1j,0,0,0,  -1-1j,0,0,0,  1+1j,0,0,0
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

def test_return_type():
    """Provjerava da funkcija vraća numpy.ndarray."""
    sts = get_short_training_sequence()
    assert isinstance(sts, np.ndarray), "Vraćeni tip nije numpy.ndarray"

def test_return_complex_dtype():
    """Provjerava da je dtype kompleksni."""
    sts = get_short_training_sequence()
    assert np.issubdtype(sts.dtype, np.complexfloating), "Vraćeni niz nije kompleksnog tipa"

def test_length_default_step():
    """Provjerava dužinu STS za step=1."""
    step = 1
    sts = get_short_training_sequence(step=step)
    expected_length = int(160 / step)
    assert len(sts) == expected_length, f"Dužina nije {expected_length}, nego {len(sts)}"

def test_length_half_step():
    """Provjerava dužinu STS za step=0.5."""
    step = 0.5
    sts = get_short_training_sequence(step=step)
    expected_length = int(160 / step)
    assert len(sts) == expected_length, f"Dužina za step={step} nije {expected_length}, nego {len(sts)}"

def test_output_values_stability():
    """Provjerava da funkcija vraća iste vrijednosti pri ponovnom pozivu."""
    sts1 = get_short_training_sequence()
    sts2 = get_short_training_sequence()
    assert np.allclose(sts1, sts2), "Sekvence nisu iste pri ponovnom pozivu"

def test_nonzero_content():
    """Provjerava da sekvenca nije sve nula."""
    sts = get_short_training_sequence()
    assert not np.allclose(sts, 0), "Sekvenca je sva nula"

def test_magnitude_consistency():
    """Provjerava da su magnitudni uzorci realni i nenegativni (ne-potpuno, ali osnovna provjera)."""
    sts = get_short_training_sequence()
    magnitudes = np.abs(sts)
    assert np.all(magnitudes >= 0), "Magnituda STS sadrži negativne vrijednosti"

def test_step_zero_raises():
    """Provjerava da step=0 podiže grešku (dieljenje nulom)."""
    with pytest.raises(ValueError):
        get_short_training_sequence(step=0)

def test_step_negative_raises():
    """Provjerava da negativan step podiže grešku."""
    with pytest.raises(ValueError):
        get_short_training_sequence(step=-1)

def test_step_non_numeric_raises():
    """Provjerava da nenumerički step podiže grešku."""
    with pytest.raises(TypeError):
        get_short_training_sequence(step="abc")

def test_very_small_step():
    """Provjerava da veoma mali step ne baca grešku i vraća niz veće dužine."""
    step = 0.01
    seq = get_short_training_sequence(step=step)
    expected_length = int(160 / step)
    assert len(seq) == expected_length
    assert np.any(np.abs(seq) > 0), "Sekvenca nije nenulta"

def test_large_step_truncates_length():
    """Provjerava da veoma veliki step vraća niz kraći od 1 LTS bloka."""
    step = 500
    seq = get_short_training_sequence(step=step)
    expected_length = int(160 / step)
    assert len(seq) == expected_length

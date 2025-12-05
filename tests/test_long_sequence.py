import numpy as np
from tx.long_sequence import get_long_training_sequence

def test_return_type():
    """Provjerava da funkcija vraća numpy.ndarray."""
    lts = get_long_training_sequence()
    assert isinstance(lts, np.ndarray), "Vraćeni tip nije numpy.ndarray"

def test_return_complex_dtype():
    """Provjerava da je dtype kompleksni."""
    lts = get_long_training_sequence()
    assert np.issubdtype(lts.dtype, np.complexfloating), "Vraćeni niz nije kompleksnog tipa"

def test_length_default_step():
    """Provjerava da je dužina sekvence ispravna za step=1."""
    lts = get_long_training_sequence()
    expected_length = 32 + 2*64  # CP od 32 + double LTS
    assert len(lts) == expected_length, f"Dužina nije {expected_length}, nego {len(lts)}"

def test_length_step_half():
    """Provjerava dužinu za step=0.5"""
    lts = get_long_training_sequence(step=0.5)
    expected_length = 64 + 2*(64//0.5)  
    assert len(lts) == expected_length, f"Dužina nije {expected_length}, nego {len(lts)}"

def test_output_values_stability():
    """Provjerava da funkcija vraća iste vrijednosti pri ponovnom pozivu."""
    lts1 = get_long_training_sequence()
    lts2 = get_long_training_sequence()
    assert np.allclose(lts1, lts2), "Sekvence nisu iste pri ponovnom pozivu"

def test_double_lts_content():
    """Provjerava da druga polovina sekvence odgovara prvoj LTS sekvenci."""
    lts = get_long_training_sequence()
    first_lts = lts[32:32+64]  # prvi LTS bez CP
    second_lts = lts[32+64:32+128]  # druga LTS
    assert np.allclose(first_lts, second_lts), "Druge LTS sekvence se ne poklapaju s prvom"

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

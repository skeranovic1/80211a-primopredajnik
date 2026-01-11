import numpy as np
import pytest
from rx.long_symbol_correlator import long_symbol_correlator

def test_raises_type_error_for_non_array_like_lts():
    """Provjerava da TypeError nastaje ako long_training_symbol nije array-like."""
    rx = np.ones(100, dtype=complex)
    with pytest.raises(TypeError):
        long_symbol_correlator(long_training_symbol=5, rx_waveform=rx, falling_edge_position=0)

def test_raises_type_error_for_non_array_like_rx():
    """Provjerava da TypeError nastaje ako rx_waveform nije array-like."""
    lts = np.ones(64, dtype=complex)
    with pytest.raises(TypeError):
        long_symbol_correlator(long_training_symbol=lts, rx_waveform=5, falling_edge_position=0)

def test_raises_value_error_for_short_lts():
    """Provjerava da ValueError nastaje ako long_training_symbol ima manje od 64 uzorka."""
    lts = np.ones(32, dtype=complex)
    rx = np.ones(100, dtype=complex)
    with pytest.raises(ValueError):
        long_symbol_correlator(long_training_symbol=lts, rx_waveform=rx, falling_edge_position=0)

def test_raises_value_error_for_short_rx():
    """Provjerava da ValueError nastaje ako rx_waveform ima manje od 64 uzorka."""
    lts = np.ones(64, dtype=complex)
    rx = np.ones(32, dtype=complex)
    with pytest.raises(ValueError):
        long_symbol_correlator(long_training_symbol=lts, rx_waveform=rx, falling_edge_position=0)

def test_raises_value_error_for_invalid_falling_edge():
    """Provjerava da ValueError nastaje ako falling_edge_position nije integer ili je izvan opsega."""
    lts = np.ones(64, dtype=complex)
    rx = np.ones(100, dtype=complex)
    # float
    with pytest.raises(ValueError):
        long_symbol_correlator(lts, rx, falling_edge_position=5.5)
    # negativno
    with pytest.raises(ValueError):
        long_symbol_correlator(lts, rx, falling_edge_position=-1)
    # preveliko
    with pytest.raises(ValueError):
        long_symbol_correlator(lts, rx, falling_edge_position=100)

def test_output_shapes_and_types():
    """Provjerava da funkcija vraća tri izlaza ispravnog tipa i dužine."""
    lts = np.exp(1j * 2 * np.pi * np.arange(64) / 64)
    rx = np.concatenate([np.zeros(50), lts, np.zeros(50)])
    peak_val, peak_pos, output_long = long_symbol_correlator(lts, rx, falling_edge_position=50)

    assert isinstance(peak_val, complex)
    assert isinstance(peak_pos, int)
    assert isinstance(output_long, np.ndarray)
    assert output_long.shape == rx.shape

def test_peak_detection_correctness():
    """Provjerava da je peak cross-korelacije detektovan blizu stvarnog LTS."""
    lts = np.exp(1j * 2 * np.pi * np.arange(64) / 64)
    rx = np.concatenate([np.zeros(50), lts, np.zeros(50)])
    peak_val, peak_pos, _ = long_symbol_correlator(lts, rx, falling_edge_position=50)

    # Pozicija peak-a treba biti unutar stvarnog LTS signala
    assert 50 <= peak_pos <= 50 + 64

def test_output_long_has_nonzero_values():
    """Provjerava da niz output_long sadrži nenula vrijednosti u LTS dijelu."""
    lts = np.exp(1j * 2 * np.pi * np.arange(64) / 64)
    rx = np.concatenate([np.zeros(50), lts, np.zeros(50)])
    _, _, output_long = long_symbol_correlator(lts, rx, falling_edge_position=50)

    # Output u LTS dijelu mora biti nenula
    assert np.any(np.abs(output_long[50:50+64]) > 0)

def test_handles_shifted_lts():
    """Provjerava da korelator ispravno detektuje LTS čak i ako je pomjeren unutar pretražnog prozora."""
    lts = np.exp(1j * 2 * np.pi * np.arange(64) / 64)
    rx = np.concatenate([np.zeros(50), np.zeros(10), lts, np.zeros(40)])
    peak_val, peak_pos, _ = long_symbol_correlator(lts, rx, falling_edge_position=50)

    assert 60 <= peak_pos <= 60 + 64
    assert abs(peak_val) > 0

def test_all_zero_lts_correlation():
    """Provjerava da funkcija radi i kada je RX signal samo nule, peak je nula."""
    lts = np.exp(1j * 2 * np.pi * np.arange(64) / 64)
    rx = np.zeros(100, dtype=complex)
    peak_val, peak_pos, output_long = long_symbol_correlator(lts, rx, falling_edge_position=0)

    assert peak_val == 0 + 0j
    assert np.all(output_long == 0)
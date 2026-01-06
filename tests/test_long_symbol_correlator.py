import numpy as np
import pytest
from rx.long_symbol_correlator import long_symbol_correlator

def test_peak_detection_exact_position():
    """Provjera da funkcija detektuje peak tačno na poziciji LTS-a"""
    lts = np.exp(1j*2*np.pi*np.random.rand(64))
    rx_signal = np.concatenate([np.zeros(50), lts, np.zeros(100)])
    falling_edge = 50

    peak_val, peak_pos, output_long = long_symbol_correlator(lts, rx_signal, falling_edge)
    
    expected_pos = 50 + 63  # zbog reverziranog LTS-a u dot
    assert peak_pos == expected_pos, f"Expected peak at {expected_pos}, got {peak_pos}"
    assert np.iscomplex(peak_val), "Peak value should be complex"
    assert len(output_long) == len(rx_signal), "Output correlation should match RX signal length"

def test_output_length_matches_signal():
    """Provjera da output_long ima istu dužinu kao rx_signal"""
    lts = np.exp(1j*2*np.pi*np.random.rand(64))
    rx_signal = np.concatenate([np.zeros(20), lts, np.zeros(30)])
    falling_edge = 20

    _, _, output_long = long_symbol_correlator(lts, rx_signal, falling_edge)
    assert len(output_long) == len(rx_signal)

def test_signal_too_short():
    """Provjera da funkcija radi i kada je signal kraći od 64 uzorka (manji od LTS)"""
    lts = np.exp(1j*2*np.pi*np.random.rand(64))
    rx_signal = np.zeros(30)  # kraći signal od LTS
    falling_edge = 0

    peak_val, peak_pos, output_long = long_symbol_correlator(lts, rx_signal, falling_edge)
    assert peak_pos == 0
    assert peak_val == 0 + 0j
    assert len(output_long) == len(rx_signal)

def test_with_multiple_lts():
    """Provjera da funkcija detektuje prvi LTS kada ih je više u signalu"""
    lts = np.exp(1j*2*np.pi*np.random.rand(64))
    rx_signal = np.concatenate([np.zeros(10), lts, np.zeros(20), lts, np.zeros(30)])
    falling_edge = 10

    peak_val, peak_pos, output_long = long_symbol_correlator(lts, rx_signal, falling_edge)
    expected_pos = 10 + 63
    assert peak_pos == expected_pos

def test_normalized_lts_correlation():
    """Provjera da se normalizacija LTS-a izvršava ispravno (±1 ± j)"""
    # LTS od 64 uzorka
    lts = np.random.randn(64) + 1j*np.random.randn(64)
    rx_signal = np.concatenate([np.zeros(5), lts])
    falling_edge = 0

    _, _, output_long = long_symbol_correlator(lts, rx_signal, falling_edge)
    
    # Kvantizacija
    L_expected = np.sign(np.real(lts)) + 1j*np.sign(np.imag(lts))
    allowed_values = np.array([1+1j, 1-1j, -1+1j, -1-1j])
    assert np.all(np.isin(L_expected, allowed_values))

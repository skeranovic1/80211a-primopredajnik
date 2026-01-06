import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from rx.estimacija_kanala import channel_estimate_and_equalizer

def generate_synthetic_signal(length=128):
    """Generiše kompleksan signal od zadate dužine."""
    np.random.seed(0)
    return np.exp(1j*2*np.pi*np.random.rand(length))

def test_channel_estimate_basic_shapes():
    """Provjera da funkcija vraća nizove odgovarajuće veličine."""
    signal = generate_synthetic_signal(200)
    lts_start = 10
    ch_est, eq_coeff = channel_estimate_and_equalizer(signal, lts_start)
    assert ch_est.shape == (64,)
    assert eq_coeff.shape == (64,)

def test_channel_estimate_dovoljna_duzina():
    """Provjera da funkcija ne pada za dovoljno dug signal."""
    signal = generate_synthetic_signal(200)
    lts_start = 50
    ch_est, eq_coeff = channel_estimate_and_equalizer(signal, lts_start)
    # Samo provjeravamo da funkcija izvršava
    assert isinstance(ch_est, np.ndarray)
    assert isinstance(eq_coeff, np.ndarray)

def test_channel_estimate_signal_with_zeros():
    """Signal sa nulama ne smije izazvati pad funkcije."""
    signal = np.zeros(200, dtype=np.complex128)
    lts_start = 10
    ch_est, eq_coeff = channel_estimate_and_equalizer(signal, lts_start)
    assert ch_est.shape == (64,)
    assert eq_coeff.shape == (64,)

def test_channel_estimate_signal_with_nan_inf():
    """Signal sa NaN i Inf ne smije izazvati pad funkcije."""
    signal = np.zeros(200, dtype=np.complex128)
    signal[50] = np.nan + 1j*np.nan
    signal[100] = np.inf + 1j*np.inf
    lts_start = 10
    ch_est, eq_coeff = channel_estimate_and_equalizer(signal, lts_start)
    assert ch_est.shape == (64,)
    assert eq_coeff.shape == (64,)

def test_channel_estimate_short_signal_raises():
    """Signal prekratak za LTS treba podići grešku."""
    signal = np.zeros(50)
    lts_start = 0
    with pytest.raises(ValueError):
        channel_estimate_and_equalizer(signal, lts_start)

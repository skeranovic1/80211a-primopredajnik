import numpy as np
import pytest
from rx.estimacija_kanala import channel_estimate_and_equalizer

def test_raises_type_error_for_non_numeric_lts_start():
    """Provjerava da TypeError nastaje ako lts_start nije broj."""
    lts_signal = np.ones(200, dtype=complex)
    with pytest.raises(TypeError):
        channel_estimate_and_equalizer(lts_signal, lts_start="string")

def test_accepts_integer_lts_start():
    """Provjerava da funkcija prihvata integer lts_start i vraća izlaz ispravno."""
    lts_signal = np.ones(200, dtype=complex)
    H, eq = channel_estimate_and_equalizer(lts_signal, lts_start=5)
    assert H.shape == (64,)
    assert eq.shape == (64,)

def test_rejects_float_lts_start():
    """Provjerava da float lts_start baca TypeError."""
    lts_signal = np.ones(200, dtype=complex)
    with pytest.raises(TypeError):
        channel_estimate_and_equalizer(lts_signal, lts_start=5.7)

def test_raises_value_error_for_too_small_signal():
    """Provjerava da ValueError nastaje ako lts_signal ima manje od 128 uzoraka."""
    lts_signal = np.ones(100, dtype=complex)
    with pytest.raises(ValueError):
        channel_estimate_and_equalizer(lts_signal)

def test_raises_value_error_for_invalid_lts_start_index():
    """Provjerava da ValueError nastaje ako je lts_start preblizu kraja signala."""
    lts_signal = np.ones(200, dtype=complex)
    with pytest.raises(ValueError):
        channel_estimate_and_equalizer(lts_signal, lts_start=150)

def test_channel_estimate_and_equalizer_shapes():
    """Happy path: provjerava da funkcija vraća H i equalizer ispravnog oblika za validan signal."""
    lts_signal = np.random.randn(200) + 1j * np.random.randn(200)
    H, eq = channel_estimate_and_equalizer(lts_signal, lts_start=10)
    assert H.shape == (64,)
    assert eq.shape == (64,)
    assert np.iscomplexobj(H)
    assert np.iscomplexobj(eq)

def test_raises_type_error_for_real_lts_signal():
    """Provjerava da lts_signal mora biti kompleksan."""
    lts_signal = np.ones(200, dtype=float)
    with pytest.raises(TypeError):
        channel_estimate_and_equalizer(lts_signal, lts_start=0)

def test_negative_lts_start_raises_value_error():
    """Provjerava da negativni lts_start baca ValueError."""
    lts_signal = np.ones(200, dtype=complex)
    with pytest.raises(ValueError):
        channel_estimate_and_equalizer(lts_signal, lts_start=-1)

def test_channel_estimate_and_equalizer_raises_type_error_for_non_ndarray():
    """Provjerava da funkcija baca TypeError ako lts_signal nije np.ndarray."""
    lts_signal = [1+1j] * 128

    with pytest.raises(TypeError, match="lts_signal mora biti np.ndarray"):
        channel_estimate_and_equalizer(lts_signal)
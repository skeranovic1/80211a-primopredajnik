import numpy as np
import pytest
from rx.pretprocessing import iq_preprocessing

def test_raises_type_error_for_invalid_fs_type():
    """Provjerava da se baca TypeError ako fs nije numeričkog tipa."""
    rx = np.ones(100)
    tx = np.ones(100)
    with pytest.raises(TypeError):
        iq_preprocessing(rx, tx, fs="1000")

def test_raises_value_error_for_non_positive_fs():
    """Provjerava da se baca ValueError ako je fs manji ili jednak nuli."""
    rx = np.ones(100)
    tx = np.ones(100)
    with pytest.raises(ValueError):
        iq_preprocessing(rx, tx, fs=0)

def test_raises_type_error_for_non_array_like_signals():
    """Provjerava da se baca TypeError ako RX ili TX signal nisu array-like."""
    with pytest.raises(TypeError):
        iq_preprocessing(rx_signal=5, tx_signal=10, fs=1e6)

def test_raises_value_error_for_empty_rx_signal():
    """Provjerava da se baca ValueError ako je RX signal prazan."""
    rx = []
    tx = np.ones(100)
    with pytest.raises(ValueError):
        iq_preprocessing(rx, tx, fs=1e6)

def test_raises_value_error_for_empty_tx_signal():
    """Provjerava da se baca ValueError ako je TX signal prazan."""
    rx = np.ones(100)
    tx = []
    with pytest.raises(ValueError):
        iq_preprocessing(rx, tx, fs=1e6)

def test_output_types_and_shapes():
    """Provjerava da funkcija vraća numpy niz i skalarnu novu frekvenciju."""
    rx = np.random.randn(200) + 1j * np.random.randn(200)
    tx = np.random.randn(200) + 1j * np.random.randn(200)

    rx_out, fs_out = iq_preprocessing(rx, tx, fs=1e6)

    assert isinstance(rx_out, np.ndarray)
    assert isinstance(fs_out, float)
    assert rx_out.ndim == 1

def test_downsampling_by_factor_two():
    """Provjerava da se RX signal decimira faktorom 2 (sa tolerancijom)."""
    rx = np.arange(100, dtype=float)
    tx = np.ones(100, dtype=float)

    rx_out, _ = iq_preprocessing(rx, tx, fs=1e6)

    assert len(rx_out) == 50
    np.testing.assert_allclose(rx_out,rx.flatten()[::2] * np.sqrt(np.mean(tx**2)) / np.sqrt(np.mean(rx**2)),rtol=1e-6, atol=1e-12)

def test_sampling_frequency_is_halved():
    """Provjerava da se frekvencija uzorkovanja prepolovi."""
    rx = np.ones(100)
    tx = np.ones(100)
    fs = 2e6

    _, fs_out = iq_preprocessing(rx, tx, fs)

    assert fs_out == fs / 2

def test_power_normalization_matches_tx_signal():
    """Provjerava da se snaga RX signala normalizira na snagu TX signala."""
    rx = np.random.randn(1000) + 1j * np.random.randn(1000)
    tx = 2 * (np.random.randn(1000) + 1j * np.random.randn(1000))

    rx_out, _ = iq_preprocessing(rx, tx, fs=1e6)

    rx_power = np.mean(np.abs(rx_out)**2)
    tx_power = np.mean(np.abs(tx)**2)

    np.testing.assert_allclose(rx_power, tx_power, rtol=1)

def test_accepts_real_valued_signals():
    """Provjerava da funkcija ispravno radi i sa realnim signalima."""
    rx = np.random.randn(200)
    tx = np.random.randn(200)

    rx_out, fs_out = iq_preprocessing(rx, tx, fs=1e6)

    assert rx_out.size == 100
    assert fs_out == 5e5

def test_no_nan_or_inf_in_output():
    """Provjerava da u izlazu nema NaN ili Inf vrijednosti."""
    rx = np.random.randn(500) + 1j * np.random.randn(500)
    tx = np.random.randn(500) + 1j * np.random.randn(500)

    rx_out, _ = iq_preprocessing(rx, tx, fs=1e6)

    assert not np.any(np.isnan(rx_out))
    assert not np.any(np.isinf(rx_out))
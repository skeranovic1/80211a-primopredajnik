import numpy as np
import pytest
from rx.pretprocessing import iq_preprocessing


def generate_valid_signals(length=1000):
    rx = np.random.randn(length) + 1j * np.random.randn(length)
    tx = np.random.randn(length) + 1j * np.random.randn(length)
    fs = 20e6
    return rx, tx, fs


def test_happy_path_basic():
    rx, tx, fs = generate_valid_signals()
    rx_out, fs_out = iq_preprocessing(rx, tx, fs)
    assert isinstance(rx_out, np.ndarray)
    assert fs_out == fs / 2
    assert len(rx_out) == len(rx) // 2


def test_rx_flattening():
    rx, tx, fs = generate_valid_signals()
    rx = rx.reshape((100, 10))
    rx_out, _ = iq_preprocessing(rx, tx, fs)
    assert rx_out.ndim == 1


def test_power_normalization_finite():
    rx, tx, fs = generate_valid_signals()
    rx_out, _ = iq_preprocessing(rx, tx, fs)
    rx_power = np.mean(np.abs(rx_out)**2)
    assert np.isfinite(rx_power)
    assert rx_power > 0


def test_decimation_factor_two():
    rx, tx, fs = generate_valid_signals(999)
    rx_out, fs_out = iq_preprocessing(rx, tx, fs)
    assert fs_out == fs / 2
    assert len(rx_out) == 500


def test_non_complex_inputs():
    rx = np.random.randn(1000)
    tx = np.random.randn(1000)
    fs = 10e6
    rx_out, fs_out = iq_preprocessing(rx, tx, fs)
    assert len(rx_out) == 500
    assert fs_out == fs / 2


def test_tx_signal_zero_power():
    rx = np.random.randn(1000) + 1j * np.random.randn(1000)
    tx = np.zeros(1000)
    fs = 10e6
    rx_out, _ = iq_preprocessing(rx, tx, fs)
    assert np.all(rx_out == 0)


def test_rx_signal_zero_power():
    rx = np.zeros(1000)
    tx = np.random.randn(1000)
    fs = 10e6
    with pytest.warns(RuntimeWarning):
        rx_out, _ = iq_preprocessing(rx, tx, fs)
    assert np.all(np.isnan(rx_out))


def test_mismatched_lengths():
    rx = np.random.randn(1000)
    tx = np.random.randn(500)
    fs = 10e6
    rx_out, fs_out = iq_preprocessing(rx, tx, fs)
    assert len(rx_out) == 500
    assert fs_out == fs / 2

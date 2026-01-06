import numpy as np
import pytest
from rx.PhaseCorrection_80211a import phase_correction_80211a



def generate_valid_inputs(num_symbols=1):
    rx_signal = np.random.randn(1000) + 1j * np.random.randn(1000)
    channel_est = np.ones(64, dtype=complex)
    equalizer_coeffs = np.ones(64, dtype=complex)
    ltpeak = 0
    return rx_signal, num_symbols, ltpeak, channel_est, equalizer_coeffs


def test_happy_path_single_symbol():
    rx_signal, num_symbols, ltpeak, channel_est, eq = generate_valid_inputs(1)
    result = phase_correction_80211a(rx_signal, num_symbols, ltpeak, channel_est, eq)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0].shape == (48,)


def test_happy_path_multiple_symbols():
    rx_signal, num_symbols, ltpeak, channel_est, eq = generate_valid_inputs(5)
    result = phase_correction_80211a(rx_signal, num_symbols, ltpeak, channel_est, eq)
    assert len(result) == 5
    for sym in result:
        assert sym.shape == (48,)


def test_max_ratio_zero_equal_weights():
    rx_signal, num_symbols, ltpeak, channel_est, eq = generate_valid_inputs(2)
    result = phase_correction_80211a(
        rx_signal,
        num_symbols,
        ltpeak,
        channel_est,
        eq,
        max_ratio=0
    )
    assert len(result) == 2


def test_custom_filter_length():
    rx_signal, num_symbols, ltpeak, channel_est, eq = generate_valid_inputs(3)
    result = phase_correction_80211a(
        rx_signal,
        num_symbols,
        ltpeak,
        channel_est,
        eq,
        L=16
    )
    assert len(result) == 3


def test_zero_symbols():
    rx_signal, num_symbols, ltpeak, channel_est, eq = generate_valid_inputs(0)
    result = phase_correction_80211a(rx_signal, 0, ltpeak, channel_est, eq)
    assert result == []


def test_invalid_channel_est_shape():
    rx_signal, num_symbols, ltpeak, channel_est, eq = generate_valid_inputs(1)
    with pytest.raises(IndexError):
        phase_correction_80211a(
            rx_signal,
            num_symbols,
            ltpeak,
            channel_est[:10],
            eq
        )


def test_invalid_equalizer_coeffs_shape():
    rx_signal, num_symbols, ltpeak, channel_est, eq = generate_valid_inputs(1)
    with pytest.raises(ValueError):
        phase_correction_80211a(
            rx_signal,
            num_symbols,
            ltpeak,
            channel_est,
            eq[:10]
        )


def test_rx_signal_too_short():
    rx_signal = np.random.randn(50) + 1j * np.random.randn(50)
    channel_est = np.ones(64, dtype=complex)
    equalizer_coeffs = np.ones(64, dtype=complex)
    with pytest.raises(ValueError):
        phase_correction_80211a(
            rx_signal,
            1,
            0,
            channel_est,
            equalizer_coeffs
        )


def test_non_complex_rx_signal():
    rx_signal = np.random.randn(1000)
    channel_est = np.ones(64, dtype=complex)
    equalizer_coeffs = np.ones(64, dtype=complex)
    result = phase_correction_80211a(
        rx_signal,
        1,
        0,
        channel_est,
        equalizer_coeffs
    )
    assert len(result) == 1


def test_output_is_independent_copies():
    rx_signal, num_symbols, ltpeak, channel_est, eq = generate_valid_inputs(2)
    result = phase_correction_80211a(rx_signal, num_symbols, ltpeak, channel_est, eq)
    assert not np.shares_memory(result[0], result[1])

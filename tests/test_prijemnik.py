import numpy as np
import pytest

from rx.prijemnik import run_rx


def test_run_rx_happy_path(monkeypatch):
    rx_40 = np.random.randn(4000) + 1j * np.random.randn(4000)
    tx_40 = np.random.randn(4000) + 1j * np.random.randn(4000)

    monkeypatch.setattr(
        "rx.prijemnik.iq_preprocessing",
        lambda rx, tx, fs: (rx[::2], 20e6),
    )
    monkeypatch.setattr(
        "rx.prijemnik.packet_detector",
        lambda rx: (None, None, 100, None),
    )
    monkeypatch.setattr(
        "rx.prijemnik.detect_frequency_offsets",
        lambda rx, idx, plot=False, fs=None: (100.0, 110.0),
    )
    monkeypatch.setattr(
        "rx.prijemnik.long_symbol_correlator",
        lambda ref, rx, fe: (None, 200, None),
    )
    monkeypatch.setattr(
        "rx.prijemnik.channel_estimate_and_equalizer",
        lambda rx, lts: (np.ones(64, complex), np.ones(64, complex)),
    )
    monkeypatch.setattr(
        "rx.prijemnik.phase_correction_80211a",
        lambda **kwargs: [np.zeros(48, complex)],
    )

    result = run_rx(rx_40, tx_40)

    assert isinstance(result, dict)
    assert result["fs"] == 20e6
    assert result["falling_edge"] == 100
    assert result["lt_peak_pos"] == 200
    assert result["lts_start"] == 137
    assert result["cfo_coarse_hz"] == 100.0
    assert result["cfo_fine_hz"] == 110.0
    assert result["corrected_symbols"][0].shape == (48,)


def test_run_rx_packet_not_detected(monkeypatch):
    rx_40 = np.random.randn(2000) + 1j * np.random.randn(2000)
    tx_40 = np.random.randn(2000) + 1j * np.random.randn(2000)

    monkeypatch.setattr(
        "rx.prijemnik.iq_preprocessing",
        lambda rx, tx, fs: (rx[::2], 20e6),
    )
    monkeypatch.setattr(
        "rx.prijemnik.packet_detector",
        lambda rx: (None, None, None, None),
    )

    with pytest.raises(RuntimeError):
        run_rx(rx_40, tx_40)


def test_run_rx_zero_payload_symbols(monkeypatch):
    rx_40 = np.random.randn(300) + 1j * np.random.randn(300)

    tx_40 = np.random.randn(1000) + 1j * np.random.randn(1000)

    monkeypatch.setattr(
        "rx.prijemnik.iq_preprocessing",
        lambda rx, tx, fs: (rx[::2], 20e6),
    )
    monkeypatch.setattr(
        "rx.prijemnik.packet_detector",
        lambda rx: (None, None, 50, None),
    )
    monkeypatch.setattr(
        "rx.prijemnik.detect_frequency_offsets",
        lambda rx, idx, plot=False, fs=None: (0.0, 0.0),
    )
    monkeypatch.setattr(
        "rx.prijemnik.long_symbol_correlator",
        lambda ref, rx, fe: (None, 60, None),
    )
    monkeypatch.setattr(
        "rx.prijemnik.channel_estimate_and_equalizer",
        lambda rx, lts: (np.ones(64, complex), np.ones(64, complex)),
    )
    monkeypatch.setattr(
        "rx.prijemnik.phase_correction_80211a",
        lambda **kwargs: [],
    )

    result = run_rx(rx_40, tx_40)

    assert result["max_symbols_in_buffer"] == 0
    assert result["corrected_symbols"] == []

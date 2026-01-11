import numpy as np
import pytest
from unittest.mock import patch
from rx.RX_802_11_a import Receiver80211a

def test_init_defaults():
    """
    Provjerava da li se Receiver80211a ispravno inicijalizira sa podrazumijevanim parametrima
    i da li su sva interna stanja postavljena na None.
    """
    rx = Receiver80211a(fs=20e6, num_symbols=10)

    assert rx.fs == 20e6
    assert rx.num_symbols == 10
    assert rx.nfft == 64
    assert rx.ncp == 16
    assert rx.nsym == 80

    assert rx.rx_signal_clean is None
    assert rx.lts_start is None
    assert rx.rx_fine is None
    assert rx.symbols_fd is None
    assert rx.channel_est is None
    assert rx.eq_coefficient is None
    assert rx.corrected_symbols is None

def test_rx_signal_clean_type_check():
    """
    Provjerava da setter rx_signal_clean prihvata samo numpy nizove i baca TypeError za neispravne tipove.
    """
    rx = Receiver80211a(20e6, 1)

    with pytest.raises(TypeError):
        rx.rx_signal_clean = "not an array"

    rx.rx_signal_clean = np.array([1, 2, 3])

def test_lts_start_validation():
    """
    Provjerava validaciju lts_start:
    - negativne vrijednosti nisu dozvoljene
    - validna vrijednost se pravilno postavlja
    """
    rx = Receiver80211a(20e6, 1)

    with pytest.raises(ValueError):
        rx.lts_start = -5

    rx.lts_start = 10
    assert rx.lts_start == 10

@pytest.mark.parametrize("attr", [
    "rx_fine",
    "symbols_fd",
    "channel_est",
    "eq_coefficient",
    "corrected_symbols"
])
def test_numpy_properties_type_check(attr):
    """
    Parametrizovani test koji provjerava da svi atributi koji očekuju numpy nizove odbacuju neispravne tipove.
    """
    rx = Receiver80211a(20e6, 1)

    with pytest.raises(TypeError):
        setattr(rx, attr, 123)

    setattr(rx, attr, np.zeros(4))

@patch("rx.RX_802_11_a.iq_preprocessing")
def test_preprocess(mock_iq):
    """
    Testira preprocess metodu:
    - da li se poziva iq_preprocessing
    - da li se rx_signal_clean i fs pravilno postavljaju
    """
    rx = Receiver80211a(20e6, 1)

    mock_signal = np.ones(100, dtype=complex)
    mock_iq.return_value = (mock_signal, 10e6)

    out = rx.preprocess(np.ones(100), np.ones(100))

    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(rx.rx_signal_clean, mock_signal)
    assert rx.fs == 10e6
    mock_iq.assert_called_once()

def test_synchronize_without_preprocess():
    """
    Provjerava da synchronize baca RuntimeError ako preprocess nije prethodno pozvan.
    """
    rx = Receiver80211a(20e6, 1)

    with pytest.raises(RuntimeError):
        rx.synchronize()

@patch("rx.RX_802_11_a.detect_frequency_offsets")
@patch("rx.RX_802_11_a.gruba_vremenska_sinhronizacija")
@patch("rx.RX_802_11_a.packet_detector")
def test_synchronize_success(mock_packet, mock_sync, mock_cfo):
    """
    Testira uspješan tok sinhronizacije:
    - detekciju paketa
    - vremensku sinhronizaciju
    - frekvencijsku korekciju
    """
    rx = Receiver80211a(20e6, 1)

    rx.rx_signal_clean = np.ones(500, dtype=complex)

    mock_packet.return_value = (None, None, 50, None)
    mock_sync.return_value = (10, None, None)
    mock_cfo.side_effect = [(1000, 0), (0, 50)]

    out = rx.synchronize()

    assert rx.lts_start == 60
    assert isinstance(out, np.ndarray)
    assert rx.rx_fine is not None

def test_extract_without_sync():
    """
    Provjerava da extract_and_equalize baca RuntimeError ako sinhronizacija nije izvršena.
    """
    rx = Receiver80211a(20e6, 1)

    with pytest.raises(RuntimeError):
        rx.extract_and_equalize()

@patch("rx.RX_802_11_a.phase_correction")
@patch("rx.RX_802_11_a.channel_estimate_and_equalizer")
@patch("rx.RX_802_11_a.remove_cp")
def test_extract_and_equalize_success(
    mock_remove_cp,
    mock_channel_est,
    mock_phase_corr
):
    """
    Testira skidanje CP-a, estimaciju kanala i ekvalizaciju uz potpuno mockovan DSP lanac.
    """
    rx = Receiver80211a(20e6, num_symbols=2)

    rx.rx_fine = np.ones(500, dtype=complex)
    rx.lts_start = 50

    mock_remove_cp.return_value = np.ones((2, 48), dtype=complex)
    mock_channel_est.return_value = (
        np.ones(48, dtype=complex),
        np.ones(48, dtype=complex)
    )
    mock_phase_corr.return_value = np.ones((2, 48), dtype=complex) * 2

    out = rx.extract_and_equalize()

    assert out.shape == (2, 48)
    assert rx.corrected_symbols is out

@patch("rx.RX_802_11_a.phase_correction")
@patch("rx.RX_802_11_a.channel_estimate_and_equalizer")
@patch("rx.RX_802_11_a.remove_cp")
@patch("rx.RX_802_11_a.detect_frequency_offsets")
@patch("rx.RX_802_11_a.gruba_vremenska_sinhronizacija")
@patch("rx.RX_802_11_a.packet_detector")
@patch("rx.RX_802_11_a.iq_preprocessing")
def test_process_signal_full_chain(
    mock_iq,
    mock_packet,
    mock_sync,
    mock_cfo,
    mock_remove_cp,
    mock_channel_est,
    mock_phase_corr
):
    """
    Integracioni unit test koji provjerava kompletan lanac obrade signala (preprocess - sync - equalize).
    """
    rx = Receiver80211a(20e6, num_symbols=1)

    mock_iq.return_value = (np.ones(1000, dtype=complex), 20e6)
    mock_packet.return_value = (None, None, 40, None)
    mock_sync.return_value = (5, None, None)
    mock_cfo.side_effect = [(100, 0), (0, 10)]
    mock_remove_cp.return_value = np.ones((1, 64), dtype=complex)
    mock_channel_est.return_value = (
        np.ones(64, dtype=complex),
        np.ones(64, dtype=complex)
    )
    mock_phase_corr.return_value = np.ones((1, 64), dtype=complex)

    out = rx.process_signal(
        np.ones(1000, dtype=complex),
        np.ones(1000, dtype=complex)
    )

    assert out.shape == (1, 64)
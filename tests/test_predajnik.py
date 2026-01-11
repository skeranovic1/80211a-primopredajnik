import numpy as np
import pytest
from unittest.mock import patch
from tx.OFDM_TX_802_11 import Transmitter80211a

def test_init_defaults():
    """Provjerava inicijalizaciju sa default parametrima."""
    tx = Transmitter80211a()
    assert tx.num_ofdm_symbols == 1
    assert tx.bits_per_symbol == 2
    assert tx.up_factor == 2
    assert tx.seed == 13
    assert tx.step == 1
    assert tx.plot is False

def test_set_get_num_ofdm_symbols():
    tx = Transmitter80211a()
    tx.num_ofdm_symbols = 5
    assert tx.num_ofdm_symbols == 5

def test_set_invalid_num_ofdm_symbols():
    tx = Transmitter80211a()
    with pytest.raises(ValueError):
        tx.num_ofdm_symbols = 0
    with pytest.raises(ValueError):
        tx.num_ofdm_symbols = -1
    with pytest.raises(ValueError):
        tx.num_ofdm_symbols = 2.5

def test_set_get_bits_per_symbol():
    tx = Transmitter80211a()
    tx.bits_per_symbol = 4
    assert tx.bits_per_symbol == 4

def test_set_invalid_bits_per_symbol():
    tx = Transmitter80211a()
    with pytest.raises(ValueError):
        tx.bits_per_symbol = 0
    with pytest.raises(ValueError):
        tx.bits_per_symbol = -1
    with pytest.raises(ValueError):
        tx.bits_per_symbol = 3.5

def test_set_get_up_factor():
    tx = Transmitter80211a()
    tx.up_factor = 8
    assert tx.up_factor == 8

def test_set_invalid_up_factor():
    tx = Transmitter80211a()
    with pytest.raises(ValueError):
        tx.up_factor = 0
    with pytest.raises(ValueError):
        tx.up_factor = -2
    with pytest.raises(ValueError):
        tx.up_factor = 1.5

def test_set_get_seed():
    tx = Transmitter80211a()
    tx.seed = 99
    assert tx.seed == 99

def test_set_invalid_seed():
    tx = Transmitter80211a()
    with pytest.raises(ValueError):
        tx.seed = -1
    with pytest.raises(ValueError):
        tx.seed = 3.14

def test_set_get_step():
    tx = Transmitter80211a()
    tx.step = 3
    assert tx.step == 3

def test_set_invalid_step():
    tx = Transmitter80211a()
    with pytest.raises(ValueError):
        tx.step = 0
    with pytest.raises(ValueError):
        tx.step = -5
    with pytest.raises(ValueError):
        tx.step = 2.7

def test_set_get_plot():
    tx = Transmitter80211a()
    tx.plot = True
    assert tx.plot is True

def test_set_invalid_plot():
    tx = Transmitter80211a()
    with pytest.raises(TypeError):
        tx.plot = 1
    with pytest.raises(TypeError):
        tx.plot = "yes"

@patch("tx.OFDM_TX_802_11.get_short_training_sequence")
@patch("tx.OFDM_TX_802_11.get_long_training_sequence")
def test_generate_training_sequences(mock_lts, mock_sts):
    """Provjerava da se trening sekvence generišu ispravno"""
    tx = Transmitter80211a(step=2)
    mock_sts.return_value = np.array([1,2])
    mock_lts.return_value = np.array([3,4])
    sts, lts = tx.generate_training_sequences()
    assert np.array_equal(sts, np.array([1,2]))
    assert np.array_equal(lts, np.array([3,4]))
    mock_sts.assert_called_once_with(2)
    mock_lts.assert_called_once_with(2)

@patch("tx.OFDM_TX_802_11.bit_sequence")
@patch("tx.OFDM_TX_802_11.Mapper_OFDM")
@patch("tx.OFDM_TX_802_11.IFFT_GI")
def test_generate_payload(mock_ifft, mock_mapper, mock_bits):
    """Provjerava generisanje OFDM payloada"""
    tx = Transmitter80211a(num_ofdm_symbols=1, bits_per_symbol=2, seed=7, plot=True)
    mock_bits.return_value = np.array([0,1,1,0])
    mock_mapper.return_value = np.array([1+1j, -1-1j])
    mock_ifft.return_value = np.array([0.5+0.5j, -0.5-0.5j])
    
    payload, bits, symbols = tx.generate_payload()
    
    np.testing.assert_array_equal(bits, np.array([0,1,1,0]))
    np.testing.assert_array_equal(symbols, np.array([1+1j, -1-1j]))
    np.testing.assert_array_equal(payload, np.array([0.5+0.5j, -0.5-0.5j]))
    mock_bits.assert_called_once_with(1, 2, 7)
    mock_mapper.assert_called_once()
    mock_ifft.assert_called_once()

@patch("tx.OFDM_TX_802_11.half_band_upsample")
@patch("tx.OFDM_TX_802_11.Transmitter80211a.generate_payload")
@patch("tx.OFDM_TX_802_11.Transmitter80211a.generate_training_sequences")
def test_generate_frame(mock_seq, mock_payload, mock_upsample):
    """Provjerava generisanje cijelog OFDM paketa"""

    tx = Transmitter80211a(up_factor=2, plot=True)

    # Mock training sequences
    mock_seq.return_value = (np.array([1,2], dtype=float), np.array([3,4], dtype=float))

    # Mock payload
    mock_payload.return_value = (
        np.array([5,6], dtype=float),
        np.array([0,1], dtype=int),
        np.array([1+1j, -1-1j], dtype=complex)
    )

    mock_upsample.return_value = (np.array([0.1,0.2,0.3], dtype=float), None)
    sample_output, bits, symbols = tx.generate_frame()
    np.testing.assert_array_equal(sample_output, np.array([0.1,0.2,0.3]))
    np.testing.assert_array_equal(bits, np.array([0,1]))
    np.testing.assert_array_equal(symbols, np.array([1+1j, -1-1j]))

    # Provjera da su mockovi pozvani
    mock_seq.assert_called_once()
    mock_payload.assert_called_once()
    mock_upsample.assert_called_once()  # samo provjerava da je pozvan jednom

    actual_call_args = mock_upsample.call_args[0][0]  # prvi argument
    expected_input = np.array([1/64, 2/64, 3/64, 4/64, 5, 6], dtype=float)
    np.testing.assert_allclose(actual_call_args, expected_input, rtol=1e-12, atol=1e-12)

    actual_kwargs = mock_upsample.call_args[1]
    assert actual_kwargs["up_factor"] == 2
    assert actual_kwargs["N"] == 31
    assert actual_kwargs["plot"] == True
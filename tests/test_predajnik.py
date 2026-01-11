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

def test_transmitter_basic_happy_path():
    """Provjera da metoda generate_frame vraća ispravne tipove i dimenzije"""
    tx = Transmitter80211a(num_ofdm_symbols=1, bits_per_symbol=2, up_factor=2, seed=13)
    sample_out, bits, symbols = tx.generate_frame()

    # provjera tipova i dtype
    assert isinstance(sample_out, np.ndarray)
    assert isinstance(symbols, np.ndarray)
    assert isinstance(bits,np.ndarray)
    assert sample_out.dtype == complex
    assert symbols.dtype == complex
    assert bits.dtype == int
    # provjera da nisu prazni
    assert len(sample_out) > 0
    assert len(symbols) > 0
    assert len(bits) > 0

def test_transmitter_deterministic_seed():
    """Isti seed daje identične izlaze"""
    tx1 = Transmitter80211a(num_ofdm_symbols=2, bits_per_symbol=2, seed=42)
    tx2 = Transmitter80211a(num_ofdm_symbols=2, bits_per_symbol=2, seed=42)

    out1, bits1, sym1 = tx1.generate_frame()
    out2, bits2, sym2 = tx2.generate_frame()

    # provjera identičnosti izlaza
    np.testing.assert_allclose(out1, out2)
    np.testing.assert_array_equal(bits1, bits2)
    np.testing.assert_allclose(sym1, sym2)

def test_transmitter_payload_length_increases_with_symbols():
    """Više OFDM simbola daje duži signal"""
    tx1 = Transmitter80211a(num_ofdm_symbols=1, bits_per_symbol=2, up_factor=2)
    tx2 = Transmitter80211a(num_ofdm_symbols=5, bits_per_symbol=2, up_factor=2)

    out1, _ , _= tx1.generate_frame()
    out2, _, _ = tx2.generate_frame()

    # provjera da signal raste s brojem simbola
    assert len(out2) > len(out1)

def test_transmitter_upsampling_factor_changes_length():
    """Veći up_factor daje više uzoraka"""
    tx1 = Transmitter80211a(num_ofdm_symbols=1, bits_per_symbol=2, up_factor=1)
    tx2 = Transmitter80211a(num_ofdm_symbols=1, bits_per_symbol=2, up_factor=2)

    out1, _, _ = tx1.generate_frame()
    out2, _, _ = tx2.generate_frame()

    # provjera da veći up_factor povećava broj uzoraka
    assert len(out2) > len(out1)

def test_transmitter_symbol_stream_length_matches_ofdm_symbols():
    """Mapper mora dati 48 data nosioca po OFDM simbolu"""
    num_symbols = 3
    bits_per_symbol = 2
    tx = Transmitter80211a(num_ofdm_symbols=num_symbols, bits_per_symbol=bits_per_symbol)
    _, _, symbols = tx.generate_frame()

    expected = num_symbols * 48
    assert len(symbols) == expected

def test_transmitter_signal_energy_nonzero():
    """Signal mora imati energiju"""
    tx = Transmitter80211a(num_ofdm_symbols=1, bits_per_symbol=2)
    out, _, _ = tx.generate_frame()

    # provjera da signal nije nula
    rms = np.sqrt(np.mean(np.abs(out)**2))
    assert rms > 0

def test_transmitter_invalid_bits_per_symbol():
    """Nevažeća modulacija"""
    # 3 bps nije podržano
    with pytest.raises(ValueError):
        tx = Transmitter80211a(num_ofdm_symbols=1, bits_per_symbol=3)
        tx.generate_frame()

def test_transmitter_negative_number_of_symbols():
    """Negativan broj OFDM simbola nije dozvoljen"""
    with pytest.raises(ValueError):
        tx = Transmitter80211a(num_ofdm_symbols=-1, bits_per_symbol=2)
        tx.generate_frame()

def test_transmitter_zero_symbols():
    """Nula simbola - očekuje se greška"""
    with pytest.raises(ValueError):
        tx = Transmitter80211a(num_ofdm_symbols=0, bits_per_symbol=2)
        tx.generate_frame()

def test_transmitter_invalid_upsampling_factor():
    """Upsampling faktor mora biti >= 1"""
    with pytest.raises(ValueError):
        tx = Transmitter80211a(num_ofdm_symbols=1, bits_per_symbol=2, up_factor=0)
        tx.generate_frame()

@pytest.mark.parametrize("bps", [1, 2, 4, 6])
def test_valid_bits_per_symbol(bps):
    """Provjera da validni bits_per_symbol daju ispravnu duljinu bitova"""
    tx = Transmitter80211a(num_ofdm_symbols=1, bits_per_symbol=bps)
    out, bits, symbols = tx.generate_frame()
    assert len(bits) == 48 * bps * tx.num_ofdm_symbols  # 48 nosioca po OFDM simbolu

@patch("tx.OFDM_TX_802_11.get_short_training_sequence")
@patch("tx.OFDM_TX_802_11.get_long_training_sequence")
def test_training_sequence_step(mock_lts, mock_sts):
    """Provjera da se funkcije za STS i LTS pozivaju i vraćaju očekivane vrijednosti"""
    mock_sts.return_value = np.array([1,2,3])
    mock_lts.return_value = np.array([4,5,6])
    tx = Transmitter80211a(step=5)
    sts, lts = tx.generate_training_sequences()
    mock_sts.assert_called_with(5)
    mock_lts.assert_called_with(5)
    np.testing.assert_array_equal(sts, [1,2,3])
    np.testing.assert_array_equal(lts, [4,5,6])

def test_invalid_seed_plot():
    """Provjera neispravnog seed-a i plot parametra"""
    with pytest.raises(ValueError):
        tx = Transmitter80211a(seed=-1)
    with pytest.raises(TypeError):
        tx = Transmitter80211a(plot="yes")

def test_generate_payload_output():
    """Provjera da generate_payload vraća ispravne duljine i dtype"""
    tx = Transmitter80211a(num_ofdm_symbols=2, bits_per_symbol=2)
    payload, bits, symbols = tx.generate_payload()
    assert len(bits) == 48 * tx.bits_per_symbol * tx.num_ofdm_symbols
    assert len(symbols) == 48 * tx.num_ofdm_symbols
    assert payload.dtype == complex

def test_frame_length_scales_with_up_factor():
    """Provjera da frame duljina raste s up_factor"""
    tx1 = Transmitter80211a(num_ofdm_symbols=1, bits_per_symbol=2, up_factor=1)
    tx2 = Transmitter80211a(num_ofdm_symbols=1, bits_per_symbol=2, up_factor=4)
    out1, _, _ = tx1.generate_frame()
    out2, _, _ = tx2.generate_frame()
    assert len(out2) > len(out1)
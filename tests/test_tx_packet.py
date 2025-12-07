import numpy as np
import pytest
from tx.OFDM_TX_802_11 import Transmitter80211a

def test_transmitter_basic_happy_path():
    """Provjera da metoda generate_frame vraća ispravne tipove i dimenzije"""
    tx = Transmitter80211a(num_ofdm_symbols=1, bits_per_symbol=2, up_factor=2, seed=13)
    sample_out, symbols = tx.generate_frame()

    assert isinstance(sample_out, np.ndarray)
    assert isinstance(symbols, np.ndarray)
    assert sample_out.dtype == complex
    assert symbols.dtype == complex
    assert len(sample_out) > 0
    assert len(symbols) > 0

def test_transmitter_deterministic_seed():
    """Isti seed daje identične izlaze"""
    tx1 = Transmitter80211a(num_ofdm_symbols=2, bits_per_symbol=2, seed=42)
    tx2 = Transmitter80211a(num_ofdm_symbols=2, bits_per_symbol=2, seed=42)

    out1, sym1 = tx1.generate_frame()
    out2, sym2 = tx2.generate_frame()

    np.testing.assert_allclose(out1, out2)
    np.testing.assert_allclose(sym1, sym2)

def test_transmitter_payload_length_increases_with_symbols():
    """Više OFDM simbola daje duži signal"""
    tx1 = Transmitter80211a(num_ofdm_symbols=1, bits_per_symbol=2, up_factor=2)
    tx2 = Transmitter80211a(num_ofdm_symbols=5, bits_per_symbol=2, up_factor=2)

    out1, _ = tx1.generate_frame()
    out2, _ = tx2.generate_frame()

    assert len(out2) > len(out1)

def test_transmitter_upsampling_factor_changes_length():
    """Veći up_factor daje više uzoraka"""
    tx1 = Transmitter80211a(num_ofdm_symbols=1, bits_per_symbol=2, up_factor=1)
    tx2 = Transmitter80211a(num_ofdm_symbols=1, bits_per_symbol=2, up_factor=2)

    out1, _ = tx1.generate_frame()
    out2, _ = tx2.generate_frame()

    assert len(out2) > len(out1)

def test_transmitter_symbol_stream_length_matches_ofdm_symbols():
    """Mapper mora dati 48 data nosioca po OFDM simbolu"""
    num_symbols = 3
    bits_per_symbol = 2
    tx = Transmitter80211a(num_ofdm_symbols=num_symbols, bits_per_symbol=bits_per_symbol)
    _, symbols = tx.generate_frame()

    expected = num_symbols * 48
    assert len(symbols) == expected

def test_transmitter_signal_energy_nonzero():
    """Signal mora imati energiju"""
    tx = Transmitter80211a(num_ofdm_symbols=1, bits_per_symbol=2)
    out, _ = tx.generate_frame()

    rms = np.sqrt(np.mean(np.abs(out)**2))
    assert rms > 0

def test_transmitter_invalid_bits_per_symbol():
    """Nevažeća modulacija"""
    with pytest.raises(ValueError):
        tx = Transmitter80211a(num_ofdm_symbols=1, bits_per_symbol=3)
        tx.generate_frame()

def test_transmitter_negative_number_of_symbols():
    """Negativan broj OFDM simbola nije dozvoljen"""
    with pytest.raises(ValueError):
        tx = Transmitter80211a(num_ofdm_symbols=-1, bits_per_symbol=2)
        tx.generate_frame()

def test_transmitter_zero_symbols():
    """Nula simbola – očekuje se greška"""
    with pytest.raises(ValueError):
        tx = Transmitter80211a(num_ofdm_symbols=0, bits_per_symbol=2)
        tx.generate_frame()

def test_transmitter_invalid_upsampling_factor():
    """Upsampling faktor mora biti >= 1"""
    with pytest.raises(ValueError):
        tx = Transmitter80211a(num_ofdm_symbols=1, bits_per_symbol=2, up_factor=0)
        tx.generate_frame()

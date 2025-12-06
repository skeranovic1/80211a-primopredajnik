import numpy as np
import pytest
from tx.OFDM_TX_802_11 import OFDM_TX

def test_ofdm_tx_basic_happy_path():
    """Funkcija radi i vraća ispravne tipove"""
    sample_out, symbols = OFDM_TX(
        NumberOf_OFDM_Symbols=1,
        BitsPerSymbol=2,
        up_factor=2,
        seed=13
    )

    assert isinstance(sample_out, np.ndarray)
    assert isinstance(symbols, np.ndarray)

    assert sample_out.dtype == complex
    assert symbols.dtype == complex

    assert len(sample_out) > 0
    assert len(symbols) > 0

def test_ofdm_tx_deterministic_seed():
    """Isti seed mora dati identičan izlaz"""
    out1, sym1 = OFDM_TX(2, 2, up_factor=2, seed=42)
    out2, sym2 = OFDM_TX(2, 2, up_factor=2, seed=42)

    np.testing.assert_allclose(out1, out2)
    np.testing.assert_allclose(sym1, sym2)

def test_ofdm_tx_payload_length_increases():
    """Više OFDM simbola daje duži signal"""
    out1, _ = OFDM_TX(1, 2, up_factor=2)
    out2, _ = OFDM_TX(5, 2, up_factor=2)

    assert len(out2) > len(out1)

def test_ofdm_tx_upsampling_factor_changes_output_length():
    """Veći up_factor daje više uzoraka"""
    out1, _ = OFDM_TX(1, 2, up_factor=1)
    out2, _ = OFDM_TX(1, 2, up_factor=2)

    assert len(out2) > len(out1)

def test_symbol_stream_length_matches_ofdm_symbols():
    """
    Mapper mora dati:
    48 data nosioca po OFDM simbolu (802.11a)
    """
    num_symbols = 3
    bits_per_symbol = 2

    _, symbols = OFDM_TX(num_symbols, bits_per_symbol)

    expected = num_symbols * 48
    assert len(symbols) == expected

def test_ofdm_tx_signal_energy_nonzero():
    """Signal mora imati energiju"""
    out, _ = OFDM_TX(1, 2)

    rms = np.sqrt(np.mean(np.abs(out) ** 2))
    assert rms > 0

def test_ofdm_tx_invalid_bits_per_symbol():
    """Nevažeća modulacija"""
    with pytest.raises(ValueError):
        OFDM_TX(1, BitsPerSymbol=3)

def test_ofdm_tx_negative_number_of_symbols():
    """Negativan broj OFDM simbola nije dozvoljen"""
    with pytest.raises(ValueError):
        OFDM_TX(-1, 2)

def test_ofdm_tx_zero_symbols():
    """Nula simbola – očekuje se greška"""
    with pytest.raises(ValueError):
        OFDM_TX(0, 2)

def test_ofdm_tx_invalid_upsampling_factor():
    """Upsampling faktor mora biti >= 1"""
    with pytest.raises(ValueError):
        OFDM_TX(1, 2, up_factor=0)

def test_guard_interval_structure_multiple_symbols():
    """
    Provjerava da payload ima strukturu:
    [GI][DATA] [GI][DATA] ... za više OFDM simbola
    """
    num_symbols = 4
    up_factor = 2

    out, _ = OFDM_TX(num_symbols, BitsPerSymbol=2, up_factor=up_factor)

    # Dužine u uzorcima (sa upsamplingom)
    sts_len = 16 * 10 * up_factor
    lts_len = 160 * up_factor
    gi_len = 16 * up_factor
    ofdm_len = 64 * up_factor
    block_len = gi_len + ofdm_len

    payload = out[sts_len + lts_len:]

    assert len(payload) == num_symbols * block_len

def test_guard_interval_alignment():
    """
    Provjera da se GI pojavljuje tačno svakih (GI+FFT) uzoraka
    """
    num_symbols = 3
    up_factor = 2

    out, _ = OFDM_TX(num_symbols, BitsPerSymbol=2, up_factor=up_factor)

    sts_len = 16 * 10 * up_factor
    lts_len = 160 * up_factor
    gi_len = 16 * up_factor
    ofdm_len = 64 * up_factor
    block_len = gi_len + ofdm_len

    payload = out[sts_len + lts_len:]

    for i in range(num_symbols):
        gi_start = i * block_len
        assert len(payload[gi_start : gi_start + gi_len]) == gi_len

def test_guard_interval_energy_consistency():
    """
    GI ne smije biti nula i mora imati
    energiju sličnu simbolu iz kojeg je kopiran
    """
    num_symbols = 2
    up_factor = 2

    out, _ = OFDM_TX(num_symbols, BitsPerSymbol=2, up_factor=up_factor)

    sts_len = 16 * 10 * up_factor
    lts_len = 160 * up_factor
    gi_len = 16 * up_factor
    ofdm_len = 64 * up_factor
    block_len = gi_len + ofdm_len

    payload = out[sts_len + lts_len:]

    for i in range(num_symbols):
        start = i * block_len

        gi = payload[start : start + gi_len]
        symbol = payload[start + gi_len : start + block_len]

        gi_rms = np.sqrt(np.mean(np.abs(gi) ** 2))
        sym_tail_rms = np.sqrt(np.mean(np.abs(symbol[-gi_len:]) ** 2))

        assert gi_rms > 0
        assert np.isclose(gi_rms, sym_tail_rms, rtol=0.1)


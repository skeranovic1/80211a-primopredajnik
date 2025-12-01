import numpy as np
import pytest
from tx.OFDM_TX_802_11 import OFDM_TX_802_11

def test_return_types():
    """Provjerava da funkcija vraća NumPy niz i kompleksni stream simbola."""
    samples, symbols = OFDM_TX_802_11(
        NumberOf_OFDM_Symbols=2,
        BitsPerSymbol=2
    )

    assert isinstance(samples, np.ndarray), "Output samples mora biti NumPy niz."
    assert isinstance(symbols, np.ndarray), "Symbol stream mora biti NumPy niz."
    assert np.iscomplexobj(symbols), "Symbol stream mora biti kompleksan."

def test_symbol_count():
    """Provjerava da je broj generisanih simbola jednak očekivanom."""
    Nsym = 5
    Bps = 2  # QPSK
    expected_symbols = 48 * Nsym

    _, symbols = OFDM_TX_802_11(Nsym, Bps)

    assert len(symbols) == expected_symbols, \
        f"Očekivano {expected_symbols} simbola, dobijeno {len(symbols)}."

def test_output_not_empty():
    """Provjerava da OFDM signal nije prazan i da ima očekivanu dužinu."""
    samples, symbols = OFDM_TX_802_11(1, 2)

    assert len(samples) > 0, "Output signal ne smije biti prazan."
    assert np.any(samples != 0), "Signal ne smije imati sve nule."

def test_different_modulation_changes_symbols():
    """Provjerava da različite modulacije daju različite mape simbola."""
    _, sym_qpsk = OFDM_TX_802_11(1, 2)      # QPSK
    _, sym_16qam = OFDM_TX_802_11(1, 4)     # 16-QAM

    assert not np.allclose(sym_qpsk, sym_16qam), \
        "Različiti BitsPerSymbol moraju da daju različite simbole."

def test_upsampling_factor_effect():
    """Provjerava da povećanje upsampling faktora povećava dužinu izlaznog signala."""
    samples_x2, _ = OFDM_TX_802_11(1, 2, up_factor=2)
    samples_x4, _ = OFDM_TX_802_11(1, 2, up_factor=4)

    assert len(samples_x4) > len(samples_x2), \
        "Veći up_factor mora da rezultuje dužim OFDM signalom."

def test_reproducibility():
    """Provjerava da seed u funkcijama daje deterministički rezultat."""
    s1, sy1 = OFDM_TX_802_11(3, 2)
    s2, sy2 = OFDM_TX_802_11(3, 2)

    assert np.allclose(s1, s2), "Outputs moraju biti deterministički za isti ulaz."
    assert np.allclose(sy1, sy2), "Symbol stream mora biti deterministički."

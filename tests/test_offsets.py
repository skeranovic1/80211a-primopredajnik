import pytest
import numpy as np
from rx.cfo import gruba_vremenska_sinhronizacija, detect_frequency_offsets

def test_gruba_raises_type_error_for_non_ndarray():
    """Provjerava TypeError ako rx_lts nije np.ndarray"""
    with pytest.raises(TypeError, match="'rx_lts' mora biti numpy.ndarray"):
        gruba_vremenska_sinhronizacija(rx_lts=[1+1j]*128)

def test_gruba_raises_value_error_for_short_signal():
    """Provjerava ValueError ako rx_lts ima manje od 128 uzoraka"""
    rx_lts = np.ones(127, dtype=complex)
    with pytest.raises(ValueError, match="mora imati barem 128 uzoraka"):
        gruba_vremenska_sinhronizacija(rx_lts)

def test_gruba_raises_value_error_for_invalid_search_win():
    """Provjerava ValueError za search_win <=0 ili veće od duljine rx_lts"""
    rx_lts = np.ones(128, dtype=complex)
    with pytest.raises(ValueError):
        gruba_vremenska_sinhronizacija(rx_lts, search_win=0)
    with pytest.raises(ValueError):
        gruba_vremenska_sinhronizacija(rx_lts, search_win=200)

def test_gruba_functionality_minimal_signal():
    """Provjerava da funkcija vraća ispravne tipove i dužine"""
    rx_lts = np.ones(128, dtype=complex)
    fft_start, timing_corr, timing_idxs = gruba_vremenska_sinhronizacija(rx_lts)
    assert isinstance(fft_start, np.int64)
    assert isinstance(timing_corr, np.ndarray)
    assert isinstance(timing_idxs, np.ndarray)
    assert len(timing_corr) == len(rx_lts)
    assert len(timing_idxs) == len(rx_lts)

def test_detect_raises_type_error_for_non_complex_input():
    """Provjerava TypeError ako RX_Input nije kompleksan"""
    RX_Input = np.ones(200, dtype=float)
    with pytest.raises(TypeError, match="mora sadržavati kompleksne uzorke"):
        detect_frequency_offsets(RX_Input, lts_start=0, fs=1e6)

def test_detect_raises_value_error_for_empty_input():
    """Provjerava ValueError za prazan RX_Input"""
    RX_Input = np.array([], dtype=complex)
    with pytest.raises(ValueError, match="ne smije biti prazan niz"):
        detect_frequency_offsets(RX_Input, lts_start=0, fs=1e6)

def test_detect_raises_value_error_for_invalid_fs():
    """Provjerava ValueError ako fs <=0"""
    RX_Input = np.ones(200, dtype=complex)
    with pytest.raises(ValueError, match="mora biti pozitivan broj"):
        detect_frequency_offsets(RX_Input, lts_start=0, fs=0)

def test_detect_raises_type_error_for_invalid_plot_type():
    """Provjerava TypeError ako plot nije bool"""
    RX_Input = np.ones(200, dtype=complex)
    with pytest.raises(TypeError, match="mora biti bool"):
        detect_frequency_offsets(RX_Input, lts_start=0, fs=1e6, plot="True")

def test_detect_functionality_minimal_input_no_plot():
    """Provjerava da funkcija vraća niz od 2 elementa i da radi bez plot"""
    RX_Input = np.ones(200, dtype=complex)
    offsets = detect_frequency_offsets(RX_Input, lts_start=64, fs=1e6, plot=False)
    assert isinstance(offsets, np.ndarray)
    assert offsets.shape == (2,)

def test_detect_functionality_minimal_input_with_plot():
    """Pokrije i granu gdje plot=True"""
    RX_Input = np.ones(200, dtype=complex)
    offsets = detect_frequency_offsets(RX_Input, lts_start=64, fs=1e6, plot=True)
    assert isinstance(offsets, np.ndarray)
    assert offsets.shape == (2,)
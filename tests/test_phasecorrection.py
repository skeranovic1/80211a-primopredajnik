import numpy as np
import pytest
from rx.PhaseCorrection_80211a import phase_correction

def test_raises_type_error_for_symbols_fd():
    """Provjerava da TypeError nastaje ako symbols_fd nije np.ndarray."""
    symbols_fd = [[1]*64]*2
    channel_est = np.ones(64, dtype=complex)
    with pytest.raises(TypeError):
        phase_correction(symbols_fd, num_symbols=2, channel_est=channel_est)

def test_raises_type_error_for_channel_est():
    """Provjerava da TypeError nastaje ako channel_est nije np.ndarray."""
    symbols_fd = np.ones((2,64), dtype=complex)
    channel_est = [1]*64
    with pytest.raises(TypeError):
        phase_correction(symbols_fd, num_symbols=2, channel_est=channel_est)

def test_raises_value_error_for_wrong_symbols_fd_shape():
    """Provjerava da ValueError nastaje ako symbols_fd nema shape (num_symbols,64)."""
    symbols_fd = np.ones((3,64), dtype=complex)
    channel_est = np.ones(64, dtype=complex)
    with pytest.raises(ValueError):
        phase_correction(symbols_fd, num_symbols=2, channel_est=channel_est)

def test_raises_value_error_for_wrong_channel_est_shape():
    """Provjerava da ValueError nastaje ako channel_est nema 64 elemenata."""
    symbols_fd = np.ones((2,64), dtype=complex)
    channel_est = np.ones(63, dtype=complex)
    with pytest.raises(ValueError):
        phase_correction(symbols_fd, num_symbols=2, channel_est=channel_est)

def test_raises_value_error_for_invalid_num_symbols():
    """Provjerava da ValueError nastaje ako num_symbols nije pozitivan int."""
    symbols_fd = np.ones((2,64), dtype=complex)
    channel_est = np.ones(64, dtype=complex)
    with pytest.raises(ValueError):
        phase_correction(symbols_fd, num_symbols=-1, channel_est=channel_est)
    with pytest.raises(ValueError):
        phase_correction(symbols_fd, num_symbols=0, channel_est=channel_est)
    with pytest.raises(ValueError):
        phase_correction(symbols_fd, num_symbols=2.5, channel_est=channel_est)

def test_raises_value_error_for_invalid_L():
    """Provjerava da ValueError nastaje ako L nije pozitivan integer."""
    symbols_fd = np.ones((2,64), dtype=complex)
    channel_est = np.ones(64, dtype=complex)
    with pytest.raises(ValueError):
        phase_correction(symbols_fd, num_symbols=2, channel_est=channel_est, L=0)
    with pytest.raises(ValueError):
        phase_correction(symbols_fd, num_symbols=2, channel_est=channel_est, L=-1)
    with pytest.raises(ValueError):
        phase_correction(symbols_fd, num_symbols=2, channel_est=channel_est, L=1.5)

def test_raises_value_error_for_invalid_max_ratio():
    """Provjerava da ValueError nastaje ako max_ratio nije 0 ili 1."""
    symbols_fd = np.ones((2,64), dtype=complex)
    channel_est = np.ones(64, dtype=complex)
    with pytest.raises(ValueError):
        phase_correction(symbols_fd, num_symbols=2, channel_est=channel_est, max_ratio=2)
    with pytest.raises(ValueError):
        phase_correction(symbols_fd, num_symbols=2, channel_est=channel_est, max_ratio=-1)

def test_output_shape_and_type():
    """Provjerava da funkcija vraća np.ndarray shape (num_symbols,48)."""
    num_symbols = 2
    symbols_fd = np.ones((num_symbols,64), dtype=complex)
    channel_est = np.ones(64, dtype=complex)

    corrected = phase_correction(symbols_fd, num_symbols, channel_est)

    assert isinstance(corrected, np.ndarray)
    assert corrected.shape == (num_symbols, 48)
    assert np.iscomplexobj(corrected)

def test_phase_correction_preserves_data_magnitude():
    """Provjerava da magnituda podnosioca ostaje u istom rasponu nakon korekcije."""
    num_symbols = 2
    symbols_fd = np.random.randn(num_symbols,64) + 1j*np.random.randn(num_symbols,64)
    channel_est = np.random.randn(64) + 1j*np.random.randn(64)

    corrected = phase_correction(symbols_fd, num_symbols, channel_est)

    # Magnituda nakon korekcije nije nula i u sličnom rasponu
    assert np.all(np.abs(corrected) > 0)

def test_phase_correction_with_max_ratio_zero():
    """Provjerava da funkcija radi ispravno kada max_ratio=0 (ponderi pilota jednaki)."""
    num_symbols = 2
    symbols_fd = np.random.randn(num_symbols,64) + 1j*np.random.randn(num_symbols,64)
    channel_est = np.random.randn(64) + 1j*np.random.randn(64)

    corrected = phase_correction(symbols_fd, num_symbols, channel_est, max_ratio=0)
    assert corrected.shape == (num_symbols,48)

def test_phase_correction_with_varied_L():
    """Provjerava da funkcija radi za različite dužine filtera L."""
    num_symbols = 2
    symbols_fd = np.random.randn(num_symbols,64) + 1j*np.random.randn(num_symbols,64)
    channel_est = np.random.randn(64) + 1j*np.random.randn(64)

    for L in [1, 4, 16]:
        corrected = phase_correction(symbols_fd, num_symbols, channel_est, L=L)
        assert corrected.shape == (num_symbols,48)

def test_multiple_symbols_consistency():
    """Provjerava da funkcija korektno procesira više simbola."""
    num_symbols = 5
    symbols_fd = np.random.randn(num_symbols,64) + 1j*np.random.randn(num_symbols,64)
    channel_est = np.ones(64, dtype=complex)

    corrected = phase_correction(symbols_fd, num_symbols, channel_est)

    assert corrected.shape == (num_symbols,48)

def test_phase_correction_invalid_num_symbols_raises():
    """Provjerava da funkcija baca ValueError kada je parametar 'num_symbols' nevažeći."""
    symbols_fd = np.zeros((1, 64), dtype=complex)
    channel_est = np.ones(64, dtype=complex)

    with pytest.raises(ValueError):
        phase_correction(symbols_fd, 0, channel_est)
    with pytest.raises(ValueError):
        phase_correction(symbols_fd, -5, channel_est)
    with pytest.raises(ValueError):
        phase_correction(symbols_fd, 1.5, channel_est)
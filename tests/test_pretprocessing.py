import numpy as np
import pytest
from rx.pretprocessing import iq_preprocessing

def test_invalid_fs_type_or_value():
    """Provjerava da fs mora biti numerički tip i > 0."""
    rx = np.ones(10)
    tx = np.ones(10)
    
    # fs nije broj
    with pytest.raises(TypeError):
        iq_preprocessing(rx, tx, fs="1000")
    
    # fs <= 0
    with pytest.raises(ValueError):
        iq_preprocessing(rx, tx, fs=0)
    with pytest.raises(ValueError):
        iq_preprocessing(rx, tx, fs=-1)

def test_invalid_rx_tx_():
    """Provjerava da se baca greška za neprihvatljive RX/TX signale ili prazne nizove."""
        # RX nije array-like
    with pytest.raises(TypeError):
        iq_preprocessing(5, np.ones(10), fs=1e6)
        # TX nije array-like
    with pytest.raises(TypeError):
        iq_preprocessing(np.ones(10), {"not": "array"}, fs=1e6)
        # RX prazan
    with pytest.raises(ValueError):
        iq_preprocessing([], np.ones(10), fs=1e6)
        # TX prazan
    with pytest.raises(ValueError):
        iq_preprocessing(np.ones(10), [], fs=1e6)

import pytest
import numpy as np
from rx.pretprocessing import iq_preprocessing

def test_invalid_rx_tx_covers_asarray():
    """Provjerava da se baca TypeError kada np.asarray ne može pretvoriti RX/TX signal."""
    
    # dict ne može np.asarray pretvoriti u 1D niz
    rx_invalid = np.ones(10)
    tx_invalid = {"not": "array"}  # ovo izaziva exception u np.asarray
    
    with pytest.raises(TypeError):
        iq_preprocessing(rx_invalid, tx_invalid, fs=1e6)
    
    # set također ne može biti array
    rx_invalid2 = {1, 2, 3}
    tx_invalid2 = np.ones(3)
    
    with pytest.raises(TypeError):
        iq_preprocessing(rx_invalid2, tx_invalid2, fs=1e6)

class NotArrayLike:
    """Objekt koji nije array-like, ali ima neki parametar."""
    def __init__(self, data):
        self.data = data
    def __array__(self, dtype=None):
        # Ovo baca grešku kada np.asarray pokuša konvertovati
        raise TypeError("Ne može se konvertovati u np.array")

def test_invalid_rx_tx_covers_asarray_exception():
    """Pokazuje da TypeError bude podignut kada np.asarray() ne može konvertovati ulaz."""
    rx = NotArrayLike(data=[1, 2, 3])
    tx = np.ones(10)
    
    with pytest.raises(TypeError, match="rx_signal i tx_signal moraju biti array-like"):
        iq_preprocessing(rx, tx, fs=1e6)

def test_output_types_and_shapes():
    """Provjerava da funkcija vraća numpy niz i float frekvenciju, te da je 1D."""
    rx = np.random.randn(200) + 1j * np.random.randn(200)
    tx = np.random.randn(200) + 1j * np.random.randn(200)
    
    rx_out, fs_out = iq_preprocessing(rx, tx, fs=1e6)
    
    assert isinstance(rx_out, np.ndarray)
    assert isinstance(fs_out, float)
    assert rx_out.ndim == 1

def test_downsampling_by_factor_two():
    """Provjerava da se RX signal decimira faktorom 2 i da je snaga normalizirana prema TX signalu."""
    rx = np.arange(100, dtype=float)
    tx = np.ones(100, dtype=float)
    
    rx_out, _ = iq_preprocessing(rx, tx, fs=1e6)
    
    # dužina mora biti prepolovljena
    assert len(rx_out) == 50
    # snaga normalizirana
    expected = rx.flatten()[::2] * np.sqrt(np.mean(tx**2)) / np.sqrt(np.mean(rx**2))
    np.testing.assert_allclose(rx_out, expected, rtol=1e-6, atol=1e-12)

def test_sampling_frequency_is_halved():
    """Provjerava da se frekvencija uzorkovanja prepolovi nakon pretprocesinga."""
    rx = np.ones(10)
    tx = np.ones(10)
    fs = 2e6

    _, fs_out = iq_preprocessing(rx, tx, fs)
    
    assert fs_out == fs / 2

def test_no_nan_or_inf_in_output():
    """Provjerava da izlaz ne sadrži NaN ili Inf vrijednosti."""
    rx = np.random.randn(500) + 1j * np.random.randn(500)
    tx = np.random.randn(500) + 1j * np.random.randn(500)
    
    rx_out, _ = iq_preprocessing(rx, tx, fs=1e6)
    
    assert not np.any(np.isnan(rx_out))
    assert not np.any(np.isinf(rx_out))

def test_accepts_real_valued_signals():
    """Provjerava da funkcija radi i sa realnim signalima, sa očekivanim downsamplingom i fs."""
    rx = np.random.randn(200)
    tx = np.random.randn(200)
    
    rx_out, fs_out = iq_preprocessing(rx, tx, fs=1e6)
    
    assert rx_out.size == 100
    assert fs_out == 5e5
import numpy as np
import pytest
from channel.AWGN import Generate_AWGN

def measure_snr_db(x, y):
    """Računa SNR između čistog i degradiranog signala."""
    signal_power = np.mean(np.abs(x) ** 2)
    noise_power = np.mean(np.abs(y - x) ** 2)
    return 10 * np.log10(signal_power / noise_power)

def test_awgn_output_length():
    """
    Generisani šum mora imati isti broj uzoraka kao ulazni signal
    i oblik koji je kompatibilan za sabiranje sa 1D signalom.
    """
    x = np.ones(1000, dtype=complex)
    n = Generate_AWGN(x, 20, 42)

    assert isinstance(n, np.ndarray)
    assert n.shape in [x.shape, (1, x.size)]

def test_awgn_snr_close_to_target():
    """Nakon što se šum doda na signal, izmjereni SNR treba biti blizu zadanog."""
    x = np.random.randn(50_000) + 1j * np.random.randn(50_000)
    target_snr = 15

    n = Generate_AWGN(x, target_snr, 123)
    y = x + n

    measured_snr = measure_snr_db(x, y)
    assert measured_snr == pytest.approx(target_snr, rel=0.1, abs=1.0)

def test_awgn_high_snr_almost_clean():
    """Za visok SNR (60 dB), greška između ulaza i izlaza mora biti vrlo mala."""
    x = np.random.randn(10_000)
    n = Generate_AWGN(x, 60, 7)
    y = x + n

    error = np.mean(np.abs(y - x))
    assert error < 0.01

def test_awgn_noise_zero_mean():
    """AWGN treba imati srednju vrijednost ≈ 0."""
    x = np.ones(50_000, dtype=complex)
    n = Generate_AWGN(x, 10, 99)

    mean_noise = np.mean(n)
    noise_power = np.mean(np.abs(n) ** 2)

    assert np.abs(mean_noise) < 0.1 * np.sqrt(noise_power)

def test_awgn_reproducible_with_same_seed():
    """Za isti seed i ulaz, generisani šum mora biti identičan."""
    x = np.random.randn(1000) + 1j * np.random.randn(1000)

    n1 = Generate_AWGN(x, 20, sd=42)
    n2 = Generate_AWGN(x, 20, sd=42)

    assert np.array_equal(n1, n2)

def test_awgn_error_monotonic_with_snr():
    """Greška mora biti veća za manji SNR."""
    x = np.random.randn(20_000) + 1j * np.random.randn(20_000)

    n_low = Generate_AWGN(x, 5, sd=1)
    n_high = Generate_AWGN(x, 20, sd=1)

    err_low = np.mean(np.abs((x + n_low) - x))
    err_high = np.mean(np.abs((x + n_high) - x))

    assert err_low > err_high

def test_awgn_nan_in_signal():
    """Input koji sadrži NaN treba izazvati grešku."""
    x = np.array([1.0, np.nan, 2.0])

    with pytest.raises(Exception):
        Generate_AWGN(x, 10, 0)

def test_awgn_empty_input():
    """
    Prazan ulazni signal treba izazvati grešku.
    """
    x = np.array([])

    with pytest.raises(Exception):
        Generate_AWGN(x, 20, 0)

@pytest.mark.parametrize("bad_input", [
    "abc",
    "123",
    None,
    5,
    3.14,
    [1, 2, 3],      # lista umjesto numpy arraya
    {"a": 1},       # dict
])

def test_awgn_invalid_input_type(bad_input):
    """
    Pogrešan tip ulaza (nije numpy array) treba izazvati grešku.
    """
    with pytest.raises(Exception):
        Generate_AWGN(bad_input, 20, 0)
import numpy as np
from channel.Multipath import GetMultipathFilter
import pytest

# Basic parameter set used across tests
FS = 20e6       # 20 MHz sampling rate
TRMS = 50e-9    # 50 ns RMS delay spread
N = 16          # number of taps


def test_multipath_length():
    """Filter must have exactly N taps."""
    h = GetMultipathFilter(FS, TRMS, N)
    assert len(h) == N


def test_multipath_complex_dtype():
    """Filter taps must be complex-valued."""
    h = GetMultipathFilter(FS, TRMS, N)
    assert np.iscomplexobj(h)


def test_multipath_not_all_zero():
    """Generated filter must not be all zeros."""
    h = GetMultipathFilter(FS, TRMS, N)
    assert np.any(np.abs(h) > 0)


def test_multipath_power_decay_on_average():
    """
    On average, earlier taps should have higher power than later ones
    due to the exponential power delay profile.
    """
    h = GetMultipathFilter(FS, TRMS, 64)
    power = np.abs(h) ** 2

    first_half_mean = np.mean(power[:8])
    last_half_mean = np.mean(power[-8:])

    assert first_half_mean > last_half_mean


def test_multipath_zero_mean_statistics():
    """
    Real and imaginary parts should be approximately zero-mean
    (Rayleigh fading assumption).
    """
    h = GetMultipathFilter(FS, TRMS, 128)

    mean_real = np.mean(np.real(h))
    mean_imag = np.mean(np.imag(h))

    assert abs(mean_real) < 0.5
    assert abs(mean_imag) < 0.5


def test_multipath_delay_spread_effect():
    """
    Larger delay spread should result in slower power decay.
    """
    h_small = GetMultipathFilter(FS, 20e-9, 64)
    h_large = GetMultipathFilter(FS, 200e-9, 64)

    power_small = np.abs(h_small) ** 2
    power_large = np.abs(h_large) ** 2

    # Compare tail power relative to first tap
    ratio_small = np.mean(power_small[-8:]) / power_small[0]
    ratio_large = np.mean(power_large[-8:]) / power_large[0]

    assert ratio_large > ratio_small

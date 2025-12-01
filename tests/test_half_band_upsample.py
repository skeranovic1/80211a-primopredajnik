import numpy as np
import pytest


import matplotlib
matplotlib.use("Agg")


import matplotlib.pyplot as plt
plt.show = lambda: None

from tx.filters import half_band_upsample


def test_output_length():
    """Testira da izlaz ima ispravnu dužinu nakon upsamplovanja i filtriranja."""
    signal = np.array([1, 2, 3, 4])
    result, h = half_band_upsample(signal, up_factor=2)

    expected_length = max(len(signal) * 2, len(h))
    assert len(result) == expected_length


def test_output_type():
    """Provjerava da je izlaz numpy array."""
    signal = np.array([1, 2, 3])
    result, h = half_band_upsample(signal)
    assert isinstance(result, np.ndarray)


def test_original_signal_not_changed():
    """Osigurava da ulazni signal nije promijenjen tokom obrade."""
    signal = np.array([1, 2, 3])
    original = signal.copy()
    _ = half_band_upsample(signal)
    assert np.array_equal(signal, original)


def test_filter_basic_properties():
    """Provjerava osnovne karakteristike filtera."""
    _, h = half_band_upsample(np.array([1, 2, 3]))
    assert np.isrealobj(h)
    assert len(h) % 2 == 1  # half-band filter ima neparnu dužinu


def test_invalid_inputs():
    """Testira reakcije na nevalidne parametre i tipove."""
    with pytest.raises(TypeError):
        half_band_upsample("abc")

    with pytest.raises(TypeError):
        half_band_upsample(None)

    with pytest.raises(ValueError):
        half_band_upsample(np.array([1, 2, 3]), up_factor=0)

    with pytest.raises(ValueError):
        half_band_upsample(np.array([1, 2, 3]), up_factor=-2)

    with pytest.raises(TypeError):
        half_band_upsample(np.array([1, 2, 3]), up_factor=2.5)


def test_plot_branch_executes():
    """
    Testira da se plot=True grana izvršava bez greške.
    plt.show() je utišan globalno, tako da nema warninga.
    """
    signal = np.array([1, 2, 3])
    result, h = half_band_upsample(signal, plot=True)

    assert isinstance(result, np.ndarray)
    assert isinstance(h, np.ndarray)

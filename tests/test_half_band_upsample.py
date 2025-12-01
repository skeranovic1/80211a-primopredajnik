import numpy as np
import pytest
from tx.filters import half_band_upsample


def test_output_length():
    """Testira da izlaz ima ispravnu dužinu nakon upsamplovanja i filtriranja."""
    signal = np.array([1, 2, 3, 4])
    result, h = half_band_upsample(signal, up_factor=2)

    # half-band konvolucija radi mode="same"
    # dužina izlaza = max(len(upsampliranog signala), len(filtera))
    expected_length = max(len(signal) * 2, len(h))

    assert len(result) == expected_length, \
        f"Očekivana dužina {expected_length}, a dobijeno {len(result)}"


def test_output_type():
    """Provjerava da je izlaz numpy array."""
    signal = np.array([1, 2, 3])
    result, h = half_band_upsample(signal)
    assert isinstance(result, np.ndarray), "Izlaz mora biti numpy array."


def test_original_signal_not_changed():
    """Osigurava da ulazni signal nije promijenjen tokom obrade."""
    signal = np.array([1, 2, 3])
    original = signal.copy()
    _ = half_band_upsample(signal)
    assert np.array_equal(signal, original), "Originalni signal ne smije biti promijenjen."


def test_filter_basic_properties():
    """Provjerava osnovne osobine filtera: realan i neparne dužine."""
    _, h = half_band_upsample(np.array([1, 2, 3]))
    assert np.isrealobj(h), "Filter mora imati realne koeficijente."
    assert len(h) % 2 == 1, "Filter half-band mora imati neparan broj koeficijenata."


def test_invalid_inputs():
    """Provjerava ponašanje funkcije sa pogrešnim tipovima ulaza i lošim parametrima."""
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

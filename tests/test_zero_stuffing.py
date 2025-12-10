import numpy as np
from tx.utilities import zero_stuffing
import pytest 

def test_zero_stuffing_basic():
    """Provjerava osnovno zero-stuffing upsampliranje sa faktorom 2."""
    signal = np.array([1, 2, 3])
    result = zero_stuffing(signal, up_factor=2)
    expected = np.array([1, 0, 2, 0, 3, 0])
    assert np.array_equal(result, expected)


def test_zero_stuffing_with_upfactor_3():
    """Provjerava zero-stuffing sa faktorom upsampliranja 3."""
    signal = np.array([5, 10])
    result = zero_stuffing(signal, up_factor=3)
    expected = np.array([5, 0, 0, 10, 0, 0])
    assert np.array_equal(result, expected)


def test_zero_stuffing_empty_signal():
    """Provjerava ponašanje funkcije za prazan ulazni signal."""
    signal = np.array([])
    result = zero_stuffing(signal, up_factor=2)
    expected = np.array([])
    assert np.array_equal(result, expected)


def test_zero_stuffing_negative_values():
    """Provjerava da li negativne vrijednosti ostaju očuvane nakon upsampliranja."""
    signal = np.array([-1, -2])
    result = zero_stuffing(signal, up_factor=2)
    expected = np.array([-1, 0, -2, 0])
    assert np.array_equal(result, expected)


def test_zero_stuffing_float_values():
    """Provjerava rad funkcije sa ulaznim signalom tipa float."""
    signal = np.array([1.5, 2.5])
    result = zero_stuffing(signal, up_factor=2)
    expected = np.array([1.5, 0, 2.5, 0])
    assert np.array_equal(result, expected)


def test_zero_stuffing_non_numpy_signal():
    """Provjerava da li se baca TypeError ako ulaz nije numpy array."""
    with pytest.raises(TypeError):
        zero_stuffing([1, 2, 3], up_factor=2)


def test_zero_stuffing_string_instead_of_signal():
    """Provjerava da li se baca TypeError ako je ulazni signal string."""
    with pytest.raises(TypeError):
        zero_stuffing("abc", up_factor=2)


def test_zero_stuffing_none_signal():
    """Provjerava da li se baca TypeError ako je ulazni signal None."""
    with pytest.raises(TypeError):
        zero_stuffing(None, up_factor=2)


def test_zero_stuffing_non_integer_upfactor():
    """Provjerava da li se baca TypeError za ne-cjelobrojni faktor upsampliranja."""
    with pytest.raises(TypeError):
        zero_stuffing(np.array([1,2,3]), up_factor=2.5)


def test_zero_stuffing_negative_upfactor():
    """Provjerava da li se baca ValueError za negativan faktor upsampliranja."""
    with pytest.raises(ValueError):
        zero_stuffing(np.array([1,2,3]), up_factor=-1)


def test_zero_stuffing_zero_upfactor():
    """Provjerava da li se baca ValueError za faktor upsampliranja jednak nuli."""
    with pytest.raises(ValueError):
        zero_stuffing(np.array([1,2,3]), up_factor=0)

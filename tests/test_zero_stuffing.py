import numpy as np
from tx.utilities import zero_stuffing
import pytest 

def test_zero_stuffing_basic():
    signal = np.array([1, 2, 3])
    result = zero_stuffing(signal, up_factor=2)
    expected = np.array([1, 0, 2, 0, 3, 0])
    assert np.array_equal(result, expected)

def test_zero_stuffing_with_upfactor_3():
    signal = np.array([5, 10])
    result = zero_stuffing(signal, up_factor=3)
    expected = np.array([5, 0, 0, 10, 0, 0])
    assert np.array_equal(result, expected)

def test_zero_stuffing_empty_signal():
    signal = np.array([])
    result = zero_stuffing(signal, up_factor=2)
    expected = np.array([])
    assert np.array_equal(result, expected)

def test_zero_stuffing_negative_values():
    signal = np.array([-1, -2])
    result = zero_stuffing(signal, up_factor=2)
    expected = np.array([-1, 0, -2, 0])
    assert np.array_equal(result, expected)

def test_zero_stuffing_float_values():
    signal = np.array([1.5, 2.5])
    result = zero_stuffing(signal, up_factor=2)
    expected = np.array([1.5, 0, 2.5, 0])
    assert np.array_equal(result, expected)

def test_zero_stuffing_non_numpy_signal():
    with pytest.raises(TypeError):
        zero_stuffing([1, 2, 3], up_factor=2)  # nije numpy array


def test_zero_stuffing_string_instead_of_signal():
    with pytest.raises(TypeError):
        zero_stuffing("abc", up_factor=2)


def test_zero_stuffing_none_signal():
    with pytest.raises(TypeError):
        zero_stuffing(None, up_factor=2)


def test_zero_stuffing_non_integer_upfactor():
    with pytest.raises(TypeError):
        zero_stuffing(np.array([1,2,3]), up_factor=2.5)


def test_zero_stuffing_negative_upfactor():
    with pytest.raises(ValueError):
        zero_stuffing(np.array([1,2,3]), up_factor=-1)


def test_zero_stuffing_zero_upfactor():
    with pytest.raises(ValueError):
        zero_stuffing(np.array([1,2,3]), up_factor=0)

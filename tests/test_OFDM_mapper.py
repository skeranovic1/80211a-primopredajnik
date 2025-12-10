import numpy as np
import pytest
from tx.OFDM_mapper import Mapper_OFDM  
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt


def test_invalid_bits():
    """Provjerava bacanje ValueError za bitove van dozvoljenog opsega {0,1}."""
    # negativni bit
    bits = np.array([-1, 0, 1])
    with pytest.raises(ValueError):
        Mapper_OFDM(bits, 1)

    # bit veći od 1
    bits = np.array([0, 2, 1])
    with pytest.raises(ValueError):
        Mapper_OFDM(bits, 1)

    # mješavina validnih i nevalidnih
    bits = np.array([0, 1, 0, 3])
    with pytest.raises(ValueError):
        Mapper_OFDM(bits, 2)


def test_non_divisible_length():
    """Provjerava da se nepotpuni bitovi na kraju ulaza ignorišu."""
    bits = np.array([0, 1, 1])  # dužina 3, BitsPerSymbol = 2
    output = Mapper_OFDM(bits, 2)
    assert len(output) == 1


def test_empty_input():
    """Provjerava da prazan ulazni niz daje prazan izlaz."""
    bits = np.array([], dtype=int)
    output = Mapper_OFDM(bits, 1)
    assert len(output) == 0


def test_large_input():
    """Provjerava ispravno mapiranje velikog broja ulaznih bitova."""
    bits = np.random.randint(0, 2, size=10**6)
    output = Mapper_OFDM(bits, 2)
    assert len(output) == 10**6 // 2


def test_invalid_type_input():
    """Provjerava bacanje IndexError ako bitovi nisu integer tipa."""
    bits = np.array([0.5, 1.0, 0])
    with pytest.raises(IndexError):
        Mapper_OFDM(bits, 1)


def test_qpsk_partial_bits():
    """Provjerava QPSK mapiranje uz ignorisanje zadnjeg nepotpunog bita."""
    bits = np.array([0, 1, 1])
    sqrt2_inv = 1 / np.sqrt(2)
    expected = np.array([-1 * sqrt2_inv + 1j * sqrt2_inv])
    output = Mapper_OFDM(bits, 2)
    np.testing.assert_array_almost_equal(output, expected)


def test_bpsk_mapping():
    """Provjerava ispravnost BPSK mapiranja."""
    bits = np.array([0, 1, 1, 0])
    expected = np.array([-1, 1, 1, -1], dtype=complex)
    output = Mapper_OFDM(bits, 1)
    np.testing.assert_array_almost_equal(output, expected)


def test_qpsk_mapping():
    """Provjerava ispravnost QPSK mapiranja."""
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
    sqrt2_inv = 1 / np.sqrt(2)
    expected = np.array([
        -sqrt2_inv - 1j * sqrt2_inv,
        -sqrt2_inv + 1j * sqrt2_inv,
        sqrt2_inv - 1j * sqrt2_inv,
        sqrt2_inv + 1j * sqrt2_inv
    ], dtype=complex)
    output = Mapper_OFDM(bits, 2)
    np.testing.assert_array_almost_equal(output, expected)


def test_16qam_mapping():
    """Provjerava ispravnost 16-QAM mapiranja."""
    bits = np.array([0,0,0,0, 0,1,1,0, 1,0,1,1, 1,1,1,1])
    sqrt10_inv = 1 / np.sqrt(10)
    expected = np.array([
        (-3 - 3j) * sqrt10_inv,
        (-1 + 1j) * sqrt10_inv,
        (1 + 3j) * sqrt10_inv,
        (3 + 3j) * sqrt10_inv
    ], dtype=complex)
    output = Mapper_OFDM(bits, 4)
    np.testing.assert_array_almost_equal(output, expected)


def test_64qam_mapping():
    """Provjerava ispravnost 64-QAM mapiranja."""
    bits = np.array([
        0,0,0,0,0,0,
        1,0,1,1,1,0
    ])
    sqrt42_inv = 1 / np.sqrt(42)
    Q64 = np.array([-7,-5,-3,-1,1,3,5,7])
    expected = np.array([
        (Q64[0] + 1j * Q64[0]) * sqrt42_inv,
        (Q64[5] + 1j * Q64[6]) * sqrt42_inv
    ], dtype=complex)
    output = Mapper_OFDM(bits, 6)
    np.testing.assert_array_almost_equal(output, expected)


def test_output_length():
    """Provjerava dužinu izlaznog niza za različite modulacije."""
    bits = np.random.randint(0, 2, 24)
    assert len(Mapper_OFDM(bits, 1)) == 24
    assert len(Mapper_OFDM(bits, 2)) == 12
    assert len(Mapper_OFDM(bits, 4)) == 6
    assert len(Mapper_OFDM(bits, 6)) == 4


def test_plot_branch_executes():
    """Provjerava da se plot=True grana izvršava bez greške."""
    bits = np.array([0, 1, 1, 0])
    output = Mapper_OFDM(bits, BitsPerSymbol=2, plot=True)
    assert isinstance(output, np.ndarray)


def test_invalid_bits_small_array():
    """Provjerava validaciju bitova na malim ulaznim nizovima."""
    bits = np.array([0, -1, 1])
    with pytest.raises(ValueError):
        Mapper_OFDM(bits, 1)

    bits = np.array([0, 2, 1])
    with pytest.raises(ValueError):
        Mapper_OFDM(bits, 1)


def test_invalid_bits_large_array_wrap():
    """Provjerava validaciju bitova na većim ulaznim nizovima."""
    bits = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    with pytest.raises(ValueError):
        Mapper_OFDM(bits, 1)


def test_bitsper_symbol_invalid():
    """Provjerava bacanje ValueError za nelegalan BitsPerSymbol."""
    bits = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError):
        Mapper_OFDM(bits, 3)


def test_number_of_symbols_zero():
    """Provjerava slučaj kada nema dovoljno bitova za jedan simbol."""
    bits = np.array([0, 1])
    output = Mapper_OFDM(bits, 4)
    assert len(output) == 0


def test_non_integer_input_type():
    """Provjerava bacanje greške za ne-integer tip ulaznih bitova."""
    bits = np.array([0.0, 1.0])
    with pytest.raises(IndexError):
        Mapper_OFDM(bits, 1)


def test_partial_symbols_16qam():
    """Provjerava ignorisanje nepotpunog 16-QAM simbola."""
    bits = np.array([0,0,0,0, 1,1,1])
    output = Mapper_OFDM(bits, 4)
    assert len(output) == 1


def test_partial_symbols_64qam():
    """Provjerava ignorisanje nepotpunog 64-QAM simbola."""
    bits = np.array([0,0,0,0,0,0, 1,1,1,1,1])
    output = Mapper_OFDM(bits, 6)
    assert len(output) == 1

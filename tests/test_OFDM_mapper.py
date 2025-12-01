import numpy as np
import pytest
from tx.OFDM_mapper import Mapper_OFDM  

def test_invalid_bits():
    # negativni bit
    bits = np.array([-1, 0, 1])
    with pytest.raises(IndexError):
        Mapper_OFDM(bits, 1)
    
    # bit veći od 1
    bits = np.array([0, 2, 1])
    with pytest.raises(IndexError):
        Mapper_OFDM(bits, 1)

def test_non_divisible_length():
    bits = np.array([0, 1, 1])  # duzina 3, BitsPerSymbol = 2
    output = Mapper_OFDM(bits, 2)
    # Funkcija će ignorirati zadnje 1 bit, tako da očekujemo 1 simbol
    assert len(output) == 1

def test_empty_input():
    bits = np.array([], dtype=int)
    output = Mapper_OFDM(bits, 1)
    assert len(output) == 0

def test_large_input():
    bits = np.random.randint(0,2, size=10**6)
    output = Mapper_OFDM(bits, 2)
    # očekujemo pola broja bitova simbol
    assert len(output) == 10**6 // 2

def test_invalid_type_input():
    bits = np.array([0.5, 1.0, 0])
    with pytest.raises(IndexError):
        Mapper_OFDM(bits, 1)

def test_qpsk_partial_bits():
    # ako broj bita nije djeljiv sa 2, zadnji bit se ignorira
    bits = np.array([0,1,1])
    sqrt2_inv = 1/np.sqrt(2)
    expected = np.array([-1*sqrt2_inv + 1j*1*sqrt2_inv])
    output = Mapper_OFDM(bits, 2)
    np.testing.assert_array_almost_equal(output, expected)

def test_bpsk_mapping():
    bits = np.array([0, 1, 1, 0])
    expected = np.array([-1, 1, 1, -1], dtype=complex)
    output = Mapper_OFDM(bits, 1)
    np.testing.assert_array_almost_equal(output, expected)

def test_qpsk_mapping():
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
    # QPSK_LUT = [-1, 1]/sqrt(2)
    sqrt2_inv = 1/np.sqrt(2)
    expected = np.array([
        -1*sqrt2_inv + 1j*-1*sqrt2_inv,  # 0,0
        -1*sqrt2_inv + 1j*1*sqrt2_inv,   # 0,1
        1*sqrt2_inv + 1j*-1*sqrt2_inv,   # 1,0
        1*sqrt2_inv + 1j*1*sqrt2_inv     # 1,1
    ], dtype=complex)
    output = Mapper_OFDM(bits, 2)
    np.testing.assert_array_almost_equal(output, expected)

def test_16qam_mapping():
    bits = np.array([0,0,0,0, 0,1,1,0, 1,0,1,1, 1,1,1,1])
    sqrt10_inv = 1/np.sqrt(10)
    expected = np.array([
        (-3 + 1j*-3)*sqrt10_inv,
        (-1 + 1j*1)*sqrt10_inv,
        (1 + 1j*3)*sqrt10_inv,
        (3 + 1j*3)*sqrt10_inv
    ], dtype=complex)
    output = Mapper_OFDM(bits, 4)
    np.testing.assert_array_almost_equal(output, expected)

def test_64qam_mapping():
    bits = np.array([
        0,0,0,0,0,0,   # simbol 1
        1,0,1,1,1,0    # simbol 2
    ])
    sqrt42_inv = 1/np.sqrt(42)
    Q64 = np.array([-7,-5,-3,-1,1,3,5,7])
    expected = np.array([
        (Q64[0*4+0*2+0] + 1j*Q64[0*4+0*2+0])*sqrt42_inv,
        (Q64[1*4+0*2+1] + 1j*Q64[1*4+1*2+0])*sqrt42_inv
    ], dtype=complex)
    output = Mapper_OFDM(bits, 6)
    np.testing.assert_array_almost_equal(output, expected)

def test_output_length():
    bits = np.arange(24)
    # BPSK
    assert len(Mapper_OFDM(bits, 1)) == 24
    # QPSK
    assert len(Mapper_OFDM(bits, 2)) == 12
    # 16-QAM
    assert len(Mapper_OFDM(bits, 4)) == 6
    # 64-QAM
    assert len(Mapper_OFDM(bits, 6)) == 4

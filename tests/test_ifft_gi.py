import numpy as np
from tx.ifft_ofdm_symbol import IFFT_GI


# Test 1 — IFFT output mora imati pravilan broj uzoraka (80 po simbolu)
def test_ifft_gi_output_length_single_symbol():
    """
    Jedan OFDM simbol: ulaz ima 48 QAM simbola,
    izlaz treba imati 80 uzoraka (16 GI + 64 IFFT).
    """
    num_symbols = 1
    num_data_carriers = 48

    symbol_stream = np.ones(num_symbols * num_data_carriers, dtype=complex)
    payload = IFFT_GI(symbol_stream)

    assert len(payload) == num_symbols * 80
    assert payload.dtype == complex


# Test 2 — IFFT output dužina mora skalirati kao N * 80 za više simbola
def test_ifft_gi_output_length_multiple_symbols():
    """
    Više OFDM simbola: provjera da se dužina skalira kao N * 80.
    """
    num_symbols = 5
    num_data_carriers = 48

    symbol_stream = np.arange(num_symbols * num_data_carriers, dtype=float) + 1j
    payload = IFFT_GI(symbol_stream)

    assert len(payload) == num_symbols * 80


# Test 3 — CP mora biti jednak zadnjih 16 uzoraka IFFT simbola (1 simbol)
def test_ifft_gi_cyclic_prefix_content_single_symbol():
    """
    Provjera da je ciklički prefiks za jedan simbol tačno
    jednak zadnjih 16 uzoraka osnovnog OFDM simbola.
    """
    num_symbols = 1
    num_data_carriers = 48

    rng = np.random.default_rng(seed=42)
    symbol_stream = rng.normal(size=num_symbols * num_data_carriers) + 1j * rng.normal(
        size=num_symbols * num_data_carriers
    )

    payload = IFFT_GI(symbol_stream)

    block = payload[0:80]
    gi = block[:16]
    ofdm_symbol = block[16:]

    assert len(ofdm_symbol) == 64
    assert np.allclose(gi, ofdm_symbol[48:64])


# Test 4 — CP mora biti ispravan i za više OFDM simbola (nema preklapanja)
def test_ifft_gi_cyclic_prefix_content_multiple_symbols():
    """
    Provjera da je za svaki simbol prefiks jednak repu tog istog simbola.
    """
    num_symbols = 3
    num_data_carriers = 48

    rng = np.random.default_rng(seed=123)
    symbol_stream = rng.normal(size=num_symbols * num_data_carriers) + 1j * rng.normal(
        size=num_symbols * num_data_carriers
    )

    payload = IFFT_GI(symbol_stream)

    for i in range(num_symbols):
        start = i * 80
        stop = start + 80
        block = payload[start:stop]

        gi = block[:16]
        ofdm_symbol = block[16:]

        assert len(ofdm_symbol) == 64
        assert np.allclose(gi, ofdm_symbol[48:64])


# Test 5 — IFFT mora biti inverzna operacija FFT-u
def test_ifft_gi_is_inverse_of_fft():
    """
    Test da ifft(fft(x)) vraća originalni vektor x.
    """
    x = np.array([1, 2, 3, 4], dtype=complex)
    X = np.fft.fft(x)
    x_rec = np.fft.ifft(X)
    assert np.allclose(x, x_rec)


# Test 6 — Ciklički prefiks mora uvijek imati dužinu 16 uzoraka
def test_ifft_gi_cp_has_correct_length():
    """
    Provjera da CP uvijek ima 16 uzoraka.
    """
    num_symbols = 2
    num_data_carriers = 48

    stream = np.ones(num_symbols * num_data_carriers, dtype=complex)
    payload = IFFT_GI(stream)

    for i in range(num_symbols):
        block = payload[i * 80 : (i + 1) * 80]
        cp = block[:16]
        assert len(cp) == 16


# Test 7 — Ukupna dužina svakog bloka mora biti tačno 80 uzoraka
def test_ifft_gi_block_total_length():
    """
    CP (16) + IFFT simbol (64) = 80 uzoraka po bloku.
    """
    num_symbols = 4
    num_data_carriers = 48

    stream = np.arange(num_symbols * num_data_carriers, dtype=complex)
    payload = IFFT_GI(stream)

    for i in range(num_symbols):
        block = payload[i * 80 : (i + 1) * 80]
        assert len(block) == 80


# Test 8 — Funkcija mora raditi stabilno za različite modulacije i seedove
def test_ifft_gi_random_seeds_and_modulations():
    """
    Test stabilnosti za različite seedove i modulacije: BPSK, QPSK, 16QAM.
    """
    seeds = [0, 42, 999]
    bits_per_symbol_list = [1, 2, 4]  # BPSK, QPSK, 16QAM

    for seed in seeds:
        for bps in bits_per_symbol_list:
            num_symbols = 2
            num_data_carriers = 48

            rng = np.random.default_rng(seed)

            # generiši random QAM simbole svih modulacija
            if bps == 1:  # BPSK
                symbols = (rng.integers(0, 2, num_symbols * num_data_carriers) * 2 - 1).astype(
                    complex
                )
            elif bps == 2:  # QPSK
                real = rng.choice([-1, 1], num_symbols * num_data_carriers)
                imag = rng.choice([-1, 1], num_symbols * num_data_carriers)
                symbols = (real + 1j * imag) / np.sqrt(2)
            else:  # 16QAM
                const = np.array([-3, -1, 1, 3]) / np.sqrt(10)
                real = rng.choice(const, num_symbols * num_data_carriers)
                imag = rng.choice(const, num_symbols * num_data_carriers)
                symbols = real + 1j * imag

            payload = IFFT_GI(symbols)

            # osnovne provjere integriteta
            assert len(payload) == num_symbols * 80
            for i in range(num_symbols):
                block = payload[i * 80 : (i + 1) * 80]
                assert len(block) == 80
                cp = block[:16]
                symbol = block[16:]
                assert np.allclose(cp, symbol[48:64])

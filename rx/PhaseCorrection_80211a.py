import numpy as np

def phase_correction_80211a(equalized_symbol,
                            equalizer_coeffs,
                            L):
    """
        Fazna korekcija za IEEE 802.11a OFDM
    - Korekcija zajedničkog faznog pomaka (CPE)
    - Praćenje faze zasnovano na pilot-tonovima
    - Izlaz: fazno ispravljeni podatkovni subnosioci (48)

    Parametri
    ----------
    equalized_symbol : np.ndarray (64,)
        FFT izlaz OFDM simbola nakon kanalske ekvilizacije
    equalizer_coeffs : np.ndarray (64,)
        Koeficijenti ekvilajzera
    average_slope_filter : np.ndarray (L,)
        Stanje filtra za usrednjavanje faznog nagiba

    Povratne vrijednosti
    --------------------
    data_symbols : np.ndarray (48,)
        Fazno ispravljeni podatkovni subnosioci
    average_slope_filter : np.ndarray
        Ažurirano stanje filtra faznog nagiba

    """

    average_slope_filter = np.zeros(L)

    # === 1. Pilot indices ===
    idx_m21 = 43
    idx_m07 = 57
    idx_p07 = 7
    idx_p21 = 21

    # === 2. Extract pilots ===
    pilot_m21 = equalized_symbol[idx_m21]
    pilot_m07 = equalized_symbol[idx_m07]
    pilot_p07 = equalized_symbol[idx_p07]
    pilot_p21 = equalized_symbol[idx_p21]

    # === 3. Pilot weights from equalizer coefficients ===
    w1 = 1.0 / np.abs(equalizer_coeffs[idx_m21])
    w2 = 1.0 / np.abs(equalizer_coeffs[idx_m07])
    w3 = 1.0 / np.abs(equalizer_coeffs[idx_p07])
    w4 = 1.0 / np.abs(equalizer_coeffs[idx_p21])

    w_sum = w1 + w2 + w3 + w4
    C1, C2, C3, C4 = w1/w_sum, w2/w_sum, w3/w_sum, w4/w_sum

    # === 4. Common Phase Error (CPE) correction ===
    averaged_pilot = (C1 * pilot_m21 +
                      C2 * pilot_m07 +
                      C3 * pilot_p07 +
                      C4 * pilot_p21)

    theta = np.angle(averaged_pilot)
    symbol_cpe_corrected = equalized_symbol * np.exp(-1j * theta)

    # === 5. Phase slope estimation ===
    pilot_m21 = symbol_cpe_corrected[idx_m21]
    pilot_m07 = symbol_cpe_corrected[idx_m07]
    pilot_p07 = symbol_cpe_corrected[idx_p07]
    pilot_p21 = symbol_cpe_corrected[idx_p21]

    slope = (-C1 * np.angle(pilot_m21) / 21
             -C2 * np.angle(pilot_m07) / 7
             +C3 * np.angle(pilot_p07) / 7
             +C4 * np.angle(pilot_p21) / 21)

    # === 6. Average slope filter ===
    average_slope_filter[1:] = average_slope_filter[:-1]
    average_slope_filter[0] = slope
    avg_slope = np.mean(average_slope_filter)

    # === 7. Apply phase slope correction ===
    step_plus = np.arange(0, 32) * avg_slope
    step_minus = np.arange(-32, 0) * avg_slope
    applied_correction = np.concatenate((step_plus, step_minus))

    symbol_phase_corrected = (
        symbol_cpe_corrected * np.exp(-1j * applied_correction)
    )

    # === 8. Extract data subcarriers (48) ===
    data_indices = np.array(
        list(range(0, 6)) +
        list(range(7, 20)) +
        list(range(21, 26)) +
        list(range(37, 42)) +
        list(range(43, 56)) +
        list(range(57, 63))
    )

    data_symbols = symbol_phase_corrected[data_indices]

    return data_symbols, average_slope_filter
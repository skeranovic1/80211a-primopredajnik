import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def long_symbol_correlator(long_training_symbol,
                           rx_waveform,
                           falling_edge_position):
    """
    Python equivalent of MATLAB LongSymbol_Correlator
    """

    # --- Sign-normalized LTS (±1 ± j) ---
    L = np.sign(np.real(long_training_symbol)) + \
        1j * np.sign(np.imag(long_training_symbol))

    rx_len = len(rx_waveform)

    output_long = np.zeros(rx_len, dtype=complex)
    lt_peak_value = 0 + 0j
    lt_peak_position = 0

    cross_correlator = np.zeros(64, dtype=complex)

    for i in range(rx_len - 64):

        # Cross-correlation
        output = np.dot(cross_correlator, np.conj(L[::-1]))

        output_long[i] = output

        # Shift register
        cross_correlator[1:] = cross_correlator[:-1]
        cross_correlator[0] = rx_waveform[i]

        # Search window for LTS
        if (i > falling_edge_position + 54) and \
           (i < falling_edge_position + 54 + 64):

            if abs(output) > abs(lt_peak_value):
                lt_peak_value = output
                lt_peak_position = i

    return lt_peak_value, lt_peak_position, output_long


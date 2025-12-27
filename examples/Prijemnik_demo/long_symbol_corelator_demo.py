import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tx.long_sequence import get_long_training_sequence
from rx.long_symbol_correlator import long_symbol_correlator


# Example usage
lts = get_long_training_sequence()
lts = get_long_training_sequence()  # 160 uzorka
lts_td = lts[32+64:32+128]  # izdvaja drugi LTS simbol, 64 uzorka, bez CP
lts_td = np.fft.ifft(lts_td)
rx = np.zeros(400, dtype=complex)
rx[150:150+64] = lts_td
rx += 0.05 * (np.random.randn(400) + 1j*np.random.randn(400))

peak_val, peak_pos, corr = long_symbol_correlator(
    lts_td,
    rx,
    falling_edge_position=100
)

print("LTS peak at:", peak_pos)
print("LTS start index:", peak_pos - 63)
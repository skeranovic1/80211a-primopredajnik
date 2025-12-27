import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tx.long_sequence import get_long_training_sequence
from rx.long_symbol_correlator import long_symbol_correlator
from rx.estimacija_kanala import channel_estimation, ofdm_eq


# --- Generisanje LTS ---
lts = get_long_training_sequence()         # 160 uzoraka LTS sa CP
lts_td = lts[32+64:32+128]                 # drugi LTS simbol, 64 uzorka, bez CP
lts_fft = np.fft.fft(lts_td, n=64)

# --- Simulacija prijemnog signala sa šumom ---
rx_len = 400
rx = np.zeros(rx_len, dtype=complex)
insertion_pos = 150
rx[insertion_pos:insertion_pos+64] = lts_td    # ubacimo LTS u rx
rx += 0.5*(np.random.randn(rx_len) + 1j*np.random.randn(rx_len))  # AWGN

peak_val, lt_peak_pos, corr = long_symbol_correlator(
    lts_td,
    rx,
    falling_edge_position=100
)
channel_est, equalizer_coeffs = channel_estimation(rx, lt_peak_pos)

# --- Prikaz rezultata ---
plt.figure(figsize=(12,4))
plt.plot(np.abs(channel_est), 'o-', label='Channel Estimate |H(f)|')
plt.title('Channel Estimate')
plt.xlabel('Subcarrier Index')
plt.ylabel('|H(f)|')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12,4))
plt.plot(np.abs(equalizer_coeffs), 'o-', label='Equalizer Coefficients |1/H(f)|')
plt.title('Equalizer Coefficients')
plt.xlabel('Subcarrier Index')
plt.ylabel('|1/H(f)|')
plt.grid(True)
plt.legend()
plt.show()

print("Prvih 8 koeficijenata kanala:", channel_est[:8])
print("Prvih 8 koeficijenata equalizera:", equalizer_coeffs[:8])


rx_eq=ofdm_eq(rx, equalizer_coeffs)

first_symbol_fft = rx_eq[0]  # prvi equalizirani OFDM simbol

plt.figure(figsize=(12,4))
plt.plot(np.real(first_symbol_fft), 'o-', label='Real part of first equalized OFDM symbol')
plt.title("Equalized OFDM Symbol (Real Part)")
plt.xlabel("Subcarrier index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12,4))
plt.plot(np.imag(first_symbol_fft), 'o-', label='Imaginary part of first equalized OFDM symbol')
plt.title("Equalized OFDM Symbol (Imaginary Part)")
plt.xlabel("Subcarrier index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()


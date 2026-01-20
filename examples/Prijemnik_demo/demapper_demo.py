import numpy as np
import matplotlib.pyplot as plt
import sys, os
from scipy.special import erfc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tx.OFDM_TX_802_11 import Transmitter80211a
from channel.Channel_Model import Channel_Model
from channel.channel_settings import ChannelSettings
from channel.channel_mode import ChannelMode
from rx.RX_802_11_a import Receiver80211a

# --- Konfiguracija ---
num_ofdm_symbols = 1000
up_factor = 2
fs = 20e6 * up_factor
snr_range = np.arange(0, 22, 1)
ber_simulated = []

print("Pokrećem simulaciju za Multipath + AWGN...")

for snr in snr_range:
    # Predajnik
    tx = Transmitter80211a(num_ofdm_symbols=num_ofdm_symbols, bits_per_symbol=2, up_factor=up_factor)
    tx_signal, original_bits, tx_symbols = tx.generate_frame()

    # Kanal: Multipath (2 tapa) + Termički šum (AWGN)
    settings = ChannelSettings(sample_rate=fs, number_of_taps=2, snr_db=snr)
    mode = ChannelMode(multipath=1, thermal_noise=1)
    channel = Channel_Model(settings, mode)
    rx_signal, _ = channel.apply(tx_signal)

    try:
        # Prijemnik
        rx = Receiver80211a(fs=fs, num_symbols=num_ofdm_symbols)
        corrected_symbols = rx.process_signal(rx_signal, tx_signal)
        
        tx_s = tx_symbols.flatten()
        rx_s = corrected_symbols.flatten()
        
        min_len = min(len(tx_s), len(rx_s))
        tx_s = tx_s[:min_len]
        rx_s = rx_s[:min_len]

        # QPSK demodulacija (hard decision)
        b0_tx, b1_tx = (tx_s.real > 0), (tx_s.imag > 0)
        b0_rx, b1_rx = (rx_s.real > 0), (rx_s.imag > 0)

        errors = np.sum(b0_tx != b0_rx) + np.sum(b1_tx != b1_rx)
        current_ber = errors / (2 * min_len)
        
    except Exception as e:
        current_ber = 0.5
    
    ber_simulated.append(current_ber)
    print(f"SNR: {snr:2d} dB | BER: {current_ber:.5f}")

# Formula: BER = 0.5 * (1 - sqrt(EbNo / (1 + EbNo)))
snr_lin = 10**(snr_range/10)
ber_theoretical_multipath = 0.5 * (1 - np.sqrt(snr_lin / (1 + snr_lin)))

plt.figure(figsize=(10, 6))
plt.semilogy(snr_range, ber_simulated, 'bo-', linewidth=2, label='Simulacija (Multipath+AWGN)')
plt.semilogy(snr_range, ber_theoretical_multipath, 'r--', linewidth=2, label='Teoretski Rayleigh Fading')
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.xlabel('SNR [dB]')
plt.ylabel('Bit Error Rate (BER)')
plt.title('BER: Simulirani Multipath kanal vs Teoretski Rayleigh model')
plt.legend()
plt.ylim([1e-5, 1])
plt.show()
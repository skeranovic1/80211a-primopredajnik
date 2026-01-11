import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tx.OFDM_TX_802_11 import Transmitter80211a
from channel.Channel_Model import Channel_Model
from channel.channel_settings import ChannelSettings
from channel.channel_mode import ChannelMode
from rx.RX_802_11_a import Receiver80211a

num_ofdm_symbols=15
up_factor=2
fs_base=20e6
fs=fs_base*up_factor

#Predajnik
tx=Transmitter80211a(
        num_ofdm_symbols=num_ofdm_symbols,
        bits_per_symbol=2,   #QPSK
        step=1,
        up_factor=up_factor,
        seed=3,
        plot=False
)
tx_signal, _, _= tx.generate_frame()

#Kanal
settings=ChannelSettings(
        sample_rate=fs,
        number_of_taps=2,
        delay_spread=10e-9,
        snr_db=10
)
mode=ChannelMode(
        multipath=1,
        thermal_noise=1
)
channel=Channel_Model(settings, mode)

#Prijemnik
rx_signal, fir_taps=channel.apply(tx_signal)
rx = Receiver80211a(
    fs=fs,
    num_symbols=num_ofdm_symbols,
    nfft=64,
    ncp=16
)

corrected_symbols = rx.process_signal(rx_signal, tx_signal)

plt.figure(figsize=(6,6))
plt.scatter(corrected_symbols.real.flatten(),corrected_symbols.imag.flatten(),s=5,alpha=0.6)
plt.axis("equal")
plt.grid(True)
plt.title(f"Konstelacija svih OFDM simbola ({num_ofdm_symbols})")
plt.show()

ekv = rx.symbols_fd * rx.eq_coefficient

plt.figure(figsize=(10,4))
plt.plot(np.fft.fftshift(np.abs(ekv[0])), label="Prije korekcije")
plt.plot(np.fft.fftshift(np.abs(corrected_symbols[0])), label="Poslije korekcije")
plt.grid()
plt.legend()
plt.title("Spektar podnosioca - prvi simbol")
plt.show()

plt.figure(figsize=(12,4))
plt.plot(np.abs(rx.channel_est), 'o-', label='|H(f)|')
plt.plot(np.abs(rx.eq_coefficient), 'x-', label='|1/H(f)|')
plt.grid()
plt.legend()
plt.title("Procjena kanala i ekvilajzer")
plt.show()
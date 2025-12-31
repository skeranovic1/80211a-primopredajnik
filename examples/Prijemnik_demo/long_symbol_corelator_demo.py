import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tx.long_sequence import get_long_training_sequence
from rx.long_symbol_correlator import long_symbol_correlator
from channel.Channel_Model import Channel_Model
from channel.channel_settings import ChannelSettings
from channel.channel_mode import ChannelMode
from tx.OFDM_TX_802_11 import Transmitter80211a
from rx.detection import packet_detector   
from rx.pretprocessing import iq_preprocessing
from scipy.signal import find_peaks
from rx.cfo import detect_frequency_offsets

num_ofdm_symbols=3
up_factor=2
fs_base=20e6    
fs=fs_base*up_factor

tx=Transmitter80211a(
        num_ofdm_symbols=num_ofdm_symbols,
        bits_per_symbol=2,   #QPSK
        step=1,
        up_factor=up_factor,
        seed=17,
        plot=False
)
tx_signal, _=tx.generate_frame()

#Kanal
settings=ChannelSettings(
        sample_rate=fs,
        number_of_taps=2,
        delay_spread=10,
        snr_db=20
)
mode=ChannelMode(
        multipath=1,
        thermal_noise=1
)
channel=Channel_Model(settings, mode)
rx_signal, _=channel.apply(tx_signal)

rx_signal, fs =iq_preprocessing(
        rx_signal=rx_signal,
        tx_signal=tx_signal,
        fs=fs
)

_, _, packet_start, _ = packet_detector(rx_signal)

FreqOffset=detect_frequency_offsets(rx_signal,packet_start)
CoarseOffset=FreqOffset[0]
print(f"Coarse CFO = {CoarseOffset:.2f} Hz")
n=np.arange(len(rx_signal))
NCO_coarse=np.exp(-1j*2*np.pi*n*CoarseOffset/fs)
rx_coarse=rx_signal*NCO_coarse

lts = get_long_training_sequence()
lts_td = lts[32+64 : 32+128]  #Drugi LTS simbol, uzimamo bez CP-a za korelaciju

peak_val, peak_pos, corr = long_symbol_correlator(lts_td,rx_coarse,falling_edge_position=packet_start)
lts_start = peak_pos - 64
print("LTS peak position:", peak_pos)
print("LTS start index:", lts_start)

lts_rx_only = rx_coarse[lts_start : lts_start + 128]  #Izdvajanje LTS dijela, oba simbola

#Automatsko pronalaženje oba LTS peaka sa find_peaks is scipy biblioteke
search_window = corr[lts_start : lts_start + 128 + 10]  #search window sa malo dodatnog prostora
peaks, _ = find_peaks(np.abs(search_window),height=0.5*np.max(np.abs(search_window)))
peaks_global = peaks + (lts_start) # prilagođavanje globalnim indeksima rx_signal
print("Detected LTS peaks (ignoring CP):", peaks_global) #ispis pozicije

#Plot korelacije
t_corr_full = np.arange(len(corr)) / fs * 1e6  # mikrosekunde
plt.figure(figsize=(12, 4))
plt.plot( np.abs(corr), label="|Cross-correlation|")
plt.title("LTS Cross-Correlation (Full RX Signal)")
plt.xlabel("Time [µs]")
plt.ylabel("|Correlation|")
plt.grid(True)
plt.tight_layout()
plt.show()
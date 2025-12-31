import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tx.OFDM_TX_802_11 import Transmitter80211a
from channel.Channel_Model import Channel_Model
from channel.channel_settings import ChannelSettings
from channel.channel_mode import ChannelMode
from rx.pretprocessing import iq_preprocessing
from rx.detection import packet_detector
from rx.cfo import detect_frequency_offsets
from tx.long_sequence import get_long_training_sequence
from rx.long_symbol_correlator import long_symbol_correlator
from rx.estimacija_kanala import channel_estimate_and_equalizer
from rx.PhaseCorrection_80211a import phase_correction_80211a

#Tx signal
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
        number_of_taps=10,
        delay_spread=10,
        snr_db=20
)
mode=ChannelMode(
        multipath=1,
        thermal_noise=1
)
channel=Channel_Model(settings, mode)
rx_signal, _=channel.apply(tx_signal)

#Rx
rx_signal, fs1 =iq_preprocessing(
        rx_signal=rx_signal,
        tx_signal=tx_signal,
        fs=fs
)

_, _, packet_start, _ = packet_detector(rx_signal)

FreqOffset=detect_frequency_offsets(rx_signal,packet_start)
CoarseOffset=FreqOffset[0]
n=np.arange(len(rx_signal))
NCO_coarse=np.exp(-1j*2*np.pi*n*CoarseOffset/fs1)
rx_coarse=rx_signal*NCO_coarse

#Fine/precizna korekcija
FreqOffset=detect_frequency_offsets(rx_coarse,packet_start)  #ponovo se pokreće detekcija za fine offset
FineOffset=FreqOffset[1]
NCO_fine=np.exp(-1j*2*np.pi*n*FineOffset/fs1)
rx_fine=rx_coarse*NCO_fine

lts = get_long_training_sequence()
lts_td = lts[32+64 : 32+128]  #Drugi LTS simbol, uzimamo bez CP-a za korelaciju
_, peak_pos, corr = long_symbol_correlator(lts_td,rx_fine,falling_edge_position=packet_start)
lts_start = peak_pos - 64

#Estimacija kanala i koeficijenti equalizera
channel_est, equalizer_coeffs = channel_estimate_and_equalizer(rx_fine, lts_start)

#Fazna korekcija
corrected_data_symbols = phase_correction_80211a(rx_fine, num_ofdm_symbols, lts_start, channel_est, equalizer_coeffs)

#Tx signal
plt.figure(figsize=(12,4))
plt.plot(np.real(tx_signal), label='Real part')
plt.plot(np.imag(tx_signal), label='Imag part')
plt.title('Transmitted OFDM Signal')
plt.xlabel('Sample index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

#Rx signal prije korekcija
plt.figure(figsize=(12,4))
plt.plot(np.real(rx_signal), label='Real part')
plt.title('Received OFDM Signal (raw)')
plt.xlabel('Sample index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

#Rx signal nakon CFO
plt.figure(figsize=(12,4))
plt.plot(np.real(rx_coarse), label='Real part')
plt.title('RX Signal after Coarse Frequency Correction')
plt.xlabel('Sample index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

#Rx signal nakon fine FO
plt.figure(figsize=(12,4))
plt.plot(np.real(rx_fine), label='Real part')
plt.title('RX Signal after Fine Frequency Correction')
plt.xlabel('Sample index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()

#LTS korelacija
plt.figure(figsize=(12,4))
plt.plot(np.abs(corr))
plt.title('LTS Correlation Magnitude')
plt.xlabel('Sample index')
plt.ylabel('|Correlation|')
plt.grid(True)
plt.show()

#Kanal i equalizer
plt.figure(figsize=(12,4))
plt.plot(np.abs(channel_est), 'o-', label='Channel Estimate |H(f)|')
plt.plot(np.abs(equalizer_coeffs), 'x-', label='Equalizer |1/H(f)|')
plt.title('Channel and Equalizer')
plt.xlabel('Subcarrier Index')
plt.ylabel('Magnitude')
plt.grid(True)
plt.legend()
plt.show()

#Ekvualizirani i fazno korigirani prvi OFDM simbol
first_symbol = corrected_data_symbols[0]
plt.figure(figsize=(12,4))
plt.plot(np.real(first_symbol), 'o-', label='Real part')
plt.title('First Equalized & Phase-Corrected OFDM Symbol')
plt.xlabel('Subcarrier index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tx.long_sequence import get_long_training_sequence
from rx.long_symbol_correlator import long_symbol_correlator

def channel_estimation(rx_waveform, lt_peak_pos):
    """
    Estimacija kanala i racunanje equalizer koeficijenata.
    """
    # Izvlacenje oba LTS simbola i prosjek
    first_lts = rx_waveform[lt_peak_pos-64 : lt_peak_pos]
    second_lts = rx_waveform[lt_peak_pos : lt_peak_pos+64]
    averaged_lts = 0.5*first_lts + 0.5*second_lts
    
    # FFT
    lts_fft = np.fft.fft(averaged_lts, n=64)
    rx_positive_tones = lts_fft[1:27]
    rx_negative_tones = lts_fft[38:64]
    rx_tones = np.concatenate([rx_negative_tones, rx_positive_tones])
    
    # Ideal tones
    all_tones = get_long_training_sequence()
    ideal_fft = np.fft.fft(all_tones[32+64:32+128], n=64)
    ideal_positive_tones = ideal_fft[1:27]
    ideal_negative_tones = ideal_fft[38:64]
    ideal_tones = np.concatenate([ideal_negative_tones, ideal_positive_tones])
    
    # Channel estimate
    channel_est = rx_tones / ideal_tones
    equalizer_coeffs = 1 / channel_est
    return channel_est, equalizer_coeffs

def equalize_ofdm_symbol(ofdm_symbol, equalizer_coeffs, n_fft=None):
    if n_fft is None:
        n_fft = len(ofdm_symbol)

    # FFT simbola
    symbol_fft = np.fft.fft(ofdm_symbol, n=n_fft)

    # Mapiranje equalizer koeficijenata na 64 subcarriera
    eq_full = np.ones(n_fft, dtype=complex)

    # Data+pilot subcarrier-i u 802.11a (0=DC, 1-26 + 38-63 = data+pilot, 27-37 null)
    data_idx = np.hstack([np.arange(1, 27), np.arange(38, 64)])

    if len(equalizer_coeffs) != len(data_idx):
        raise ValueError(f"Equalizer length {len(equalizer_coeffs)} ne odgovara broju data subcarrier-a {len(data_idx)}")

    eq_full[data_idx] = equalizer_coeffs

    # Equalizacija
    equalized_symbol = symbol_fft * eq_full
    return equalized_symbol

def ofdm_eq(rx_signal, equalizer_coeffs, symbol_len=64):
    """
    Equalizuje cijeli OFDM paket simbol-po-simbol koristeći equalizer koeficijente.

    Parameters:
    - rx_signal: kompleksni primljeni signal (bez CP)
    - equalizer_coeffs: koeficijenti equalizera (za data + pilot subcarriere, npr. 52)
    - symbol_len: dužina OFDM simbola (default 64 za 802.11a)

    Returns:
    - equalized_symbols: lista FFT-ova equaliziranih simbola
    """
    num_symbols = len(rx_signal) // symbol_len
    equalized_symbols = []

    for i in range(num_symbols):
        start = i * symbol_len
        symbol = rx_signal[start:start+symbol_len]
        equalized_fft = equalize_ofdm_symbol(symbol, equalizer_coeffs, n_fft=symbol_len)
        equalized_symbols.append(equalized_fft)

    return equalized_symbols

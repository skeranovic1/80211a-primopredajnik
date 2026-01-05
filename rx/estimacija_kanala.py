import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tx.long_sequence import get_long_training_sequence

def channel_estimate_and_equalizer(signal,lts_start):
    """
    Procjena kanala i računanje koeficijenata ekvalajzera na osnovu dugih trening simbola (LTS).

    Parametri
    signal : np.ndarray
        Primljeni (RX) signal, oblika (1, N) ili (N,)
    lts_start : int
        Indeks početka LTS-a 

    Povratne vrijednosti
    channel_estimate : np.ndarray
        Procijenjeni frekvencijski odziv kanala
    equalizer_coefficients : np.ndarray
        Koeficijenti ekvilajzera (1 / procjena kanala)
    """
    #1D signal
    signal = np.squeeze(signal)

    #Izdvajanje dugih trening simbola i njihovo usrednjavanje
    first_long_symbol = signal[lts_start : lts_start+64]
    second_long_symbol = signal[lts_start+64 : lts_start+128]
    averaged_long_training_symbol = (0.5 * first_long_symbol + 0.5 * second_long_symbol)

    #FFT i izdvajanje podnosača (tonova)
    fft_of_long_training_symbol = 1/64 * np.fft.fft(averaged_long_training_symbol)

    rx_positive_tones = fft_of_long_training_symbol[1:27]
    rx_negative_tones = fft_of_long_training_symbol[38:64]

    #Idealni tonovi
    all_tones = get_long_training_sequence()
    ideal_lts_td = all_tones[32 + 64 : 32 + 128]  # 64 uzorka bez CP (LTS)
    ideal_lts_fd = 1/64 * np.fft.fft(ideal_lts_td)

    ideal_positive_tones = ideal_lts_fd[1:27]
    ideal_negative_tones = ideal_lts_fd[38:64]

    channel_est_pos = rx_positive_tones / ideal_positive_tones
    channel_est_neg = rx_negative_tones / ideal_negative_tones
    channel_estimate = np.concatenate((np.zeros(1), channel_est_pos, np.zeros(11), channel_est_neg))

    #Estimacija kanala i koeficijenti ekvalajzera
    equalizer_coeff_pos = 1.0 / channel_est_pos
    equalizer_coeff_neg = 1.0 / channel_est_neg
    equalizer_coefficients = np.concatenate((np.zeros(1), equalizer_coeff_pos, np.zeros(11), equalizer_coeff_neg))
    
    return channel_estimate, equalizer_coefficients

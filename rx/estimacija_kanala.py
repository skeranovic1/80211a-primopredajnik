import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tx.long_sequence import get_long_training_sequence
import matplotlib.pyplot as plt

def channel_estimate_and_equalizer(lts_signal, lts_start=0, plot=False):
    """
    Procjenjuje kanal i kreira ZF (Zero-Forcing) equalizer koristeći Long Training Sequence (LTS).

    Funkcija koristi dva uzastopna LTS simbola iz primljenog signala za procjenu kanala u frekvencijskom domenu i kreira equalizer 
    koji se koristi za korekciju efekata kanala u OFDM sistemima (802.11).

    Parameters
    lts_signal : np.ndarray
        Primljeni signal koji sadrži barem 128 uzoraka LTS (2 LTS simbola po 64 uzorka).
    lts_start : int, optional
        Početni indeks prvog LTS simbola unutar 'lts_signal'. Default je 0.
   
    Returns
    H : np.ndarray, shape (64,)
        Procjena kanala u frekventnom domenu za 64 subnosioca.
    equalizer : np.ndarray, shape (64,)
        ZF equalizer kreiran iz procijenjenog kanala.

    Raises
    TypeError
        Ako 'lts_signal' nije numpy array ili nije kompleksnog tipa.
        Ako 'lts_start' nije integer.
    ValueError
        Ako 'lts_signal' sadrži manje od 128 uzoraka (dva LTS simbola).
        Ako 'lts_start' nije validan indeks (previše blizu kraja signala da se formiraju 2 LTS simbola).
    """
    #Provjere ulaza
    if not isinstance(lts_signal, np.ndarray):
        raise TypeError("lts_signal mora biti np.ndarray")
    if not np.iscomplexobj(lts_signal):
        raise TypeError("lts_signal mora sadržavati kompleksne vrijednosti")
    if not isinstance(lts_start, int):
        raise TypeError("lts_start mora biti int")
    if len(lts_signal) < 128:
        raise ValueError("lts_signal mora sadržavati barem 128 uzoraka (2 LTS simbola)")
    if lts_start < 0 or lts_start > len(lts_signal) - 128:
        raise ValueError("lts_start nije validan: preblizu kraja signala za 2 LTS simbola")

    #Glavna funkcionalnost
    NFFT = 64

    #Primljeni LTS, dijeli se na dva simbola 
    lts1_td = lts_signal[lts_start : lts_start + NFFT]
    lts2_td = lts_signal[lts_start + NFFT : lts_start + 2*NFFT]
    lts_avg_td = 0.5 * (lts1_td + lts2_td) #srednja vrijednost
    lts_fd = (np.fft.fft(lts_avg_td))

    #Referentni LTS, uzima se samo jedan simbol za racunanje korelacije
    lts_ref_td = 1/64*get_long_training_sequence()[32:32+64]
    lts_ref_fd = (np.fft.fft(lts_ref_td))

    #Aktivni podnosioci (802.11a)
    active = np.r_[1:27, 38:64]

    #Racunanje procjene kanala
    H = np.zeros(NFFT, dtype=complex)
    H[active] = lts_fd[active] / lts_ref_fd[active]

    #ZF equalizer
    eps = 1e-8
    equalizer = np.conj(H) / (np.abs(H)**2 + eps)

    return H, equalizer
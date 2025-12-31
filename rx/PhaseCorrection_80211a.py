import numpy as np
import matplotlib.pyplot as plt

def phase_correction_80211a(rx_signal, num_symbols,ltpeak,channel_est,equalizer_coeffs,L=8, max_ratio=1):
    """
    Fazna korekcija za IEEE 802.11a OFDM sistem.

    Implementira:
    - Korekciju zajedničke fazne greške (CPE - Common Phase Error)
    - Praćenje i korekciju faznog nagiba na osnovu pilot-tonova
    - Izlaz su fazno ispravljeni podatkovni podnosioci (48)

    Parametri
    rx_signal : np.ndarray
        Primljeni signal u vremenskom domenu
    num_symbols : int
        Broj OFDM simbola za obradu
    ltpeak : int
        Indeks početka LTS-a (detektovani vrh dugog trening simbola)
    channel_est : np.ndarray (64,)
        Procijenjeni frekvencijski odziv kanala
    equalizer_coeffs : np.ndarray (64,)
        Koeficijenti kanalskog ekvilajzera
    L : int, opcionalno
        Dužina filetra za usrednjavanje faznog nagiba
    max_ratio : int, opcionalno
        Ako je 1 koristi ponderisanje pilota po snazi,
        ako je 0 svi piloti imaju jednake težine

    Povratne vrijednosti
    corrected_symbols : list[np.ndarray]
        Lista fazno ispravljenih OFDM simbola (48 podnosioca po simbolu)
    """
    #Pozicije pilota
    idx_m21 = 38
    idx_m07 = 52
    idx_p07 = 11
    idx_p21 = 25

    pilot_m21_snaga=abs(channel_est[idx_m21])
    pilot_m07_snaga=abs(channel_est[idx_m07])
    pilot_p07_snaga=abs(channel_est[idx_p07])
    pilot_p21_snaga=abs(channel_est[idx_p21])

    snaga_pilota=pilot_m07_snaga+pilot_m21_snaga+pilot_p07_snaga+pilot_p21_snaga
    
    if(max_ratio==0):
        C1, C2, C3, C4=1/4, 1/4, 1/4, 1/4
    else: 
        C1= pilot_m21_snaga/snaga_pilota
        C2= pilot_m07_snaga/snaga_pilota
        C3= pilot_p07_snaga/snaga_pilota
        C4= pilot_p21_snaga/snaga_pilota

    average_slope_filter = np.zeros(L)
    corrected_symbols=[]

    for i in range (num_symbols):
        start=ltpeak+2*64+i*80
        stop=start+64
        trenutni_simbol=rx_signal[start:stop]
        trenutni_fft=1/64*np.fft.fft(trenutni_simbol)
        equalized_symbol=trenutni_fft*equalizer_coeffs

        #Ekstrakcija pilota
        pilot_m21 = equalized_symbol[idx_m21]
        pilot_m07 = equalized_symbol[idx_m07]
        pilot_p07 = equalized_symbol[idx_p07]
        pilot_p21 = equalized_symbol[idx_p21]

        #CPE korekcija
        averaged_pilot=(C1*pilot_m21+C2*pilot_m07+C3*pilot_p07+C4*pilot_p21)
        theta = np.angle(averaged_pilot)
        corr_symbol1 = equalized_symbol * np.exp(-1j * theta)

        #Phase slope 
        pilot_m21 = pilot_m21*np.conj(averaged_pilot)
        pilot_m07 = pilot_m07*np.conj(averaged_pilot)
        pilot_p07 = pilot_p07*np.conj(averaged_pilot)
        pilot_p21 = pilot_p21*np.conj(averaged_pilot)

        slope = (-C1*np.angle(pilot_m21)/21- C2*np.angle(pilot_m07)/7+ C3*np.angle(pilot_p07)/7+ C4*np.angle(pilot_p21)/21)

        #Average slope filter 
        average_slope_filter[1:] = average_slope_filter[:-1]
        average_slope_filter[0] = slope
        avg_slope = np.sum(average_slope_filter)/L

        #Korekcija
        step_plus = np.arange(0, 32) * avg_slope
        step_minus = np.arange(-32, 0) * avg_slope
        applied_correction = np.concatenate((step_plus, step_minus))
        corr_symbol2=corr_symbol1*np.exp(-1j*applied_correction)
        equalizer_coeffs=equalizer_coeffs*np.exp(-1j*applied_correction/L)

        #Ekstrakcija data podnosioca
        start=i*48
        stop=start+48
        data_indices=np.array([
        6,7,8,9,10,
        12,13,14,15,16,17,18,19,20,21,22,23,24,
        26,27,28,29,30,31,32,33,34,35,36,37,39,
        40,41,42,43,44,45,46,47,48,49,50,51,
        53,54,55,56,57
        ])

        corrected_sym = corr_symbol2[data_indices]
        corrected_symbols.append(corrected_sym)

    return corrected_symbols
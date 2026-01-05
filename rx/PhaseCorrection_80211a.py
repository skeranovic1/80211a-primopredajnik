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
    idx_p1 = 11
    idx_p2 = 25
    idx_p3 = 38
    idx_p4 = 52

    pilot_p1_snaga=abs(channel_est[idx_p1])
    pilot_p2_snaga=abs(channel_est[idx_p2])
    pilot_p3_snaga=abs(channel_est[idx_p3])
    pilot_p4_snaga=abs(channel_est[idx_p4])

    snaga_pilota=pilot_p1_snaga+pilot_p2_snaga+pilot_p3_snaga+pilot_p4_snaga
    
    if(max_ratio==0):
        C1, C2, C3, C4=1/4, 1/4, 1/4, 1/4
    else: 
        C1= pilot_p1_snaga/snaga_pilota
        C2= pilot_p2_snaga/snaga_pilota
        C3= pilot_p3_snaga/snaga_pilota
        C4= pilot_p4_snaga/snaga_pilota

    average_slope_filter = np.zeros(L)
    corrected_symbols=[]

    # FFT bin -> subcarrier index k (za fazni nagib)
    # k = idx za 0..31, a k = idx-64 za 32..63 (negativni tonovi)
    k_vec = np.fft.fftfreq(64) * 64  # [0..31, -32..-1]

    # k vrijednosti za vaše pilot binove
    pilot_bins = np.array([idx_p1, idx_p2, idx_p3, idx_p4], dtype=int)
    pilot_k = k_vec[pilot_bins]

    # Indeksi data podnosioca (vaš TX mapping)
    data_indices=np.array([
        6,7,8,9,10,
        12,13,14,15,16,17,18,19,20,21,22,23,24,
        26,27,28,29,30,31,32,33,34,35,36,37,39,
        40,41,42,43,44,45,46,47,48,49,50,51,
        53,54,55,56,57
    ])

    for i in range (num_symbols):
        CP = 16
        SYM = 80
        start=ltpeak+2*64+i*SYM+CP
        stop=start+64
        trenutni_simbol=rx_signal[start:stop]
        trenutni_fft=1/64*np.fft.fft(trenutni_simbol)
        equalized_symbol=trenutni_fft*equalizer_coeffs

        #Ekstrakcija pilota
        pilot_1 = equalized_symbol[idx_p1]
        pilot_2 = equalized_symbol[idx_p2]
        pilot_3 = equalized_symbol[idx_p3]
        pilot_4 = equalized_symbol[idx_p4]

        #CPE korekcija
        averaged_pilot=(C1*pilot_1+C2*pilot_2+C3*pilot_3+C4*pilot_4)
        theta = np.angle(averaged_pilot)
        corr_symbol1 = equalized_symbol * np.exp(-1j * theta)

        #Phase slope 
        # (nakon skidanja CPE, faze pilota fitamo na liniju: phase ≈ slope*k + const)
        pilots_cpe_removed = np.array([
            pilot_1*np.conj(averaged_pilot),
            pilot_2*np.conj(averaged_pilot),
            pilot_3*np.conj(averaged_pilot),
            pilot_4*np.conj(averaged_pilot)
        ], dtype=complex)

        pilot_phase = np.unwrap(np.angle(pilots_cpe_removed))

        # Least-squares fit: pilot_phase = a*pilot_k + b  -> slope=a
        A = np.vstack([pilot_k, np.ones_like(pilot_k)]).T
        slope, _ = np.linalg.lstsq(A, pilot_phase, rcond=None)[0]

        #Average slope filter 
        average_slope_filter[1:] = average_slope_filter[:-1]
        average_slope_filter[0] = slope
        avg_slope = np.sum(average_slope_filter)/L

        #Korekcija
        applied_correction = avg_slope * k_vec
        corr_symbol2=corr_symbol1*np.exp(-1j*applied_correction)
        equalizer_coeffs=equalizer_coeffs*np.exp(-1j*applied_correction/L)

        #Ekstrakcija data podnosioca
        corrected_sym = corr_symbol2[data_indices]
        corrected_symbols.append(corrected_sym)

    return corrected_symbols

import numpy as np
from rx.detection import packet_detector
import matplotlib.pyplot as plt

import numpy as np
from rx.detection import packet_detector
import matplotlib.pyplot as plt

def gruba_vremenska_sinhronizacija(rx_lts, search_win=32):
    """
    Izvršava grubu vremensku sinhronizaciju 802.11a OFDM signala koristeći Long Training Sequence (LTS).

    Ova funkcija detektuje početak korisnog dijela LTS signala u primljenom OFDM paketu. 
    Metoda se zasniva na korelaciji između dva uzastopna LTS simbola i pronalasku maksimuma korelacije unutar inicijalnog pretražnog prozora.

    Parametri
    rx_lts : np.ndarray
        Primljeni kompleksni uzorci koji sadrže LTS. Očekuje se barem 2*64 uzorka (2 LTS simbola po 64 korisna uzorka).
    search_win : int, opcionalno
        Broj uzoraka od početka 'rx_lts' unutar kojih se traži maksimum korelacije. Default je 32.

    Povratne vrijednosti
    fft_start : int
        Indeks unutar 'rx_lts' koji označava početak korisnog dijela LTS simbola (koristi se za FFT start pri OFDM demodulaciji).
    timing_corr : np.ndarray
        Niz apsolutnih vrijednosti korelacije između dva LTS simbola, iste dužine kao 'rx_lts'.
    timing_idxs : np.ndarray
        Niz indeksa koji odgovara pozicijama u 'timing_corr', koristan za vizualizaciju i debug.
    
    Raises
    TypeError
        Ako 'rx_lts' nije numpy array ili nije kompleksnog tipa.
    ValueError
        Ako 'rx_lts' sadrži manje od 128 uzoraka.
        Ako 'search_win' nije pozitivan ili veći od dužine 'rx_lts'.
    """
    #Provjere ulaznih podataka
    if not isinstance(rx_lts, np.ndarray):
        raise TypeError(f"'rx_lts' mora biti numpy.ndarray, a dobiveno je {type(rx_lts)}")
    
    if rx_lts.size < 2*64:
        raise ValueError(f"'rx_lts' mora imati barem 128 uzoraka (2 LTS simbola), a dobiveno je {rx_lts.size}")
    
    if not isinstance(search_win, int) or search_win <= 0:
        raise ValueError(f"'search_win' mora biti pozitivan cijeli broj, a dobiveno je {search_win}")
    
    if search_win > len(rx_lts):
        raise ValueError(f"'search_win' ({search_win}) ne može biti veći od dužine 'rx_lts' ({len(rx_lts)})")
    
    #Glavna funkcionalnost
    N = len(rx_lts)
    timing_corr = np.zeros(N, dtype=np.float64)
    
    LTS_len = 64  #dva LTS simbola po 64 uzorka, bey CP

    for n in range(N - 2*LTS_len + 1):
        corr = np.sum(rx_lts[n:n+LTS_len] * np.conj(rx_lts[n+LTS_len:n+2*LTS_len]))
        timing_corr[n] = np.abs(corr)

    #Pretraga maksimuma unutar pocetnog search window-a
    max_idx = np.argmax(timing_corr[:search_win])
    fft_start = max_idx  #indeks pocetka korisnog dijela LTS-a
    
    timing_idxs = np.arange(len(timing_corr))
    return fft_start, timing_corr, timing_idxs

def detect_frequency_offsets(RX_Input, lts_start, fs, plot=False):
    """
    Detektuje frekvencijski ofset nosioca (CFO) primljenog 802.11a OFDM signala.

    Funkcija izračunava i grubi (coarse) i precizni (fine) frekvencijski ofset koristeći metodu automatske korelacije:
      - Grubi CFO se procjenjuje na osnovu Short Training Sequence (STS).
      - Precizni CFO se procjenjuje na osnovu Long Training Sequence (LTS).

    Parametri
    RX_Input : array_like
        Kompleksni primljeni uzorci signala.
    lts_start : int
        Indeks u RX_Input koji označava početak korisnog dijela prvog LTS simbola (64 uzorka, bez CP).
    fs : float, optional
        Frekvencija uzorkovanja [Hz]. Zadano 20e6.
    plot : bool, optional
        Ako je True, prikazuje grafove autokorelacije korištene za procjenu coarse i fine CFO.
        Zadano False.

    Povratne vrijednosti
    FrequencyOffsets : ndarray, shape (2,)
        Niz koji sadrži:
            FrequencyOffsets[0] : grubi CFO u Hz
            FrequencyOffsets[1] : precizni CFO u Hz

    Raises
    TypeError
        Ako RX_Input nije np.ndarray ili nije kompleksnog tipa.
    ValueError
        Ako lts_start nije validan indeks u RX_Input.
        Ako fs nije pozitivan broj.
    
    Napomene
    - Funkcija pretpostavlja da je gruba vremenska sinhronizacija već izvršena i da 'lts_start' precizno pokazuje početak LTS simbola.
    """
    #Provjere ulaznih podataka
    RX_Input = np.asarray(RX_Input)

    if RX_Input.size == 0:
        raise ValueError("'RX_Input' ne smije biti prazan niz")

    if not np.iscomplexobj(RX_Input):
        raise TypeError("'RX_Input' mora sadržavati kompleksne uzorke")


    if not isinstance(fs, (int, float)) or fs <= 0:
        raise ValueError(f"'fs' mora biti pozitivan broj, a dobiveno je {fs}")

    if not isinstance(plot, bool):
        raise TypeError(f"'plot' mora biti bool, a dobiveno je {type(plot)}")
    
    #Glavna funkcionalnost
    RX_Input = np.asarray(RX_Input)
    N = len(RX_Input)

    #1=coarse/gruba, 2=fine/precizna 
    FrequencyOffsets = np.zeros(2)

    #Coarse/gruba
    AutoCorr_Est = np.zeros(N, dtype=complex)
    Delay16 = np.zeros(16, dtype=complex)
    SlidingAverage1 = np.zeros(32, dtype=complex)

    for i in range(N):
        RX_Input_16 = Delay16[-1]
        Delay16[1:] = Delay16[:-1]
        Delay16[0] = RX_Input[i]

        Temp = RX_Input[i] * np.conj(RX_Input_16)
        SlidingAverage1[1:] = SlidingAverage1[:-1]
        SlidingAverage1[0] = Temp

        AutoCorr_Est[i] = np.sum(SlidingAverage1) / 32

    idx_coarse = int(np.clip(lts_start - 32 - 50, 0, N - 1))
    Theta = np.angle(AutoCorr_Est[idx_coarse])
    FrequencyOffsets[0] = Theta * fs / (2 * np.pi * 16)

    #Plot 
    if plot:
        plt.figure(figsize=(12,3))
        plt.plot(np.abs(AutoCorr_Est), label='|R(n)| - LTS')
        plt.axvline(idx_coarse, color='r', linestyle='--', label='Coarse CFO index')
        plt.title("Grubi CFO: LTS autokorelacija")
        plt.xlabel("Uzorke")
        plt.ylabel("|R(n)|")
        plt.grid(True)
        plt.legend()
        plt.show()

    #Fine/precizna 
    AutoCorr_Est_Fine = np.zeros(N, dtype=complex)
    Delay64 = np.zeros(64, dtype=complex)
    SlidingAverage2 = np.zeros(64, dtype=complex)

    for i in range(N):
        RX_Input_64 = Delay64[-1]
        Delay64[1:] = Delay64[:-1]
        Delay64[0] = RX_Input[i]

        Temp = RX_Input[i] * np.conj(RX_Input_64)
        SlidingAverage2[1:] = SlidingAverage2[:-1]
        SlidingAverage2[0] = Temp

        AutoCorr_Est_Fine[i] = np.sum(SlidingAverage2) / 64

    idx_fine = int(np.clip(lts_start + 64, 0, N - 1))
    Theta = np.angle(AutoCorr_Est_Fine[idx_fine])
    FrequencyOffsets[1] = Theta * fs / (2 * np.pi * 64)

    #Plot 
    if plot:
        plt.figure(figsize=(12,3))
        plt.plot(np.abs(AutoCorr_Est_Fine), label='|R(n)| - LTS')
        plt.axvline(idx_fine, color='r', linestyle='--', label='Fine CFO index')
        plt.title("Fini CFO: LTS autokorelacija")
        plt.xlabel("Uzorke")
        plt.ylabel("|R(n)|")
        plt.grid(True)
        plt.legend()
        plt.show()

    return FrequencyOffsets

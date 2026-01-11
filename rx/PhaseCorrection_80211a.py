import numpy as np

def phase_correction(symbols_fd, num_symbols, channel_est, L=8, max_ratio=1):
    """
    Fazna korekcija pilota za IEEE 802.11a OFDM signal.

    Funkcija izvršava:
    - Korekciju zajedničke fazne greške (CPE - Common Phase Error)
    - Praćenje i korekciju faznog nagiba po pilot-tonovima
    - Izlaz su fazno ispravljeni podatkovni podnosioci (48 DATA podnosioca po simbolu)

    Parametri
    symbols_fd : np.ndarray
        FFT simboli (bez CP + izvrsena ekvalizacija), shape (num_symbols, 64)
    num_symbols : int
        Broj OFDM simbola za obradu
    channel_est : np.ndarray
        Procijenjeni frekvencijski odziv kanala, shape (64,)
    L : int, optional
        Dužina filtera za prosjek faznog nagiba. Default je 8.
    max_ratio : int, optional
        Ako je 1 koristi ponderisanje pilota po snazi,
        ako je 0 svi piloti imaju jednake težine. Default je 1.

    Povratna vrijednost
    corrected_symbols : np.ndarray
        Niz shape (num_symbols, 48) sa fazno ispravljenim data subnosiocima.

    Raises
    TypeError
        Ako symbols_fd ili channel_est nisu np.ndarray
    ValueError
        Ako symbols_fd nema shape (num_symbols, 64)
        Ako channel_est nema 64 elemenata
        Ako num_symbols nije pozitivan integer
        Ako L nije pozitivan integer
        Ako max_ratio nije 0 ili 1
    """
    #Provjera ulaza
    if not isinstance(num_symbols, int) or num_symbols <= 0:
        raise ValueError("num_symbols mora biti pozitivan integer.")
    if not isinstance(L, int) or L <= 0:
        raise ValueError("L mora biti pozitivan integer.")
    if max_ratio not in (0, 1):
        raise ValueError("max_ratio mora biti 0 ili 1.")
    if not isinstance(symbols_fd, np.ndarray):
        raise TypeError("symbols_fd mora biti numpy.ndarray tipa.")
    if not isinstance(channel_est, np.ndarray):
        raise TypeError("channel_est mora biti numpy.ndarray tipa.")
    if symbols_fd.shape != (num_symbols, 64):
        raise ValueError(f"symbols_fd mora imati shape ({num_symbols}, 64).")
    if channel_est.shape != (64,):
        raise ValueError("channel_est mora imati shape (64,).")
        
    #Glavna funkcionalnost
    idx_pilots = np.array([11, 25, 38, 52])
    pilot_magnitudes = np.abs(channel_est[idx_pilots])
    total_mag = np.sum(pilot_magnitudes)

    if max_ratio==0:
        C = np.ones(4)/4
    else:
        C = pilot_magnitudes / total_mag

    average_slope_filter = np.zeros(L) #Inicijalizacija povratnih varijabli
    corrected_symbols = []

    #FFT bin -> indeks subnosioca k, ravnomjerno rasporedeni 
    k_vec = np.fft.fftfreq(64) * 64

    #k vrijednosti pilota
    pilot_k = k_vec[idx_pilots]

    #Indeksi data podnosioca (48)
    idx_data=np.array([
        6,7,8,9,10,
        12,13,14,15,16,17,18,19,20,21,22,23,24,
        26,27,28,29,30,31,32,33,34,35,36,37,39,
        40,41,42,43,44,45,46,47,48,49,50,51,
        53,54,55,56,57
    ])

    for i in range(num_symbols):
        sym = symbols_fd[i]

        # Ekstrakcija pilota svakog simbola
        pilots = sym[idx_pilots]

        #CPE detekcija i korekcija
        averaged_pilot = np.sum(C * pilots)
        theta = np.angle(averaged_pilot)
        sym_cpe = sym * np.exp(-1j*theta)

        #Phase slope
        pilots_cpe_removed = pilots * np.conj(averaged_pilot)
        pilot_phase = np.unwrap(np.angle(pilots_cpe_removed))
        A = np.vstack([pilot_k, np.ones_like(pilot_k)]).T
        slope, _ = np.linalg.lstsq(A, pilot_phase, rcond=None)[0]

        #Prosjek nagiba
        average_slope_filter[1:] = average_slope_filter[:-1]
        average_slope_filter[0] = slope
        avg_slope = np.sum(average_slope_filter)/L

        #Primjena korekcije po binovima
        applied_correction = np.exp(-1j*avg_slope*k_vec)
        sym_corrected = sym_cpe * applied_correction

        #Ekstrakcija podataka na podnosioca
        corrected_symbols.append(sym_corrected[idx_data])

    return np.array(corrected_symbols)

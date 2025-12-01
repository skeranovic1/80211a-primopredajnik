import numpy as np

def IFFT_GI(symbol_stream):
    """
    Funkcija koja prima stream QAM simbola i vraća OFDM signal
    sa guard intervalom za 802.11a/g/n Wi-Fi standard.
    
    Ulaz: stream od N × 48 QAM simbola
    Izlaz: OFDM signal sa guard intervalom - 80 uzoraka po simbolu
    """
    
    # broj ofdm simbola dobijemo kao duljinu streama podijeljenu s 48
    # jer svaki OFDM simbol ima 48 nosioca za podatke
    num_symbols = len(symbol_stream) // 48
    
    # output payload: 80 samples per symbol
    # 64 uzoraka (IFFT) + 16 uzoraka (guard interval) = 80 uzoraka
    payload = np.zeros(num_symbols * 80, dtype=complex)

    # indeksiranje za 48 nosioca podataka
    # Ovo su indeksi u 64-point IFFT-u gdje se postavljaju podaci
    # MATLAB/Octave koristi 1-based indekse, mi prevodimo na 0-based
    IFFT_index = np.array(
        list(range(1,7)) +      # nosioci 1-6
        list(range(8,21)) +     # nosioci 8-20
        list(range(22,27)) +    # nosioci 22-26
        list(range(38,43)) +    # nosioci 38-42  
        list(range(44,57)) +    # nosioci 44-56
        list(range(58,64))      # nosioci 58-63
    ) - 1   # konverzija iz MATLAB 1-based u Python 0-based indekse
    
    # pozicije pilot nosioca (prema 802.11 standardu)
    # MATLAB: [7 21 43 57], Python: [6, 20, 42, 56]
    pilot_idx = np.array([7, 21, 43, 57]) - 1

    # Procesuiramo svaki OFDM simbol posebno
    for i in range(num_symbols):
        # uzimamo 48 QAM simbola za trenutni OFDM simbol
        start = i * 48
        stop = start + 48
        current_input = symbol_stream[start:stop]

        # kreiramo ulaz za 64-point IFFT
        IFFT_input = np.zeros(64, dtype=complex)

        # postavljamo podatke na odgovarajuće nosioce
        IFFT_input[IFFT_index] = current_input

        # ubacujemo 4 pilot nosioca (vrijednost = 1)
        # Pilot nosioci se koriste za sinc hronizaciju i estimaciju kanala
        IFFT_input[pilot_idx] = 1 + 0j

        # izvodi se 64-point IFFT
        # Ovo pretvara iz frekvencijske u vremensku domenu
        IFFT_output = np.fft.ifft(IFFT_input)

        # guard interval = zadnjih 16 uzoraka IFFT izlaza
        # Ovo je cyclic prefix koji štiti od intersymbol interference (ISI)
        GI = IFFT_output[48:64]

        # kreiramo cijeli simbol: GI + cijeli IFFT izlaz
        # Format: [guard interval (16)] + [cijeli simbol (64)] = 80 uzoraka
        block = np.concatenate([GI, IFFT_output])

        # pohranjujemo u izlazni payload
        payload[i*80 : (i+1)*80] = block

    return payload
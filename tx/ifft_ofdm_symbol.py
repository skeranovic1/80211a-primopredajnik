import numpy as np
import matplotlib.pyplot as plt

def IFFT_GI(symbol_stream, plot=False):
    """
    Generiše OFDM simbole primjenom IFFT-a i dodavanjem zaštitnog intervala (GI).

    Funkcija prima niz od 48-tonski modulisanih simbola za svaki OFDM simbol,
    raspoređuje ih na odgovarajuće frekvencijske podnosioca prema IEEE 802.11a
    strukturi, ubacuje pilot-nosioca, vrši IFFT i dodaje ciklički prefiks dužine 16.
    Rezultat je niz kompleksnih uzoraka u vremenskom domenu.

    Parameteri
    symbol_stream : np.ndarray
        Kompleksni ulazni niz modulisani podataka. Dužina mora biti
        'N * 48', gdje je 'N' broj OFDM simbola.
    plot : bool, optional
        Ako je True, prikazuju se grafovi različitih faza obrade.

    Povratna vrijednost
    payload : np.ndarray
        Kompleksni niz u vremenskom domenu dužine 'N * 80', gdje je:
        - 64 uzorka IFFT izlaz (OFDM simbol),
        - 16 uzoraka ciklički prefiks (GI).
    """
    if not isinstance(symbol_stream, np.ndarray):
        raise ValueError("Ulaz mora biti numpy array.")

    if symbol_stream.size == 0:
        raise ValueError("Ulazni niz ne smije biti prazan.")

    if not np.issubdtype(symbol_stream.dtype, np.number):
        raise TypeError("Ulazni niz mora sadržavati numeričke vrijednosti (int ili complex).")

    #Koliko OFDM simbola se nalazi u ulaznom streamu
    num_symbols=len(symbol_stream)//48

    #Rezervacija prostora za završni niz (svaki simbol ima 80 uzoraka)
    payload=np.zeros(num_symbols*80,dtype=complex)

    #Indeksi data podnosioca 
    IFFT_index=np.array([
        6,7,8,9,10,
        12,13,14,15,16,17,18,19,20,21,22,23,24,
        26,27,28,29,30,31,32,33,34,35,36,37,39,
        40,41,42,43,44,45,46,47,48,49,50,51,
        53,54,55,56,57
    ])

    #Indeksi 4 pilot-nosioca
    pilot_idx=np.array([11, 25, 38, 52])

    #Procesiranje svakog OFDM simbola
    for i in range(num_symbols):
        # Izdvajanje 48 ulaznih simbola za tekući OFDM simbol
        start = i * 48
        stop = start + 48
        current_input = symbol_stream[start:stop]

        #Crtanje ulaznih simbola
        if plot:
            plt.figure(figsize=(12,4))
            plt.subplot(2,1,1)
            plt.stem(np.real(current_input))
            plt.title("Ulazni stream - realni dio")
            plt.grid(True)
            plt.subplot(2,1,2)
            plt.stem(np.imag(current_input))
            plt.grid(True)
            plt.title("Ulazni stream - imaginarni dio")
            plt.show(block=False)

        #Frekvencijski OFDM okvir (64 podnosioca)
        IFFT_input=np.zeros(64, dtype=complex)

        #Upis 48 data simbola na odgovarajuće podnosioce
        IFFT_input[IFFT_index]=current_input
        proba=IFFT_input.copy()  #kopija za prikaz pilota

        #Prikaz smještanja data simbola
        if plot:
            plt.figure(figsize=(12,4))
            plt.subplot(2,1,1)
            plt.stem(np.real(IFFT_input))
            plt.title("Podaci na pozicijama - realni dio")
            plt.grid(True)
            plt.subplot(2,1,2)
            plt.stem(np.imag(IFFT_input))
            plt.title("Podaci na pozicijama - imaginarni dio")
            plt.grid(True)
            plt.show(block=False)

        #Ubacivanje pilot-nosioca 
        IFFT_input[pilot_idx]=1+0j
        provjera=IFFT_input-proba  #razlika je samo na mjestima pilota

        #Prikaz samo pilot simbola
        if plot:
            plt.figure(figsize=(12,4))
            plt.subplot(2,1,1)
            plt.axhline(y=1, color='green', linestyle='--')
            plt.stem(np.real(provjera))
            plt.title("Samo piloti na pozicijama - realni dio")
            plt.grid(True)
            plt.subplot(2,1,2)
            plt.stem(np.imag(provjera))
            plt.title("Samo piloti na pozicijama - imaginarni dio")
            plt.grid(True)
            plt.show(block=False)

        #Prikaz kompletnog OFDM frekvencijskog okvira
        if plot:
            plt.figure(figsize=(12,4))
            plt.subplot(2,1,1)
            plt.axhline(y=1, color='green', linestyle='--')
            plt.stem(np.real(IFFT_input))
            plt.title("Cijeli OFDM signal u frekvencijskoj domeni - realni dio")
            plt.grid(True)
            plt.subplot(2,1,2)
            plt.stem(np.imag(IFFT_input))
            plt.title("Cijeli OFDM signal u frekvencijskoj domeni - imaginarni dio")
            plt.grid(True)
            plt.show(block=False)

        #IFFT: prelazak u vremensku domenu
        IFFT_output=np.fft.ifft(IFFT_input)

        #Prikaz vremenske domene
        if plot:
            plt.figure(figsize=(12,4))
            plt.plot(np.real(IFFT_output))
            plt.grid(True)
            plt.plot(np.imag(IFFT_output))
            plt.title("OFDM simbol u vremenskoj domeni")
            plt.show(block=False)

        #Dodavanje cikličkog prefiksa dužine 16
        #GI se uzima iz zadnjeg dijela IFFT izlaza (48–63)
        GI=IFFT_output[48:64]

        #Formiranje jednog OFDM bloka: GI + 64 uzorka
        block=np.hstack([GI, IFFT_output])

        #Upis u finalni izlazni stream
        payload[i*80:(i+1)*80]=block

        #Prikaz cijelog payload-a
        if plot:
            plt.figure(figsize=(12,4))
            plt.subplot(3,1,1)
            plt.title(f"OFDM simbola broj {i+1} - realni dio")
            plt.stem(np.real(payload))
            plt.grid(True)
            plt.subplot(3,1,2)
            plt.stem(np.real(payload))
            plt.xlim(-1,17)
            plt.title("Dodani ciklički prefiks (GI) - realni dio")
            plt.grid(True)
            plt.subplot(3,1,3)
            plt.stem(np.real(payload))
            plt.xlim(63,81)
            plt.title("Zadnji dio OFDM simbola - realni dio")
            plt.grid(True)
            plt.show(block=False)

    return payload

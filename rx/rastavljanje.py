import numpy as np 

def remove_cp(rx, start, num_symbols, NSYM, NFFT, NCP):
    """
    Skida Cyclic Prefix (CP) iz OFDM simbola i vrši FFT za svaki simbol.

    Parametri
    rx : array-like
        Kompleksni bazni uzorci primljenog signala.
    start : int
        Indeks početka prvog OFDM simbola (može biti unutar CP-a).
    num_symbols : int
        Broj OFDM simbola za obradu.
    NSYM : int
        Dužina jednog OFDM simbola u vremenskom domenu uključujući CP.
    NCP : int
        Dužina Cyclic Prefix-a.
    NFFT : int
        Broj FFT tačaka (korisna dužina OFDM simbola).

    Povratne vrijednosti
    symbols_fd : np.ndarray, shape (num_symbols, NFFT)
        FFT rezultati OFDM simbola nakon uklanjanja CP-a.

    Raises
    TypeError
        Ako 'rx' nije array-like ili 'start', 'num_symbols', 'NSYM', 'NCP', 'NFFT' nisu cijeli brojevi.
    ValueError
        Ako su vrijednosti negativne ili ako 'rx' nema dovoljno uzoraka za zadani broj simbola.
    """
    #Provjere ulaza
    if not isinstance(rx, (np.ndarray, list, tuple)):
        raise TypeError("rx mora biti array-like (np.ndarray, list ili tuple).")
    rx = np.asarray(rx)
        
    for name, val in zip(["num_symbols", "NSYM", "NCP", "NFFT"],[num_symbols, NSYM, NCP, NFFT]):
        if not isinstance(val, int):
            raise TypeError(f"{name} mora biti int tipa.")
        if val < 0:
            raise ValueError(f"{name} ne smije biti negativan.")
    
    if NCP + NFFT > NSYM:
        raise ValueError("NCP + NFFT ne smije biti veće od NSYM")
    #Provjera da li ima dovoljno uzoraka
    required_len = start + num_symbols * NSYM
    if required_len > len(rx):
        raise ValueError(f"Nema dovoljno uzoraka u rx za {num_symbols} simbola. \nPotrebno: {required_len}, dostupno: {len(rx)}")
    
    #Glavna funkcionalnost
    symbols_fd = [] #izlaz

    for k in range(num_symbols):
        sym_start = start + k*NSYM  #pocetak trenutnog simbola (ukljucujuci CP) i postavljanje pozicija
        sym_cp_removed_start = sym_start + NCP 
        sym_cp_removed_end = sym_cp_removed_start + NFFT
        
        sym_td = rx[sym_cp_removed_start : sym_cp_removed_end]  #skidanje CP-a
        sym_fd = np.fft.fft(sym_td, NFFT)  #FFT
        symbols_fd.append(sym_fd) #Dodavanje na izlaznu varijablu 

    return np.array(symbols_fd)
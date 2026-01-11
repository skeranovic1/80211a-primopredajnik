import numpy as np

def iq_preprocessing(rx_signal, tx_signal, fs):
    """
    Priprema IQ signala prije daljnje obrade.

    Uključuje:
    - Pretvaranje signala u 1D niz
    - Normalizaciju snage primljenog signala u odnosu na poslani
    - Decimaciju signala za faktor 2 (smanjenje frekvencije uzorkovanja)

    Parametri
    rx_signal : array-like
        Primljeni (RX) IQ signal
    tx_signal : array-like
        Poslani (TX) IQ signal (referenca za normalizaciju)
    fs : float
        Frekvencija uzorkovanja [Hz]

    Povratne vrijednosti
    rx_signal : np.ndarray
        Predobrađeni primljeni IQ signal
    fs : float
        Nova (smanjena) frekvencija uzorkovanja

    Raises
    TypeError
        Ako rx_signal ili tx_signal nisu array-like
        Ako fs nije float ili int
    ValueError
        Ako rx_signal ili tx_signal nemaju elemente
        Ako fs <= 0
    """
    #Provjere ulaza
    if not isinstance(fs, (int, float)):
        raise TypeError("fs mora biti float ili int tipa.")
    if fs <= 0:
        raise ValueError("fs mora biti veći od 0.")
    
    try:
        rx_signal = np.asarray(rx_signal).flatten()
        tx_signal = np.asarray(tx_signal).flatten()
    except Exception as e:
        raise TypeError("rx_signal i tx_signal moraju biti array-like.") from e

    if rx_signal.size == 0:
        raise ValueError("rx_signal ne smije biti prazan.")
    if tx_signal.size == 0:
        raise ValueError("tx_signal ne smije biti prazan.")

    #Glavna funkcionalnost
    rx_signal = np.asarray(rx_signal).flatten()

    #Normalizacija
    rx_signal *= np.sqrt(np.mean(np.abs(tx_signal)**2)) / \
                 np.sqrt(np.mean(np.abs(rx_signal)**2))

    #Downsampling x2
    rx_signal = rx_signal[::2]
    fs = fs / 2

    return rx_signal, fs

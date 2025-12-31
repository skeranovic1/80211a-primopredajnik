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
    """
    rx_signal = np.asarray(rx_signal).flatten()

    # normalizacija
    rx_signal *= np.sqrt(np.mean(np.abs(tx_signal)**2)) / \
                 np.sqrt(np.mean(np.abs(rx_signal)**2))

    # decimacija ×2
    rx_signal = rx_signal[::2]
    fs = fs / 2

    return rx_signal, fs

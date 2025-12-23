import numpy as np

def iq_preprocessing(rx_signal, tx_signal, fs):
    rx_signal = np.asarray(rx_signal).flatten()

    # normalizacija
    rx_signal *= np.sqrt(np.mean(np.abs(tx_signal)**2)) / \
                 np.sqrt(np.mean(np.abs(rx_signal)**2))

    # decimacija ×2
    rx_signal = rx_signal[::2]
    fs = fs / 2

    return rx_signal, fs

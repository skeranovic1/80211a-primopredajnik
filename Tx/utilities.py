import numpy as np
import matplotlib.pyplot as plt

def zero_stuffing(signal, up_factor=2):
    """
    Zero-stuffing upsampling helper funkcija
    Ubacuje nule između originalnih uzoraka
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal must be a numpy array.")

    if not isinstance(up_factor, int):
        raise TypeError("up_factor must be an integer.")

    if up_factor <= 0:
        raise ValueError("up_factor must be a positive integer.")

    upsamplirano = np.zeros(len(signal) * up_factor)
    upsamplirano[::up_factor] = signal
    
    return upsamplirano

def spektar(x, fs, label):
    """
    Funkcija koja crta spektar signala x
    fs: frekvencija uzorkovanja
    label: oznaka za legendu
    """
    N = len(x)
    X = np.fft.fftshift(np.fft.fft(x))
    f = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    magnitude = np.abs(X) / (N/2)
    plt.plot(f, magnitude, label=label)
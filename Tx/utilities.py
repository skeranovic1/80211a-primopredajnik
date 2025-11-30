import numpy as np
import matplotlib.pyplot as plt

def zero_stuffing(signal, up_factor=2):
    """
    Zero-stuffing upsampling helper funkcija
    Ubacuje nule između originalnih uzoraka
    """
    upsamplirano = np.zeros(len(signal) * up_factor)
    upsamplirano[::up_factor] = signal
    
    return upsamplirano

def spektar(x, fs, label):
    N = len(x)
    X = np.fft.fftshift(np.fft.fft(x))
    f = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    magnitude = np.abs(X) / (N/2)
    plt.plot(f, magnitude, label=label)
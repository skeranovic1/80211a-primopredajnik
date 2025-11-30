import numpy as np
import matplotlib.pyplot as plt

def half_band_upsample(signal, up_factor=2, N=31, plot=False):
    n = np.arange(N)
    Arg = n/2 - (N-1)/4
    Hann = np.hanning(N+2)[1:-1]
    h = np.sinc(Arg) * np.sqrt(Hann)
    
    # Zero-stuffing
    upsampled = np.zeros(len(signal) * up_factor)
    upsampled[::up_factor] = signal
    
    # Filtriranje konvolucijom sa h
    filtered = np.convolve(upsampled, h, mode='same')

    # Crtanje
    if plot:
        # Frekvencijski odziv
        Frezolucija = 0.002
        frekvencije = np.arange(-0.5, 0.5+Frezolucija, Frezolucija)
        frekvencijski_odziv = np.zeros(len(frekvencije), dtype=complex)

        for i, f in enumerate(frekvencije):
            analysistone = np.exp(-1j*2*np.pi*f*n)
            frekvencijski_odziv[i] = (1/N)*np.dot(h, analysistone)

        log_odziv = 20*np.log10(np.abs(frekvencijski_odziv))
        log_odziv -= np.max(log_odziv)
        
        # Plot
        plt.figure(figsize=(10, 6))
        # Impulsni odziv
        plt.subplot(1,2,1)
        markerline, stemlines, baseline = plt.stem(n, h)
        plt.setp(markerline, marker='o', markersize=6, markerfacecolor='blue')
        plt.setp(stemlines, color='blue', linewidth=1.5)
        plt.setp(baseline, visible=False)
        plt.title('Impulsni odziv half-band filtera')
        plt.xlabel('n')
        plt.ylabel('h[n]')
        plt.grid(True)
        plt.xlim([-0.5, N-0.5])
        
        # Frekvencijski odziv
        plt.subplot(1,2,2)
        plt.plot(frekvencije*400e6, log_odziv, 'k')
        plt.title('Frekvencijski odziv half-band filtera')
        plt.xlabel('Frekvencija (Hz)')
        plt.ylabel('Amplituda (dB)')
        plt.ylim([-60, 5])
        plt.tight_layout()
        plt.show()
    
    return filtered, h

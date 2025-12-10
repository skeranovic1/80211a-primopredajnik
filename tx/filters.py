import numpy as np
import matplotlib.pyplot as plt
from .utilities import zero_stuffing


def half_band_upsample(signal, up_factor=2, N=31, plot=False):
    """
    Vrši upsampliranje diskretnog signala korištenjem half-band FIR filtra.

    Funkcija povećava frekvenciju uzorkovanja ulaznog signala ubacivanjem
    nula između uzoraka (zero-stuffing), a zatim primjenjuje half-band
    FIR filter radi uklanjanja neželjenih spektralnih komponenti nastalih
    procesom upsampliranja.

    Parametri
    signal : array-like
        Ulazni diskretni signal koji se upsamplira.
    up_factor : int, opcionalno
        Faktor upsampliranja. Podrazumijevana vrijednost je 2.
    N : int, opcionalno
        Dužina half-band FIR filtera. Podrazumijevana vrijednost je 31.
    plot : bool, opcionalno
        Ako je True, funkcija prikazuje impulsni i frekvencijski odziv
        half-band filtra.

    Povratne vrijednosti
    filtrirano : numpy.ndarray
        Upsamplirani i filtrirani izlazni signal.
    h : numpy.ndarray
        Impulsni odziv dizajniranog half-band FIR filtra.

    Izuzeci
    ValueError
        - Ako je faktor upsampliranja manji ili jednak nuli.
    TypeError
        - Ako ulazni signal nije numeričkog tipa.

    Napomene
    - Half-band filteri imaju graničnu frekvenciju na polovini Nyquistove
      frekvencije i često se koriste za interpolaciju sa faktorom dva.
    - Filter je dizajniran korištenjem sinc funkcije sa Hanning (Hann) prozorom.
    - Prikaz frekvencijskog odziva služi isključivo za analizu i vizualizaciju.
    """
    # Dizajn half-band filtera
    n = np.arange(N)
    Arg = n/2 - (N-1)/4
    Hann = np.hanning(N+2)[1:-1]
    h = np.sinc(Arg) * np.sqrt(Hann)
    
    # Upsampling
    upsamplirano = zero_stuffing(signal, up_factor)
    
    # Filtriranje konvolucijom sa h
    filtrirano = np.convolve(upsamplirano, h, mode='same')

    # Crtanje
    if plot:
        # Frekvencijski odziv
        Frezolucija = 0.002
        frekvencije = np.arange(-0.5, 0.5+Frezolucija, Frezolucija)
        frekvencijski_odziv = np.zeros(len(frekvencije), dtype=complex)

        for i, f in enumerate(frekvencije):
            tone = np.exp(-1j*2*np.pi*f*n)
            frekvencijski_odziv[i] = (1/N)*np.dot(h, tone)

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

  
    return filtrirano, h

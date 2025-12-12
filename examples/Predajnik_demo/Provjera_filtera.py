import numpy as np
import matplotlib.pyplot as plt
from tx.filters import half_band_upsample
from tx.utilities import zero_stuffing
from tx.utilities import spektar

def filter_plot():
    """
    Demonstrira proces upsampliranja sinusnog signala sa faktorom 2 korištenjem zero stuffing metode,
    zatim filtrira upsampleirani signal half-band filterom.

    Uključuje prikaz originalnog, upsampleiranog i filtriranog signala u vremenskom domenu,
    kao i prikaz njihovih spektara u frekvencijskom domenu.

    Koraci:
    - Generisanje sinusnog signala od 50 Hz uzorkovanog na 1000 Hz
    - Upsampliranje signala ubacivanjem nula (zero stuffing)
    - Filtriranje upsampliranog signala half-band filterom
    - Vizualizacija signala i njihovih spektara

    Parametri
        Nema ulaznih parametara.

    Povratna vrijednost
        Nema povratne vrijednosti, samo prikazuje grafike.
    """
    fs=1000
    up_factor=2

    #Originalni signal
    t=np.arange(0, 1, 1/fs)
    f1=50  
    signal=np.sin(2*np.pi*f1*t) 
    
    #Upsampliranje
    upsampled=zero_stuffing(signal, up_factor)
    t_up=np.arange(len(upsampled))/(fs * up_factor)

    #Filtriranje half-band filterom
    filtrirano, h =half_band_upsample(signal, up_factor=up_factor, N=31, plot=True)

    t_filt=np.arange(len(filtrirano))/(fs * up_factor)

    #Prikazivanje rezultata
    plt.figure(figsize=(12,5))

    #Prvi subplot
    plt.subplot(1,2,1)
    plt.plot(t, signal, 'o-', 'b')
    plt.xlim(0, 1/f1)
    plt.title('Originalni signal')
    plt.subplot(1,2,2)
    plt.stem(t_up, upsampled, 'r')
    plt.title('Upsamplirani signal')
    plt.xlabel('Vrijeme [s]')
    plt.ylabel('Amplituda')
    plt.xlim(0, 1/f1)
    plt.grid(True)
    
    #Drugi plot: filtrirani signal
    plt.figure(figsize=(12,5))
    plt.plot(t, signal,'b', label='Originalni signal')
    plt.title('Originalni i filtrirani signal')
    plt.plot(t_filt, filtrirano,'r', label='Filtrirani signal')
    plt.xlabel('Vrijeme [s]')
    plt.ylabel('Amplituda')
    plt.grid(True)
    plt.xlim(0, 1/f1)

    #Treci plot: spektar
    plt.figure()
    spektar(signal, fs, 'Originalni signal')
    spektar(filtrirano, fs*up_factor, 'Upsamplirani + filtrirani')
    spektar(upsampled, fs*up_factor, 'Upsamplirani signal')
    plt.title('Spektar signala')
    plt.xlabel('Frekvencija [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.grid(True)
    plt.legend()
    plt.xlim(45,55)
    plt.show()

if __name__ == "__main__":
    filter_plot()
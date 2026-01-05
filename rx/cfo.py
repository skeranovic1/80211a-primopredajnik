import numpy as np
from rx.detection import packet_detector
import matplotlib.pyplot as plt

def gruba_vremenska_sinhronizacija(rx_lts, search_win=32):
    """
    Izvršava grubu vremensku sinhronizaciju 802.11a OFDM signala.


    """
    N = len(rx_lts)
    timing_corr = np.zeros(N, dtype=np.float64)
    
    # LTS se sastoji od 2 ponovljena simbola po 64 uzorka (korisni dio)
    LTS_len = 64
    
    for n in range(N - 2*LTS_len + 1):  # +1 da uključi zadnji mogući par
        corr = np.sum(rx_lts[n:n+LTS_len] * np.conj(rx_lts[n+LTS_len:n+2*LTS_len]))
        timing_corr[n] = np.abs(corr)

    
    # Pretraga maksimuma unutar početnog search window
    max_idx = np.argmax(timing_corr[:search_win])
    fft_start = max_idx  # indeks početka korisnog OFDM dijela unutar rx_lts
    
    timing_idxs = np.arange(len(timing_corr))
    return fft_start, timing_corr, timing_idxs



def detect_frequency_offsets(RX_Input, lts_start, plot=False):
    """
    Detektuje frekvencijski ofset nosioca (CFO) primljenog 802.11a OFDM signala.

    Funkcija izračunava i grubi (coarse) i precizni (fine) frekvencijski ofset 
    koristeći metodu automatske korelacije:
      - Grubi CFO se procjenjuje na osnovu Short Training Sequence (STS).
      - Precizni CFO se procjenjuje na osnovu Long Training Sequence (LTS).

    Parametri
    RX_Input : array_like
        Kompleksni primljeni uzorci signala.
    lts_start : int
        Indeks padajuće ivice STS-a, dobijen iz packet detector funkcije.
    plot : bool, optional
        Ako je True, prikazuje grafove realnog i imaginarnog dijela
        automatske korelacije korištene za procjenu coarse i fine CFO.
        Zadana vrijednost je False.

    Povratna vrijednost
    FrequencyOffsets : ndarray, shape (2,)
        Niz koji sadrži:
            FrequencyOffsets[0] : grubi CFO u Hz
            FrequencyOffsets[1] : precizni CFO u Hz
    """
    RX_Input=np.asarray(RX_Input)
    N=len(RX_Input)

    #1=coarse/gruba, 2=fine/precizna 
    FrequencyOffsets=np.zeros(2)

    #Coarse/gruba
    AutoCorr_Est=np.zeros(N,dtype=complex)
    Delay16=np.zeros(16,dtype=complex)
    SlidingAverage1=np.zeros(32,dtype=complex)

    for i in range(N):
        RX_Input_16=Delay16[-1]
        Delay16[1:]=Delay16[:-1]
        Delay16[0]=RX_Input[i]

        Temp=RX_Input[i]*np.conj(RX_Input_16)
        SlidingAverage1[1:]=SlidingAverage1[:-1]
        SlidingAverage1[0]=Temp

        AutoCorr_Est[i]=np.sum(SlidingAverage1)/32

    idx_coarse=lts_start-32-50
    Theta=np.angle(AutoCorr_Est[idx_coarse])
    FrequencyOffsets[0]=Theta*20e6/(2*np.pi*16)

    #Plot 
    if plot:
        #plt.figure()
    #    plt.plot(np.real(AutoCorr_Est), 'r', label='Real')
      #  plt.plot(np.imag(AutoCorr_Est), 'b', label='Imag')
     #   plt.stem([idx_coarse], [1], 'k', markerfmt='ko', basefmt=" ")
       # plt.grid(True)
       # plt.legend()

        """plt.figure()
        plt.plot(np.abs(AutoCorr_Est))
        plt.axvline(idx_coarse, color='r', linestyle='--', label='Odabrani indeks')
        plt.title("Autokorelacija STS - coarse CFO")
        plt.xlabel("Uzorke")
        plt.ylabel("|R(n)|")
        plt.legend()
        plt.grid(True)
        plt.show()"""


    #Fine/precizna 
    AutoCorr_Est_Fine=np.zeros(N, dtype=complex)
    Delay64=np.zeros(64, dtype=complex)
    SlidingAverage2=np.zeros(64, dtype=complex)

    for i in range(N):
        RX_Input_64=Delay64[-1]
        Delay64[1:]=Delay64[:-1]
        Delay64[0]=RX_Input[i]

        Temp=RX_Input[i]*np.conj(RX_Input_64)
        SlidingAverage2[1:]=SlidingAverage2[:-1]
        SlidingAverage2[0]=Temp

        AutoCorr_Est_Fine[i]=np.sum(SlidingAverage2)/64

    idx_fine=lts_start+64
    Theta=np.angle(AutoCorr_Est_Fine[idx_fine])
    FrequencyOffsets[1]=Theta*20e6/(2*np.pi*64)

    #Plot 
    if plot:
        """plt.figure()
        plt.plot(np.real(AutoCorr_Est_Fine), 'r', label='Real')
        plt.plot(np.imag(AutoCorr_Est_Fine), 'b', label='Imag')
        plt.stem([idx_fine], [1], 'k', markerfmt='ko', basefmt=" ")
        plt.grid(True)
        plt.legend()"""

        # Fini CFO
        plt.figure(figsize=(12,3))
        plt.plot(np.abs(AutoCorr_Est_Fine), label='|R(n)| - LTS')
        plt.axvline(idx_fine, color='r', linestyle='--', label='Fine CFO index')
        plt.title("Fini CFO: LTS autokorelacija")
        plt.xlabel("Uzorke")
        plt.ylabel("|R(n)|")
        plt.grid(True)
        plt.legend()
        plt.show()
    return FrequencyOffsets
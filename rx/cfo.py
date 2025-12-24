import numpy as np
from rx.detection import packet_detector
import matplotlib.pyplot as plt

def gruba_vremenska_sinhronizacija(rx_signal, delay_correction=12):
    """
    Izvršava grubu vremensku sinhronizaciju 802.11a OFDM signala.

    Funkcija detektuje kraj Short Training Sequence (STS) u primljenom signalu
    koristeći packet detector i vraća indeks početka korisnog dijela paketa
    (podataka OFDM simbola), uz mogućnost kompenzacije pomakom (delay).

    Parametri
    rx_signal : array_like
        Kompleksni primljeni uzorci signala.
    delay_correction : int, optional
        Broj uzoraka za kompenzaciju pomaka od kraja STS do početka korisnog dijela paketa.
        Zadana vrijednost je 12.

    Povratna vrijednost
    start_index : int
        Indeks u 'rx_signal' koji označava početak korisnog dijela paketa.
        Ako izračunata vrijednost padne ispod 0, vraća se 0.

    Podiznim funkcijama
    Funkcija koristi 'packet_detector' za detekciju STS sekvence.
    
    Izuzeci
    RuntimeError
        Ako paket nije detektovan (falling edge nije pronađen), funkcija podiže izuzetak.
    """
    rx_signal = np.asarray(rx_signal)
    _, _, falling_edge, _ = packet_detector(rx_signal)

    if falling_edge is None:
        raise RuntimeError("Paket nije detektovan - gruba vremenska sinhronizacija nije uspjela.")

    start_index = falling_edge - delay_correction

    if start_index < 0:
        start_index = 0

    return start_index

def detect_frequency_offsets(RX_Input, FallingEdgePosition, plot=False):
    """
    Detektuje frekvencijski ofset nosioca (CFO) primljenog 802.11a OFDM signala.

    Funkcija izračunava i grubi (coarse) i precizni (fine) frekvencijski ofset 
    koristeći metodu automatske korelacije:
      - Grubi CFO se procjenjuje na osnovu Short Training Sequence (STS).
      - Precizni CFO se procjenjuje na osnovu Long Training Sequence (LTS).

    Parametri
    RX_Input : array_like
        Kompleksni primljeni uzorci signala.
    FallingEdgePosition : int
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

    idx_coarse=FallingEdgePosition-50-1
    Theta=np.angle(AutoCorr_Est[idx_coarse])
    FrequencyOffsets[0]=Theta*20e6/(2*np.pi*16)

    #Plot 
    if plot:
        plt.figure()
        plt.plot(np.real(AutoCorr_Est), 'r', label='Real')
        plt.plot(np.imag(AutoCorr_Est), 'b', label='Imag')
        plt.stem([idx_coarse], [1], 'k', markerfmt='ko', basefmt=" ")
        plt.grid(True)
        plt.legend()

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

    idx_fine=FallingEdgePosition+125-1
    Theta=np.angle(AutoCorr_Est_Fine[idx_fine])
    FrequencyOffsets[1]=Theta*20e6/(2*np.pi*64)

    #Plot 
    if plot:
        plt.figure()
        plt.plot(np.real(AutoCorr_Est_Fine), 'r', label='Real')
        plt.plot(np.imag(AutoCorr_Est_Fine), 'b', label='Imag')
        plt.stem([idx_fine], [1], 'k', markerfmt='ko', basefmt=" ")
        plt.grid(True)
        plt.legend()

    return FrequencyOffsets


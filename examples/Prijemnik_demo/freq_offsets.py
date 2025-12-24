import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from channel.Channel_Model import Channel_Model
from channel.channel_settings import ChannelSettings
from channel.channel_mode import ChannelMode
from tx.OFDM_TX_802_11 import Transmitter80211a
from rx.detection import packet_detector
from rx.pretprocessing import iq_preprocessing
from rx.cfo import detect_frequency_offsets
from tx.utilities import plot_konstelaciju

def main():
    """
    Simulira prijem i djelomicnu obradu 802.11a OFDM signala.

    Glavne funkcionalnosti:
    1. Generiše OFDM paket sa predajnika (Transmitter80211a).
    2. Prolazi signal kroz kanal sa šumom i multipathom (Channel_Model).
    3. Primenjuje pretprocesing na primljeni signal (iq_preprocessing).
    4. Detektuje paket i STS koristeći packet_detector.
    5. Izračunava grubi (coarse) i precizni (fine) frekvencijski ofset (CFO) i koriguje fazu signala.
    6. Prikazuje grafove:
       - Paket detektor (|R| / P) i zastavica paketa.
       - Realni dio primljenog signala sa STS oznakom.
       - Faza RX signala prije korekcije, nakon grube i nakon fine CFO korekcije.
    """
    num_ofdm_symbols=2500
    up_factor=2
    fs_base=20e6
    fs=fs_base*up_factor

    #Predajnik
    tx=Transmitter80211a(
        num_ofdm_symbols=num_ofdm_symbols,
        bits_per_symbol=2,   #QPSK
        step=1,
        up_factor=up_factor,
        seed=17,
        plot=False
    )
    tx_signal, _=tx.generate_frame()

    #Kanal
    settings=ChannelSettings(
        sample_rate=fs,
        number_of_taps=2,
        delay_spread=10,
        snr_db=20
    )
    mode=ChannelMode(
        multipath=1,
        thermal_noise=1
    )
    channel=Channel_Model(settings, mode)
    rx_signal, _=channel.apply(tx_signal)

    #Prijemnik - pretprocesing, detekcija i korekcija
    rx_signal, fs1=iq_preprocessing(
        rx_signal=rx_signal,
        tx_signal=tx_signal,
        fs=fs
    )
    comparison_ratio, packet_flag, falling_edge, _=packet_detector(rx_signal)
    if falling_edge is None:
        print("Paket nije detektovan → nema CFO")
        return

    N=len(rx_signal)
    t=np.arange(N)/fs1*1e6

    plt.figure(figsize=(14, 8))
    plt.subplot(3,1,1)
    plt.plot(t,comparison_ratio)
    plt.axhline(0.85,color='r',linestyle='--')
    plt.axhline(0.65,color='orange',linestyle='--')
    plt.title("Paket detektor: |R| / P")
    plt.grid(True)
    plt.subplot(3,1,2)
    plt.step(t,packet_flag,where='post')
    plt.title("Packet flag/zastavica")
    plt.yticks([0,1])
    plt.grid(True)
    plt.subplot(3,1,3)
    plt.plot(t,np.real(rx_signal))
    plt.axvline(falling_edge/fs1*1e6,color='r',linestyle='--',label="Kraj STS-a")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #Detekcija frekvencijskih ofseta
    FreqOffset=detect_frequency_offsets(rx_signal,falling_edge)
    CoarseOffset=FreqOffset[0]
    print(f"Coarse CFO = {CoarseOffset:.2f} Hz")

    #Coarse/gruba korekcija
    n=np.arange(len(rx_signal))
    NCO_coarse=np.exp(-1j*2*np.pi*n*CoarseOffset/fs1)
    rx_coarse=rx_signal*NCO_coarse

    # Fine/precizna korekcija
    FreqOffset=detect_frequency_offsets(rx_coarse,falling_edge)  #ponovo se pokreće detekcija za fine offset
    FineOffset=FreqOffset[1]
    NCO_fine=np.exp(-1j*2*np.pi*n*FineOffset/fs1)
    rx_fine=rx_coarse*NCO_fine
    print(f"Fine CFO = {FineOffset:.2f} Hz")

    plt.figure(figsize=(12, 10)) 
    plt.subplot(3,1,1) #Prije korekcije
    plt.plot(np.unwrap(np.angle(rx_signal)))
    plt.title("Faza RX signala PRIJE CFO korekcije")
    plt.xlabel("Uzorci")
    plt.ylabel("Faza [rad]")
    plt.grid(True)
    plt.subplot(3,1,2) #Nakon grube korekcije
    plt.plot(np.unwrap(np.angle(rx_coarse)))
    plt.title("Faza RX signala poslije grube korekcije")
    plt.xlabel("Uzorci")
    plt.ylabel("Faza [rad]")
    plt.grid(True)
    plt.subplot(3,1,3) #Nakon fine korekcije
    plt.plot(np.unwrap(np.angle(rx_fine)))
    plt.title("Faza RX signala poslije precizne korekcije")
    plt.xlabel("Uzorci")
    plt.ylabel("Faza [rad]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()    

if __name__ == "__main__":
    main()

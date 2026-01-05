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
from rx.cfo import detect_frequency_offsets, gruba_vremenska_sinhronizacija
from tx.utilities import plot_konstelaciju
def plot_constellations(rx_signal, rx_coarse):
    plt.figure(figsize=(12,6))

    # Prije korekcije
    plt.subplot(1, 2, 1)
    plt.plot(rx_signal.real, rx_signal.imag, 'o', markersize=2, alpha=0.6)
    plt.title("Konstelacija prije korekcije CFO")
    plt.xlabel("Realni dio")
    plt.ylabel("Imaginari dio")
    plt.grid(True)
    plt.axis('equal')

    # Nakon korekcije
    plt.subplot(1, 2, 2)
    plt.plot(rx_coarse.real, rx_coarse.imag, 'o', markersize=2, alpha=0.6)
    plt.title("Konstelacija nakon grube CFO korekcije")
    plt.xlabel("Realni dio")
    plt.ylabel("Imaginari dio")
    plt.grid(True)
    plt.axis('equal')

    plt.tight_layout()
    plt.show()


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
    num_ofdm_symbols=3
    up_factor=2
    fs_base=20e6
    fs=fs_base*up_factor

    #Predajnik
    tx=Transmitter80211a(
        num_ofdm_symbols=num_ofdm_symbols,
        bits_per_symbol=2,   #QPSK
        step=1,
        up_factor=up_factor,
        seed=16,
        plot=False
    )
    tx_signal, _= tx.generate_frame()

    #Kanal
    settings=ChannelSettings(
        sample_rate=fs,
        number_of_taps=2,
        delay_spread=10,
        snr_db=10
    )
    mode=ChannelMode(
        multipath=0,
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
    comparison_ratio, packet_flag, falling_edge, _ = packet_detector(rx_signal)
    print("Kraj detektovane short training sekvence:", falling_edge)
    print(f"Očekivani kraj STS-a (sample): {160}")
    print(f"Greska pri detekciji STS-a: {160-falling_edge} uzoraka")
    
    #Symbol timing
    start_lts = falling_edge
    end_lts = falling_edge + 160  # 32 CP + 2x64 korisna
    rx_lts = rx_signal[start_lts:end_lts]

    pravi_pocetak_lts, timing_corr, timing_idxs = gruba_vremenska_sinhronizacija(rx_lts, search_win=32)

    # FFT start u globalnim indeksima
    lts_start = start_lts + pravi_pocetak_lts
    ideal_lts_start = 160+32

    timing_error = ideal_lts_start  - lts_start
    print("Detektovani pocetak korisnog dijela LTS-a:", lts_start)
    print("Idealni pocetak korisnog dijela LTS-a:", ideal_lts_start)
    print(f"Symbol timing greška: {timing_error} uzoraka ({timing_error/fs1*1e6:.2f} µs)")

    plt.figure(figsize=(10,4))
    plt.plot(timing_idxs, timing_corr, label="|C(n)| - korelacija LTS")
    plt.axvline(pravi_pocetak_lts, color='r', linestyle='--', label="Detektovani FFT start")
    plt.title("Gruba vremenska sinhronizacija LTS")
    plt.xlabel("Sample index (relativno LTS)")
    plt.ylabel("|C(n)|")
    plt.legend()
    plt.grid(True)
    plt.show()

    N=len(rx_signal)
    t=np.arange(N)/fs_base*1e6

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
    plt.axvline(falling_edge/fs_base*1e6,color='r',linestyle='--',label="Kraj STS-a")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    #Detekcija frekvencijskih ofseta
    FreqOffset=detect_frequency_offsets(rx_signal,lts_start)
    CoarseOffset=FreqOffset[0]
    print(f"Coarse CFO = {CoarseOffset:.2f} Hz")

    #Coarse/gruba korekcija
    n=np.arange(len(rx_signal))
    NCO_coarse=np.exp(-1j*2*np.pi*n*CoarseOffset/fs_base)
    rx_coarse=rx_signal*NCO_coarse

    phase_before = np.unwrap(np.angle(rx_signal))
    phase_after = np.unwrap(np.angle(rx_coarse))

    # Plot faze
    plt.figure(figsize=(12, 5))
    plt.plot(phase_before, label="Faza prije korekcije", alpha=0.7)
    plt.plot(phase_after, label="Faza nakon grube CFO korekcije", alpha=0.7)
    plt.title("Faza RX signala prije i nakon grube CFO korekcije")
    plt.xlabel("Uzorke")
    plt.ylabel("Faza [rad]")
    plt.legend()
    plt.grid(True)
    plt.show()

    #plot_constellations(rx_signal, rx_coarse)

    # Fine/precizna korekcija
    FreqOffset=detect_frequency_offsets(rx_coarse,lts_start, plot=True)  #ponovo se pokreće detekcija za fine offset
    FineOffset=FreqOffset[1]
    NCO_fine=np.exp(-1j*2*np.pi*n*FineOffset/fs_base)
    rx_fine=rx_coarse*NCO_fine
    print(f"Fine CFO = {FineOffset:.3f} Hz")

    phase_after_fine= np.unwrap(np.angle(rx_fine))
    plt.figure(figsize=(12, 5))
    plt.plot(phase_before, label="Faza prije korekcije", alpha=0.7)
    plt.plot(phase_after, label="Faza nakon grube CFO korekcije", alpha=0.7)
    plt.plot(phase_after_fine, label="Faza nakon fine CFO korekcije", alpha=0.7)
    plt.title("Faza RX signala prije i nakon grube CFO korekcije")
    plt.xlabel("Uzorke")
    plt.ylabel("Faza [rad]")
    plt.legend()
    plt.grid(True)
    plt.show()

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

    # --- Konstelacije ---
    plt.figure(figsize=(12,4))

    plt.subplot(1, 3, 1)
    plt.plot(rx_signal.real, rx_signal.imag, 'o', markersize=2, alpha=0.6)
    plt.title("Konstelacija prije korekcije")
    plt.xlabel("Realni dio"); plt.ylabel("Imaginari dio")
    plt.grid(True); plt.axis('equal')

    plt.subplot(1, 3, 2)
    plt.plot(rx_coarse.real, rx_coarse.imag, 'o', markersize=2, alpha=0.6)
    plt.title("Nakon coarse CFO")
    plt.xlabel("Realni dio"); plt.ylabel("Imaginari dio")
    plt.grid(True); plt.axis('equal')

    plt.subplot(1, 3, 3)
    plt.plot(rx_fine.real, rx_fine.imag, 'o', markersize=2, alpha=0.6)
    plt.title("Nakon fine CFO")
    plt.xlabel("Realni dio"); plt.ylabel("Imaginari dio")
    plt.grid(True); plt.axis('equal')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

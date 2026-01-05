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

def test_packet_detector(rx_signal, fs, title=""):
    """
    Testira i prikazuje rad packet_detector funkcije na primljenom signalu.

    Funkcija izvršava packet_detector na RX signalu, ispisuje detektovani kraj STS-a,
    grešku u odnosu na očekivani kraj STS-a, te prikazuje grafove:
      1. Realni dio RX signala sa oznakama pravog i detektovanog kraja STS.
      2. Odnos |R| / P korišten za detekciju paketa.
      3. Paket flag (binarna zastavica prisustva paketa).

    Parametri
    rx_signal : array_like
        Kompleksni primljeni signal.
    fs : float
        Frekvencija uzorkovanje RX signala.
    title : str, optional
        Naslov za grafove. Default je prazno.

    Povratna vrijednost
    Funkcija ne vraća ništa; služi za testiranje i vizualizaciju detektora.
    """
    comparison_ratio, packet_flag, falling_edge, _ = packet_detector(rx_signal)
    N=len(rx_signal)
    t=np.arange(N)/fs*1e6  #u µs
    
    STS_len=16*10   #uzorci ali ovdje bez upsamplinga jer smo u funkciju vratili normalan fs
    sts_end_time=STS_len/fs*1e6
    sts_end_sample=STS_len

    print(f"Očekivani kraj STS-a (sample): {sts_end_sample}")
    print(f"Detektovani falling edge (sample): {falling_edge}")

    if falling_edge is not None:
        error=falling_edge-sts_end_sample
        print(f"Greška detekcije: {error} uzoraka ({error/fs*1e6:.2f} µs)")
    else:
        print("Paket nije detektovan")

    #plot
    fig, axs=plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    #Rx signal
    axs[0].plot(t, np.real(rx_signal), label="Rx (real)")
    axs[0].axvline(sts_end_time, color='green', linestyle='--', label='Pravi kraj STS')
    if falling_edge is not None:
        axs[0].axvline(falling_edge/fs*1e6, color='red', linestyle='--', label='Detektovan kraj STS')
    axs[0].set_title(f"RX signal – {title}")
    axs[0].set_ylabel("Amplituda")
    axs[0].grid(True)
    axs[0].legend()

    #Odnosi za poredenje
    axs[1].plot(t, comparison_ratio, label="|R| / P")
    axs[1].axhline(0.85, color='red', linestyle='--', label='Upper threshold')
    axs[1].axhline(0.65, color='orange', linestyle='--', label='Lower threshold')
    axs[1].set_ylabel("Ratio")
    axs[1].grid(True)
    axs[1].legend()

    #Packet flag
    axs[2].step(t, packet_flag, where='post', label="packet_det_flag")
    axs[2].set_ylabel("Flag")
    axs[2].set_xlabel("Vrijeme [µs]")
    axs[2].set_yticks([0, 1])
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plt.show()

def main():
    num_ofdm_symbols=1
    up_factor=2
    fs_base=20e6
    fs=fs_base*up_factor

    tx=Transmitter80211a(
        num_ofdm_symbols=num_ofdm_symbols,
        bits_per_symbol=2,   #QPSK
        step=1,
        up_factor=up_factor,
        seed=50,
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

    rx_signal, fs =iq_preprocessing(
        rx_signal=rx_signal,
        tx_signal=tx_signal,
        fs=fs
    )

    test_packet_detector(
        rx_signal,
        fs,
        title="AWGN + multipath kanal"
    )

if __name__ == "__main__":
    main()
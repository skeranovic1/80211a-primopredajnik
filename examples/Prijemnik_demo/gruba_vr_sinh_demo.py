import numpy as np
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
from tx.OFDM_TX_802_11 import Transmitter80211a
from channel.Channel_Model import Channel_Model
from channel.channel_settings import ChannelSettings
from channel.channel_mode import ChannelMode
from rx.pretprocessing import iq_preprocessing
from rx.cfo import gruba_vremenska_sinhronizacija

def main():
    #Parametri
    num_ofdm_symbols = 3
    up_factor = 2
    fs_base = 20e6
    fs = fs_base * up_factor

    #Tx
    tx = Transmitter80211a(
        num_ofdm_symbols=num_ofdm_symbols,
        bits_per_symbol=2,
        step=1,
        up_factor=up_factor,
        seed=42,
        plot=False
    )

    tx_signal, _ = tx.generate_frame()
    tx_signal = np.asarray(tx_signal).flatten()

    #Kanal
    settings = ChannelSettings(
        sample_rate=fs,
        number_of_taps=1,
        delay_spread=10,
        snr_db=20
    )

    mode = ChannelMode(multipath=1, thermal_noise=1)
    channel = Channel_Model(settings, mode)
    rx_signal, _ = channel.apply(tx_signal)

    #Pretprocesing
    rx_signal, fs = iq_preprocessing(rx_signal, tx_signal, fs)

    #Gruba vremenska sinhronizacija
    start_idx = gruba_vremenska_sinhronizacija(rx_signal)

    print("Početak korisnog dijela paketa (sample):", start_idx)

    #Vizualna provjera
    t = np.arange(len(rx_signal)) / fs * 1e6
    plt.figure(figsize=(12, 4))
    plt.plot(t, np.real(rx_signal), label="RX signal (real)")
    plt.axvline(start_idx / fs * 1e6, color="red", linestyle="--",label="Detektovani početak paketa")
    plt.xlabel("Vrijeme [µs]")
    plt.ylabel("Amplituda")
    plt.title("Gruba vremenska sinhronizacija")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

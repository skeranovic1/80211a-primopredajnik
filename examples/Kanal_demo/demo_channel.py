import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)
#Tx
from tx.OFDM_TX_802_11 import Transmitter80211a

#CHANNEL
from channel.Channel_Model import Channel_Model
from channel.channel_settings import ChannelSettings
from channel.channel_mode import ChannelMode


def oznaci_okvir(ax, up_factor, num_ofdm_symbols, fs):
    """
    Oznake STS, LTS, payload i GI – ISTO kao u demo.py
    """
    STS_len = 16 * up_factor * 10
    LTS_len = 160 * up_factor
    OFDM_symbol_len = 64 * up_factor
    GI_len = 16 * up_factor

    sts_end = STS_len / fs
    lts_end = (STS_len + LTS_len) / fs
    payload_end = (STS_len + LTS_len +
                   num_ofdm_symbols * (OFDM_symbol_len + GI_len)) / fs

    ax.axvline(sts_end * 1e6, color='green', linestyle='--', label='Kraj STS')
    ax.axvline(lts_end * 1e6, color='orange', linestyle='--', label='Kraj LTS')
    ax.axvline(payload_end * 1e6, color='purple', linestyle='--', label='Kraj payload')

    for i in range(num_ofdm_symbols):
        gi_start = (STS_len + LTS_len +
                    i * (GI_len + OFDM_symbol_len)) / fs
        gi_end = gi_start + GI_len / fs
        ax.axvspan(gi_start * 1e6, gi_end * 1e6,
                   color='green', alpha=0.25)


def testiranje():
    num_ofdm_symbols = 2
    up_factor = 2

    tx = Transmitter80211a(
        num_ofdm_symbols=num_ofdm_symbols,
        bits_per_symbol=2,   # QPSK
        step=1,
        up_factor=up_factor,
        seed=42,
        plot=False
    )

    signal_tx, _ = tx.generate_frame()
    signal_tx = np.asarray(signal_tx).flatten()

    fs_base = 20e6
    fs = fs_base * up_factor
    t = np.arange(len(signal_tx)) / fs

    settings = ChannelSettings(
        sample_rate=fs,
        number_of_taps=2,
        delay_spread=10e-9,
        snr_db=20
    )

    mode_awgn = ChannelMode(multipath=0, thermal_noise=1)
    channel_awgn = Channel_Model(settings, mode_awgn)
    rx_awgn, _ = channel_awgn.apply(signal_tx)
    rx_awgn = np.asarray(rx_awgn).flatten()

    # skaliranje radi istog vizuelnog nivoa
    rx_awgn *= np.sqrt(np.mean(np.abs(signal_tx)**2)) / np.sqrt(np.mean(np.abs(rx_awgn)**2))

    # 4. SLUČAJ 2: AWGN + MULTIPATH
    mode_mp = ChannelMode(multipath=1, thermal_noise=1)
    channel_mp = Channel_Model(settings, mode_mp)
    rx_mp, _ = channel_mp.apply(signal_tx)
    rx_mp = np.asarray(rx_mp).flatten()

    rx_mp *= np.sqrt(np.mean(np.abs(signal_tx)**2)) / np.sqrt(np.mean(np.abs(rx_mp)**2))

    # 5. PLOT 1 — PRIJE KANALA vs AWGN
    fig1, ax1 = plt.subplots(figsize=(16,6))
    ax1.plot(t * 1e6, np.real(signal_tx), label='Tx – prije kanala')
    ax1.plot(t * 1e6, np.real(rx_awgn), label='Rx – AWGN', alpha=0.8)

    oznaci_okvir(ax1, up_factor, num_ofdm_symbols, fs)

    ax1.set_title("OFDM signal u vremenskom domenu – AWGN")
    ax1.set_xlabel("Vrijeme [µs]")
    ax1.set_ylabel("Amplituda")
    ax1.grid(True)
    ax1.legend()
    plt.tight_layout()
    plt.show()

    #PLOT 2 — PRIJE KANALA vs AWGN + MULTIPATH
    fig2, ax2 = plt.subplots(figsize=(16,6))
    ax2.plot(t * 1e6, np.real(signal_tx), label='Tx – prije kanala')
    ax2.plot(t * 1e6, np.real(rx_mp), label='Rx – AWGN + multipath', alpha=0.8)

    oznaci_okvir(ax2, up_factor, num_ofdm_symbols, fs)

    ax2.set_title("OFDM signal u vremenskom domenu – AWGN + multipath")
    ax2.set_xlabel("Vrijeme [µs]")
    ax2.set_ylabel("Amplituda")
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    testiranje()

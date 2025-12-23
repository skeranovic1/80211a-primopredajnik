import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import matplotlib.pyplot as plt
from tx.OFDM_mapper import Mapper_OFDM
from channel.Channel_Model import Channel_Model
from channel.channel_settings import ChannelSettings
from channel.channel_mode import ChannelMode

def plot_constellation_side_by_side(tx, rx, title_tx, title_rx, max_points=2000, seed=123):
    tx = np.asarray(tx).reshape(-1)
    rx = np.asarray(rx).reshape(-1)
    n = len(tx)
    if n == 0:
        return
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        tx_p = tx[idx]
        rx_p = rx[idx]
    else:
        tx_p = tx
        rx_p = rx

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].scatter(tx_p.real, tx_p.imag, s=8)
    axs[0].axhline(0, color="black", linewidth=0.5)
    axs[0].axvline(0, color="black", linewidth=0.5)
    axs[0].grid(True)
    axs[0].set_aspect("equal", "box")
    axs[0].set_title(title_tx)
    axs[0].set_xlabel("I")
    axs[0].set_ylabel("Q")

    axs[1].scatter(rx_p.real, rx_p.imag, s=8)
    axs[1].axhline(0, color="black", linewidth=0.5)
    axs[1].axvline(0, color="black", linewidth=0.5)
    axs[1].grid(True)
    axs[1].set_aspect("equal", "box")
    axs[1].set_title(title_rx)
    axs[1].set_xlabel("I")
    axs[1].set_ylabel("Q")

    plt.tight_layout()
    plt.show()

def channel_out_from_bits(bits, BitsPerSymbol, snr_db, multipath, thermal_noise):
    #Mapper (QPSK)
    tx = Mapper_OFDM(bits, BitsPerSymbol, plot=False)

    #Channel settings + mode
    settings = ChannelSettings(snr_db=snr_db)
    mode = ChannelMode(multipath=multipath, thermal_noise=thermal_noise)

    #Channel apply metoda
    ch = Channel_Model(settings, mode)
    rx, fir_taps = ch.apply(tx)

    rx = np.asarray(rx).reshape(-1)
    return tx, rx, fir_taps


if __name__ == "__main__":
    np.random.seed(42)

    # QPSK
    BitsPerSymbol = 2
    n_bits = 20000
    snr_db = 12

    bits = np.random.randint(0, 2, n_bits)

    #1) AWGN 
    tx1, rx1, _ = channel_out_from_bits(
        bits, BitsPerSymbol, snr_db,
        multipath=0,
        thermal_noise=1
    )
    plot_constellation_side_by_side(
        tx1, rx1,
        title_tx="TX (Channel In) | QPSK",
        title_rx=f"RX (Channel Out) | AWGN | SNR={snr_db} dB "
    )
    #2) Multipath + AWGN 
    tx2, rx2, _ = channel_out_from_bits(
        bits, BitsPerSymbol, snr_db,
        multipath=1,
        thermal_noise=1
    )
    plot_constellation_side_by_side(
    tx1, rx1,
    title_tx="TX (Channel In) | QPSK",
    title_rx=f"RX (Channel Out) | Multipath + AWGN | SNR={snr_db} dB"
    )


import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tx.OFDM_mapper import Mapper_OFDM
from channel.Channel_Model import Channel_Model
from channel.channel_settings import ChannelSettings
from channel.channel_mode import ChannelMode

def channel_out_from_bits(bits, BitsPerSymbol, snr_db, multipath, thermal_noise):
    """
    Mapper + Channel pipeline
    """
    # Mapper (QPSK)
    tx = Mapper_OFDM(bits, BitsPerSymbol, plot=False)

    # Channel settings + mode
    settings = ChannelSettings(snr_db=snr_db)
    mode = ChannelMode(multipath=multipath, thermal_noise=thermal_noise)

    # Channel apply
    ch = Channel_Model(settings, mode)
    rx, fir_taps = ch.apply(tx)

    tx = np.asarray(tx).reshape(-1)
    rx = np.asarray(rx).reshape(-1)
    return tx, rx, fir_taps

def plot_spectrum_before_after(tx, rx, fs, title_tx="TX Spectrum", title_rx="RX Spectrum"):
    """
    Prikazuje spektralne profile signala prije i poslije kanala.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # TX spektar
    axs[0].magnitude_spectrum(tx, Fs=fs, scale='linear', color='blue', label='TX')
    axs[0].set_title(title_tx)
    axs[0].set_xlabel("Frequency (Hz)")
    axs[0].set_ylabel("Magnitude (dB)")
    axs[0].grid(True)
    axs[0].legend()

    # RX spektar
    axs[1].magnitude_spectrum(rx, Fs=fs, scale='linear', color='orange', label='RX')
    axs[1].set_title(title_rx)
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Magnitude (dB)")
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)

    # Parametri
    BitsPerSymbol = 2
    n_bits = 20000
    snr_db = 8
    Fs = 20e6  # Sampling frequency

    bits = np.random.randint(0, 2, n_bits)

    # ===== 1) AWGN only =====
    tx1, rx1, _ = channel_out_from_bits(
        bits, BitsPerSymbol, snr_db,
        multipath=0,
        thermal_noise=1
    )

    plot_spectrum_before_after(
        tx1, rx1,
        Fs,
        title_tx="TX Spectrum | QPSK",
        title_rx=f"RX Spectrum | AWGN | SNR={snr_db} dB"
    )

    # ===== 2) Multipath + AWGN =====
    tx2, rx2, _ = channel_out_from_bits(
        bits, BitsPerSymbol, snr_db,
        multipath=1,
        thermal_noise=1
    )

    plot_spectrum_before_after(
        tx2, rx2,
        Fs,
        title_tx="TX Spectrum | QPSK",
        title_rx=f"RX Spectrum | Multipath + AWGN | SNR={snr_db} dB"
    )

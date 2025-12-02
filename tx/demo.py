import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt

from tx.utilities import bit_sequence
from tx.OFDM_mapper import Mapper_OFDM
from tx.ifft_ofdm_symbol import IFFT_GI
from tx.short_sequence import get_short_training_sequence
from tx.long_sequence import get_long_training_sequence   # ← DODANO
from tx.filters import half_band_upsample


def tx_input_output_demo():
    """
    Najjednostavniji mogući prikaz predajnika:

        Ulaz: bitovi
        Izlaz: finalni interpolirani signal (DAC → RF)
        + STS (short preamble)
        + LTS (long preamble)
        + dodatni prikaz: zoom OFDM payloada
    """

    # 1) Generisanje bitova
    bits = bit_sequence(5, 2, 0)  # 5 OFDM simbola, QPSK

    # 2) Mapiranje → QPSK → IFFT → OFDM signal
    qpsk = Mapper_OFDM(bits, 2)
    ofdm = IFFT_GI(qpsk)

    # 3) Preambule
    sts = get_short_training_sequence(1)     # 10 × STS
    lts = get_long_training_sequence(1)      # CP + LTS1 + LTS2  (64+32 sample)

    # 4) TX paket = preambule + payload
    tx_packet = np.concatenate((sts, lts, ofdm))

    # 5) Interpolacija x2 (zero-stuffing + half-band filter)
    up2 = np.zeros(len(tx_packet) * 2)
    up2[::2] = tx_packet
    tx_output, _ = half_band_upsample(up2, up_factor=1, N=31, plot=False)

    # 6) Grafički prikaz
    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    # Prikaz ulaznih bitova
    axs[0].stem(bits[:80])
    axs[0].set_title("ULAZ U PREDAJNIK — Bitovi")
    axs[0].grid(True)

    # Prikaz STS + LTS + OFDM payload
    axs[1].plot(np.real(tx_output[:1200]))
    axs[1].set_title("IZLAZ IZ PREDAJNIKA — STS + LTS + OFDM")
    axs[1].grid(True)

    # Zoom OFDM signala
    start = 700
    stop = 1200
    axs[2].plot(np.real(tx_output[start:stop]))
    axs[2].set_title("ZOOM — OFDM payload nakon interpolacije")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tx_input_output_demo()

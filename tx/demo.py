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
from tx.filters import half_band_upsample


def tx_input_output_demo():
    """
    Najjednostavniji mogući prikaz predajnika:

        Ulaz: bitovi
        Izlaz: finalni interpolirani signal (DAC → RF)
        + dodatni prikaz: zoom OFDM payloada

    """


    bits = bit_sequence(5, 2, 0)  # 5 OFDM simbola, QPSK

   
    qpsk = Mapper_OFDM(bits, 2)       # QPSK mapiranje
    ofdm = IFFT_GI(qpsk)              # OFDM IFFT+GI
    sts = get_short_training_sequence(1)

    # Kompletan TX paket (preambula + payload)
    tx_packet = np.concatenate((sts, ofdm))

    # Interpolacija / upsampling x2
    up2 = np.zeros(len(tx_packet) * 2)
    up2[::2] = tx_packet
    tx_output, _ = half_band_upsample(up2, up_factor=1, N=31, plot=False)

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    # === SUBPLOT 1: Bitovi (ulaz predajnika) ===
    axs[0].stem(bits[:80])
    axs[0].set_title("ULAZ U PREDAJNIK — Bitovi")
    axs[0].set_xlabel("Sample")
    axs[0].set_ylabel("Vrijednost")
    axs[0].grid(True)

    # === SUBPLOT 2: Kompletan TX izlaz (STS + OFDM) ===
    axs[1].plot(np.real(tx_output[:600]))
    axs[1].set_title("IZLAZ IZ PREDAJNIKA — Interpolirani TX signal (STS + OFDM)")
    axs[1].set_xlabel("Sample")
    axs[1].set_ylabel("Amplituda")
    axs[1].grid(True)

    # === SUBPLOT 3: Zoom OFDM payloada ===
    start = 320   # Početak OFDM payloada (poslije STS-a)
    stop = 600    # Dio OFDM signala za pregled

    axs[2].plot(np.real(tx_output[start:stop]))
    axs[2].set_title("ZOOM — OFDM payload nakon interpolacije")
    axs[2].set_xlabel("Sample")
    axs[2].set_ylabel("Amplituda")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tx_input_output_demo()

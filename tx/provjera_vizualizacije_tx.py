import numpy as np
import matplotlib.pyplot as plt

from tx.vizualizacija_tx import (
    plot_time_domain_tx,
    plot_spectrum_tx,
    plot_constellation_tx,
)

from tx.OFDM_mapper import Mapper_OFDM
from tx.long_sequence import get_long_training_sequence
from tx.short_sequence import get_short_training_sequence


def main():
    fs = 20e6  # 20 MHz sample rate

    # ==================================
    # 1) PRAVI TX QPSK simboli tima
    # ==================================
    bits = np.random.randint(0, 2, size=2*1000)
    qpsk_symbols = Mapper_OFDM(bits, 2)

    plot_constellation_tx(qpsk_symbols, title="QPSK - konstelacioni dijagram (PRAVI TX)")

    # ==================================
    # 2) PRAVA 802.11a SHORT SEQUENCE
    # ==================================
    sts = get_short_training_sequence()

    plot_time_domain_tx(sts, fs, title_prefix="Short Training Sequence")
    plot_spectrum_tx(sts, fs, title_prefix="Short Training Sequence")

    # ==================================
    # 3) PRAVA 802.11a LONG SEQUENCE
    # ==================================
    lts = get_long_training_sequence()

    plot_time_domain_tx(lts, fs, title_prefix="Long Training Sequence")
    plot_spectrum_tx(lts, fs, title_prefix="Long Training Sequence")

    plt.show()


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from tx.vizualizacija_tx import (
    plot_time_domain_tx,
    plot_spectrum_tx,
    plot_constellation_tx  
)


def main():
    
    fs = 20e6  # 20 MHz
    n = 2048
    t = np.arange(n) / fs

    f1 = 2e6   # 2 MHz
    f2 = 5e6   # 5 MHz

    signal = np.exp(1j * 2 * np.pi * f1 * t) + 0.5 * np.exp(1j * 2 * np.pi * f2 * t)

    # TG1-57: vremenski domen
    plot_time_domain_tx(signal, fs, title_prefix="Testni TX signal")

    # TG1-56: FFT spektar
    plot_spectrum_tx(signal, fs, title_prefix="Testni TX signal")

    
    # TEST 3: TG1-55 – QPSK konstelacija
    
    num_symbols = 500

    # nasumični bitovi (2 bita po simbolu)
    bits = np.random.randint(0, 2, size=2 * num_symbols)
    bit_pairs = bits.reshape((-1, 2))

    # Gray mapping za QPSK
    mapping = {
        (0, 0): 1 + 1j,
        (0, 1): -1 + 1j,
        (1, 1): -1 - 1j,
        (1, 0): 1 - 1j,
    }

    symbols = np.array([mapping[tuple(bp)] for bp in bit_pairs])
  
    # TG1-55: konstelacioni dijagram
    plot_constellation_tx(symbols, title="QPSK - konstelacioni dijagram (test)")

    # prikaži sve figure
    plt.show()


if __name__ == "__main__":
    main()

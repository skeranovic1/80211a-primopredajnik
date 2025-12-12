import numpy as np
import matplotlib.pyplot as plt
from tx.OFDM_mapper import Mapper_OFDM  
from tx.utilities import bit_sequence

def mapper_test(bit_seq):
    """
    Mapira ulazne bitove u OFDM simbole za BPSK, QPSK, 16-QAM i 64-QAM
    i prikazuje konstelacije za sve modulacije.

    Parametri
    bit_seq : array-like
        Niz ulaznih bitova (0 ili 1)
    """
    modulacije = {1: "BPSK", 2: "QPSK", 4: "16-QAM", 6: "64-QAM"}

    plt.figure(figsize=(12,12))

    for i, (BitsPerSymbol, naziv) in enumerate(modulacije.items(), 1):
        # Provjeri da imamo dovoljno bitova za trenutnu modulaciju
        n_symbols = len(bit_seq) // BitsPerSymbol
        if n_symbols == 0:
            continue
        input_bits = bit_seq[:n_symbols * BitsPerSymbol]

        simbole = Mapper_OFDM(input_bits, BitsPerSymbol, plot=False)

        plt.subplot(2,2,i)
        plt.scatter(simbole.real, simbole.imag, color='blue')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.title(f"{naziv} ({BitsPerSymbol}-bit po simbolu)")
        plt.xlabel("Realni dio")
        plt.ylabel("Imaginarni dio")
        plt.grid(True)
        plt.gca().set_aspect("equal", "box")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    np.random.seed(42) 
    test_bits = np.random.randint(0, 2, 2000) 
    mapper_test(test_bits)

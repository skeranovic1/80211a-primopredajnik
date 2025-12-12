import numpy as np
import matplotlib.pyplot as plt
from tx.ifft_ofdm_symbol import IFFT_GI
from tx.OFDM_mapper import Mapper_OFDM
from tx.utilities import bit_sequence

def main():
    """
    Demonstrira generisanje i prikaz OFDM payload signala.

    Funkcija:
    - Generiše nasumične ulazne bitove.
    - Mapira bitove u QPSK simbole koristeći Mapper_OFDM.
    - Primjenjuje IFFT i dodaje guard interval koristeći IFFT_GI.
    - Prikazuje realni i imaginarni dio OFDM payload signala u vremenskom domenu.

    Napomene
    - Plot prikazuje strukturu OFDM payload-a po uzorcima.
    - Parametri simulacije (broj simbola, seme generatora) su fiksni unutar funkcije.
    """
    # Generišemo nasumične ulazne bitove
    input_bits = bit_sequence(2, 2, sd=41)
    symbols = Mapper_OFDM(input_bits, 2)
    payload = IFFT_GI(symbols, plot=False)  

    #Crtanje cijelog payload-a
    plt.figure(figsize=(14,5))
    plt.subplot(2,1,1)
    plt.stem(np.real(payload), 'b', label='Real')
    plt.subplot(2,1,2)
    plt.stem(np.imag(payload), 'r', label='Imag')
    plt.title(f"OFDM payload za {2} simbola")
    plt.xlabel("Uzorci")
    plt.ylabel("Amplituda")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()


import numpy as np
import matplotlib.pyplot as plt
from ifft_ofdm_symbol import IFFT_GI
from OFDM_mapper import Mapper_OFDM
from utilities import bit_sequence

def main():
    # Generišemo nasumične ulazne bitove
    input_bits = bit_sequence(2, 2, sd=41)
    symbols = Mapper_OFDM(input_bits, 2)
    payload = IFFT_GI(symbols, plot=False)  

    #Crtanje cijelog payload-a
    plt.figure(figsize=(14,5))
    plt.subplot(2,1,1)
    plt.plot(np.real(payload), 'b', label='Real')
    plt.subplot(2,1,2)
    plt.plot(np.imag(payload), 'r', label='Imag')
    plt.title(f"OFDM payload za {2} simbola")
    plt.xlabel("Uzorci")
    plt.ylabel("Amplituda")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()


from OFDM_TX_802_11 import OFDM_TX
import numpy as np
import matplotlib.pyplot as plt

def tx_input_output_demo1():
    """
    Najjednostavniji mogući prikaz predajnika:

        Ulaz: bitovi
        Izlaz: finalni interpolirani signal (DAC → RF)
        + STS (short preamble)
        + LTS (long preamble)
        + dodatni prikaz: zoom OFDM payloada
    """
    Sample_Output, Symbol_Stream = OFDM_TX(3, 2, 2)  #1000 OFDM simbola, QPSK, upsampling x2

    #Plotovanje rezultata
    plt.figure(figsize=(14,5))
    plt.subplot(2,1,1)
    plt.plot(np.real(Sample_Output), 'b', label='Real')
    plt.subplot(2,1,2)
    plt.plot(np.imag(Sample_Output), 'r', label='Imag')
    plt.title(f"Generisani OFDM paket sa 3 simbola i QPSK modulacijom")
    plt.xlabel("Uzorci")
    plt.ylabel("Amplituda")
    plt.grid(True)
    plt.show()

    #Oznaka početka payload-a (nakon Short i Long Training Sequence)
    training_len = len(Sample_Output) - len(Symbol_Stream) * 80 * 2
    plt.axvline(x=training_len, color='green', linestyle='--', label='Početak payload-a')

    #Ispis informacija
    print(f"Broj OFDM simbola: 3")
    print(f"Broj kompleksnih simbola: {len(Symbol_Stream)}")
    print(f"Dužina finalnog uzorka: {len(Sample_Output)}")



from OFDM_TX_802_11 import OFDM_TX
import numpy as np
import matplotlib.pyplot as plt

def tx_input_output_demo():
    """
    Prikaz predajnika sa oznakama za STS, LTS, payload i GI unutar payload-a.
    """
    num_symbols = 1
    up_factor = 2
    Sample_Output, Symbol_Stream = OFDM_TX(num_symbols, 2, up_factor)  # QPSK

    # Dužine komponenti (pretpostavljamo standardne dužine za 20MHz 802.11a)
    STS_len = 16 * up_factor * 10  # 10 short training simboli po 16 uzoraka
    LTS_len = 64 * up_factor * 2   # 2 long training simbola po 64 uzorka
    OFDM_symbol_len = 64 * up_factor
    GI_len = 16 * up_factor

    plt.figure(figsize=(16,6))
    plt.plot(np.real(Sample_Output), 'b', label='Real dio')
    #plt.plot(np.imag(Sample_Output), 'r', label='Imag dio')

    # ----- Oznake granica -----
    sts_end = STS_len
    lts_end = STS_len + LTS_len
    payload_end = len(Sample_Output)

    plt.axvline(x=0, color='black', linestyle='--')
    plt.text(0, 1.2*np.max(np.abs(Sample_Output)), 'Start', rotation=90)

    plt.axvline(x=sts_end, color='green', linestyle='--', linewidth=1.5)
    plt.text(sts_end, 1.2*np.max(np.abs(Sample_Output)), 'Kraj STS', rotation=90)

    plt.axvline(x=lts_end, color='orange', linestyle='--', linewidth=1.5)
    plt.text(lts_end, 1.2*np.max(np.abs(Sample_Output)), 'Kraj LTS', rotation=90)

    plt.axvline(x=payload_end, color='purple', linestyle='--', linewidth=1.5)
    plt.text(payload_end, 1.2*np.max(np.abs(Sample_Output)), 'Kraj payload', rotation=90)

    # ----- Granice GI unutar payload-a -----
    # Payload počinje od lts_end
    for i in range(num_symbols):
        gi_start = lts_end + i*(GI_len + OFDM_symbol_len)
        gi_end = gi_start + GI_len
        plt.axvspan(gi_start, gi_end, color='yellow', alpha=0.3)
        plt.text(gi_start + GI_len/2, 1.1*np.max(np.abs(Sample_Output)), f'GI {i+1}', 
                 rotation=0, ha='center')

    plt.title(f"OFDM paket sa {num_symbols} simbol(a) i QPSK modulacijom")
    plt.xlabel("Uzorci")
    plt.ylabel("Amplituda")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    tx_input_output_demo()

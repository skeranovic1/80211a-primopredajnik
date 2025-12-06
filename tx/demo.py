from OFDM_TX_802_11 import OFDM_TX
import numpy as np
import matplotlib.pyplot as plt

def tx_input_output_demo():
    num_symbols=5
    up_factor=2
    fs_base=20e6             # 20 MHz (802.11a)
    fs=fs_base*up_factor   # upsamplirani rate

    Sample_Output, _ =OFDM_TX(num_symbols, 2, up_factor, seed=17)

    t=np.arange(len(Sample_Output))/fs   #vremenska osa u sekundama za plot

    #Dužine komponenti za plot oznake
    STS_len=16*up_factor*10      #10 kratkih simbola = 160 uzoraka
    LTS_len=160*up_factor        #160 * up_factor
    OFDM_symbol_len=64*up_factor #dužina OFDM simbola = 64
    GI_len=16*up_factor          #dužina guard intervala = 16

    plt.figure(figsize=(16,6))
    plt.plot(t*1e6, np.real(Sample_Output), label='Realni dio')  #u µs za laksi prikaz

    #Oznake za krajeve sekvenci
    sts_end=STS_len/fs
    lts_end=(STS_len+LTS_len)/fs
    payload_end=len(Sample_Output)/fs
    plt.axvline(sts_end*1e6, color='green', linestyle='--', label='Kraj STS')
    plt.axvline(lts_end*1e6, color='orange', linestyle='--', label='Kraj LTS')
    plt.axvline(payload_end*1e6, color='purple', linestyle='--', label='Kraj payload')

    #GI označavanje
    for i in range(num_symbols):
        gi_start=(STS_len+LTS_len+i*(GI_len+OFDM_symbol_len))/fs
        gi_end=(STS_len+LTS_len+i*(GI_len+OFDM_symbol_len)+GI_len)/fs
        plt.axvspan(gi_start*1e6, gi_end*1e6, color='green', alpha=0.3)

    plt.title(f"OFDM signal u vremenskoj domeni ({num_symbols} simbol, QPSK)")
    plt.xlabel("Vrijeme [µs]")
    plt.ylabel("Amplituda")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    tx_input_output_demo()

from tx.OFDM_TX_802_11 import Transmitter80211a
import numpy as np
import matplotlib.pyplot as plt

def demo():
    """
    Demonstrira generisanje i prikaz OFDM signala koristeći Transmitter80211a.

    Funkcija kreira instancu TX modula za 802.11a standard sa zadanim
    parametrima (QPSK modulacija, broj OFDM simbola, faktor upsampliranja),
    generiše OFDM okvir i prikazuje realni dio signala u vremenskom domenu
    sa označenim sekvencama STS, LTS, payload i guard intervalima (GI).

    Napomene
    - Vrijeme na x-osi je u mikrosekundama (µs) radi preglednosti.
    """
    # Kreiranje TX instance sa parametrima
    num_ofdm_symbols=3
    up_factor=2
    tx = Transmitter80211a(
        num_ofdm_symbols=num_ofdm_symbols,
        bits_per_symbol=2,  #QPSK
        step=1,
        up_factor=up_factor,
        seed=42,
        plot=False
    )

    signal, _ = tx.generate_frame()
    fs_base=20e6 
    fs=fs_base*2   # upsamplirani rate
    t=np.arange(len(signal))/fs   #vremenska osa u sekundama za plot

    #Dužine komponenti za plot oznake
    STS_len=16*up_factor*10      #10 kratkih simbola = 160 uzoraka
    LTS_len=160*up_factor        #160 * up_factor
    OFDM_symbol_len=64*up_factor #dužina OFDM simbola = 64
    GI_len=16*up_factor          #dužina guard intervala = 16

    plt.figure(figsize=(16,6))
    plt.plot(t*1e6, np.real(signal), label='Realni dio')  #u µs za laksi prikaz

    #Oznake za krajeve sekvenci
    sts_end=STS_len/fs
    lts_end=(STS_len+LTS_len)/fs
    payload_end=len(signal)/fs
    plt.axvline(sts_end*1e6, color='green', linestyle='--', label='Kraj STS')
    plt.axvline(lts_end*1e6, color='orange', linestyle='--', label='Kraj LTS')
    plt.axvline(payload_end*1e6, color='purple', linestyle='--', label='Kraj payload')

    #GI označavanje
    for i in range(num_ofdm_symbols):
        gi_start=(STS_len+LTS_len+i*(GI_len+OFDM_symbol_len))/fs
        gi_end=(STS_len+LTS_len+i*(GI_len+OFDM_symbol_len)+GI_len)/fs
        plt.axvspan(gi_start*1e6, gi_end*1e6, color='green', alpha=0.3)

    plt.title(f"OFDM signal u vremenskoj domeni ({num_ofdm_symbols} simbola, QPSK)")
    plt.xlabel("Vrijeme [µs]")
    plt.ylabel("Amplituda")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
  
if __name__ == "__main__":
    demo()

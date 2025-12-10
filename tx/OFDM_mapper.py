import numpy as np
import matplotlib.pyplot as plt

def Mapper_OFDM(InputBits, BitsPerSymbol, plot=False):
    """
    Mapira ulazne bitove u kompleksne QAM simbole za OFDM modulaciju.

    Funkcija podržava sljedeće modulacije:
    - BPSK  (1 bit po simbolu)
    - QPSK  (2 bita po simbolu)
    - 16-QAM (4 bita po simbolu)
    - 64-QAM (6 bita po simbolu)

    Parametri
    InputBits : array-like
        Niz ulaznih bitova (0 ili 1) koji se mapiraju u simbole.
    BitsPerSymbol : int
        Broj bitova po jednom OFDM simbolu (1, 2, 4 ili 6).
    plot : bool, opcionalno
        Ako je True, funkcija prikazuje konstelacioni dijagram rezultujućih simbola.

    Povratna vrijednost
    OutputSymbols : numpy.ndarray
        Niz kompleksnih simbola generisanih iz ulaznih bitova.

    Izuzeci
    ValueError
        - Ako InputBits sadrži vrijednosti različite od 0 ili 1.
        - Ako BitsPerSymbol nije 1, 2, 4 ili 6.
    IndexError
        - Ako su ulazni bitovi tipa koji nije integer.

    Napomene
    - Funkcija koristi lookup tabele (LUT) za mapiranje bitova u simbole.
    - Plotanje je opciono i služi za vizualizaciju konstelacije.
    """

    InputBits = np.asarray(InputBits)

    if np.any(InputBits < 0) or np.any(InputBits > 1):
        raise ValueError("Ulazni biti moraju biti 0 ili 1")
    
    #Prazan ulaz
    if InputBits.size == 0:
        return np.zeros(0, dtype=complex)

    #Float → uvijek error (test_invalid_type_input)
    if not np.issubdtype(InputBits.dtype, np.integer):
        raise IndexError("Biti  moraju biti integeri.")
 
    #LUT tabele
    BPSK_LUT  = np.array([-1, 1])
    QPSK_LUT  = np.array([-1, 1]) / np.sqrt(2)
    QAM16_LUT = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    QAM64_LUT = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) / np.sqrt(42)

    NumberOfSymbols = np.floor(len(InputBits) // BitsPerSymbol)
    OutputSymbols = np.zeros(NumberOfSymbols, dtype=complex)

    for i in range(NumberOfSymbols):
        bg = InputBits[i * BitsPerSymbol : (i + 1) * BitsPerSymbol]

        if BitsPerSymbol == 1:#BPSK
            Symbol = BPSK_LUT[bg[0]]

        elif BitsPerSymbol == 2:#QPSK
            Symbol = QPSK_LUT[bg[0]] + 1j * QPSK_LUT[bg[1]]

        elif BitsPerSymbol == 4:#16-QAM
            I = bg[0] * 2 + bg[1]
            Q = bg[2] * 2 + bg[3]
            Symbol = QAM16_LUT[I] + 1j * QAM16_LUT[Q]

        elif BitsPerSymbol == 6:#64-QAM
            I = bg[0] * 4 + bg[1] * 2 + bg[2]
            Q = bg[3] * 4 + bg[4] * 2 + bg[5]
            Symbol = QAM64_LUT[I] + 1j * QAM64_LUT[Q]

        else:
            raise ValueError("Broj bita po simbola mora biti 1,2,4 ili 6") #dozvoljene modulacije
        
        OutputSymbols[i] = Symbol

    # Plot grana
    if plot and NumberOfSymbols > 0:
        plt.figure()
        plt.scatter(OutputSymbols.real, OutputSymbols.imag)
        plt.axhline(0, color="black", linewidth=0.5)
        plt.axvline(0, color="black", linewidth=0.5)
        plt.grid(True)
        plt.title(f"Konstelacioni dijagram za {BitsPerSymbol}-bitnu modulaciju")
        plt.gca().set_aspect("equal", "box")

    return OutputSymbols

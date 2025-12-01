import numpy as np
import matplotlib.pyplot as plt


def Mapper_OFDM(InputBits, BitsPerSymbol, plot=False):
    """
    Mapira bitove u kompleksne QAM simbole za OFDM.
    """

    InputBits = np.asarray(InputBits)

    if np.any(InputBits < 0) or np.any(InputBits > 1):
        raise ValueError("InputBits must be 0 or 1")
    
    # Prazan ulaz
    if InputBits.size == 0:
        return np.zeros(0, dtype=complex)

    # Float → uvijek error (test_invalid_type_input)
    if not np.issubdtype(InputBits.dtype, np.integer):
        raise IndexError("Bits must be integer type.")

    # Odobrene modulacije
    if BitsPerSymbol not in (1, 2, 4, 6):
        raise ValueError("BitsPerSymbol must be 1,2,4,6.")
    
    # LUT tabele
    BPSK_LUT  = np.array([-1, 1])
    QPSK_LUT  = np.array([-1, 1]) / np.sqrt(2)
    QAM16_LUT = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    QAM64_LUT = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) / np.sqrt(42)

    NumberOfSymbols = len(InputBits) // BitsPerSymbol
    OutputSymbols = np.zeros(NumberOfSymbols, dtype=complex)

    for i in range(NumberOfSymbols):
        bg = InputBits[i * BitsPerSymbol : (i + 1) * BitsPerSymbol]

        if BitsPerSymbol == 1:
            Symbol = BPSK_LUT[bg[0]]

        elif BitsPerSymbol == 2:
            Symbol = QPSK_LUT[bg[0]] + 1j * QPSK_LUT[bg[1]]

        elif BitsPerSymbol == 4:
            I = bg[0] * 2 + bg[1]
            Q = bg[2] * 2 + bg[3]
            Symbol = QAM16_LUT[I] + 1j * QAM16_LUT[Q]

        elif BitsPerSymbol == 6:
            I = bg[0] * 4 + bg[1] * 2 + bg[2]
            Q = bg[3] * 4 + bg[4] * 2 + bg[5]
            Symbol = QAM64_LUT[I] + 1j * QAM64_LUT[Q]

        OutputSymbols[i] = Symbol

    # Plot grana
    if plot and NumberOfSymbols > 0:
        plt.figure()
        plt.scatter(OutputSymbols.real, OutputSymbols.imag)
        plt.axhline(0, color="black", linewidth=0.5)
        plt.axvline(0, color="black", linewidth=0.5)
        plt.grid(True)
        plt.title(f"Constellation for {BitsPerSymbol}-bit modulation")
        plt.gca().set_aspect("equal", "box")

    return OutputSymbols

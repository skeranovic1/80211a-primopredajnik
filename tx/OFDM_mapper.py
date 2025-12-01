import numpy as np

def Mapper_OFDM(InputBits, BitsPerSymbol):
    # BitsPerSymbol: 1, 2, 4, 6 --> BPSK, QPSK, 16QAM, 64QAM

    BPSK_LUT  = np.array([-1,  1])
    QPSK_LUT  = np.array([-1,  1]) / np.sqrt(2)
    QAM16_LUT = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    QAM64_LUT = np.array([-7, -5, -3, -1,  1,  3,  5,  7]) / np.sqrt(42)

    NumberOfSymbols = len(InputBits) // BitsPerSymbol
    OutputSymbols = np.zeros(NumberOfSymbols, dtype=complex)

    for i in range(NumberOfSymbols):
        Start = i * BitsPerSymbol
        Stop = Start + BitsPerSymbol
        BitGroup = InputBits[Start:Stop]

        if BitsPerSymbol == 1:
            Symbol = BPSK_LUT[BitGroup[0]]
        elif BitsPerSymbol == 2:
            Symbol = QPSK_LUT[BitGroup[0]] + 1j * QPSK_LUT[BitGroup[1]]
        elif BitsPerSymbol == 4:
            Symbol = QAM16_LUT[BitGroup[0]*2 + BitGroup[1]] + 1j * QAM16_LUT[BitGroup[2]*2 + BitGroup[3]]
        elif BitsPerSymbol == 6:
            Symbol = QAM64_LUT[BitGroup[0]*4 + BitGroup[1]*2 + BitGroup[2]] + 1j * QAM64_LUT[BitGroup[3]*4 + BitGroup[4]*2 + BitGroup[5]]

        OutputSymbols[i] = Symbol

    return OutputSymbols
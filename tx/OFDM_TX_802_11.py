import numpy as np
#from scipy.signal import lfilter, hann

def OFDM_TX_802_11(NumberOf_OFDM_Symbols, BitsPerSymbol):
    # BitsPerSymbol = 1,2,4,6 = BPSK, QPSK, 16QAM, 64QAM

    # 1. Compute Short and Long Training Sequences at 20MHz (Step = 1/2)
    Step = 1
    ShortTrainingSequence = get_short_training_sequence(Step)
    LongTrainingSequence = get_long_training_sequence(Step)

    # 2. Generating Random bits for Mapping Operation
    sd=0
    NumberOfBits = (48 * BitsPerSymbol) * NumberOf_OFDM_Symbols
    Source_Bits = bit_sequence(NumberOf_OFDM_Symbols, BitsPerSymbol,sd)
    Symbol_Stream = Mapper_OFDM(Source_Bits, BitsPerSymbol)

    # 3. Generating the Payload
    Payload = IFFT_GI(Symbol_Stream)

    # 4. Zero-Stuffing Operation
    Packet_20MHz = np.concatenate((ShortTrainingSequence, LongTrainingSequence, Payload))
    up_factor = 2  
    Packet_Zero_Stuffed = zero_stuffing(Packet_20MHz, up_factor)
    Packet_Zero_Stuffed[::2] = Packet_20MHz

    # 5. Computing Halfband filter coefficients
    N = 31  # Number of Taps

    # 6. The Halfband filtering operation
    Sample_Output,h = half_band_upsample(signal, up_factor=1, N=31, plot=False)

    return Sample_Output, Symbol_Stream
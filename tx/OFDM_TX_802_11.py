import numpy as np
from .short_sequence import get_short_training_sequence
from .long_sequence import get_long_training_sequence
from .OFDM_mapper import Mapper_OFDM
from .utilities import bit_sequence
from .ifft_ofdm_symbol import IFFT_GI
from .filters import half_band_upsample

def OFDM_TX_802_11(NumberOf_OFDM_Symbols, BitsPerSymbol, up_factor=2):
    """
    Generiše OFDM signal za 802.11 standard.
    Signal se automatski upsampluje korištenjem half-band filtera.
    
    Parametri:
    - NumberOf_OFDM_Symbols : broj OFDM simbola u paketu
    - BitsPerSymbol         : modulacija (1=BPSK, 2=QPSK, 4=16-QAM, 6=64-QAM)
    - up_factor             : faktor upsamplovanja (default=2)
    
    Vraća:
    - Sample_Output         : upsample-ovan i filtriran OFDM signal
    - Symbol_Stream         : kompleksni simboli generisani Mapper_OFDM
    """

    #1. Generisanje Short i Long Training Sequence
    Step = 1
    ShortTrainingSequence = get_short_training_sequence(Step)
    LongTrainingSequence = get_long_training_sequence(Step)

    #2. Generisanje random bita
    sd = 0 #seed za generator slučajnih brojeva
    NumberOfBits = (48 * BitsPerSymbol) * NumberOf_OFDM_Symbols # 48 nosioca po OFDM simbolu
    Source_Bits = bit_sequence(NumberOf_OFDM_Symbols, BitsPerSymbol, sd) # Izvorni biti

    #3. Mapper u kompleksne QAM simbole
    Symbol_Stream = Mapper_OFDM(Source_Bits, BitsPerSymbol)

    #4. Generisanje payload-a (IFFT + GI)
    Payload = IFFT_GI(Symbol_Stream)

    #5. Kreiranje kompletnog paketa (training + payload)
    Packet_20MHz = np.concatenate((ShortTrainingSequence, LongTrainingSequence, Payload))

    #6. Upsampling i filtriranje pomoću half-band filtera
    Sample_Output, h = half_band_upsample(Packet_20MHz, up_factor=up_factor, N=31, plot=False)

    return Sample_Output, Symbol_Stream

import numpy as np
from .short_sequence import get_short_training_sequence
from .long_sequence import get_long_training_sequence
from .OFDM_mapper import Mapper_OFDM
from .utilities import bit_sequence
from .ifft_ofdm_symbol import IFFT_GI
from .filters import half_band_upsample

class Transmitter80211a:
    """
    OFDM Transmitter za IEEE 802.11a standard.
    
    Atributi:
    - num_ofdm_symbols : broj OFDM simbola u paketu
    - bits_per_symbol  : modulacija (1=BPSK, 2=QPSK, 4=16-QAM, 6=64-QAM)
    - up_factor        : faktor upsamplovanja
    - seed             : sjeme za generator nasumičnih bita
    - step             : korak za training sekvence
    - plot             : ako je True, prikazuju se svi plotovi
    """
    def __init__(self, num_ofdm_symbols=1, bits_per_symbol=2, up_factor=2, seed=13, step=1, plot=False):
        self.num_ofdm_symbols = num_ofdm_symbols
        self.bits_per_symbol = bits_per_symbol
        self.up_factor = up_factor
        self.seed = seed
        self.step = step
        self.plot = plot  

    def generate_training_sequences(self):
        """Generiše Short i Long Training Sequence"""
        sts=get_short_training_sequence(self.step)
        lts=get_long_training_sequence(self.step)
        return sts, lts
    
    def generate_payload(self):
        """Generiše OFDM simbol payload"""
        bits=bit_sequence(self.num_ofdm_symbols, self.bits_per_symbol, self.seed)
        symbols=Mapper_OFDM(bits, self.bits_per_symbol, plot=self.plot)
        payload=IFFT_GI(symbols, plot=self.plot)
        return payload, symbols
    
    def generate_frame(self):
        """Generiše kompletan OFDM paket sa training sekvencama i upsamplingom"""
        sts, lts=self.generate_training_sequences()
        payload, symbols=self.generate_payload()
        packet_20MHz=np.concatenate((sts, lts, payload))
        sample_output, _ =half_band_upsample(packet_20MHz, up_factor=self.up_factor, N=31, plot=self.plot)
        return sample_output, symbols

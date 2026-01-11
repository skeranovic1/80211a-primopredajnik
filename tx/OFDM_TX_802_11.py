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
    
    @property
    def num_ofdm_symbols(self):
        return self._num_ofdm_symbols

    @num_ofdm_symbols.setter
    def num_ofdm_symbols(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("num_ofdm_symbols mora biti pozitivan cijeli broj")
        self._num_ofdm_symbols = value

    @property
    def bits_per_symbol(self):
        return self._bits_per_symbol

    @bits_per_symbol.setter
    def bits_per_symbol(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("bits_per_symbol mora biti pozitivan cijeli broj")
        self._bits_per_symbol = value

    @property
    def up_factor(self):
        return self._up_factor

    @up_factor.setter
    def up_factor(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("up_factor mora biti pozitivan cijeli broj")
        self._up_factor = value

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError("seed mora biti cijeli broj >= 0")
        self._seed = value

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("step mora biti pozitivan cijeli broj")
        self._step = value

    @property
    def plot(self):
        return self._plot

    @plot.setter
    def plot(self, value):
        if not isinstance(value, bool):
            raise TypeError("plot mora biti bool")
        self._plot = value

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
        return payload, bits, symbols
    
    def generate_frame(self):
        """Generiše kompletan OFDM paket sa training sekvencama i upsamplingom"""
        sts, lts=self.generate_training_sequences()
        sts=sts/64
        lts=lts/64
        payload, bits, symbols =self.generate_payload()
        packet_20MHz=np.concatenate((sts, lts, payload))
        sample_output, _ =half_band_upsample(packet_20MHz, up_factor=self.up_factor, N=31, plot=self.plot)
        return sample_output, bits, symbols
    


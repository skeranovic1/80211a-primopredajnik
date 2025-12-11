class ChannelSettings:
    """
    Settings definira parametre simulacije kanala.
    Attributes:
        SampleRate (float): frekvencija uzorkovanja u Hz
        NumberOfTaps (int): broj tapova za multipath model
        DelaySpread (float): maksimalno kašnjenje multipath u sekundama
        SNR_dB (float): Signal-to-Noise Ratio u dB
    """

    def __init__(self, sample_rate=40e6, number_of_taps=40, delay_spread=150e-9, snr_db=35):
        self.SampleRate = sample_rate
        self.NumberOfTaps = number_of_taps
        self.DelaySpread = delay_spread
        self.SNR_dB = snr_db

    @property
    def SampleRate(self):
        return self._sample_rate

    @SampleRate.setter
    def SampleRate(self, value):
        if value > 0:
            self._sample_rate = value
        else:
            raise ValueError("SampleRate mora biti veći od 0")

    @property
    def NumberOfTaps(self):
        return self._number_of_taps

    @NumberOfTaps.setter
    def NumberOfTaps(self, value):
        if isinstance(value, int) and value > 0:
            self._number_of_taps = value
        else:
            raise ValueError("NumberOfTaps mora biti pozitivan cijeli broj")

    @property
    def DelaySpread(self):
        return self._delay_spread

    @DelaySpread.setter
    def DelaySpread(self, value):
        if value >= 0:
            self._delay_spread = value
        else:
            raise ValueError("DelaySpread mora biti ≥ 0")

    @property
    def SNR_dB(self):
        return self._snr_db

    @SNR_dB.setter
    def SNR_dB(self, value):
        self._snr_db = value

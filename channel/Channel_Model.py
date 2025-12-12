import numpy as np
from .Multipath import GetMultipathFilter
from .AWGN import Generate_AWGN
import scipy.signal as sc


class Channel_Model:
    """
    OFDM Transmitter za IEEE 802.11a standard.

    Atributi:
    - NumberOfTaps
    - DelaySpread
    - SampleRate
    - SNR_dB
    """

    def __init__(self, settings, mode):
        """
        settings: objekt sa poljima:
            - NumberOfTaps
            - DelaySpread
            - SampleRate
            - SNR_dB

        mode: objekt klase ChannelMode s poljima:
            - Multipath (0/1)
            - ThermalNoise (0/1)
        """
        self.settings = settings
        self.mode = mode

    def apply(self, tx_samples):
        """
        Primjenjuje model kanala na ulazne uzorke.
        """

        # 1. Mode parametri
        multipath_select = self.mode.Multipath
        awgn_select = self.mode.ThermalNoise

        # 2. Settings parametri
        N = self.settings.NumberOfTaps
        delay_spread = self.settings.DelaySpread
        sample_rate = self.settings.SampleRate
        snr_db = self.settings.SNR_dB

        # 3. Multipath FIR filter
        fir_taps = GetMultipathFilter(sample_rate, delay_spread, N)

        # Uključivanje multipath fadinga
        if multipath_select == 1:
            tx_samples = sc.lfilter(fir_taps, 1, tx_samples)

        # Normalizacija energije
        var_out = np.var(tx_samples)
        if var_out > 0:
            tx_samples = tx_samples / np.sqrt(var_out)

        # 4. AWGN
        if awgn_select == 1:
            tx_samples = tx_samples + Generate_AWGN(tx_samples, snr_db)

        return tx_samples, fir_taps

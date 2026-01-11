import numpy as np
from rx.pretprocessing import iq_preprocessing
from rx.detection import packet_detector
from rx.cfo import detect_frequency_offsets, gruba_vremenska_sinhronizacija
from rx.estimacija_kanala import channel_estimate_and_equalizer
from rx.PhaseCorrection_80211a import phase_correction
from rx.rastavljanje import remove_cp

class Receiver80211a:
    def __init__(self, fs, num_symbols, nfft=64, ncp=16):
        self.fs = fs
        self.num_symbols = num_symbols
        self.nfft = nfft
        self.ncp = ncp
        self.nsym = nfft + ncp
        
        # Interni statusi za provjere i cuvanje
        self.rx_signal_clean = None
        self.lts_start = None
        self.rx_fine = None
        self.symbols_fd = None
        self.channel_est = None
        self.eq_coefficient = None
        self.corrected_symbols = None
    
    @property
    def rx_signal_clean(self):
        return self._rx_signal_clean

    @rx_signal_clean.setter
    def rx_signal_clean(self, value):
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError("rx_signal_clean mora biti numpy niz ili None")
        self._rx_signal_clean = value

    @property
    def lts_start(self):
        return self._lts_start

    @lts_start.setter
    def lts_start(self, value):
        if value is not None:
            value = int(value)
            if value < 0:
                raise ValueError("lts_start mora biti >= 0")
        self._lts_start = value

    @property
    def rx_fine(self):
        return self._rx_fine

    @rx_fine.setter
    def rx_fine(self, value):
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError("rx_fine mora biti numpy niz ili None")
        self._rx_fine = value

    @property
    def symbols_fd(self):
        return self._symbols_fd

    @symbols_fd.setter
    def symbols_fd(self, value):
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError("symbols_fd mora biti numpy niz ili None")
        self._symbols_fd = value

    @property
    def channel_est(self):
        return self._channel_est

    @channel_est.setter
    def channel_est(self, value):
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError("channel_est mora biti numpy niz ili None")
        self._channel_est = value

    @property
    def eq_coefficient(self):
        return self._eq_coefficient

    @eq_coefficient.setter
    def eq_coefficient(self, value):
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError("eq_coefficient mora biti numpy niz ili None")
        self._eq_coefficient = value

    @property
    def corrected_symbols(self):
        return self._corrected_symbols

    @corrected_symbols.setter
    def corrected_symbols(self, value):
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError("corrected_symbols mora biti numpy niz ili None")
        self._corrected_symbols = value
        
    def preprocess(self, rx_signal_raw, tx_signal_ref):
        """IQ balansiranje i inicijalna obrada."""
        self.rx_signal_clean, self.fs = iq_preprocessing(
            rx_signal=rx_signal_raw, 
            tx_signal=tx_signal_ref, 
            fs=self.fs
        )
        return self.rx_signal_clean
    
    def synchronize(self):
        """Detekcija paketa, precizni tajming i CFO korekcija."""
        if self.rx_signal_clean is None:
            raise RuntimeError("Greška: Morate prvo pozvati preprocess()!")
        
        #Gruba detekcija (STS)
        _, _, start_sts_end, _ = packet_detector(self.rx_signal_clean)
        
        #Vremenska sinhronizacija 
        rx_lts_section = self.rx_signal_clean[start_sts_end : start_sts_end + 160]
        pravi_pocetak_relativni, _, _ = gruba_vremenska_sinhronizacija(rx_lts_section, search_win=32)
        self.lts_start = start_sts_end + pravi_pocetak_relativni
        
        #Frekvencijska sinhronizacija (Coarse + Fine)
        n = np.arange(len(self.rx_signal_clean))
        
        #Coarse
        f_off = detect_frequency_offsets(self.rx_signal_clean, self.lts_start, self.fs)
        rx_c = self.rx_signal_clean * np.exp(-1j * 2 * np.pi * n * f_off[0] / self.fs)
        
        #Fine
        f_off_f = detect_frequency_offsets(rx_c, self.lts_start, self.fs)
        self.rx_fine = rx_c * np.exp(-1j * 2 * np.pi * n * f_off_f[1] / self.fs)
        
        return self.rx_fine

    def extract_and_equalize(self):
        """Skidanje CP-a i estimacija kanala."""
        if self.rx_fine is None or self.lts_start is None:
            raise RuntimeError("Greška: Pokušaj ekvalizacije prije sinhronizacije!")
    
        data_start = self.lts_start + 128  # Početak podataka je nakon 2x64 LTS simbola
        
        #Skidanje CP-a
        self.symbols_fd = remove_cp(self.rx_fine, data_start, self.num_symbols, self.nsym, self.nfft, self.ncp)
        
        #Estimacija kanala pomoću LTS-a
        samo_lts = self.rx_fine[self.lts_start : data_start]
        self.channel_est, self.eq_coefficient = channel_estimate_and_equalizer(samo_lts)
        
        #Ekvalizacija i fazna korekcija (CPE)
        ekvalizirani = self.symbols_fd * self.eq_coefficient
        self.corrected_symbols = phase_correction(ekvalizirani, self.num_symbols, self.channel_est)

        return self.corrected_symbols

    def process_signal(self, rx, tx):
        """Glavni lanac obrade."""
        self.preprocess(rx, tx)
        self.synchronize()
        self.extract_and_equalize()

        return self.corrected_symbols
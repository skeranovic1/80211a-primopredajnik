import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from rx.pretprocessing import iq_preprocessing
from rx.detection import packet_detector
from rx.cfo import detect_frequency_offsets
from rx.long_symbol_correlator import long_symbol_correlator
from rx.estimacija_kanala import channel_estimate_and_equalizer
from rx.PhaseCorrection_80211a import phase_correction_80211a

from tx.long_sequence import get_long_training_sequence
from rx.prijemnik import run_rx
from tx.OFDM_TX_802_11 import Transmitter80211a

from channel.Channel_Model import Channel_Model
from channel.channel_settings import ChannelSettings
from channel.channel_mode import ChannelMode

# ---------------- helperi ----------------
def apply_cfo(x, cfo_hz, fs):
    n = np.arange(len(x))
    return x * np.exp(1j * 2 * np.pi * cfo_hz * n / fs)

def apply_cfo_correction(x, cfo_hz, fs):
    n = np.arange(len(x))
    return x * np.exp(-1j * 2 * np.pi * cfo_hz * n / fs)

def add_awgn(x, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    sig_pow = np.mean(np.abs(x) ** 2)
    snr_lin = 10 ** (snr_db / 10)
    noise_pow = sig_pow / snr_lin
    w = (rng.normal(0, np.sqrt(noise_pow / 2), size=x.shape) +
         1j * rng.normal(0, np.sqrt(noise_pow / 2), size=x.shape))
    return x + w

def extract_equalized_data_symbols(rx_td, lts_start, eq_coeffs, num_symbols):
    """
    Iz rx signala u TD izvuče data simbole (48) nakon FFT+EQ,
    bez phase correction (CPE/slope) -> idealno da CFO vidiš kao rotaciju.
    """
    CP = 16
    SYM = 80

    data_indices = np.array([
        6,7,8,9,10,
        12,13,14,15,16,17,18,19,20,21,22,23,24,
        26,27,28,29,30,31,32,33,34,35,36,37,39,
        40,41,42,43,44,45,46,47,48,49,50,51,
        53,54,55,56,57
    ], dtype=int)

    payload_start = lts_start + 2 * 64

    out = []
    for i in range(num_symbols):
        start = payload_start + i * SYM + CP
        stop = start + 64
        if stop > len(rx_td):
            break

        sym_td = rx_td[start:stop]
        sym_fd = (1/64) * np.fft.fft(sym_td)
        sym_eq = sym_fd * eq_coeffs
        out.append(sym_eq[data_indices])

    if len(out) == 0:
        return np.zeros(0, dtype=complex)

    return np.concatenate(out)

# ============================================================
# 1. Overall Simulation Setup
# ============================================================

 # ------------------------------------------------------------
    # A) TX
    # ------------------------------------------------------------
num_payload_symbols = 30
bits_per_symbol = 2  # QPSK

tx_obj = Transmitter80211a(
        num_ofdm_symbols=num_payload_symbols,
        bits_per_symbol=bits_per_symbol,
        up_factor=2,
        seed=13,
        step=1,
        plot=False
    )

tx_40mhz, tx_symbols = tx_obj.generate_frame()
fs40 = 40e6

    # ------------------------------------------------------------
    # B) Kanal + CFO + šum
    # ------------------------------------------------------------
settings = ChannelSettings(
        sample_rate=fs40,
        number_of_taps=2,
        delay_spread=10e-9,
        snr_db=10
    )

mode_awgn = ChannelMode(multipath=0, thermal_noise=1)
channel_awgn = Channel_Model(settings, mode_awgn)
rx_40mhz,FIR_Taps = channel_awgn.apply(tx_40mhz)
rx_40mhz = np.asarray(rx_40mhz).flatten()

true_cfo_hz = 2500.0
rx_40mhz = apply_cfo(rx_40mhz, true_cfo_hz, fs40)

    # ------------------------------------------------------------
    # C) RUN RX chain
    # ------------------------------------------------------------
res = run_rx(
        rx_40mhz=rx_40mhz,
        tx_40mhz=tx_40mhz,
        num_symbols_req=num_payload_symbols,
        fs_in=fs40,
        plot=False
    )

fs = float(res["fs"])  # 20 MHz nakon decimacije
lts_start = int(res["lts_start"])
eq_coeffs = res["equalizer_coeffs"]

cfo_coarse = float(res.get("cfo_coarse_hz", 0.0))
cfo_fine = float(res.get("cfo_fine_hz", 0.0))
    # BITNO: ako ti fine u funkciji nije residual nego opet TOTAL,
    # onda ovdje uzmi samo residual nakon coarse:
cfo_fine_res = float(res.get("cfo_fine_res_hz", cfo_fine - cfo_coarse))

cfo_total_est = cfo_coarse + cfo_fine_res



    # ------------------------------------------------------------
    # D) CFO KONSTELACIJE (FFT+EQ, bez phase correction)
    # ------------------------------------------------------------
rx_20, _ = iq_preprocessing(rx_40mhz, tx_40mhz, fs=fs40)

    # 1) bez CFO korekcije
rx_td_0 = rx_20

    # 2) nakon coarse CFO korekcije
rx_td_1 = apply_cfo_correction(rx_20, cfo_coarse, fs)

    # 3) nakon coarse + fine residual CFO korekcije
rx_td_2 = apply_cfo_correction(rx_td_1, cfo_fine_res, fs)

K = 10  # broj OFDM simbola za plot CFO-konstelacije
s0 = extract_equalized_data_symbols(rx_td_0, lts_start, eq_coeffs, K)
s1 = extract_equalized_data_symbols(rx_td_1, lts_start, eq_coeffs, K)
s2 = extract_equalized_data_symbols(rx_td_2, lts_start, eq_coeffs, K)

    # ------------------------------------------------------------
    # E) Standardno: RX nakon eq + phase correction (tvoje corrected_symbols)
    # ------------------------------------------------------------
corrected = res["corrected_symbols"]
Corrected_Symbols = np.concatenate(corrected)
TX_Symbol_Stream = tx_symbols[:len(Corrected_Symbols)]

mask = np.abs(Corrected_Symbols) > 1e-6
Corrected_Symbols = Corrected_Symbols[mask]
TX_Symbol_Stream = TX_Symbol_Stream[mask]

# ============================================================
# 5. Performance Evaluation (EVM)
# ============================================================

ErrorVectors = TX_Symbol_Stream[:len(Corrected_Symbols)] - Corrected_Symbols
Average_ErrorVectorPower = np.mean(np.abs(ErrorVectors) ** 2)

EVM_dB = 10 * np.log10(Average_ErrorVectorPower)
print(f"EVM = {EVM_dB:.2f} dB")

# -------------------------------
# EVM vs Time
# -------------------------------
NumberOf_OFDMSymbols=num_payload_symbols
Error_Time = np.zeros(num_payload_symbols)

for i in range(num_payload_symbols):
    s = i * 48
    e = s + 48
    ev = TX_Symbol_Stream[s:e] - Corrected_Symbols[s:e]
    Error_Time[i] = np.mean(np.abs(ev) ** 2)

EVM_Time_dB = 10 * np.log10(Error_Time)

# -------------------------------
# EVM vs Frequency
# -------------------------------
Error_Frequency = np.zeros(17)

for i in range(num_payload_symbols):
    s = i * 17
    e = s + 17
    ev = TX_Symbol_Stream[s:e] - Corrected_Symbols[s:e]
    Error_Frequency += np.abs(ev) ** 2 / num_payload_symbols

EVM_Frequency_dB = 10 * np.log10(Error_Frequency)

# ============================================================
# Plots
# ============================================================

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(EVM_Frequency_dB, 'k.')
plt.title("EVM vs Frequency")
plt.ylabel("dB")
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(EVM_Time_dB, 'k')
plt.title("EVM vs Time")
plt.xlabel("OFDM Symbols")
plt.ylabel("dB")
plt.grid()

plt.subplot(2, 2, 3)
plt.stem(np.abs(FIR_Taps))
plt.title("FIR Taps")
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(
    np.real(Corrected_Symbols),
    np.imag(Corrected_Symbols),
    'k.',
    markersize=3
)
plt.title("Constellation")
plt.axis("equal")
plt.grid()

plt.tight_layout()
plt.show()

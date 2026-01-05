import numpy as np

from rx.pretprocessing import iq_preprocessing
from rx.detection import packet_detector
from rx.cfo import detect_frequency_offsets
from rx.long_symbol_correlator import long_symbol_correlator
from rx.estimacija_kanala import channel_estimate_and_equalizer
from rx.PhaseCorrection_80211a import phase_correction_80211a

from tx.long_sequence import get_long_training_sequence


def apply_cfo_correction(x, cfo_hz, fs):
    n = np.arange(len(x))
    return x * np.exp(-1j * 2 * np.pi * cfo_hz * n / fs)


def get_lts_64_reference():
    lts_all = get_long_training_sequence(step=1)  # CP32 + 2*64
    return lts_all[32:32+64]  # 64 uzorka bez CP


def run_rx(rx_40mhz, tx_40mhz, num_symbols_req=None, fs_in=40e6, plot=False):
    # 1) IQ preprocessing (40->20)
    rx, fs = iq_preprocessing(rx_40mhz, tx_40mhz, fs=fs_in)

    # 2) Packet detection (STS)
    _, _, falling_edge, _ = packet_detector(rx)
    if falling_edge is None:
        raise RuntimeError("Packet detector nije našao falling edge (STS).")

    # 3) Coarse CFO (STS) + korekcija
    cfo_coarse = float(detect_frequency_offsets(rx, falling_edge, plot=False, fs=fs)[0])
    rx_cfo1 = apply_cfo_correction(rx, cfo_coarse, fs)

    # 4) LTS korelacija -> lts_start
    lts_ref_64 = get_lts_64_reference()
    _, lt_peak_pos, _ = long_symbol_correlator(lts_ref_64, rx_cfo1, falling_edge)

    lts_start = int(lt_peak_pos - 63)
    if lts_start < 0:
        lts_start = 0

    # 5) Fine CFO (LTS) + korekcija
    cfo_fine = float(detect_frequency_offsets(rx_cfo1, lts_start, plot=plot, fs=fs)[1])
    cfo_fine_res = cfo_fine - cfo_coarse
    rx_cfo2 = apply_cfo_correction(rx_cfo1, cfo_fine_res, fs)
    rx_cfo2 = apply_cfo_correction(rx_cfo1, cfo_fine, fs)

    # 6) Kanal + EQ (na 2x64 LTS)
    channel_est, equalizer_coeffs = channel_estimate_and_equalizer(rx_cfo2, lts_start)

    # 7) Koliko payload simbola možemo izvući (80 = 16CP + 64)
    CP = 16
    SYM = 80
    payload_start = lts_start + 2 * 64

    max_symbols = int((len(rx_cfo2) - (payload_start + CP + 64)) // SYM)
    if max_symbols < 0:
        max_symbols = 0

    if num_symbols_req is None:
        num_symbols = max_symbols
    else:
        num_symbols = int(min(num_symbols_req, max_symbols))

    # 8) Phase correction (piloti + data indeksi su unutra)
    corrected_symbols = phase_correction_80211a(
        rx_signal=rx_cfo2,
        num_symbols=num_symbols,
        ltpeak=lts_start,
        channel_est=channel_est,
        equalizer_coeffs=equalizer_coeffs,
        L=8,
        max_ratio=1
    )

    return {
    "fs": fs,
    "falling_edge": int(falling_edge),
    "lt_peak_pos": int(lt_peak_pos),
    "lts_start": int(lts_start),
    "cfo_coarse_hz": cfo_coarse,
    "cfo_fine_hz": cfo_fine,
    "cfo_fine_res_hz": float(cfo_fine_res),
    "max_symbols_in_buffer": int(max_symbols),
    "corrected_symbols": corrected_symbols,
    "channel_est": channel_est,
    "equalizer_coeffs": equalizer_coeffs,
}


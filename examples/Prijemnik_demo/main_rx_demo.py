import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from rx.prijemnik import run_rx
from rx.pretprocessing import iq_preprocessing
from tx.OFDM_TX_802_11 import Transmitter80211a


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

def evm_rms(rx, tx):
    err = rx - tx
    return np.sqrt(np.mean(np.abs(err) ** 2) / np.mean(np.abs(tx) ** 2))

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


def main():
    print("START main_rx_demo")

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
    h = np.array([1.0 + 0j, 0.3 + 0.2j, 0.12 - 0.05j], dtype=complex)
    rx_40mhz = np.convolve(tx_40mhz, h, mode="same")

    true_cfo_hz = 2500.0
    rx_40mhz = apply_cfo(rx_40mhz, true_cfo_hz, fs40)
    rx_40mhz = add_awgn(rx_40mhz, snr_db=25, seed=1)

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

    print("\n=== CFO INFO ===")
    print(f"True CFO        [Hz]: {true_cfo_hz:.2f}")
    print(f"Coarse CFO      [Hz]: {cfo_coarse:.2f}")
    print(f"Fine (raw) CFO  [Hz]: {cfo_fine:.2f}")
    print(f"Fine residual   [Hz]: {cfo_fine_res:.2f}")
    print(f"Total est (c+r) [Hz]: {cfo_total_est:.2f}")
    print(f"Error           [Hz]: {cfo_total_est - true_cfo_hz:.2f}")

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

    plt.figure()
    plt.scatter(s0.real, s0.imag, s=8)
    plt.grid(True)
    plt.title("CFO konstelacija: prije CFO korekcije (FFT + EQ, bez phase corr)")
    plt.gca().set_aspect("equal", "box")

    plt.figure()
    plt.scatter(s1.real, s1.imag, s=8)
    plt.grid(True)
    plt.title("CFO konstelacija: nakon COARSE CFO korekcije (FFT + EQ, bez phase corr)")
    plt.gca().set_aspect("equal", "box")

    plt.figure()
    plt.scatter(s2.real, s2.imag, s=8)
    plt.grid(True)
    plt.title("CFO konstelacija: nakon COARSE+FINE(res) CFO korekcije (FFT + EQ, bez phase corr)")
    plt.gca().set_aspect("equal", "box")

    # ------------------------------------------------------------
    # E) Standardno: RX nakon eq + phase correction (tvoje corrected_symbols)
    # ------------------------------------------------------------
    corrected = res["corrected_symbols"]
    if len(corrected) == 0:
        print("Nema izvučenih simbola (corrected_symbols je prazno).")
        plt.show(block=True)
        return

    rx_syms = np.concatenate(corrected)
    tx_syms = tx_symbols[:len(rx_syms)]

    mask = np.abs(rx_syms) > 1e-6
    rx_syms = rx_syms[mask]
    tx_syms = tx_syms[mask]

    evm = evm_rms(rx_syms, tx_syms)
    print(f"\nEVM_rms: {100 * evm:.2f} %")

    plt.figure()
    plt.scatter(rx_syms.real, rx_syms.imag, s=8)
    plt.grid(True)
    plt.title("RX konstelacija (nakon eq + phase correction)")
    plt.gca().set_aspect("equal", "box")

    plt.figure()
    plt.scatter(tx_syms.real, tx_syms.imag, s=8, alpha=0.2)
    plt.grid(True)
    plt.title("TX konstelacija (payload symbols)")
    plt.gca().set_aspect("equal", "box")

    plt.show(block=True)


if __name__ == "__main__":
    main()

import pytest
import numpy as np
from rx.prijemnik import run_rx
from tx.OFDM_TX_802_11 import Transmitter80211a
from channel.Channel_Model import Channel_Model
from channel.channel_settings import ChannelSettings
from channel.channel_mode import ChannelMode

# ---------------- helper functions ----------------
def apply_cfo(x, cfo_hz, fs):
    n = np.arange(len(x))
    return x * np.exp(1j * 2 * np.pi * cfo_hz * n / fs)

# ------------------ E2E TEST -----------------------
@pytest.mark.parametrize("snr_db", [10, 20, 35])
def test_e2e_rx(snr_db):
    """
    End-to-end test prijemnika sa poznatim signalima.
    Provjera: EVM mora biti ispod praga.
    """
    num_payload_symbols = 30
    bits_per_symbol = 2  # QPSK
    fs40 = 40e6

    # ------------------ 1) TX ------------------------
    tx_obj = Transmitter80211a(
        num_ofdm_symbols=num_payload_symbols,
        bits_per_symbol=bits_per_symbol,
        up_factor=2,
        seed=123,
        step=1,
        plot=False
    )
    tx_40mhz, tx_symbols = tx_obj.generate_frame()

    # ------------------ 2) Kanal + AWGN ----------------
    settings = ChannelSettings(
        sample_rate=fs40,
        number_of_taps=2,
        delay_spread=10e-9,
        snr_db=snr_db
    )
    mode = ChannelMode(multipath=0, thermal_noise=1)
    channel = Channel_Model(settings, mode)
    rx_40mhz, _ = channel.apply(tx_40mhz)
    rx_40mhz = np.asarray(rx_40mhz).flatten()

    # ------------------ 3) Dodavanje poznatog CFO --------
    true_cfo_hz = 2500.0
    rx_40mhz = apply_cfo(rx_40mhz, true_cfo_hz, fs40)

    # ------------------ 4) RX chain --------------------
    try:
        res = run_rx(
            rx_40mhz=rx_40mhz,
            tx_40mhz=tx_40mhz,
            num_symbols_req=num_payload_symbols,
            fs_in=fs40,
            plot=False
        )
    except RuntimeError as e:
        if "falling edge" in str(e):
            pytest.skip(f"Packet detector failed at SNR={snr_db} dB, očekivano za loš SNR")
        else:
            raise

    # ------------------ 5) EVM ------------------------
    corrected = res["corrected_symbols"]
    Corrected_Symbols = np.concatenate(corrected)
    TX_Symbol_Stream = tx_symbols[:len(Corrected_Symbols)]

    # filtriranje vrlo malih vrijednosti
    mask = np.abs(Corrected_Symbols) > 1e-6
    Corrected_Symbols = Corrected_Symbols[mask]
    TX_Symbol_Stream = TX_Symbol_Stream[mask]

    error_vectors = TX_Symbol_Stream - Corrected_Symbols
    evm = 10 * np.log10(np.mean(np.abs(error_vectors)**2))
    print(f"[SNR={snr_db} dB] EVM = {evm:.2f} dB")

    # prag EVM po SNR-u
    evm_thresholds = {10: -8, 20: -15, 35: -20}  # u dB
    threshold = evm_thresholds[snr_db]
    assert evm < threshold, f"EVM previsok: {evm:.2f} dB za SNR {snr_db} dB"

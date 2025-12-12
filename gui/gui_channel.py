import sys
import os
import numpy as np
import tkinter as tk
from tkinter import ttk

# ===== PATH FIX =====
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

# ===== MATPLOTLIB (STABILNO SA TKINTEROM) =====
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ===== TX =====
from tx.OFDM_TX_802_11 import Transmitter80211a

# ===== CHANNEL =====
from channel.Channel_Model import Channel_Model
from channel.channel_settings import ChannelSettings
from channel.channel_mode import ChannelMode


def oznaci_okvir(ax, up_factor, num_ofdm_symbols, fs):
    STS_len = 16 * up_factor * 10
    LTS_len = 160 * up_factor
    OFDM_symbol_len = 64 * up_factor
    GI_len = 16 * up_factor

    sts_end = STS_len / fs
    lts_end = (STS_len + LTS_len) / fs
    payload_end = (STS_len + LTS_len +
                   num_ofdm_symbols * (OFDM_symbol_len + GI_len)) / fs

    ax.axvline(sts_end * 1e6, color='green', linestyle='--', label='Kraj STS')
    ax.axvline(lts_end * 1e6, color='orange', linestyle='--', label='Kraj LTS')
    ax.axvline(payload_end * 1e6, color='purple', linestyle='--', label='Kraj payload')

    for i in range(num_ofdm_symbols):
        gi_start = (STS_len + LTS_len +
                    i * (GI_len + OFDM_symbol_len)) / fs
        gi_end = gi_start + GI_len / fs
        ax.axvspan(gi_start * 1e6, gi_end * 1e6,
                   color='green', alpha=0.25)


class OFDMGui:
    def __init__(self, root):
        self.root = root
        root.title("OFDM Channel Demo")

        # ===== KONTROLE =====
        ctrl = ttk.Frame(root, padding=10)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)

        self.snr = tk.IntVar(value=20)
        self.multipath = tk.BooleanVar(value=True)
        self.taps = tk.IntVar(value=2)
        self.delay = tk.IntVar(value=10)

        # ---- SNR ----
        ttk.Label(ctrl, text="SNR [dB]").pack(anchor="w")

        self.snr_scale = ttk.Scale(
            ctrl, from_=0, to=40,
            variable=self.snr,
            orient="horizontal"
        )
        self.snr_scale.pack(fill=tk.X)

        # LABEL KOJI PRIKAZUJE SNR VRIJEDNOST
        self.snr_value_label = ttk.Label(
            ctrl, text=f"{self.snr.get()} dB"
        )
        self.snr_value_label.pack(anchor="e", pady=(0, 10))

        # automatsko ažuriranje labela kad se pomjera slider
        self.snr.trace_add(
            "write",
            lambda *args: self.snr_value_label.config(
                text=f"{self.snr.get()} dB"
            )
        )

        # ---- MULTIPATH ----
        ttk.Checkbutton(
            ctrl, text="Multipath",
            variable=self.multipath
        ).pack(pady=5)

        # ---- TAPS ----
        ttk.Label(ctrl, text="Broj tapova").pack(anchor="w")
        ttk.Spinbox(
            ctrl, from_=1, to=50,
            textvariable=self.taps
        ).pack(fill=tk.X)

        # ---- DELAY ----
        ttk.Label(ctrl, text="Delay spread [ns]").pack(anchor="w")
        ttk.Spinbox(
            ctrl, from_=1, to=500,
            textvariable=self.delay
        ).pack(fill=tk.X)

        ttk.Button(
            ctrl, text="Pokreni simulaciju",
            command=self.run
        ).pack(pady=10)

        # ===== FIGURA =====
        fig = Figure(figsize=(10, 4))
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.get_tk_widget().pack(
            side=tk.RIGHT, fill=tk.BOTH, expand=True
        )

    def run(self):
        self.ax.clear()

        # ===== ISTO KAO demo_channel.py =====
        num_ofdm_symbols = 2
        up_factor = 2

        tx = Transmitter80211a(
            num_ofdm_symbols=num_ofdm_symbols,
            bits_per_symbol=2,
            step=1,
            up_factor=up_factor,
            seed=42,
            plot=False
        )

        tx_sig, _ = tx.generate_frame()
        tx_sig = np.asarray(tx_sig).flatten()

        fs = 20e6 * up_factor
        t = np.arange(len(tx_sig)) / fs

        settings = ChannelSettings(
            sample_rate=fs,
            number_of_taps=self.taps.get(),
            delay_spread=self.delay.get() * 1e-9,
            snr_db=self.snr.get()
        )

        mode = ChannelMode(
            multipath=1 if self.multipath.get() else 0,
            thermal_noise=1
        )

        channel = Channel_Model(settings, mode)
        rx, _ = channel.apply(tx_sig)
        rx = np.asarray(rx).flatten()

        # ===== NORMALIZACIJA =====
        rx *= np.sqrt(np.mean(np.abs(tx_sig)**2)) / np.sqrt(np.mean(np.abs(rx)**2))

        self.ax.plot(t * 1e6, np.real(tx_sig), label="Tx – prije kanala")
        self.ax.plot(t * 1e6, np.real(rx), label="Rx – poslije kanala", alpha=0.8)

        oznaci_okvir(self.ax, up_factor, num_ofdm_symbols, fs)

        self.ax.set_title(
            f"SNR={self.snr.get()} dB | multipath={self.multipath.get()} | "
            f"taps={self.taps.get()} | delay={self.delay.get()} ns"
        )
        self.ax.set_xlabel("Vrijeme [µs]")
        self.ax.set_ylabel("Amplituda")
        self.ax.grid(True)
        self.ax.legend()

        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = OFDMGui(root)
    root.mainloop()

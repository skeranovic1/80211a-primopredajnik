import sys
import os
import numpy as np
import tkinter as tk
from tkinter import ttk
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tx.OFDM_mapper import Mapper_OFDM
from channel.Channel_Model import Channel_Model
from channel.channel_settings import ChannelSettings
from channel.channel_mode import ChannelMode

class ConstellationGuiOnePlot:
    """GUI za prikaz konstelacije QPSK simbola kroz različite kanale."""
    def __init__(self, root):
        """Inicijalizacija GUI-ja i početnih parametara."""
        self.root = root
        root.title("Constellation Demo – Ideal / AWGN / Multipath+AWGN")

        ctrl = ttk.Frame(root, padding=10)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)
        self.mode = tk.StringVar(value="ideal")
        self.snr = tk.IntVar(value=12)

        # Parametri za multipath kanal
        self.taps = tk.IntVar(value=2)
        self.delay = tk.IntVar(value=10)  # u ns
        self.max_points = tk.IntVar(value=2000)  # maksimalan broj tačaka u plotu

        ttk.Label(ctrl, text="Scenario").pack(anchor="w", pady=(0, 4))
        ttk.Radiobutton(ctrl, text="Ideal (Channel In)", variable=self.mode, value="ideal").pack(anchor="w")
        ttk.Radiobutton(ctrl, text="AWGN", variable=self.mode, value="awgn").pack(anchor="w")
        ttk.Radiobutton(ctrl, text="Multipath + AWGN", variable=self.mode, value="mp").pack(anchor="w")

        ttk.Separator(ctrl).pack(fill=tk.X, pady=10)

        #SNR slider
        ttk.Label(ctrl, text="SNR [dB]").pack(anchor="w")
        ttk.Scale(ctrl, from_=-5, to=20, variable=self.snr, orient="horizontal").pack(fill=tk.X)
        self.snr_value_label = ttk.Label(ctrl, text=f"{self.snr.get()} dB")
        self.snr_value_label.pack(anchor="e", pady=(0, 10))
        #automatska update labela kad se slider pomjeri
        self.snr.trace_add("write", lambda *args: self.snr_value_label.config(text=f"{self.snr.get()} dB"))

        ttk.Separator(ctrl).pack(fill=tk.X, pady=10)

        #Multipath parametri
        ttk.Label(ctrl, text="Multipath parametri").pack(anchor="w", pady=(0, 4))
        ttk.Label(ctrl, text="Broj tapova").pack(anchor="w")
        self.taps_spin = ttk.Spinbox(ctrl, from_=1, to=50, textvariable=self.taps)
        self.taps_spin.pack(fill=tk.X)
        ttk.Label(ctrl, text="Delay spread [ns]").pack(anchor="w")
        self.delay_spin = ttk.Spinbox(ctrl, from_=1, to=500, textvariable=self.delay)
        self.delay_spin.pack(fill=tk.X)

        ttk.Separator(ctrl).pack(fill=tk.X, pady=10)

        #Maksimalni broj tačaka u plotu
        ttk.Label(ctrl, text="Max tačaka za plot").pack(anchor="w")
        ttk.Spinbox(ctrl, from_=200, to=20000, increment=200, textvariable=self.max_points).pack(fill=tk.X)

        #Dugme za prikaz
        ttk.Button(ctrl, text="Prikaži", command=self.run).pack(pady=12)

        #Automatski update pri promjeni scenarija
        self.mode.trace_add("write", lambda *args: self._on_mode_change())

        #Figura
        fig = Figure(figsize=(8, 7))
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        #Priprema TX simbola (QPSK)
        self._prepare_tx()

        #Inicijalno enable/disable multipath kontrole
        self._update_mp_controls()

        #Inicijalni plot
        self.run()

    def _prepare_tx(self):
        """Generiše inicijalne TX simbole za QPSK i priprema sample rate."""
        np.random.seed(42)
        n_bits = 40000
        bits = np.random.randint(0, 2, n_bits)
        BitsPerSymbol = 2  # QPSK
        tx = Mapper_OFDM(bits, BitsPerSymbol, plot=False)
        self.tx = np.asarray(tx).reshape(-1)
        self.fs = 20e6  # sample rate za kanal

    def _subsample(self, x, nmax, seed=0):
        """Smanjuje broj tačaka za scatter plot radi brzine."""
        x = np.asarray(x).reshape(-1)
        if len(x) <= nmax:
            return x
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x), size=nmax, replace=False)
        return x[idx]

    def _apply_channel(self, tx_symbols, scenario, snr_db):
        """Primjenjuje odabrani kanal na TX simbole i vraća RX simbole."""
        if scenario == "ideal":
            return None

        settings = ChannelSettings(
            sample_rate=self.fs,
            number_of_taps=self.taps.get(),
            delay_spread=self.delay.get() * 1e-9,
            snr_db=snr_db
        )

        if scenario == "awgn":
            mode = ChannelMode(multipath=0, thermal_noise=1)
        else:  # "mp"
            mode = ChannelMode(multipath=1, thermal_noise=1)

        ch = Channel_Model(settings, mode)
        rx, _ = ch.apply(tx_symbols)
        return np.asarray(rx).reshape(-1)

    def _update_mp_controls(self):
        """Enable/disable multipath kontrole po potrebi."""
        self.taps_spin.configure(state="normal")
        self.delay_spin.configure(state="normal")

    def _on_mode_change(self):
        """Automatski update kontrole i plot kada se promijeni scenario."""
        self._update_mp_controls()
        self.run()

    def run(self):
        """Crtanje scatter plot konstelacije za izabrani scenario."""
        self.ax.clear()

        scenario = self.mode.get()
        snr_db = self.snr.get()
        nmax = self.max_points.get()

        #Smanjenje broja tačaka za brzinu prikaza
        tx_p = self._subsample(self.tx, nmax, seed=1)

        #Uvijek prikazujemo ulaz u kanal (ideal)
        self.ax.scatter(tx_p.real, tx_p.imag, s=8, label="Channel In")

        #Primjena kanala ako je odabran
        rx = self._apply_channel(self.tx, scenario, snr_db)
        if rx is not None:
            rx_p = self._subsample(rx, nmax, seed=2)
            if scenario == "awgn":
                label = "Channel Out | AWGN"
                title = f"AWGN | SNR={snr_db} dB"
            else:
                label = "Channel Out | Multipath + AWGN"
                title = f"Multipath + AWGN | SNR={snr_db} dB | taps={self.taps.get()} | delay={self.delay.get()} ns"

            self.ax.scatter(rx_p.real, rx_p.imag, s=8, alpha=0.8, label=label)
        else:
            title = "Ideal (Channel In)"

        #Crtanje osi i grid-a
        self.ax.axhline(0, color="black", linewidth=0.5)
        self.ax.axvline(0, color="black", linewidth=0.5)
        self.ax.grid(True)
        self.ax.set_aspect("equal", "box")
        self.ax.set_title(title)
        self.ax.set_xlabel("I")
        self.ax.set_ylabel("Q")
        self.ax.legend(loc="lower right")

        #Osvježi canvas
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = ConstellationGuiOnePlot(root)
    root.mainloop()

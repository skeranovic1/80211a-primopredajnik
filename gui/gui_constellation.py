import sys
import os
import numpy as np
import tkinter as tk
from tkinter import ttk
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tx.OFDM_TX_802_11 import Transmitter80211a
from channel.Channel_Model import Channel_Model
from channel.channel_settings import ChannelSettings
from channel.channel_mode import ChannelMode
from rx.RX_802_11_a import Receiver80211a

import sys
import os
import numpy as np
import tkinter as tk
from tkinter import ttk

# Matplotlib za GUI
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Importi tvojih modula
from tx.OFDM_TX_802_11 import Transmitter80211a
from channel.Channel_Model import Channel_Model
from channel.channel_settings import ChannelSettings
from channel.channel_mode import ChannelMode
from rx.RX_802_11_a import Receiver80211a

class ConstellationGui:
    def __init__(self, root):
        self.root = root
        self.root.title("802.11a Constellation Viewer")

        # --- Fiksni parametri sustava ---
        self.num_ofdm_symbols = 50
        self.up_factor = 2
        self.fs = 20e6 * self.up_factor
        self.fixed_taps = 5
        self.fixed_delay = 50e-9 

        # --- GUI Varijable ---
        self.view_type = tk.StringVar(value="TX")  
        self.modulation = tk.StringVar(value="QPSK")
        self.channel_type = tk.StringVar(value="AWGN") 
        self.rx_phase_mode = tk.StringVar(value="Before") 
        self.snr = tk.IntVar(value=15)

        self._setup_ui()
        self._update_all()

    def _setup_ui(self):
        # Glavni kontejner
        ctrl = ttk.Frame(self.root, padding=10)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)

        # 1. Mod odabira (TX / RX)
        ttk.Label(ctrl, text="Prikaz:", font=('Arial', 10, 'bold')).pack(anchor="w")
        ttk.Radiobutton(ctrl, text="TX (Idealni simboli)", variable=self.view_type, value="TX", 
                        command=self._on_view_change).pack(anchor="w")
        ttk.Radiobutton(ctrl, text="RX (Prijemnik)", variable=self.view_type, value="RX", 
                        command=self._on_view_change).pack(anchor="w")
        
        ttk.Separator(ctrl).pack(fill=tk.X, pady=10)

        # 2. Modulacija (Zajedničko)
        ttk.Label(ctrl, text="Modulacija:").pack(anchor="w")
        mod_combo = ttk.Combobox(ctrl, textvariable=self.modulation, 
                                 values=["BPSK", "QPSK", "16-QAM", "64-QAM"], state="readonly")
        mod_combo.pack(fill=tk.X, pady=5)
        mod_combo.bind("<<ComboboxSelected>>", lambda e: self._update_all())

        # 3. RX Specifični okvir
        self.rx_frame = ttk.LabelFrame(ctrl, text="Kanal & Faza (Samo za RX)", padding=10)
        self.rx_frame.pack(fill=tk.X, pady=10)

        ttk.Label(self.rx_frame, text="Model kanala:").pack(anchor="w")
        ttk.Radiobutton(self.rx_frame, text="AWGN", variable=self.channel_type, 
                        value="AWGN", command=self._update_all).pack(anchor="w")
        ttk.Radiobutton(self.rx_frame, text="AWGN + Multipath", variable=self.channel_type, 
                        value="MP", command=self._update_all).pack(anchor="w")

        ttk.Label(self.rx_frame, text="SNR [dB]:").pack(anchor="w", pady=(5,0))
        self.snr_scale = ttk.Scale(self.rx_frame, from_=0, to=30, variable=self.snr, 
                                   orient="horizontal", command=lambda s: self._update_all())
        self.snr_scale.pack(fill=tk.X)
        self.snr_label = ttk.Label(self.rx_frame, text=f"{self.snr.get()} dB")
        self.snr_label.pack(anchor="e")

        ttk.Separator(self.rx_frame).pack(fill=tk.X, pady=10)

        ttk.Label(self.rx_frame, text="Faza prikaza:").pack(anchor="w")
        ttk.Radiobutton(self.rx_frame, text="Prije korekcije", variable=self.rx_phase_mode, 
                        value="Before", command=self._update_all).pack(anchor="w")
        ttk.Radiobutton(self.rx_frame, text="Poslije korekcije", variable=self.rx_phase_mode, 
                        value="After", command=self._update_all).pack(anchor="w")

        # Matplotlib Figura
        self.fig = Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Inicijalno onemogući RX opcije ako je TX default
        self._on_view_change()

    def _on_view_change(self):
        state = "normal" if self.view_type.get() == "RX" else "disabled"
        # Isključi/uključi sve unutar RX frejma
        for child in self.rx_frame.winfo_children():
            try: child.configure(state=state)
            except: pass
        self._update_all()

    def _get_bits_per_symbol(self):
        mapping = {"BPSK": 1, "QPSK": 2, "16-QAM": 4, "64-QAM": 6}
        return mapping[self.modulation.get()]

    def _update_all(self):
        self.snr_label.config(text=f"{self.snr.get()} dB")
        
        # 1. Kreiranje TX signala
        tx_obj = Transmitter80211a(
            num_ofdm_symbols=self.num_ofdm_symbols,
            bits_per_symbol=self._get_bits_per_symbol(),
            up_factor=self.up_factor,
            seed=3,
            plot=False
        )
        # Uzimamo treći parametar - idealni simboli u frekv. domenu
        self.tx_signal_time, _, self.tx_symbols_fd = tx_obj.generate_frame()

        # 2. Logika za odabir podataka za plot
        if self.view_type.get() == "TX":
            data_to_plot = self.tx_symbols_fd.flatten()
            title = f"TX Konstelacija: {self.modulation.get()}"
        else:
            # Primjeni kanal i pokreni Receiver
            rx_signal_raw = self._apply_channel()
            
            rx = Receiver80211a(fs=self.fs, num_symbols=self.num_ofdm_symbols)
            rx.process_signal(rx_signal_raw, self.tx_signal_time)
            
            if self.rx_phase_mode.get() == "Before":
                # symbols_fd * eq_coefficient (prema tvom kodu)
                data_to_plot = (rx.symbols_fd * rx.eq_coefficient).flatten()
                title = "RX: Nakon ekvalizacije (prije CPE)"
            else:
                # corrected_symbols (nakon phase_correction)
                data_to_plot = rx.corrected_symbols.flatten()
                title = "RX: Nakon fazne korekcije (CPE)"

        self._plot_data(data_to_plot, title)

    def _apply_channel(self):
        settings = ChannelSettings(
            sample_rate=self.fs,
            number_of_taps=self.fixed_taps,
            delay_spread=self.fixed_delay,
            snr_db=self.snr.get()
        )
        is_mp = (self.channel_type.get() == "MP")
        mode = ChannelMode(multipath=is_mp, thermal_noise=True)
        ch = Channel_Model(settings, mode)
        rx_sig, _ = ch.apply(self.tx_signal_time)
        return np.asarray(rx_sig).flatten()

    def _plot_data(self, data, title):
        self.ax.clear()
        
        # Plotanje simbola
        self.ax.scatter(data.real, data.imag, s=12, alpha=0.7, edgecolors='none')
        
        # --- KLJUČNI DIO ZA FIKSNU VELIČINU ---
        # Postavljamo fiksne granice bez obzira na snagu signala
        # Za 64-QAM normalizirani simboli idu do cca 1.5, tako da je 2.0 sigurno.
        limit = 1.8 
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_aspect('equal', adjustable='box') # Kvadratični prikaz
        
        self.ax.set_title(title, pad=15)
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.axhline(0, color='black', lw=1)
        self.ax.axvline(0, color='black', lw=1)
        
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    # Postavi prozor na solidnu veličinu pri paljenju
    root.geometry("900x700")
    app = ConstellationGui(root)
    root.mainloop()
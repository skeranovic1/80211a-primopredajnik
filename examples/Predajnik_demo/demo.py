import tkinter as tk
from tkinter import ttk
from tx.OFDM_TX_802_11 import Transmitter80211a
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from tx.filters import half_band_upsample

class OFDMGUI:
    """
    GUI aplikacija za generisanje i prikaz OFDM signala IEEE 802.11a standarda.
    Koristi Transmitter80211a klasu za generisanje STS, LTS, payload i finalnog okvira.
    Omogućava odabir segmenta za prikaz i x-os u mikrosekundama.
    """

    def __init__(self, root):
        """
        Inicijalizacija GUI-ja:
        - root: Tkinter glavni prozor
        """
        self.root = root
        self.root.title("802.11a OFDM Generator Okvira")
        self.root.state('zoomed')  # Postavlja prozor na full-screen

        #Stil i padding
        self.font_style = ("Times New Roman", 14)
        self.padx = 5
        self.pady = 5

        #Kontrolni panel
        control_frame = tk.Frame(root, height=100)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        #Broj OFDM simbola
        tk.Label(control_frame, text="Broj OFDM simbola:", font=self.font_style).grid(row=0, column=0, padx=self.padx, pady=self.pady)
        self.num_ofdm_entry = tk.Entry(control_frame, width=6, font=self.font_style)
        self.num_ofdm_entry.insert(0, "3")
        self.num_ofdm_entry.grid(row=0, column=1, padx=self.padx, pady=self.pady)

        #Modulacija
        tk.Label(control_frame, text="Bita po simbolu:", font=self.font_style).grid(row=0, column=2, padx=self.padx, pady=self.pady)
        self.bps_entry = tk.Entry(control_frame, width=6, font=self.font_style)
        self.bps_entry.insert(0, "2")  # QPSK
        self.bps_entry.grid(row=0, column=3, padx=self.padx, pady=self.pady)

        #Faktor upsampliranje
        tk.Label(control_frame, text="Faktor upsamplovanja:", font=self.font_style).grid(row=0, column=4, padx=self.padx, pady=self.pady)
        self.up_entry = tk.Entry(control_frame, width=6, font=self.font_style)
        self.up_entry.insert(0, "2")
        self.up_entry.grid(row=0, column=5, padx=self.padx, pady=self.pady)

        #Sjeme za generator bita
        tk.Label(control_frame, text="Sjeme (seed):", font=self.font_style).grid(row=0, column=6, padx=self.padx, pady=self.pady)
        self.seed_entry = tk.Entry(control_frame, width=6, font=self.font_style)
        self.seed_entry.insert(0, "42")
        self.seed_entry.grid(row=0, column=7, padx=self.padx, pady=self.pady)

        #Sample rate za x-osu
        tk.Label(control_frame, text="Sample rate (Hz):", font=self.font_style).grid(row=0, column=8, padx=self.padx, pady=self.pady)
        self.fs_entry = tk.Entry(control_frame, width=10, font=self.font_style)
        self.fs_entry.insert(0, "20000000")
        self.fs_entry.grid(row=0, column=9, padx=self.padx, pady=self.pady)

        #Dropdown za odabir dijela okvira za prikaz
        tk.Label(control_frame, text="Prikaži segment:", font=self.font_style).grid(row=0, column=10, padx=self.padx, pady=self.pady)
        self.plot_option = ttk.Combobox(control_frame, values=["STS", "LTS", "Payload", "Finalni okvir"], font=self.font_style, width=15)
        self.plot_option.current(3)  # Default postavljen na finalni okvir
        self.plot_option.grid(row=0, column=11, padx=self.padx, pady=self.pady)

        #Dugme za generisanje i plotanje
        self.generate_btn = tk.Button(control_frame, text="Generiši okvir", font=("Times New Roman",16), bg="lightblue", command=self.generate_frame)
        self.generate_btn.grid(row=0, column=12, padx=10, pady=5)

        #Matplotlib canvas
        self.fig, self.ax = plt.subplots(figsize=(16,6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def generate_frame(self):
        """
        Funkcija koja generiše OFDM signal i prikazuje ga na canvas-u.
        - Učitava parametre sa GUI-ja
        - Generiše STS, LTS, payload i finalni okvir
        - Prikazuje I/Q komponente u mikrosekundama
        - Dodaje oznake STS, LTS, payload i GI samo za finalni okvir
        """
        #Učitavanje parametara
        num_ofdm_symbols = int(self.num_ofdm_entry.get())
        bps = int(self.bps_entry.get())
        up_factor = int(self.up_entry.get())
        seed = int(self.seed_entry.get())
        fs_base = float(self.fs_entry.get())
        fs = fs_base * up_factor  #upsamplovani sample rate

        #Kreiranje Transmittera
        tx = Transmitter80211a(
            num_ofdm_symbols=num_ofdm_symbols,
            bits_per_symbol=bps,
            up_factor=up_factor,
            seed=seed,
            plot=False
        )

        #Generisanje komponenti
        sts, lts = tx.generate_training_sequences()
        sts /= 64
        lts /= 64
        payload, _ = tx.generate_payload()
        final_frame, _ = tx.generate_frame()

        option = self.plot_option.get()

        #Upamplovanje odabranog segmenta
        if option == "STS":
            signal, _ = half_band_upsample(sts, up_factor=up_factor, N=31, plot=False)
        elif option == "LTS":
            signal, _ = half_band_upsample(lts, up_factor=up_factor, N=31, plot=False)
        elif option == "Payload":
            signal, _ = half_band_upsample(payload, up_factor=up_factor, N=31, plot=False)
        else:
            signal = final_frame

        #Dužine komponenti za oznake (samo za finalni okvir)
        STS_len = 16 * 10 * up_factor
        LTS_len = 160 * up_factor
        OFDM_symbol_len = 64 * up_factor
        GI_len = 16 * up_factor

        t = np.arange(len(signal)) / fs * 1e6

        #Plotanje
        self.ax.clear()
        self.ax.plot(t, np.real(signal), label='I komponenta')
        self.ax.plot(t, np.imag(signal), label='Q komponenta', linestyle='--')

        if option == "Finalni okvir":
            #Oznake krajeva STS, LTS, payload
            sts_end = STS_len / fs * 1e6
            lts_end = (STS_len + LTS_len) / fs * 1e6
            payload_end = len(signal) / fs * 1e6
            self.ax.axvline(sts_end, color='green', linestyle='--', label='Kraj STS')
            self.ax.axvline(lts_end, color='orange', linestyle='--', label='Kraj LTS')
            self.ax.axvline(payload_end, color='purple', linestyle='--', label='Kraj payload')

            #GI intervali za svaki OFDM simbol
            for i in range(num_ofdm_symbols):
                gi_start = (STS_len + LTS_len + i*(GI_len+OFDM_symbol_len)) / fs * 1e6
                gi_end = gi_start + GI_len / fs * 1e6
                self.ax.axvspan(gi_start, gi_end, color='green', alpha=0.3)

        self.ax.set_title(f"OFDM signal ({option})", fontsize=16)
        self.ax.set_xlabel("Vrijeme [µs]", fontsize=14)
        self.ax.set_ylabel("Amplituda", fontsize=14)
        self.ax.grid(True)
        self.ax.legend(fontsize=12)
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = OFDMGUI(root)
    root.mainloop()

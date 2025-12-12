import tkinter as tk
from tkinter import ttk
from tx.OFDM_TX_802_11 import Transmitter80211a
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class OFDMGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("802.11a OFDM Generator Okvira")
        self.root.state('zoomed')  # Maksimalni prozor

        # Font i padding za kontrolni panel
        self.font_style = ("Arial", 14)
        self.padx = 5
        self.pady = 5

        # Frame za kontrolu
        control_frame = tk.Frame(root, height=100)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Parametri TX-a
        tk.Label(control_frame, text="Broj OFDM simbola:", font=self.font_style).grid(row=0, column=0, padx=self.padx, pady=self.pady)
        self.num_ofdm_entry = tk.Entry(control_frame, width=6, font=self.font_style)
        self.num_ofdm_entry.insert(0, "1")
        self.num_ofdm_entry.grid(row=0, column=1, padx=self.padx, pady=self.pady)

        tk.Label(control_frame, text="Bita po simbolu:", font=self.font_style).grid(row=0, column=2, padx=self.padx, pady=self.pady)
        self.bps_entry = tk.Entry(control_frame, width=6, font=self.font_style)
        self.bps_entry.insert(0, "2")
        self.bps_entry.grid(row=0, column=3, padx=self.padx, pady=self.pady)

        tk.Label(control_frame, text="Faktor upsamplovanja:", font=self.font_style).grid(row=0, column=4, padx=self.padx, pady=self.pady)
        self.up_entry = tk.Entry(control_frame, width=6, font=self.font_style)
        self.up_entry.insert(0, "2")
        self.up_entry.grid(row=0, column=5, padx=self.padx, pady=self.pady)

        tk.Label(control_frame, text="Sjeme (seed):", font=self.font_style).grid(row=0, column=6, padx=self.padx, pady=self.pady)
        self.seed_entry = tk.Entry(control_frame, width=6, font=self.font_style)
        self.seed_entry.insert(0, "13")
        self.seed_entry.grid(row=0, column=7, padx=self.padx, pady=self.pady)

        tk.Label(control_frame, text="Prikaži signal:", font=self.font_style).grid(row=0, column=8, padx=self.padx, pady=self.pady)
        self.plot_option = ttk.Combobox(control_frame, values=["STS", "LTS", "Payload", "Finalni okvir"], font=self.font_style, width=15)
        self.plot_option.current(3)  # default: Finalni okvir
        self.plot_option.grid(row=0, column=9, padx=self.padx, pady=self.pady)

        tk.Label(control_frame, text="Sample rate (Hz):", font=self.font_style).grid(row=0, column=10, padx=self.padx, pady=self.pady)
        self.fs_entry = tk.Entry(control_frame, width=10, font=self.font_style)
        self.fs_entry.insert(0, "20000000")  # 20 MHz
        self.fs_entry.grid(row=0, column=11, padx=self.padx, pady=self.pady)

        # Dugme Generate veće
        self.generate_btn = tk.Button(control_frame, text="Generiši okvir", font=("Arial", 16), bg="lightblue", command=self.generate_frame)
        self.generate_btn.grid(row=0, column=12, padx=10, pady=5)

        # Canvas zauzima cijeli preostali prostor
        self.fig, self.ax = plt.subplots(figsize=(12,6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def generate_frame(self):
        num_ofdm = int(self.num_ofdm_entry.get())
        bps = int(self.bps_entry.get())
        up = int(self.up_entry.get())
        seed = int(self.seed_entry.get())
        option = self.plot_option.get()
        fs = float(self.fs_entry.get())

        # Kreiranje Transmittera
        tx = Transmitter80211a(num_ofdm_symbols=num_ofdm,
                            bits_per_symbol=bps,
                            up_factor=up,
                            seed=seed,
                            plot=False)

        # Generisanje STS, LTS, Payload i Finalnog okvira
        sts, lts = tx.generate_training_sequences()
        sts /= 64
        lts /= 64
        payload, symbols = tx.generate_payload()
        final_frame, _ = tx.generate_frame()

        # Odabir signala
        if option == "STS":
            data = sts
        elif option == "LTS":
            data = lts
        elif option == "Payload":
            data = payload
        else:
            data = final_frame

        # x-os u mikrosekundama (svaki signal pravilno)
        t = np.arange(len(data)) / fs 

        # Plot
        self.ax.clear()
        self.ax.plot(t, np.real(data), label='I komponenta')
        self.ax.plot(t, np.imag(data), label='Q komponenta', linestyle='--')
        self.ax.set_title(option, fontsize=16)
        self.ax.set_xlabel("Vrijeme [µs]", fontsize=14)
        self.ax.set_ylabel("Amplituda", fontsize=14)
        self.ax.legend(fontsize=12)
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = OFDMGUI(root)
    root.mainloop()

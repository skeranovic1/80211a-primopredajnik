import sys
import os
import numpy as np
import matplotlib.pyplot as plt
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QRadioButton, QLineEdit, QGroupBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from tx.OFDM_TX_802_11 import Transmitter80211a

class VisualPopup(QMainWindow):
    def __init__(self, title, plot_func, data):
        super().__init__()
        self.setWindowTitle(title)
        self.setGeometry(200, 200, 800, 600)
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)
        plot_func(self.ax, data)
        self.canvas.draw()

class STSPopup(QMainWindow):
    def __init__(self, tx_instance):
        super().__init__()
        self.tx = tx_instance
        self.setWindowTitle("Analiza STS Sekvence")
        self.setFixedSize(900, 700)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        controls = QHBoxLayout()
        self.rb1 = QRadioButton("Step 1")
        self.rb1.setChecked(True)
        self.rb05 = QRadioButton("Step 0.5")
        self.rb1.toggled.connect(lambda: self.update_plot(1) if self.rb1.isChecked() else None)
        self.rb05.toggled.connect(lambda: self.update_plot(0.5) if self.rb05.isChecked() else None)
        controls.addWidget(QLabel("Step:"))
        controls.addWidget(self.rb1)
        controls.addWidget(self.rb05)
        layout.addLayout(controls)
        self.fig, (self.ax_real, self.ax_imag) = plt.subplots(2, 1)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self.update_plot(1)

    def update_plot(self, step_val):
        from tx.short_sequence import get_short_training_sequence
        sts = get_short_training_sequence(step_val)
        self.ax_real.clear()
        self.ax_real.plot(sts.real, color='blue')
        self.ax_real.set_title(f"STS Real (Step={step_val})")
        self.ax_imag.clear()
        self.ax_imag.plot(sts.imag, color='red')
        self.ax_imag.set_title(f"STS Imag (Step={step_val})")
        self.fig.tight_layout()
        self.canvas.draw()

class LTSPopup(QMainWindow):
    def __init__(self, tx_instance):
        super().__init__()
        self.tx = tx_instance
        self.setWindowTitle("Analiza LTS Sekvence")
        self.setFixedSize(900, 700)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        controls = QHBoxLayout()
        self.rb1 = QRadioButton("Step 1")
        self.rb1.setChecked(True)
        self.rb05 = QRadioButton("Step 0.5")
        self.rb1.toggled.connect(lambda: self.update_plot(1) if self.rb1.isChecked() else None)
        self.rb05.toggled.connect(lambda: self.update_plot(0.5) if self.rb05.isChecked() else None)
        controls.addWidget(QLabel("Step:"))
        controls.addWidget(self.rb1)
        controls.addWidget(self.rb05)
        layout.addLayout(controls)
        self.fig, (self.ax_real, self.ax_imag) = plt.subplots(2, 1)
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self.update_plot(1)

    def update_plot(self, step_val):
        from tx.long_sequence import get_long_training_sequence
        lts = get_long_training_sequence(step_val)
        self.ax_real.clear()
        self.ax_real.plot(lts.real, color='blue')
        self.ax_real.set_title(f"LTS Real (Step={step_val})")
        self.ax_imag.clear()
        self.ax_imag.plot(lts.imag, color='red')
        self.ax_imag.set_title(f"LTS Imag (Step={step_val})")
        self.fig.tight_layout()
        self.canvas.draw()

class PayloadDeepDivePopup(QMainWindow):
    def __init__(self, tx_instance):
        super().__init__()
        self.tx = tx_instance
        self.setWindowTitle("Dubinska analiza: Realni dio signala")
        self.setFixedSize(1000, 1000)
        
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        
        # 6 subplota za 6 faza transformacije
        self.fig, self.axs = plt.subplots(6, 1, figsize=(8, 18))
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)
        
        self.draw_deep_dive()

    def draw_deep_dive(self):
        # Poziv tvoje metode
        payload, bits, symbols = self.tx.generate_payload()
        
        # Tvoji ispravni indeksi
        data_pos = np.array([6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31,
                             32,33,34,35,36,37,39,40,41,42,43,44,45,46,47,48,49,50,51,53,54,55,56,57])
        pilot_pos = np.array([11, 25, 38, 52])

        # 1. SIROVI PODACI
        self.axs[0].clear()
        self.axs[0].stem(bits[:48], linefmt='gray', markerfmt='bo', basefmt="black")
        self.axs[0].set_title("1. Sirova bit sekvenca (Ulaz)")

        # 2. MAPIRANI SIMBOLI (Realni dio)
        self.axs[1].clear()
        self.axs[1].stem(range(len(symbols[:48])), np.real(symbols[:48]), linefmt='C1-', markerfmt='C1o')
        self.axs[1].set_title("2. Simboli nakon Mapper-a (Realni dio)")

        # 3. POZICIJE PILOTA
        self.axs[2].clear()
        p_grid = np.zeros(64)
        p_grid[pilot_pos] = 1.0
        self.axs[2].stem(range(64), p_grid, linefmt='red', markerfmt='ro')
        self.axs[2].set_title("3. Pozicije pilota (Realni dio)")

        # 4. DATA NA POZICIJAMA
        self.axs[3].clear()
        f_grid_data = np.zeros(64, dtype=complex)
        f_grid_data[data_pos] = symbols[:48]
        self.axs[3].stem(range(64), np.real(f_grid_data), linefmt='C0-', markerfmt='C0o')
        self.axs[3].set_title("4. Data na IFFT indeksima (Realni dio)")

        # 5. DATA + PILOTI (Kompletan grid - Frekvencija)
        self.axs[4].clear()
        full_grid = f_grid_data.copy()
        full_grid[pilot_pos] = 1.0
        m, s, b = self.axs[4].stem(range(64), np.real(full_grid), linefmt='gray', markerfmt=' ')
        plt.setp(m, alpha=0.3); plt.setp(s, alpha=0.3)
        self.axs[4].stem(data_pos, np.real(full_grid[data_pos]), linefmt='C0-', markerfmt='C0o', label='Data')
        self.axs[4].stem(pilot_pos, np.real(full_grid[pilot_pos]), linefmt='red', markerfmt='ro', label='Piloti')
        self.axs[4].set_title("5. Kompletan grid (Frekvencijski domen)")

        # 6. KONAČAN SIMBOL (Vremenski domen - tačno jedan simbol)
        self.axs[5].clear()
        
        # Uzimamo tačno 80 uzoraka (jer je up_factor=1 u ovoj fazi)
        # Ali koristimo len(payload) ako želiš biti sigurna
        single_symbol_len = 80 
        final_signal = np.real(payload[:single_symbol_len])
        
        # Ovdje koristimo np.arange koji garantuje istu dužinu kao signal
        x_axis = np.arange(len(final_signal))
        
        markerline, stemlines, baseline = self.axs[5].stem(x_axis, final_signal, linefmt='purple')
        plt.setp(markerline, markersize=3)
        
        # Označavamo CP (prvih 16 uzoraka)
        self.axs[5].axvspan(0, 16, color='yellow', alpha=0.2, label='CP')
        self.axs[5].set_title(f"6. Vremenski domen: Jedan simbol ({len(final_signal)} uzoraka)")
        self.axs[5].legend(loc='upper right')

        self.fig.tight_layout()
        self.canvas.draw()

class PayloadDeepDivePopup(QMainWindow):
    def __init__(self, tx_instance, data): # Dodaj 'data' ovdje
        super().__init__()
        self.tx = tx_instance
        self.p_data = data # Sačuvaj podatke iz pipeline-a
        self.setWindowTitle("Dubinska analiza: Realni dio signala")
        self.setFixedSize(1000, 1000)
        
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        
        self.fig, self.axs = plt.subplots(6, 1, figsize=(8, 18))
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)
        
        self.draw_deep_dive()

    def draw_deep_dive(self):
        # VAŽNO: Koristimo već postojeće podatke umjesto da ponovo generišemo
        payload = self.p_data['payload']
        bits = self.p_data['bits']
        symbols = self.p_data['symbols']
        
        data_pos = np.array([6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31,
                             32,33,34,35,36,37,39,40,41,42,43,44,45,46,47,48,49,50,51,53,54,55,56,57])
        pilot_pos = np.array([11, 25, 38, 52])

        # Koristimo np.real() svuda da izbjegnemo ComplexWarning
        self.axs[0].stem(bits[:48], linefmt='gray', markerfmt='bo', basefmt="black")
        self.axs[0].set_title("1. Sirova bit sekvenca (Ulaz)")

        self.axs[1].stem(range(len(symbols[:48])), np.real(symbols[:48]), linefmt='C1-', markerfmt='C1o')
        self.axs[1].set_title("2. Simboli nakon Mapper-a (Realni dio)")

        p_grid = np.zeros(64)
        p_grid[pilot_pos] = 1.0
        self.axs[2].stem(range(64), p_grid, linefmt='red', markerfmt='ro')
        self.axs[2].set_title("3. Pozicije pilota (Realni dio)")

        f_grid_data = np.zeros(64, dtype=complex)
        f_grid_data[data_pos] = symbols[:48]
        self.axs[3].stem(range(64), np.real(f_grid_data), linefmt='C0-', markerfmt='C0o')
        self.axs[3].set_title("4. Data na IFFT indeksima (Realni dio)")

        full_grid = f_grid_data.copy()
        full_grid[pilot_pos] = 1.0
        self.axs[4].stem(data_pos, np.real(full_grid[data_pos]), linefmt='C0-', markerfmt='C0o', label='Data')
        self.axs[4].stem(pilot_pos, np.real(full_grid[pilot_pos]), linefmt='red', markerfmt='ro', label='Piloti')
        self.axs[4].set_title("5. Kompletan grid (Frekvencijski domen)")

        final_signal = np.real(payload[:80])
        self.axs[5].stem(np.arange(len(final_signal)), final_signal, linefmt='purple')
        self.axs[5].axvspan(0, 16, color='yellow', alpha=0.2, label='CP')
        self.axs[5].set_title("6. Vremenski domen: Jedan simbol (sa CP)")
        
        self.fig.tight_layout()
        self.canvas.draw()

class FilterPopup(QMainWindow):
    def __init__(self, tx_instance, data):
        super().__init__()
        self.tx = tx_instance
        self.p_data = data
        self.setWindowTitle("Analiza Half-Band Filtra")
        self.setFixedSize(1000, 900)
        
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        
        # Ovdje kreiramo grafike
        self.fig, self.axs = plt.subplots(4, 1, figsize=(8, 12))
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)
        
        self.draw_filter_process()

    def draw_filter_process(self):
        # Uzimanje podataka (np.real je ključan ovdje)
        packet_raw = np.real(self.p_data['full_frame'])
        filtered_signal = np.real(self.p_data['filtered_signal'])
        h = self.p_data['filter_h']
        up = self.tx.up_factor
        
        n_show = 80 
        
        # 1. Ulaz
        self.axs[0].clear()
        self.axs[0].stem(packet_raw[:n_show])
        self.axs[0].set_title("1. Ulaz u filter (Originalna brzina)")

        # 2. Izlaz
        self.axs[1].clear()
        self.axs[1].plot(filtered_signal[:n_show * up], color='red', marker='o', markersize=2)
        self.axs[1].set_title("2. Izlaz iz filtra (Upsampled & Filtered)")

        # 3. Impulsni odziv
        self.axs[2].clear()
        self.axs[2].stem(h, linefmt='C2-')
        self.axs[2].set_title("3. Impulsni odziv filtra (h)")

        # 4. Spektar filtra
        self.axs[3].clear()
        H_freq = np.fft.fftshift(np.fft.fft(h, 1024))
        f = np.linspace(-up/2, up/2, 1024)
        self.axs[3].plot(f, 20*np.log10(np.abs(H_freq) + 1e-6), color='purple')
        self.axs[3].set_title("4. Frekvencijska karakteristika filtra")
        
        self.fig.tight_layout()
        self.canvas.draw()

class MainTXDashboard(QMainWindow):
    def __init__(self, tx_instance):
        super().__init__()
        self.tx = tx_instance
        self.popups = []
        self.current_frame = np.array([]) # Ovdje gradimo signal korak po korak
        self.pipeline_data = {}           # Podaci za popup prozore
        
        self.setWindowTitle("802.11a Transmitter Step-by-Step Lab")
        self.setFixedSize(1200, 650)
        
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)
        
        # 1. Naslov
        header = QLabel("IEEE 802.11a Physical Layer Pipeline")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 24px; font-weight: bold; color: #2c3e50; margin: 10px;")
        self.main_layout.addWidget(header)

        # 2. Kontrole (Ova metoda je falila!)
        self.setup_controls()

        # 3. Pipeline dugmići
        self.block_layout = QHBoxLayout()
        self.btns = [] 
        self.setup_progressive_pipeline()
        self.main_layout.addLayout(self.block_layout)

        # 4. Finalni grafikon
        self.fig, self.ax = plt.subplots(figsize=(10, 3))
        self.canvas = FigureCanvas(self.fig)
        self.main_layout.addWidget(self.canvas)
        self.ax.set_title("Status: Waiting for STS generation...")
        self.fig.tight_layout()

    def setup_controls(self):
        """Metoda koja je uzrokovala AttributeError."""
        control_group = QGroupBox("Postavke")
        control_layout = QHBoxLayout()
        
        lbl = QLabel("Broj OFDM simbola:")
        self.symbol_input = QLineEdit(str(self.tx.num_ofdm_symbols))
        self.symbol_input.setFixedWidth(50)
        
        btn_reset = QPushButton("RESET PIPELINE")
        btn_reset.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; padding: 5px;")
        btn_reset.clicked.connect(self.reset_ui)
        
        control_layout.addWidget(lbl)
        control_layout.addWidget(self.symbol_input)
        control_layout.addStretch()
        control_layout.addWidget(btn_reset)
        control_group.setLayout(control_layout)
        self.main_layout.addWidget(control_group)

    def setup_progressive_pipeline(self):
        # Konfiguracija: Ime, Funkcija, Pastelna boja
        configs = [
            ("1. CREATE STS", self.step_sts, "#A2D2FF"),
            ("2. CREATE LTS", self.step_lts, "#BDE0FE"),
            ("3. GENERATE PAYLOAD", self.step_payload, "#FFC8DD"),
            ("4. HALF-BAND FILTER", self.step_filter, "#B9FBC0")
        ]

        for i, (name, func, color) in enumerate(configs):
            btn = QPushButton(name)
            btn.setFixedSize(220, 80)
            btn.clicked.connect(func)
            
            # Sačuvamo boju u atribut dugmeta da je možemo vratiti kasnije
            btn.setProperty("active_color", color) 
            
            if i > 0:
                btn.setEnabled(False)
                btn.setStyleSheet("background-color: #ecf0f1; color: #bdc3c7; border-radius: 12px;")
            else:
                btn.setStyleSheet(f"background-color: {color}; color: #444; font-weight: bold; border-radius: 12px;")
            
            self.btns.append(btn)
            self.block_layout.addWidget(btn)
            if i < len(configs)-1:
                lbl = QLabel("→")
                lbl.setStyleSheet("font-size: 20px; color: #bdc3c7;")
                self.block_layout.addWidget(lbl)

    def reset_ui(self):
        """Vraća sve na početak."""
        self.current_frame = np.array([])
        self.pipeline_data = {}
        self.ax.clear()
        self.ax.set_title("Status: Pipeline reset. Start with STS.")
        self.canvas.draw()
        
        for i, btn in enumerate(self.btns):
            if i > 0:
                btn.setEnabled(False)
                btn.setStyleSheet("background-color: #ecf0f1; color: #bdc3c7; border-radius: 12px;")
            else:
                btn.setEnabled(True)

    def unlock_next(self, index):
        if index < len(self.btns):
            btn = self.btns[index]
            color = btn.property("active_color")
            btn.setEnabled(True)
            btn.setStyleSheet(f"background-color: {color}; color: #444; font-weight: bold; border-radius: 12px;")

    def update_dashboard_plot(self, title):
        self.ax.clear()
        
        # Uzimamo samo realni dio prije samog plotanja
        # np.real() eliminiše ComplexWarning u potpunosti
        real_signal = np.real(self.current_frame)
        
        self.ax.plot(real_signal, color='#2c3e50', linewidth=0.8)
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.2)
        
        # Postavljamo fiksne granice y-ose radi stabilnosti prikaza (opciono)
        if len(real_signal) > 0:
            margin = 0.1
            self.ax.set_ylim(np.min(real_signal) - margin, np.max(real_signal) + margin)
            
        self.canvas.draw()

    def step_sts(self):
        from tx.short_sequence import get_short_training_sequence
        sts = np.real(get_short_training_sequence(1)) / 64
        self.pipeline_data['sts_raw'] = sts
        self.current_frame = sts
        self.update_dashboard_plot("1. STS Generated")
        self.unlock_next(1)
        self.open_sts_visualizer()

    def step_lts(self):
        from tx.long_sequence import get_long_training_sequence
        lts = np.real(get_long_training_sequence(1)) / 64
        self.pipeline_data['lts_raw'] = lts
        self.current_frame = np.concatenate([self.current_frame, lts])
        self.update_dashboard_plot("2. STS + LTS Joined")
        self.unlock_next(2)
        self.open_lts_visualizer()

    def step_payload(self):
        try:
            self.tx.num_ofdm_symbols = int(self.symbol_input.text())
        except: pass
        payload, bits, symbols = self.tx.generate_payload()
        self.pipeline_data.update({'bits': bits, 'symbols': symbols, 'payload': payload})
        self.current_frame = np.concatenate([self.current_frame, np.real(payload)])
        self.update_dashboard_plot("3. Full Frame (Pre-filter)")
        self.unlock_next(3)
        self.open_payload_visualizer()

    def step_filter(self):
        # 1. Prvo spremimo trenutni spojeni frejm u rječnik pod ključem koji FilterPopup traži
        self.pipeline_data['full_frame'] = self.current_frame
        
        # 2. Pozivamo filtriranje
        filtered, h = self.tx.apply_filter(self.current_frame)
        
        # 3. Spremamo filtrirane d
        self.pipeline_data.update({'filtered_signal': filtered, 'filter_h': h})
        
        # 4. Ažuriramo glavni prikaz i otvaramo popup
        self.current_frame = filtered
        self.update_dashboard_plot("4. FINAL OUTPUT (Filtered & Upsampled)")
        self.open_filter_visualizer()

    # --- POPUP POZIVI (Prosljeđuju pipeline_data) ---
    def open_sts_visualizer(self):
        self.win = STSPopup(self.tx) # Ili STSPopup(self.tx, self.pipeline_data['sts_raw']) zavisno od tvoje klase
        self.popups.append(self.win); self.win.show()

    def open_lts_visualizer(self):
        self.win = LTSPopup(self.tx)
        self.popups.append(self.win); self.win.show()


    def open_payload_visualizer(self):
        # Proslijeđujemo cijeli rječnik podataka
        self.win = PayloadDeepDivePopup(self.tx, self.pipeline_data)
        self.popups.append(self.win)
        self.win.show()

    def open_filter_visualizer(self):
        # Ovdje je bila greška - falio je argument 'self.pipeline_data'
        self.win = FilterPopup(self.tx, self.pipeline_data)
        self.popups.append(self.win)
        self.win.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    tx = Transmitter80211a(num_ofdm_symbols=5, bits_per_symbol=2, up_factor=2)
    gui = MainTXDashboard(tx)
    gui.show()
    sys.exit(app.exec_())
import numpy as np
import matplotlib.pyplot as plt


def plot_time_domain_tx(signal, fs, title_prefix="TX signal"):
    """
    TG1-57:
    Vizualizacija realnog i imaginarnog dijela TX signala u vremenskom domenu.
    """
    n = len(signal)
    t = np.arange(n) / fs

    plt.figure(figsize=(10, 5))
    plt.plot(t, np.real(signal), label="Realni dio (I)")
    if np.iscomplexobj(signal):
        plt.plot(t, np.imag(signal), "--", label="Imaginarni dio (Q)")
    plt.xlabel("Vrijeme [s]")
    plt.ylabel("Amplituda")
    plt.title(f"{title_prefix} – vremenski domen")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_spectrum_tx(signal, fs, title_prefix="TX signal"):
    """
    TG1-56:
    FFT spektar TX signala u frekvencijskom domenu (jednostrani, u dB).
    """
    n = len(signal)

    # FFT
    S = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, d=1 / fs)

    # Uzmemo samo pozitivne frekvencije (jednostrani spektar)
    mask = freqs >= 0
    freqs_pos = freqs[mask]
    mag = np.abs(S[mask])

    # Pretvaranje u dB (mala konstanta da izbjegnemo log(0))
    mag_db = 20 * np.log10(mag + 1e-12)

    plt.figure(figsize=(10, 5))
    plt.plot(freqs_pos, mag_db)
    plt.xlabel("Frekvencija [Hz]")
    plt.ylabel("Magnituda [dB]")
    plt.title(f"{title_prefix} – FFT spektar")
    plt.grid(True)
    plt.tight_layout()

def plot_constellation_tx(symbols, title="TX signal - konstelacioni dijagram (QPSK)"):
    """
    TG1-55:
    Konstelacioni dijagram TX simbola (npr. QPSK).
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(symbols), np.imag(symbols), s=10)

    # Ose kroz nulu radi lakše čitanja
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.axvline(0, color="gray", linewidth=0.8)

    plt.xlabel("I komponenta")
    plt.ylabel("Q komponenta")
    plt.title(title)
    plt.grid(True)
    plt.gca().set_aspect("equal", "box")  # da bude krug -> kvadrat
    plt.tight_layout()


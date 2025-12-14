import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tx.utilities import spektar, plot_konstelaciju


def test_coverage_spektar_executes_lines():
    # samo da "prođe" kroz N=len, fft, fftfreq, abs i plt.plot
    fs = 1000.0
    x = np.random.randn(256) + 1j * np.random.randn(256)
    spektar(x, fs, label="cov")


def test_coverage_plot_konstelaciju_executes_lines():
    # samo da "prođe" kroz figure, scatter, axhline/axvline, label/title/grid/aspect/tight_layout
    symbols = np.array([1+1j, -1+1j, -1-1j, 1-1j], dtype=complex)
    plot_konstelaciju(symbols, "cov")
    plt.close("all")

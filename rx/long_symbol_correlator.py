import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def long_symbol_correlator(long_training_symbol,rx_waveform, falling_edge_position):
    """
    Detektuje poziciju Long Training Symbol (LTS) u primljenom OFDM signalu koristeći
    klizni cross-korelator sa sign-normalizovanom verzijom LTS-a.

    Funkcija koristi 64-uzorka pomični prozor (shift register) i kvantizovanu
    verziju LTS-a (±1 ± j) za procjenu cross-korelacije. Na osnovu maksimalne vrednosti
    izlaza korelatora unutar definisanog pretražnog prozora ('falling_edge_position')
    detektuje poziciju LTS-a sa preciznošću od jednog uzorka.

    Parametri
    long_training_symbol : array_like
            Kompleksni Long Training Symbol (LTS) koji se koristi kao referenca
            za cross-korelaciju. Može sadržavati samo jedan OFDM LTS simbol bez CP.
    rx_waveform : array_like
            Kompleksni primljeni signal u kojem se traži LTS.
    falling_edge_position : int
            Indeks približne početne pozicije paketa (crude timing reference) oko koje
            se traži LTS peak. Omogućava ograničenje pretražnog prozora.

    Povratna vrijednost
    lt_peak_value : complex
            Vrednost cross-korelacije na detektovanom LTS peak-u (kompleksna amplitude i faza).
    lt_peak_position : int
            Indeks u 'rx_waveform' gde je detektovan peak cross-korelacije, tj. pozicija
            LTS simbola.
    output_long : ndarray
            Niz kompleksnih vrednosti cross-korelacije kroz ceo prijemni signal.
            Može se koristiti za vizualizaciju i dalju analizu.
    """
    #Normalizovani LTS 
    L = np.sign(np.real(long_training_symbol)) + \
        1j * np.sign(np.imag(long_training_symbol))

    rx_len = len(rx_waveform)

    output_long = np.zeros(rx_len, dtype=complex)
    lt_peak_value = 0 + 0j
    lt_peak_position = 0

    cross_correlator = np.zeros(64, dtype=complex)

    for i in range(rx_len):

        # Shift registar
        cross_correlator[1:] = cross_correlator[:-1]
        cross_correlator[0] = rx_waveform[i]

        #Kros-korelacija
        output = np.dot(cross_correlator, np.conj(L[::-1]))
        output_long[i] = output

        # Search window za LTS
        if (i > falling_edge_position + 54) and \
           (i < falling_edge_position + 54 + 64):

            if abs(output) > abs(lt_peak_value):
                lt_peak_value = output
                lt_peak_position = i

    return lt_peak_value, lt_peak_position, output_long

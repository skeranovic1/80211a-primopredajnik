import numpy as np
from rx.detection import packet_detector


def gruba_vremenska_sinhronizacija(rx_signal, delay_correction=12):
    """
    Gruba vremenska sinhronizacija 802.11a signala.

    Detektuje kraj short training sekvence (STS) koristeći packet detector
    i vraća indeks početka korisnog dijela paketa.

    """

    rx_signal = np.asarray(rx_signal)

    _, _, falling_edge, _ = packet_detector(rx_signal)

    if falling_edge is None:
        raise RuntimeError("Paket nije detektovan – gruba vremenska sinhronizacija nije uspjela.")

    start_index = falling_edge - delay_correction

    if start_index < 0:
        start_index = 0

    return start_index

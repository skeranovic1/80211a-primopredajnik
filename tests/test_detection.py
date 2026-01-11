import numpy as np
import pytest
from rx.detection import packet_detector  

def test_raises_value_error_for_short_input():
    """Provjerava da se baca ValueError ako je ulazni signal kraći od minimalno dozvoljene dužine."""
    rx = np.ones(100, dtype=np.complex128)  
    with pytest.raises(ValueError):
        packet_detector(rx)

def test_raises_type_error_for_real_input():
    """Provjerava da se baca TypeError ako ulazni signal nije kompleksnog tipa."""
    rx = np.ones(500)  # realni signal
    with pytest.raises(TypeError):
        packet_detector(rx)

def test_output_shapes_and_types():
    """Provjerava da su izlazni nizovi ispravnih dimenzija i očekivanih tipova podataka."""
    rx = (np.random.randn(500) + 1j * np.random.randn(500)).astype(np.complex128)

    comparison_ratio, packet_flag, _, autocorr = packet_detector(rx)

    assert comparison_ratio.shape == rx.shape
    assert packet_flag.shape == rx.shape
    assert autocorr.shape == rx.shape

    assert comparison_ratio.dtype == float
    assert packet_flag.dtype == int
    assert autocorr.dtype == np.complex128

def test_noise_only_no_packet_detected():
    """Provjerava da se na čistom šumu ne detektuje paket niti falling edge."""
    np.random.seed(0)
    rx = (np.random.randn(600) + 1j * np.random.randn(600)) * 0.1

    _, packet_flag, falling_edge, _ = packet_detector(rx)

    assert np.all(packet_flag == 0)
    assert falling_edge is None

def test_packet_detection_on_periodic_signal():
    """Provjerava da se periodični STS signal ispravno detektuje kao paket i da se pronađe falling edge."""
    np.random.seed(1)

    noise = (np.random.randn(200) + 1j * np.random.randn(200)) * 0.05

    sts = np.exp(1j * 2 * np.pi * np.arange(16) / 16)
    sts_repeated = np.tile(sts, 25)  # 400 uzoraka

    rx = np.concatenate([noise, sts_repeated, noise])

    _, packet_flag, falling_edge, _ = packet_detector(rx)

    assert np.any(packet_flag == 1)
    assert falling_edge is not None
    assert 200 < falling_edge < 700

def test_hysteresis_behavior():
    """Provjerava da histereza sprječava višestruka lažna uključivanja i isključivanja detekcije."""
    sts = np.exp(1j * 2 * np.pi * np.arange(16) / 16)
    rx = np.tile(sts, 30)  # čisti paket

    rx = np.pad(rx, (200, 200))  # šum implicitno nula

    _, packet_flag, _, _ = packet_detector(rx)

    transitions = np.diff(packet_flag)
    rising_edges = np.sum(transitions == 1)
    falling_edges = np.sum(transitions == -1)

    assert rising_edges <= 1
    assert falling_edges <= 1

def test_no_nan_or_inf_in_outputs():
    """Provjerava da funkcija ne proizvodi NaN ili Inf vrijednosti u numeričkim izlazima."""
    rx = (np.random.randn(800) + 1j * np.random.randn(800))

    comparison_ratio, _, _, autocorr = packet_detector(rx)

    assert not np.any(np.isnan(comparison_ratio))
    assert not np.any(np.isinf(comparison_ratio))
    assert not np.any(np.isnan(autocorr))

def test_minimum_length_input_allowed():
    """Provjerava da je ulaz tačno minimalne dozvoljene dužine prihvaćen bez greške."""
    rx = (np.random.randn(400) + 1j * np.random.randn(400))
    comparison_ratio, _, _, _ = packet_detector(rx)

    assert len(comparison_ratio) == 400

def test_packet_without_falling_edge():
    """Provjerava da se falling edge ne vraća ako paket traje do kraja signala."""
    sts = np.exp(1j * 2 * np.pi * np.arange(16) / 16)
    rx = np.tile(sts, 40)  # paket do kraja signala

    _, packet_flag, falling_edge, _ = packet_detector(rx)

    assert np.any(packet_flag == 1)
    assert falling_edge is None

def test_multiple_packets_last_falling_edge_returned():
    """Provjerava da se u signalu sa više paketa vraća indeks posljednjeg falling edge-a."""
    sts = np.exp(1j * 2 * np.pi * np.arange(16) / 16)

    rx = np.concatenate([
        np.zeros(200),
        np.tile(sts, 20),
        np.zeros(100),
        np.tile(sts, 20),
        np.zeros(200)
    ])

    _, _, falling_edge, _ = packet_detector(rx)

    assert falling_edge is not None
    assert falling_edge > 300

def test_all_zero_input():
    """Provjerava da nulti signal ne uzrokuje detekciju paketa niti numeričke probleme."""
    rx = np.zeros(500, dtype=np.complex128)

    comparison_ratio, packet_flag, falling_edge, _ = packet_detector(rx)

    assert np.all(comparison_ratio == 0)
    assert np.all(packet_flag == 0)
    assert falling_edge is None
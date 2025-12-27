import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tx.long_sequence import get_long_training_sequence
from rx.long_symbol_correlator import long_symbol_correlator

def channel_estimation(rx_waveform, lt_peak_pos):

    """
    Izračunava procjenu kanala i koeficijenata equalizera na osnovu Long Training Symbol (LTS).
    
    Funkcija izdvaja dva uzastopna LTS simbola iz primljenog signala, vrši njihovo usrednjavanje
    za smanjenje uticaja šuma, i kroz FFT analizu dobija frekvencijski odziv kanala. Na osnovu
    odnosa primljenih i idealnih tonova na 52 data+pilot subcarriera izračunava se kompleksni
    frekvencijski odziv kanala i koeficijenti equalizera.

    Parameters
    ----------
    rx_waveform : array_like
            Kompleksni primljeni signal u vremenskom domenu.
    lt_peak_pos : int
            Indeks početka LTS simbola u 'rx_waveform' (tačan timing pozicija dobijen LTS korrelacijom).

    Returns
    -------
    channel_est : ndarray
            Kompleksni frekvencijski odziv kanala na 52 data+pilot subcarriera.
            Dimenzija (52,) - sadrži komponente za sve aktivne OFDM tone.
    equalizer_coeffs : ndarray
            Koeficijenti equalizera (recipročne vrijednosti channel_est) za kompenzaciju
            amplitude i faze uvedenih kanalom. Dimenzija (52,).
    
    Notes
    -----
    - Pretpostavlja 802.11a/g/n OFDM format sa 64-point FFT i 52 aktivna subcarriera
    - DC subcarrier (indeks 0) i guard band (indeksi 27-37) se izostavljaju
    """
    # Izvlacenje oba LTS simbola i prosjek
    first_lts = rx_waveform[lt_peak_pos-64 : lt_peak_pos]
    second_lts = rx_waveform[lt_peak_pos : lt_peak_pos+64]
    averaged_lts = 0.5*first_lts + 0.5*second_lts
    
    # FFT
    lts_fft = np.fft.fft(averaged_lts, n=64)
    rx_positive_tones = lts_fft[1:27]
    rx_negative_tones = lts_fft[38:64]
    rx_tones = np.concatenate([rx_negative_tones, rx_positive_tones])
    
    # Ideal tones
    all_tones = get_long_training_sequence()
    ideal_fft = np.fft.fft(all_tones[32+64:32+128], n=64)
    ideal_positive_tones = ideal_fft[1:27]
    ideal_negative_tones = ideal_fft[38:64]
    ideal_tones = np.concatenate([ideal_negative_tones, ideal_positive_tones])
    
    # Channel estimate
    channel_est = rx_tones / ideal_tones
    equalizer_coeffs = 1 / channel_est
    return channel_est, equalizer_coeffs

def equalize_ofdm_symbol(ofdm_symbol, equalizer_coeffs, n_fft=None):
    
    """
    Primjenjuje frekvencijski equalizer na pojedinačni OFDM simbol.
    
    Funkcija vrši FFT transformaciju vremenskog OFDM simbola u frekvencijski domen,
    primjenjuje kompleksne koeficijente equalizera na odgovarajuće subcarriere
    (uključujući data i pilot tone-ove), i na taj način kompenzuje amplitude i fazne
    distorzije kanala. DC subcarrier i guard band ostaju nepromijenjeni.

    Parameters
    
    ofdm_symbol : array_like
            Kompleksni OFDM simbol u vremenskom domenu (bez Cyclic Prefix-a).
            Dužina treba da odgovara n_fft ili da bude kraća (zero-padding).
    equalizer_coeffs : array_like
            Kompleksni koeficijenti equalizera za 52 data+pilot subcarriera.
    n_fft : int, optional
            Broj FFT tačaka (default: len(ofdm_symbol)). Koristi se za zero-padding
            ako je simbol kraći od FFT veličine.

    Returns
    
    equalized_symbol : ndarray
            Kompleksni frekvencijski sadržaj equaliziranog OFDM simbola.
            Dimenzija (n_fft,) - sadrži FFT koeficijente svih subcarriera.
    """

    if n_fft is None:
        n_fft = len(ofdm_symbol)

    # FFT simbola
    symbol_fft = np.fft.fft(ofdm_symbol, n=n_fft)

    # Mapiranje equalizer koeficijenata na 64 subcarriera
    eq_full = np.ones(n_fft, dtype=complex)

    # Data+pilot subcarrier-i u 802.11a (0=DC, 1-26 + 38-63 = data+pilot, 27-37 null)
    data_idx = np.hstack([np.arange(1, 27), np.arange(38, 64)])

    if len(equalizer_coeffs) != len(data_idx):
        raise ValueError(f"Equalizer length {len(equalizer_coeffs)} ne odgovara broju data subcarrier-a {len(data_idx)}")

    eq_full[data_idx] = equalizer_coeffs

    # Equalizacija
    equalized_symbol = symbol_fft * eq_full
    return equalized_symbol

def ofdm_eq(rx_signal, equalizer_coeffs, symbol_len=64):
    """
    Equalizuje kompletan OFDM paket simbol-po-simbol koristeći zadate koeficijente equalizera.
    
    Funkcija dijeli primljeni signal na pojedinačne OFDM simbole (pretpostavljajući
    da je Cyclic Prefix već uklonjen), za svaki simbol poziva 'equalize_ofdm_symbol'
    i vraća listu frekvencijskih reprezentacija svih equaliziranih simbola.

    Parameters
    
    rx_signal : array_like
            Kompleksni primljeni signal u vremenskom domenu (bez Cyclic Prefix-a).
            Dužina treba da bude višekratnik od 'symbol_len'.
    equalizer_coeffs : array_like
            Koeficijenti equalizera za 52 data+pilot subcarriera. Isti koeficijenti
            se primjenjuju na sve OFDM simbole u paketu.
    symbol_len : int, optional
            Dužina OFDM simbola bez CP (default: 64 za 802.11a/g/n).

    Returns
    
    equalized_symbols : list of ndarray
            Lista frekvencijskih reprezentacija equaliziranih OFDM simbola.
            Svaki element je kompleksni niz dužine 'symbol_len' koji sadrži
            FFT koeficijente subcarriera za dati simbol.

    """
    num_symbols = len(rx_signal) // symbol_len
    equalized_symbols = []

    for i in range(num_symbols):
        start = i * symbol_len
        symbol = rx_signal[start:start+symbol_len]
        equalized_fft = equalize_ofdm_symbol(symbol, equalizer_coeffs, n_fft=symbol_len)
        equalized_symbols.append(equalized_fft)

    return equalized_symbols
